"""
ncp_loader.py - NCP file discovery.

classify_extracted_ncp_files reads a flat input/ncp*.zip (extract_ncp_zip first) into one pool of files per profile
(NcpPools), whatever the TSO of a file; build_ncp_files reads an Instance/ folder
({Country}/NetworkCode/cimxml/{Country}_{profile}.xml; bare country names, display names carry a "TSO-" prefix).
Required profiles AE, CO, RA (none at all -> ValueError); optional SHS, SIS, ER (missing -> logged once; PATL then falls
back to the CGM static limits).
"""

from __future__ import annotations

import logging
import re
import shutil
from pathlib import Path
from typing import Optional

from pre_milp import ncp_header

log = logging.getLogger(__name__)

REQUIRED_PROFILES = {"AE", "CO", "RA"}
# ER (EquipmentReliability): GeneratingUnit start-up / shut-down costs, the fallback of the
# SIS UnitCostSchedule values
OPTIONAL_PROFILES = {"SHS", "SIS", "ER"}
ALL_PROFILES = REQUIRED_PROFILES | OPTIONAL_PROFILES

# what a missing optional profile means (logged per TSO); ER is commonly absent, so info only
_MISSING_OPTIONAL_PROFILE = {
    "SHS": (logging.WARNING, "SHS fallback to CGM static value"),
    "SIS": (logging.WARNING, "no power bids, no GLSK, no unit costs from SIS"),
    "ER": (logging.INFO, "no GeneratingUnit start-up/shut-down costs from ER"),
}


# profiles of the NCP 2.2 legacy form that are never read (one log line per profile and run)
UNUSED_PROFILES = ("SSI", "IAM", "PS")


def _log_missing_optional_profiles(tso_display: str, present_profiles) -> None:
    for profile in sorted(OPTIONAL_PROFILES - set(present_profiles)):
        level, meaning = _MISSING_OPTIONAL_PROFILE[profile]
        log.log(level, "%s: optional profile %s not found — %s", tso_display, profile, meaning)

# NCP content that is not parsed and is skipped without a warning: *_SSI.xml (Steady State
# Instruction, a different profile from *_SIS.xml) and Org-NineRealms_CD_OR.xml (object registry).
_KNOWN_IRRELEVANT_EXACT_NAMES = {"Org-NineRealms_CD_OR.xml"}


def _classify_ncp_zip_entry(filename: str, known_tsos: set[str]) -> tuple[str, Optional[str], Optional[str]]:
    """Classify one file name of a flat input/ncp*.zip: (category, tso or None, profile or None),
    category one of "recognized", "organisation", "known_irrelevant", "unrecognized".
    known_tsos: bare country names (e.g. "Belgovia"), not "TSO-" display names."""
    if filename == "Org-NineRealms_CD.xml":
        return ("organisation", None, None)

    if filename in _KNOWN_IRRELEVANT_EXACT_NAMES or filename.endswith("_SSI.xml"):
        return ("known_irrelevant", None, None)

    if not filename.endswith(".xml") or "_" not in filename:
        return ("unrecognized", None, None)

    stem = filename[: -len(".xml")]

    # match by TSO-name prefix: suffixes may contain underscores ("Svedala_RAS_Jotunheim_agreed.xml")
    for country in known_tsos:
        prefix = f"{country}_"
        if not stem.startswith(prefix):
            continue
        profile = stem[len(prefix):]
        if profile in ALL_PROFILES:
            return ("recognized", country, profile)
        # a known TSO's file of a profile that is not parsed (ER, PS, RAS, RAS_FAP, OR, AS, ...)
        return ("known_irrelevant", None, None)

    country, _, profile = stem.rpartition("_")
    if "-" in country:
        # "Country-Neighbour" prefix: a cross-TSO contingency export, not parsed
        return ("known_irrelevant", None, None)

    return ("unrecognized", None, None)


def _find_organisation_file(extract_dir: Path) -> Optional[Path]:
    """The Organisation profile: Org-NineRealms_CD.xml, else a file whose header is the CD profile."""
    named = extract_dir / "Org-NineRealms_CD.xml"
    if named.is_file():
        return named
    for path in sorted(extract_dir.rglob("*.xml")):
        header = ncp_header.read_header(path)
        if header is not None and header.profile == "CD":
            return path
    return None


def _classify_ncp_file(path: Path, known_tsos: set[str]) -> tuple[str, Optional[str], Optional[str]]:
    """Classify one extracted file: (category, TSO display name or None, profile or None), categories as in
    _classify_ncp_zip_entry. The profile comes from the file header. With known_tsos (an organisation file) the TSO is
    the name prefix "<Country>_" as before; without, it is the creator EIC of the header. A file whose header names no
    profile is classified by its name (the form before the header rule)."""
    header = ncp_header.read_header(path) if path.suffix == ".xml" else None
    if header is None or header.profile is None:
        category, country, profile = _classify_ncp_zip_entry(path.name, known_tsos)
        return (category, f"TSO-{country}" if country else None, profile)
    profile = header.profile
    if profile == "CD":
        return ("organisation", None, None)
    if profile == "OR":
        return ("known_irrelevant", None, None)
    if known_tsos:
        stem = path.name[: -len(".xml")]
        for country in known_tsos:
            if stem.startswith(f"{country}_"):
                if profile in ALL_PROFILES:
                    return ("recognized", f"TSO-{country}", profile)
                return ("known_irrelevant", None, profile)
        category, country, _ = _classify_ncp_zip_entry(path.name, known_tsos)
        return (category, None, profile if category == "known_irrelevant" else None)
    if profile not in ALL_PROFILES:
        return ("known_irrelevant", None, profile)
    if header.creator is None:
        log.warning("%s: no TSO read from the header (dcterms:creator %s)", path.name, header.creator_problem or "missing")
        return ("unrecognized", None, None)
    return ("recognized", header.creator, profile)


def clear_directory(directory: Path) -> None:
    """Remove everything inside `directory` (a generated extraction folder); a missing folder is left alone."""
    if not directory.is_dir():
        return
    for entry in directory.iterdir():
        if entry.is_dir():
            shutil.rmtree(entry)
        else:
            entry.unlink()


def extract_ncp_zip(ncp_zip_path: Path, extract_dir: Path, require_organisation: bool = True) -> Optional[Path]:
    """Extract an input/ncp*.zip and return the path of the Organisation profile.

    Runs before classify_extracted_ncp_files, which needs the tso_name_map built from that
    profile. Duplicate base names are checked on the zip's entry list before extraction
    (extractall would silently overwrite one with the other).
    Raises FileNotFoundError (no zip, or no Organisation profile unless require_organisation is
    False, which returns None then) or ValueError (duplicate).
    """
    import zipfile

    if not ncp_zip_path.exists():
        raise FileNotFoundError(f"NCP zip not found: {ncp_zip_path}")

    with zipfile.ZipFile(ncp_zip_path) as zf:
        entry_names = [name for name in zf.namelist() if not name.endswith("/")]

        basenames_seen: dict[str, str] = {}
        basename_duplicates: list[str] = []
        for name in entry_names:
            basename = Path(name).name
            if basename in basenames_seen and basenames_seen[basename] != name:
                basename_duplicates.append(
                    f"'{basenames_seen[basename]}' and '{name}' both extract to the same filename '{basename}'"
                )
            else:
                basenames_seen[basename] = name

        if basename_duplicates:
            raise ValueError(
                "extract_ncp_zip: the NCP zip contains multiple entries "
                "that would extract to the same filename — refusing to silently "
                "let one overwrite the other: " + "; ".join(basename_duplicates)
            )

        extract_dir.mkdir(parents=True, exist_ok=True)
        zf.extractall(extract_dir)

    org_cd_path = _find_organisation_file(extract_dir)
    if org_cd_path is None:
        if not require_organisation:
            return None
        raise FileNotFoundError(
            f"extract_ncp_zip: no file named exactly 'Org-NineRealms_CD.xml' "
            f"found in {ncp_zip_path} — TSO resolution has no source to work "
            f"from without it."
        )
    return org_cd_path


class NcpPools(dict):
    """{profile: [files]}: every file of a profile, whatever its header creator or file-name TSO. file_tso: {file: the TSO
    of the file (its name prefix with an Organisation file, else its header creator)}: the TSO of an object that names none."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.file_tso: dict = {}

    def by_label(self) -> dict[str, dict[str, list]]:
        """{file TSO: {profile: [files]}}, files in name order."""
        result: dict[str, dict[str, list]] = {}
        for profile, files in self.items():
            for path in files:
                result.setdefault(self.file_tso.get(path, str(path.name)), {}).setdefault(profile, []).append(path)
        return result

    def label_eics(self) -> dict[str, str]:
        """{file TSO: EIC} where a file of the TSO names its creator in the header (dcterms:creator)."""
        result: dict[str, str] = {}
        for label, profiles in sorted(self.by_label().items()):
            for profile in sorted(profiles):
                for path in profiles[profile]:
                    header = ncp_header.read_header(path)
                    if header is not None and header.creator:
                        result.setdefault(label, header.creator)
        return result


def log_pool_summary(pools: NcpPools, objects_by_profile: dict) -> None:
    """One line per profile: the files read and, where objects carry a TSO, the objects per TSO (counts)."""
    for profile in sorted(pools):
        counts = objects_by_profile.get(profile)
        per_tso = ", ".join(f"{tso}: {count}" for tso, count in sorted(counts.items())) if counts else ""
        log.info("NCP profile %s: %d file(s)%s", profile, len(pools[profile]), f", objects per TSO - {per_tso}" if per_tso else "")


def classify_extracted_ncp_files(
    extract_dir: Path,
    tso_name_map,
) -> NcpPools:
    """Classify the files extracted by extract_ncp_zip into one pool per profile (NcpPools). "recognized" files are
    pooled whatever their TSO; "organisation" and "known_irrelevant" files are skipped; "unrecognized" files are listed in
    one warning. Raises ValueError when AE, CO or RA has no file at all; a missing SHS, SIS or ER is not an error.
    An object id in two files of one profile is refused later (ncp_parser.check_unique_ids)."""
    if isinstance(tso_name_map, dict):
        tso_display_names = set(tso_name_map.values()) or set(tso_name_map.keys())
    else:
        tso_display_names = set(tso_name_map)
    known_tsos = {_country_from_display(name) for name in tso_display_names}

    pools = NcpPools()
    unrecognized: list[str] = []
    unused_files: dict[str, int] = {}

    for extracted_file in sorted(extract_dir.rglob("*")):
        if not extracted_file.is_file():
            continue
        category, tso_display, profile = _classify_ncp_file(extracted_file, known_tsos)

        if category == "known_irrelevant" and profile in UNUSED_PROFILES:
            unused_files[profile] = unused_files.get(profile, 0) + 1
        if category in ("organisation", "known_irrelevant"):
            continue
        elif category == "unrecognized":
            unrecognized.append(extracted_file.name)
        else:  # "recognized"
            pools.setdefault(profile, []).append(extracted_file)
            pools.file_tso[extracted_file] = tso_display

    for profile in UNUSED_PROFILES:
        if profile in unused_files:
            log.info("NCP profile %s is not used: %d file(s) skipped", profile, unused_files[profile])

    if unrecognized:
        log.warning(
            "classify_extracted_ncp_files: %d file(s) did not match the expected "
            "{TSO}_{PROFILE}.xml naming for a known TSO and recognized "
            "profile - skipped: %s",
            len(unrecognized), unrecognized,
        )

    missing_required = REQUIRED_PROFILES - pools.keys()
    if missing_required:
        raise ValueError(
            "classify_extracted_ncp_files: missing required NCP profile(s): " + ", ".join(sorted(missing_required))
        )
    _log_missing_optional_profiles("NCP files", pools.keys())
    return pools


def build_ncp_files(
    instance_root: Path,
    tso_name_map,
) -> dict[str, dict[str, Path]]:
    """Scan instance_root and return NCP_FILES for the TSOs that have a NetworkCode/cimxml/
    folder. tso_name_map: TSO mRID -> display name (tso_name_resolver.build_tso_name_map).
    Raises FileNotFoundError (no instance_root) or ValueError (a required profile is missing).
    """
    if not instance_root.exists():
        raise FileNotFoundError(f"Instance root not found: {instance_root}")

    # tso_name_map: dict (tso_name -> tso_name) or an iterable of display names
    if isinstance(tso_name_map, dict):
        tso_display_names = set(tso_name_map.values()) or set(tso_name_map.keys())
    else:
        try:
            tso_display_names = set(tso_name_map)
        except Exception:
            raise ValueError("tso_name_map must be a dict or iterable of TSO display names")
    ncp_files: dict[str, dict[str, Path]] = {}
    errors: list[str] = []

    for tso_display in sorted(tso_display_names):
        country = _country_from_display(tso_display)
        cimxml_dir = instance_root / country / "NetworkCode" / "cimxml"

        if not cimxml_dir.exists():
            log.info(
                "%s: no NetworkCode/cimxml/ directory — TSO provides no NCP data, skipped",
                tso_display,
            )
            continue

        profiles: dict[str, Path] = {}
        for profile in ALL_PROFILES:
            candidate = cimxml_dir / f"{country}_{profile}.xml"
            if candidate.exists():
                profiles[profile] = candidate
                log.debug("%s: found %s → %s", tso_display, profile, candidate.name)

        # Hard error on missing required profiles (only for TSOs with a cimxml dir)
        missing_required = REQUIRED_PROFILES - profiles.keys()
        if missing_required:
            errors.append(
                f"{tso_display}: missing required profile(s): "
                f"{', '.join(sorted(missing_required))}"
            )

        _log_missing_optional_profiles(tso_display, profiles.keys())

        ncp_files[tso_display] = profiles
        log.info(
            "%s: %d profile(s) found (%s)",
            tso_display,
            len(profiles),
            ", ".join(sorted(profiles)),
        )

    if errors:
        raise ValueError(
            "NCP file discovery failed — missing required files:\n"
            + "\n".join(f"  • {e}" for e in errors)
        )

    log.info(
        "NCP_FILES built: %d TSOs with NCP data (of %d in CGM)",
        len(ncp_files),
        len(tso_display_names),
    )
    return ncp_files


def _country_from_display(tso_display: str) -> str:
    """Strip a leading 'TSO-': "TSO-Belgovia" -> "Belgovia"."""
    return tso_display.removeprefix("TSO-")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse, sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="NCP file discovery")
    parser.add_argument("instance_root", type=Path,
                        help="Path to Instance/ directory")
    parser.add_argument("org_cd", type=Path,
                        help="Path to Org-NineRealms_CD.xml")
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from pre_milp.tso_name_resolver import build_tso_name_map

    tso_name_map = build_tso_name_map(args.org_cd)
    ncp_files = build_ncp_files(args.instance_root, tso_name_map)

    print("\nNCP_FILES:")
    for tso, profiles in sorted(ncp_files.items()):
        print(f"  {tso}:")
        for profile, path in sorted(profiles.items()):
            print(f"    {profile}: {path.name}")

    print(f"\nTSOs with NCP data: {len(ncp_files)}")