"""
scanned_elements.py — scanned-element list (ae_type SCANNED) with the limits at the run timestamp:
    patl_scanned_ka = patl_cgm_ka * (1 + virtual_load_margin_pct / 100)
    effective_limit_ka = max(patl_scanned_ka, lf_initial[branch_id]),
    = patl_scanned_ka when lf_initial is None or the branch is absent from it.
"""

from __future__ import annotations

import csv
import logging
import sys
from pathlib import Path
from typing import Optional

# Ensure src/ is on the path when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_model import AEType, AssessedElement, ScannedElement
from text_encoding import TEXT_ENCODING

log = logging.getLogger(__name__)

# RAO control parameters — defaults per spec
DEFAULT_SOFT    = False
DEFAULT_PENALTY = 0.0


def build_scanned_element_list(
    ae_flat_list:   list[AssessedElement],
    timestamp:      str,
    lf_initial:     Optional[dict[str, float]] = None,
    soft:           bool  = DEFAULT_SOFT,
    penalty:        float = DEFAULT_PENALTY,
) -> list[ScannedElement]:
    """Build the ScannedElementList from the SCANNED entries of ae_flat_list.

    lf_initial: optional {branch_mrid: i_ka} at the run timestamp; without it (or without the
    branch) effective_limit_ka = patl_scanned_ka. soft / penalty (EUR): soft-constraint
    settings. The limits of the result are keyed by the run timestamp and always set (no None).
    """
    scanned_aes = [ae for ae in ae_flat_list if ae.ae_type == AEType.SCANNED]

    if not scanned_aes:
        log.info("No scanned AEs found in AEFlatList — ScannedElementList is empty")
        return []

    result: list[ScannedElement] = []

    for ae in scanned_aes:
        if ae.branch_mrid is None:
            log.warning("Scanned AE %s has no branch_mrid — skipped", ae.mrid)
            continue

        # normalPositiveVirtualMargin is not parsed (absent from the NCP files): 0 (spec default)
        virtual_margin_pct: int = 0

        patl_cgm_ka:    dict[str, float] = {}
        patl_scanned_ka: dict[str, float] = {}
        effective_limit_ka: dict[str, float] = {}

        cgm_ka = ae.patl_ka.get(timestamp, 0.0)
        patl_cgm_ka[timestamp] = cgm_ka

        scanned_ka = cgm_ka * (1 + virtual_margin_pct / 100.0)
        patl_scanned_ka[timestamp] = scanned_ka

        # Effective limit: max(patl_scanned, initial_loading) per spec
        if lf_initial is not None:
            initial_ka = lf_initial.get(ae.branch_mrid)
            if initial_ka is not None:
                eff_ka = max(scanned_ka, initial_ka)
                if initial_ka > scanned_ka:
                    log.debug(
                        "Scanned AE %s: initial loading %.3f kA exceeds "
                        "patl_scanned %.3f kA — using initial as limit",
                        ae.mrid, initial_ka, scanned_ka,
                    )
            else:
                eff_ka = scanned_ka  # branch not in lf_initial → conservative
        else:
            eff_ka = scanned_ka  # no lf_initial provided → conservative

        effective_limit_ka[timestamp] = eff_ka

        result.append(ScannedElement(
            mrid=ae.mrid,
            name=ae.name,
            tso=ae.tso,
            branch_mrid=ae.branch_mrid,
            virtual_load_margin_pct=virtual_margin_pct,
            patl_cgm_ka=patl_cgm_ka,
            patl_scanned_ka=patl_scanned_ka,
            effective_limit_ka=effective_limit_ka,
            soft=soft,
            penalty=penalty,
        ))

    log.info(
        "ScannedElementList built: %d entries (soft=%s, penalty=%.1f EUR)",
        len(result), soft, penalty,
    )
    return result


def write_scanned_element_csv(
    scanned_list: list[ScannedElement],
    output_path:  Path,
    timestamp: str,
    lf_initial: Optional[dict[str, float]] = None,
    branch_mrid_to_name: Optional[dict[str, str]] = None,
) -> None:
    """Write scanned_element_list.csv: [branch_mrid, branch_name, tso, timestamp, patl_cgm_ka,
    virtual_load_margin_pct, patl_scanned_ka, initial_loading_ka, effective_limit_ka].
    branch_name falls back to the element's own name; initial_loading_ka is blank without lf_initial.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "branch_mrid", "branch_name", "tso", "timestamp",
        "patl_cgm_ka", "virtual_load_margin_pct",
        "patl_scanned_ka", "initial_loading_ka", "effective_limit_ka",
    ]
    branch_mrid_to_name = branch_mrid_to_name or {}
    with open(output_path, "w", newline="", encoding=TEXT_ENCODING) as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for sc in scanned_list:
            initial_loading_ka = ""
            if lf_initial is not None:
                val = lf_initial.get(sc.branch_mrid)
                if val is not None:
                    initial_loading_ka = val
            writer.writerow({
                "branch_mrid":            sc.branch_mrid,
                "branch_name":            branch_mrid_to_name.get(sc.branch_mrid, sc.name),
                "tso":                    sc.tso,
                "timestamp":              timestamp,
                "patl_cgm_ka":            sc.patl_cgm_ka.get(timestamp, 0.0),
                "virtual_load_margin_pct": sc.virtual_load_margin_pct,
                "patl_scanned_ka":        sc.patl_scanned_ka.get(timestamp, 0.0),
                "initial_loading_ka":     initial_loading_ka,
                "effective_limit_ka":     sc.effective_limit_ka.get(timestamp, 0.0),
            })
    log.info("Written: %s (%d rows)", output_path, len(scanned_list))


# ---------------------------------------------------------------------------
# CLI entry point — for manual inspection
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Scanned element list")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--soft",    action="store_true", default=False)
    parser.add_argument("--penalty", type=float, default=0.0)
    parser.add_argument("--timestamp", default="1030Z")
    args = parser.parse_args()

    # Minimal smoke: build from empty list
    result = build_scanned_element_list([], args.timestamp, soft=args.soft, penalty=args.penalty)
    write_scanned_element_csv(result, args.output_dir / "scanned_element_list.csv", args.timestamp)
    print(f"ScannedElementList: {len(result)} entries")