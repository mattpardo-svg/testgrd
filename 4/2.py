"""cgmes_loader.py — CGM load, base-case AC load flow and network tables (currents, static
PATL, PST degrees per tap, names, equipment types). pypowsybl is imported lazily so tests
without it can import the module.
"""

import csv
import logging
import math
from pathlib import Path
from typing import Optional, Tuple


from pre_milp.lf_parameters import build_loadflow_parameters
from text_encoding import TEXT_ENCODING

log = logging.getLogger(__name__)


def extract_cgm_zip(cgm_zip_path: Path, extract_dir: Path) -> None:
    """Extract an input/cgm*.zip to extract_dir; a zip inside it (one zip per profile) is extracted next to itself
    into a folder of its name, at any depth. pypowsybl reads the zip itself; the extraction gives the
    EQ XML files that the tap-changer and CurrentLimit side maps parse directly."""
    import zipfile

    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(cgm_zip_path) as zf:
        zf.extractall(extract_dir)
    pending = sorted(extract_dir.rglob("*.zip"))
    while pending:
        inner = pending.pop(0)
        target = inner.with_suffix("")
        with zipfile.ZipFile(inner) as zf:
            zf.extractall(target)
        pending.extend(sorted(target.rglob("*.zip")))


def _model_profiles(path: Path) -> list[str]:
    """The md:Model.profile values of the file header (md:FullModel); [] for a file without a header."""
    import xml.etree.ElementTree as ET

    profiles: list[str] = []
    depth = 0
    try:
        for event, element in ET.iterparse(str(path), events=("start", "end")):
            local = element.tag.rsplit("}", 1)[-1]
            if event == "start":
                depth += 1
                continue
            depth -= 1
            if depth == 1:
                break                           # the end of the first object, the header or not
            if local == "Model.profile" and element.text:
                profiles.append(element.text.strip())
    except (ET.ParseError, OSError, UnicodeDecodeError):
        return []
    return profiles


def _is_equipment_profile(url: str) -> bool:
    """CGMES 3.0 ".../CoreEquipment-EU/<v>" or CGMES 2.4.15 ".../EquipmentCore/<v>"."""
    return any(part in ("CoreEquipment-EU", "EquipmentCore") for part in url.split("/"))


def _is_boundary(profiles: list[str]) -> bool:
    """A boundary equipment file (EquipmentBoundary profile); the CGMES 3.0 boundary and common data files declare the
    CoreEquipment profile and are listed with the EQ files (they hold no TSO equipment)."""
    return any("EquipmentBoundary" in url for url in profiles)


def find_eq_files(extract_dir: Path) -> list[Path]:
    """Every EQ file of the extracted CGM at any depth, sorted by name, without the boundary files: the header profile
    decides (CGMES 3.0 CoreEquipment or 2.4.15 EquipmentCore), the file name ("..._EQ_<version>.xml") only for a file
    without one."""
    found = []
    for path in extract_dir.rglob("*.xml"):
        profiles = _model_profiles(path)
        if _is_boundary(profiles):
            continue
        if any(_is_equipment_profile(profile) for profile in profiles) or (not profiles and "_EQ_" in path.name):
            found.append(path)
    return sorted(found, key=lambda path: (path.name, str(path)))


def find_eq_file_for_tso(extract_dir: Path, country: str) -> Optional[Path]:
    """A TSO's EQ file in the extracted CGM ("{timestamp}__{TSO}_EQ_{version}.xml"), matched on
    "_{country}_EQ" so an HVDC file ("__HVDC-Espheim-Svedala_EQ_1.xml") does not match.
    None when not found (the caller logs and continues)."""
    pattern = f"_{country}_EQ"
    matches = sorted(f for f in extract_dir.iterdir() if f.is_file() and pattern in f.name)
    return matches[0] if matches else None


def load_cgm(cgm_zip_override: Path):
    """Load the CGM zip cgm_zip_override (input/cgm*.zip) with pypowsybl.network.load(...)
    and return the Network. There is no default source."""
    if cgm_zip_override is None or not Path(cgm_zip_override).is_file():
        raise FileNotFoundError(f"CGM zip not found: {cgm_zip_override}")
    try:
        import pypowsybl as pp
    except Exception as e:
        raise ImportError("pypowsybl is required to load the CGM") from e
    network = pp.network.load(
        str(cgm_zip_override),
        parameters={
            "iidm.import.cgmes.source-for-iidm-id": "mRID",
            "iidm.import.cgmes.cgm-with-subnetworks": "false",
        },
    )
    if network is None:
        raise RuntimeError("pypowsybl failed to load the CGM package")
    return network


def run_initial_loadflow(network, use_reactive_limits: bool = False, extra_lf_params: dict = None):
    """Run the initial AC load flow; halt if it does not converge.
    extra_lf_params: extra LoadFlowParameters kwargs (voltage_init_mode, distributed_slack, ...).
    use_reactive_limits defaults to False, unlike the spec's respectReactivePowerLimits=true:
    on this dataset enforcing reactive limits makes most contingency load flows diverge
    (several generators need far more reactive power than their limits).
    """
    try:
        import pypowsybl as pp
    except Exception as e:
        raise ImportError("pypowsybl is required to run loadflow") from e

    lf_result = pp.loadflow.run_ac(
        network,
        parameters=build_loadflow_parameters(use_reactive_limits, extra_lf_params),
    )
    status = None
    try:
        status = getattr(lf_result[0], "status", None)
    except Exception:
        status = getattr(lf_result, "status", None)

    status_name = getattr(status, "name", None) or str(status)
    if "CONVERGED" not in str(status_name).upper():
        log.fatal("Initial AC load flow did not converge: %s", status_name)
        raise RuntimeError(f"AC load flow did not converge (status={status!r})")

    log.info("Initial AC load flow converged: %s", status_name)
    return lf_result


def run_initial_dc_loadflow(network, use_reactive_limits: bool = False, extra_lf_params: dict = None):
    """calculation_mode dc: the DC load flow with the pipeline's load-flow parameters (distributed
    slack as configured); halt if it does not converge, as the AC initial load flow does. The DC
    load flow leaves p1/p2 on the branches; q, i and the bus voltages are NaN."""
    try:
        import pypowsybl as pp
    except Exception as e:
        raise ImportError("pypowsybl is required to run loadflow") from e

    lf_result = pp.loadflow.run_dc(
        network,
        parameters=build_loadflow_parameters(use_reactive_limits, extra_lf_params),
    )
    status = getattr(lf_result[0], "status", None)
    status_name = getattr(status, "name", None) or str(status)
    if "CONVERGED" not in str(status_name).upper():
        log.fatal("Initial DC load flow did not converge: %s", status_name)
        raise RuntimeError(f"DC load flow did not converge (status={status!r})")

    log.info("Initial DC load flow converged: %s", status_name)
    return lf_result


def _has_flow(p_mw) -> bool:
    """A branch side with an active power value: pandas gives NaN (not None) for a disconnected side."""
    return p_mw is not None and not math.isnan(float(p_mw))


def _boundary_branch_values(network, column: str, nan: float, convert=None) -> dict[str, dict[int, float]]:
    """{branch: {1: value, 2: value}} of the dangling lines and the tie lines, from the dangling line column `column`
    ("i" in A, "p" in MW; `convert(branch, side, value)` for the DC currents). A dangling line has one network side,
    side 1; side 2 is NaN. A tie line takes side 1 from its first and side 2 from its second dangling line."""
    result: dict[str, dict[int, float]] = {}
    try:
        dangling = network.get_dangling_lines()
        for mrid, row in dangling.iterrows():
            value = row[column]
            if convert is not None and _has_flow(value):
                value = convert(str(mrid), 1, value)
            result[str(mrid)] = {1: nan if value is None else float(value), 2: nan}
        for tie_id, row in network.get_tie_lines().iterrows():
            first, second = result.get(str(row["dangling_line1_id"])), result.get(str(row["dangling_line2_id"]))
            result[str(tie_id)] = {1: first[1] if first else nan, 2: second[1] if second else nan}
            if convert is not None:
                for side, own in ((1, first), (2, second)):
                    if own and _has_flow(own[1]):
                        raw = dangling.loc[row[f"dangling_line{side}_id"], column]
                        result[str(tie_id)][side] = convert(str(tie_id), side, raw)
    except Exception:
        log.exception("Failed to extract dangling line / tie line values (%s)", column)
    return result


def build_lf_initial_dc(network, dc_converter) -> dict[str, dict[int, float]]:
    """calculation_mode dc: branch currents (A) of both sides from the active power p1/p2 left by the
    DC load flow, converted by dc_converter (dc_conversion, the one MW -> A rule). Same shape as
    build_lf_initial; a side without p (disconnected) is NaN, never 0.0."""
    lf_initial: dict[str, dict[int, float]] = {}
    nan = float("nan")
    for table in (network.get_lines(), network.get_2_windings_transformers()):
        for mrid, row in table.iterrows():
            lf_initial[str(mrid)] = {
                side: dc_converter.current(str(mrid), side, row[f"p{side}"]) if _has_flow(row[f"p{side}"]) else nan
                for side in (1, 2)
            }
    lf_initial.update(_boundary_branch_values(network, "p", nan, dc_converter.current))
    return lf_initial


def build_lf_initial(network, dc_converter=None) -> dict[str, dict[int, float]]:
    """Extract initial branch currents (A) from the pypowsybl network results.

    Both sides are kept (never max(i1, i2)): PATL limits are side-specific and
    every consumer picks the side of the branch with data_model.select_by_side.

    Returns dict[branch_mrid, {1: i1_ampere, 2: i2_ampere}] for both
    lines and 2-winding transformers. A missing i1/i2 value is NaN (with a
    warning), never 0.0, so the model's missing-current exclusion applies.
    dc_converter (calculation_mode dc): the currents come from the DC p1/p2 (build_lf_initial_dc);
    None = the AC results, as always.
    """
    if dc_converter is not None:
        return build_lf_initial_dc(network, dc_converter)
    lf_initial: dict[str, dict[int, float]] = {}
    nan = float("nan")

    def _current(value) -> float:
        return nan if value is None else float(value)

    def _extract_currents(source, description: str) -> None:
        if source is None:
            return
        missing: list[str] = []
        if hasattr(source, "index"):
            for mrid in source.index:
                row = source.loc[mrid] if hasattr(source, "loc") else None
                i1 = _current(row.get("i1")) if row is not None else nan
                i2 = _current(row.get("i2")) if row is not None else nan
                if row is None or "i1" not in row or "i2" not in row:
                    missing.append(str(mrid))
                lf_initial[str(mrid)] = {1: i1, 2: i2}
        else:
            for item in source:
                mrid = getattr(item, "mrid", getattr(item, "id", str(item)))
                i1, i2 = _current(getattr(item, "i1", None)), _current(getattr(item, "i2", None))
                if getattr(item, "i1", None) is None or getattr(item, "i2", None) is None:
                    missing.append(str(mrid))
                lf_initial[str(mrid)] = {1: i1, 2: i2}
        if missing:
            log.warning(
                "build_lf_initial: %d branch(es) of the %s without an i1/i2 current - set to NaN "
                "(excluded from the model): %s", len(missing), description, sorted(missing),
            )

    try:
        lines_results = None
        if hasattr(network, "get_lines_results"):
            try:
                lines_results = network.get_lines_results()
            except Exception:
                lines_results = None

        if lines_results is not None:
            _extract_currents(lines_results, "line results")
        else:
            _extract_currents(network.get_lines(), "lines")
    except Exception:
        log.exception("Failed to extract line currents from network line results")

    try:
        trafos_results = None
        if hasattr(network, "get_2_windings_transformers_results"):
            try:
                trafos_results = network.get_2_windings_transformers_results()
            except Exception:
                trafos_results = None

        if trafos_results is not None:
            _extract_currents(trafos_results, "transformer results")
        else:
            _extract_currents(network.get_2_windings_transformers(), "2-winding transformers")
    except Exception:
        log.exception("Failed to extract transformer currents from network transformer results")

    lf_initial.update(_boundary_branch_values(network, "i", float("nan")))
    return lf_initial


_REGULAR_LIMIT_ELEMENTS = ("LINE", "TWO_WINDINGS_TRANSFORMER")
_BOUNDARY_LIMIT_ELEMENTS = ("BOUNDARY_LINE", "DANGLING_LINE")        # the pypowsybl name differs between versions


def _nominal_kv_by_branch_side(network) -> dict[tuple[str, int], float]:
    """{(branch id, side 1 or 2): nominal voltage of the voltage level of that side, in kV} for lines, 2W transformers,
    dangling lines (side 1) and tie lines (the voltage level of each dangling line)."""
    kv_of_level = network.get_voltage_levels()["nominal_v"]
    result: dict[tuple[str, int], float] = {}
    for table in (network.get_lines(), network.get_2_windings_transformers()):
        for mrid, row in table.iterrows():
            for side in (1, 2):
                result[(str(mrid), side)] = float(kv_of_level[row[f"voltage_level{side}_id"]])
    dangling = network.get_dangling_lines()
    for mrid, row in dangling.iterrows():
        result[(str(mrid), 1)] = float(kv_of_level[row["voltage_level_id"]])
    for tie_id, row in network.get_tie_lines().iterrows():
        for side in (1, 2):
            result[(str(tie_id), side)] = float(kv_of_level[dangling.loc[row[f"dangling_line{side}_id"], "voltage_level_id"]])
    return result


def _apparent_power_limits_as_current(network, limits):
    """(rows, number of limits converted): for each branch side that has no CURRENT limit, its APPARENT_POWER limits
    (value in MVA) as CURRENT rows of the same name and duration, I [A] = S [MVA] * 1000 / (sqrt(3) * U [kV]), with U the
    nominal voltage of the voltage level of that side. A CURRENT limit on the side always wins. Rows of a side whose
    voltage is unknown are left out."""
    has_current = set(zip(limits.loc[limits["type"] == "CURRENT", "branch_id"], limits.loc[limits["type"] == "CURRENT", "side_number"]))
    apparent = limits[limits["type"] == "APPARENT_POWER"]
    wanted = apparent[[(b, s) not in has_current for b, s in zip(apparent["branch_id"], apparent["side_number"])]].copy()
    if wanted.empty:
        return wanted, 0
    kv = _nominal_kv_by_branch_side(network)
    voltages = [kv.get((b, int(s)), float("nan")) for b, s in zip(wanted["branch_id"], wanted["side_number"])]
    wanted["value"] = [mva * 1000.0 / (math.sqrt(3.0) * u) for mva, u in zip(wanted["value"], voltages)]
    wanted["type"] = "CURRENT"
    wanted = wanted[[u == u for u in voltages]]
    return wanted, int((wanted["element_type"] != "TIE_LINE").sum())


def build_branch_limits(network, report: bool = False):
    """The operational limits of the branches of the pipeline as a table with the extra columns branch_id and
    side_number (1 or 2): lines and 2W transformers on their sides ONE / TWO; a dangling line (pypowsybl gives its
    limits the side NONE) on side 1, its network side; a tie line on the limits of its first (side 1) and second
    (side 2) dangling line. A side with APPARENT_POWER limits (MVA) and no CURRENT limit gets them as CURRENT limits in
    amperes (_apparent_power_limits_as_current). report: log how many were converted. None when the network has no
    limits or fetching them fails."""
    import pandas as pd

    try:
        table = network.get_operational_limits()
    except Exception:
        log.exception("build_branch_limits: failed to fetch operational limits")
        return None
    if table is None or table.empty:
        return None
    table = table.reset_index()
    element_types = table["element_type"].astype(str)
    regular = table[element_types.isin(_REGULAR_LIMIT_ELEMENTS)].copy()
    regular["side_number"] = regular["side"].astype(str).map({"ONE": 1, "TWO": 2})
    boundary = table[element_types.isin(_BOUNDARY_LIMIT_ELEMENTS)].copy()
    boundary["side_number"] = 1
    frames = [regular, boundary]
    try:
        ties = network.get_tie_lines()
    except Exception:
        ties = None
    if ties is not None:
        for tie_id, row in ties.iterrows():
            for side in (1, 2):
                part = boundary[boundary["element_id"] == row[f"dangling_line{side}_id"]].copy()
                part["element_id"], part["element_type"], part["side_number"] = str(tie_id), "TIE_LINE", side
                frames.append(part)
    result = pd.concat(frames, ignore_index=True)
    result["branch_id"] = result["element_id"].astype(str)
    try:
        converted, count = _apparent_power_limits_as_current(network, result)
    except Exception:
        log.exception("build_branch_limits: failed to convert the apparent power limits")
        return result
    if count:
        result = pd.concat([result, converted], ignore_index=True)
        if report:
            log.info(
                "build_cgm_static_patl: %d apparent power limit(s) (MVA) of %d branch side(s) without a current limit converted to "
                "amperes, I = S / (sqrt(3) * U_nominal) with U in kV", count, converted[["branch_id", "side_number"]].drop_duplicates().shape[0],
            )
    return result


def _outside_main_component(network, row) -> str:
    """", not in the main network component (component N)" when a bus of the branch is in a connected or synchronous
    component other than the main one (0, the largest, which the load flow solves); "" otherwise or when it cannot be told."""
    try:
        buses = network.get_buses()
        for column in ("bus1_id", "bus2_id", "bus_id"):
            bus_id = row.get(column) if column in row else None
            if not bus_id or bus_id not in buses.index:
                continue
            for kind in ("connected_component", "synchronous_component"):
                number = buses.loc[bus_id, kind]
                if number == number and int(number) != 0:
                    return f", not in the main network component (component {int(number)})"
    except Exception:
        pass
    return ""


def describe_branch(network, branch_mrid: str) -> str:
    """What a branch is and whether the network has it connected, for a warning: "dangling line, connected"; a zero or very
    small impedance (|r| + |x| < 1e-6 ohm) is said too."""
    for accessor, label, flags in (
        ("get_lines", "line", ("connected1", "connected2")),
        ("get_2_windings_transformers", "2W transformer", ("connected1", "connected2")),
        ("get_dangling_lines", "dangling line", ("connected",)),
        ("get_tie_lines", "tie line", ("connected1", "connected2")),
    ):
        try:
            table = getattr(network, accessor)()
            if branch_mrid in table.index:
                row = table.loc[branch_mrid]
                connected = all(bool(row[flag]) for flag in flags if flag in row)
                text = f"{label}, {'connected' if connected else 'disconnected'}"
                text += _outside_main_component(network, row)
                if "r" in row and "x" in row and abs(float(row["r"])) + abs(float(row["x"])) < 1e-6:
                    text += ", zero or very small impedance"
                return text
        except Exception:
            continue
    return "not in the network"


def phase_transformer_mrids(network) -> set[str]:
    """The PowerTransformers with a phase tap changer (pypowsybl indexes phase tap changers by transformer)."""
    try:
        return {str(mrid) for mrid in network.get_phase_tap_changers().index}
    except Exception:
        log.exception("phase_transformer_mrids: failed to read the phase tap changers")
        return set()


def one_side_branch_mrids(network) -> set[str]:
    """The branches with one network side: the dangling lines. Their only current, limit and sensitivity are on side 1."""
    try:
        return {str(mrid) for mrid in network.get_dangling_lines().index}
    except Exception:
        log.exception("one_side_branch_mrids: failed to read the dangling lines")
        return set()


def build_cgm_static_patl(network) -> dict[str, dict[int, float]]:
    """Static CGM PATL per branch and side (the fallback when the NCP gives no PATL): permanent
    (acceptable_duration == -1) CURRENT limits of lines, 2W transformers (both sides), dangling lines (side 1, the network
    side) and tie lines (see build_branch_limits). {} when the network has no limits or fetching them fails."""
    limits_df = build_branch_limits(network, report=True)
    if limits_df is None:
        return {}

    static_patl: dict[str, dict[int, float]] = {}
    kinds: dict[str, str] = {}
    for _, row in limits_df.iterrows():
        if str(row.get("type")) != "CURRENT":
            continue
        try:
            acceptable_duration = int(row.get("acceptable_duration"))
        except (TypeError, ValueError):
            continue
        if acceptable_duration != -1:
            continue

        side_number = row.get("side_number")
        if side_number != side_number or side_number is None:         # NaN: a side that is neither ONE nor TWO
            log.warning(
                "build_cgm_static_patl: element %s has unrecognized side %r — skipped",
                row.get("element_id"), row.get("side"),
            )
            continue

        branch_id = str(row.get("branch_id"))
        static_patl.setdefault(branch_id, {})[int(side_number)] = float(row.get("value"))
        kinds[branch_id] = str(row.get("element_type"))

    by_kind = ", ".join(f"{count} {kind}" for kind, count in sorted(
        {kind: sum(1 for k in kinds.values() if k == kind) for kind in set(kinds.values())}.items()))
    log.info("build_cgm_static_patl: resolved static PATL for %d branch(es) (%s)", len(static_patl), by_kind or "none")
    return static_patl

def build_pst_degrees_per_tap(network, pst_ids: list[str]) -> dict[str, float]:
    """Average phase shift per tap step (degrees) per PST mRID, from
    get_phase_tap_changer_steps()['alpha']: (alpha_last - alpha_first) / (num_steps - 1).
    A PST with fewer than 2 steps or no steps is omitted (warning); non-uniform steps
    (tolerance 1e-6) give a warning and still the average. {} if the table cannot be read."""
    try:
        steps_df = network.get_phase_tap_changer_steps()
    except Exception:
        log.exception("build_pst_degrees_per_tap: failed to fetch phase tap changer steps")
        return {}

    if steps_df is None or "alpha" not in getattr(steps_df, "columns", []):
        log.warning("build_pst_degrees_per_tap: no 'alpha' column in phase tap changer steps table")
        return {}

    degrees_per_tap: dict[str, float] = {}
    for pst_mrid in pst_ids:
        if pst_mrid not in steps_df.index:
            log.warning(
                "build_pst_degrees_per_tap: PST %s not found in phase tap changer "
                "steps table — omitted",
                pst_mrid,
            )
            continue

        # pypowsybl's steps table is not sorted by tap position: sort by the position index
        # first, otherwise the increments are between non-consecutive taps
        pst_rows = steps_df.loc[[pst_mrid]].sort_index()
        alpha_values = [float(v) for v in pst_rows["alpha"].tolist()]
        if len(alpha_values) < 2:
            log.warning(
                "build_pst_degrees_per_tap: PST %s has fewer than 2 tap steps — omitted",
                pst_mrid,
            )
            continue

        average_degrees_per_tap = (alpha_values[-1] - alpha_values[0]) / (len(alpha_values) - 1)

        consecutive_increments = [
            alpha_values[index + 1] - alpha_values[index] for index in range(len(alpha_values) - 1)
        ]
        if any(abs(increment - average_degrees_per_tap) > 1e-6 for increment in consecutive_increments):
            log.warning(
                "build_pst_degrees_per_tap: PST %s tap steps are not uniformly spaced "
                "— using average increment %.6f deg/tap",
                pst_mrid, average_degrees_per_tap,
            )

        degrees_per_tap[pst_mrid] = average_degrees_per_tap

    return degrees_per_tap


def print_network_summary(network, converged: bool, load_flow: str = "") -> None:
    """Log a one-line-per-category summary of the network (element counts); never raises.
    load_flow: "DC" labels the converged flag as the DC load flow's (calculation_mode dc)."""
    def _count(accessor_name: str) -> Optional[int]:
        try:
            accessor = getattr(network, accessor_name)
            return len(accessor())
        except Exception:
            return None

    counts = {
        "buses": _count("get_buses"),
        "lines": _count("get_lines"),
        "2-winding transformers": _count("get_2_windings_transformers"),
        "PSTs (phase tap changers)": _count("get_phase_tap_changers"),
        "HVDC lines": _count("get_hvdc_lines"),
        "generators": _count("get_generators"),
    }

    log.info(
        "Network summary: " + (f"{load_flow} load flow " if load_flow else "") + "converged=%s | %s",
        converged,
        ", ".join(f"{name}={count if count is not None else 'n/a'}" for name, count in counts.items()),
    )


def write_lf_basecase_csv(
    network,
    lf_initial: dict[str, dict[int, float]],
    patl: dict[str, dict[str, float]],
    timestamp: str,
    converged: bool,
    output_path: Path,
) -> None:
    """lf_basecase.csv: one row per branch, N-0 current against PATL. The display current is
    max(i1, i2) (not side-resolved). patl: {branch: {timestamp: patl_kA}} (side-resolved).
    Columns [branch_mrid, branch_name, timestamp, i_ka, patl_ka, loading_pct, converged];
    patl_ka and loading_pct are empty when the branch has no PATL at this timestamp.
    """
    branch_mrid_to_name = build_branch_mrid_to_name(network)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding=TEXT_ENCODING) as f:
        writer = csv.writer(f)
        writer.writerow([
            "branch_mrid", "branch_name", "timestamp", "i_ka", "patl_ka", "loading_pct", "converged",
        ])
        for branch_mrid in sorted(lf_initial):
            per_side_currents = lf_initial[branch_mrid]
            i_ampere = max(per_side_currents.get(1, 0.0), per_side_currents.get(2, 0.0))
            i_ka = i_ampere / 1000.0

            patl_ka = patl.get(branch_mrid, {}).get(timestamp)
            loading_pct = (i_ka / patl_ka) * 100.0 if patl_ka else ""

            writer.writerow([
                branch_mrid, branch_mrid_to_name.get(branch_mrid, branch_mrid), timestamp,
                i_ka, patl_ka if patl_ka is not None else "", loading_pct, converged,
            ])

    log.info("Written base-case load flow CSV: %s", output_path)


def build_branch_mrid_to_name(network) -> dict[str, str]:
    """Return mapping from branch mRID to branch name for lines, 2W transformers, dangling lines and tie lines."""
    name_to_mrid: dict[str, str] = {}
    try:
        lines = network.get_lines()
        if hasattr(lines, "index"):
            for mrid in lines.index:
                row = lines.loc[mrid] if hasattr(lines, "loc") else None
                name = str(row.get("name", mrid)) if row is not None else str(mrid)
                name_to_mrid[str(mrid)] = name
        else:
            for item in lines:
                mrid = getattr(item, "mrid", getattr(item, "id", str(item)))
                name = getattr(item, "name", str(mrid))
                name_to_mrid[str(mrid)] = str(name)
    except Exception:
        log.exception("Failed to build branch name mapping from network.get_lines()")

    try:
        trafos = network.get_2_windings_transformers()
        if hasattr(trafos, "index"):
            for mrid in trafos.index:
                row = trafos.loc[mrid] if hasattr(trafos, "loc") else None
                name = str(row.get("name", mrid)) if row is not None else str(mrid)
                name_to_mrid[str(mrid)] = name
        else:
            for item in trafos:
                mrid = getattr(item, "mrid", getattr(item, "id", str(item)))
                name = getattr(item, "name", str(mrid))
                name_to_mrid[str(mrid)] = str(name)
    except Exception:
        log.exception("Failed to build branch name mapping from network.get_2_windings_transformers()")

    for accessor_name in ("get_dangling_lines", "get_tie_lines"):
        try:
            table = getattr(network, accessor_name)()
            for mrid, row in table.iterrows():
                name_to_mrid[str(mrid)] = str(row.get("name", mrid))
        except Exception:
            log.exception("Failed to build branch name mapping from network.%s()", accessor_name)

    return name_to_mrid


def build_equipment_mrid_to_type(network) -> dict[str, str]:
    """{equipment mRID: CoMo EquipmentType} for 8 of the spec's 9 types: line (a dangling line and a tie line are lines too), transformer
    (2-winding), gen, load, switch, shuntCompensator, staticVarCompensator, busbar.
    "bus" is not mapped: pypowsybl's buses are a computed view that cannot be disconnected.
    HVDC equipment is out of scope. An mRID not in the result is excluded by the caller with a reason.
    """
    accessor_to_type = [
        ("get_lines", "line"),
        ("get_2_windings_transformers", "transformer"),
        ("get_dangling_lines", "line"),
        ("get_tie_lines", "line"),
        ("get_generators", "gen"),
        ("get_loads", "load"),
        ("get_switches", "switch"),
        ("get_shunt_compensators", "shuntCompensator"),
        ("get_static_var_compensators", "staticVarCompensator"),
        ("get_busbar_sections", "busbar"),
    ]

    mrid_to_type: dict[str, str] = {}
    for accessor_name, spec_type in accessor_to_type:
        try:
            df = getattr(network, accessor_name)()
            for mrid in df.index:
                mrid_to_type[str(mrid)] = spec_type
        except Exception:
            log.exception("Failed to build equipment-type mapping from network.%s()", accessor_name)

    return mrid_to_type


def extract_equipment_dicts(network) -> Tuple[dict, dict, dict]:
    """Extract branch, tap changer and generating unit dicts from Network.

    The returned dicts are intentionally minimal — they provide the keys and
    a small set of attributes used by downstream parsers and tests.
    """
    branch_dict = {}
    tap_changer_dict = {}
    generating_unit_dict = {}

    # Lines
    try:
        lines = network.get_lines()
        if hasattr(lines, "index"):
            for mrid in lines.index:
                branch_dict[str(mrid)] = {"name": str(mrid), "kind": "line"}
        else:
            for item in lines:
                mrid = getattr(item, "mrid", getattr(item, "id", str(item)))
                branch_dict[str(mrid)] = {"name": str(mrid), "kind": "line"}
    except Exception:
        log.exception("extract_equipment_dicts: failed to read network.get_lines()")

    # Phase tap changers
    try:
        ptcs = network.get_phase_tap_changers()
        if hasattr(ptcs, "index"):
            for mrid in ptcs.index:
                row = ptcs.loc[mrid] if hasattr(ptcs, "loc") else None
                tap_changer_dict[str(mrid)] = {
                    "low_tap": int(row.get("low_tap", 0)) if row is not None else None,
                    "high_tap": int(row.get("high_tap", 0)) if row is not None else None,
                    "tap": int(row.get("tap", 0)) if row is not None else None,
                }
        else:
            for item in ptcs:
                mrid = getattr(item, "mrid", getattr(item, "id", str(item)))
                tap_changer_dict[str(mrid)] = {"low_tap": None, "high_tap": None, "tap": None}
    except Exception:
        log.exception("extract_equipment_dicts: failed to read network.get_phase_tap_changers()")

    # Ratio / general tap changers (some CGMs use RatioTapChanger or similar)
    try:
        rtcs = None
        if hasattr(network, "get_ratio_tap_changers"):
            try:
                rtcs = network.get_ratio_tap_changers()
            except Exception:
                rtcs = None

        if rtcs is not None:
            if hasattr(rtcs, "index"):
                for mrid in rtcs.index:
                    row = rtcs.loc[mrid] if hasattr(rtcs, "loc") else None
                    tap_changer_dict[str(mrid)] = {
                        "low_tap": int(row.get("low_tap", row.get("low_step", 0))) if row is not None else None,
                        "high_tap": int(row.get("high_tap", row.get("high_step", 0))) if row is not None else None,
                        "tap": int(row.get("tap", row.get("r0", 0))) if row is not None else None,
                    }
            else:
                def _first_set(item, *names):
                    # explicit None check: a tap position of 0 is a value, not a missing one
                    for name in names:
                        value = getattr(item, name, None)
                        if value is not None:
                            return int(value)
                    return None

                for item in rtcs:
                    mrid = getattr(item, "mrid", getattr(item, "id", str(item)))
                    tap_changer_dict[str(mrid)] = {
                        "low_tap": _first_set(item, "low_tap", "low_step"),
                        "high_tap": _first_set(item, "high_tap", "high_step"),
                        "tap": _first_set(item, "tap", "r0"),
                    }
    except Exception:
        log.exception("extract_equipment_dicts: failed to read network.get_ratio_tap_changers()")

    # Generators
    try:
        gens = network.get_generators()
        if hasattr(gens, "index"):
            for mrid in gens.index:
                row = gens.loc[mrid] if hasattr(gens, "loc") else None
                unit = {"name": str(mrid), "target_p": float(row.get("target_p", 0.0)) if row is not None else 0.0}
                # SSH ACDCTerminal.connected (pypowsybl "connected"): an unconnected unit is not
                # part of the redispatch model; absent column = connected
                connected = row.get("connected", True) if row is not None else True
                unit["connected"] = True if connected is None or connected != connected else bool(connected)
                # a missing or NaN Pmax/Pmin is 0.0 (the unit is then fixed), one warning per unit
                defaulted = []
                for key in ("max_p", "min_p"):
                    value = row.get(key) if row is not None else None
                    if value is None or value != value:
                        unit[key] = 0.0
                        defaulted.append(key)
                    else:
                        unit[key] = float(value)
                if defaulted:
                    log.warning(
                        "generating unit %s has no %s in the CGM - defaulted to 0.0 "
                        "(its Pmax/Pmin redispatch limits are then 0.0)", mrid, " / ".join(defaulted),
                    )
                generating_unit_dict[str(mrid)] = unit
        else:
            for item in gens:
                mrid = getattr(item, "mrid", getattr(item, "id", str(item)))
                generating_unit_dict[str(mrid)] = {"name": str(mrid)}
    except Exception:
        log.exception("extract_equipment_dicts: failed to read network.get_generators()")

    return branch_dict, tap_changer_dict, generating_unit_dict
