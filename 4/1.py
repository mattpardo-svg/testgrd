"""run_pipeline.py — end-to-end driver: data ingestion (CGM, NCP), preprocessing (PST, RD,
N-X), the nested completeness / linearisation loop, and the result exports.

Parameters come from config/rao_config.yaml and config/load_flow_config.yaml (both required,
no fallback; config/*.default.yaml restores them). TSO names come from the Organisation
profile (tso_name_resolver.build_tso_name_map). The equipment-mRID set for AE parsing is
built from the network's lines and 2-winding transformers.
lf_results_branches.csv / lf_results_buses.csv are not produced.
"""

import collections
import datetime
import functools
import logging
import sys
from pathlib import Path
from typing import Optional


def _find_repo_root(start: Path) -> Path:
    """Walk upward from `start` to the first directory containing src/rao/data_model.py
    (the repo-root marker, independent of where this file lives)."""
    candidate = start.resolve()
    for _ in range(8):
        if (candidate / "src" / "rao" / "data_model.py").is_file():
            return candidate
        if candidate.parent == candidate:
            break
        candidate = candidate.parent
    raise RuntimeError(
        f"Could not locate the repo root (a directory containing "
        f"src/rao/data_model.py) by walking up from {start} — "
        f"checked up to 8 parent directories."
    )


REPO_ROOT = _find_repo_root(Path(__file__).parent)
SRC_ROOT = REPO_ROOT / "src" / "rao"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

# REPO_ROOT on sys.path too, so "from src.X import ..." style imports also resolve
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_model import (  # noqa: E402
    AcVerificationStatus,
    AssessedElement,
    Contingency,
    PST,
    RAKind,
    RDUnit,
    PowerBid,
)
from pre_milp import (  # noqa: E402
    run_time,
    cgmes_loader,
    tso_name_resolver,
    glsk_resolver,
    ncp_loader,
    ncp_parser,
    scanned_elements,
    inventory_summary,
    n0_sensitivity,
    hvdc_units,
    ct_units,
    pst_bounds,
    workflow_gates,
    pst_groups,
    lf_parameters,
    dc_conversion,
    log_summary,
)
from pre_milp.operators import OperatorDirectory  # noqa: E402
from milp import milp_builder, milp_solver  # noqa: E402
from milp.milp_builder import (  # noqa: E402
    CtMilpInputs,
    HvdcMilpInputs,
    NxMilpInputs,
    PstMilpInputs,
    RdMilpInputs,
    _get_monitored_branch_limits_and_ae_types,
)
from pre_milp.nx_security_analysis import compute_lf_nx, compute_tatl_data  # noqa: E402
from pre_milp.nx_sensitivity import (  # noqa: E402
    compute_sensitivities_nx, prune_sensitivity_rd_nx,
)
from pre_milp.cra_preprocessing import (  # noqa: E402
    build_curative_pst_group_constraints,
    compute_pst_cra_availability,
    compute_rd_cra_availability,
    validate_pst_group_kind,
    write_pst_cra_availability_csv,
)
from milp import ac_convergence_loop, dc_pass, nx_constraint_filter, post_ra_congestion_check, single_pass  # noqa: E402
from milp.ac_convergence_loop import NxLoopHooks, SENSITIVITY_INVALID_REASON  # noqa: E402
from milp.operating_point_state import OperatingPointState  # noqa: E402
from pre_milp import sensitivity_checks  # noqa: E402
import run_profile  # noqa: E402
from pre_milp.pst_support import exclude_three_winding_psts  # noqa: E402
from pre_milp.sensitivity_checks import SensitivityGuardError, guard_share_error, log_invalid  # noqa: E402
from post_milp import ct_results_export, nx_results_export, n0_results_export  # noqa: E402
from post_milp.n0_results_export import write_completeness_iterations_csv  # noqa: E402
from output_file_registry import delete_registered_output_files  # noqa: E402
from text_encoding import TEXT_ENCODING, not_utf8_message  # noqa: E402

log = logging.getLogger("run_pipeline")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

INPUT_DIR = REPO_ROOT / "input"
OUTPUT_DIR = REPO_ROOT / "output"


def _find_unique_zip_by_prefix(folder: Path, prefix: str) -> Path:
    """The one file in `folder` whose name starts with `prefix` (case-insensitive), e.g.
    "cgm" matches "cgm.zip" and "CGM_XXX.zip". Zero or several matches raise (no guessing)."""
    if not folder.is_dir():
        raise FileNotFoundError(
            f"Required input folder not found: {folder}. This pipeline reads "
            f"its CGM and NCP data from {INPUT_DIR}/ only: place one cgm*.zip "
            f"and one ncp*.zip there."
        )
    matches = sorted(
        f for f in folder.iterdir()
        if f.is_file() and f.suffix.lower() == ".zip" and f.name.lower().startswith(prefix.lower())
    )
    if not matches:
        raise FileNotFoundError(
            f"No file starting with '{prefix}' (case-insensitive) found in {folder} "
            f"— expected exactly one, e.g. '{prefix}.zip' or '{prefix.upper()}_XXX.zip'."
        )
    if len(matches) > 1:
        raise ValueError(
            f"Expected exactly one file starting with '{prefix}' in {folder}, "
            f"found {len(matches)}: {[m.name for m in matches]} — remove all but "
            f"the one that should actually be used."
        )
    return matches[0]


START_HOUR = "1030Z"       # the run timestamp when the CGM has no scenario time inside the NCP series

import yaml  # noqa: E402

RAO_CONFIG_PATH = REPO_ROOT / "src" / "rao" / "config" / "rao_config.yaml"
LOAD_FLOW_CONFIG_PATH = REPO_ROOT / "src" / "rao" / "config" / "load_flow_config.yaml"


def _load_yaml_config(path: Path) -> dict:
    """Load a required YAML config file — NO fallback to hardcoded
    defaults. Raises immediately and clearly if the file is missing,
    rather than silently using some in-code default, per explicit
    project decision: this config file is meant to be the single source
    of truth, committed to the repo, always present.
    """
    if not path.is_file():
        default_path = path.with_name(path.stem + ".default" + path.suffix)
        raise FileNotFoundError(
            f"Required config file not found: {path}. There is no fallback "
            f"to hardcoded defaults — this file must exist. If it's been "
            f"deleted or corrupted, restore it from {default_path}, which "
            f"holds a known-good baseline."
        )
    try:
        with open(path, encoding=TEXT_ENCODING) as f:
            return yaml.safe_load(f)
    except UnicodeDecodeError as exc:
        raise ValueError(not_utf8_message(path, exc)) from None


def _build_extra_lf_params(load_flow_config: dict) -> dict:
    """Load-flow YAML dict -> pp.loadflow.Parameters kwargs. voltage_init_mode, balance_type
    and component_mode are converted to pypowsybl enums: a plain string is accepted at
    construction but breaks the object later (AttributeError on repr)."""
    import pypowsybl as pp

    config = dict(load_flow_config)  # shallow copy — never mutate the loaded dict itself
    config["voltage_init_mode"] = getattr(pp.loadflow.VoltageInitMode, config["voltage_init_mode"])
    config["balance_type"] = getattr(pp.loadflow.BalanceType, config["balance_type"])
    config["component_mode"] = getattr(pp.loadflow.ComponentMode, config["component_mode"])
    return config


class ConfigKeyError(KeyError):
    """A key the code reads is missing from rao_config.yaml (clear message instead of a bare KeyError)."""

    def __str__(self) -> str:
        return str(self.args[0])


def _config_value(config: dict, keys: tuple, config_path: Path):
    """Return config[keys[0]][keys[1]]...; if a key is missing raise ConfigKeyError naming the full
    dotted key and the file (and the default file that holds the documented value)."""
    node = config
    for key in keys:
        if not isinstance(node, dict) or key not in node:
            default_path = config_path.with_name(config_path.stem + ".default" + config_path.suffix)
            raise ConfigKeyError(
                f"Missing config key '{'.'.join(keys)}' in {config_path}. Add it; "
                f"{default_path.name} holds the documented default."
            )
        node = node[key]
    return node


_NON_NEGATIVE_CONFIG_KEYS = [
    ("thresholds", "min_loading_threshold_pct"),
    ("thresholds", "min_loading_threshold_pct_nx"),
    ("thresholds", "sensitivity_threshold_pct_pst"),
    ("thresholds", "sensitivity_threshold_pct_rd"),
    ("thresholds", "sensitivity_threshold_pct_pst_nx"),
    ("thresholds", "sensitivity_threshold_pct_rd_nx"),
    ("thresholds", "max_change_ra_sum_threshold"),
    ("thresholds", "max_change_ra"),
    ("thresholds", "max_change_overload"),
    ("sensitivity_guard", "absolute_floor_ampere"),
    ("nx_constraint_filter", "margin_ampere"),
]
_SHARE_CONFIG_KEYS = [("sensitivity_guard", "max_dropped_share")]               # a number from 0 to 1
_AT_LEAST_ONE_INTEGER_CONFIG_KEYS = [
    ("iteration_caps", "max_linearisation_iterations"),
    ("iteration_caps", "max_completeness_iterations"),
    ("iteration_caps", "incremental_ac_fallback_max_steps"),
    ("performance", "lf_thread_count"),
    ("curative_ra", "tatl_fallback_duration_s"),
]
# null (not set) or a number: > 0 for the time limit, >= 0 for the gap
_OPTIONAL_POSITIVE_CONFIG_KEYS = [("performance", "solver_time_limit_seconds")]
_OPTIONAL_NON_NEGATIVE_CONFIG_KEYS = [("performance", "solver_mip_rel_gap")]
_BOOLEAN_CONFIG_KEYS = [("outputs", "write_nx_sensitivity_csvs"), ("nx_constraint_filter", "enabled")]

# workflow options: the RA types the optimisation may use and the calculation mode
RA_TYPE_VALUES = ("pst", "rd", "hvdc", "countertrade")
# the two HVDC penalty references (EUR/MW); read and checked only when hvdc is in ra_types
HVDC_PENALTY_REFERENCE_KEYS = (
    ("penalty_costs", "penalty_reference_hvdc_pra"), ("penalty_costs", "penalty_reference_hvdc_cra"),
)
# the two countertrade penalty references (EUR/MW); read and checked only when countertrade is in ra_types
CT_PENALTY_REFERENCE_KEYS = (
    ("penalty_costs", "penalty_reference_ct_pra"), ("penalty_costs", "penalty_reference_ct_cra"),
)
CALCULATION_MODE_VALUES = ("ac_iterative", "ac_single_pass", "dc")
# calculation modes that are reserved: named, but stopped at startup (none since dc is built)
CALCULATION_MODE_RESERVED = ()
# replaced by workflow_toggles.ra_types
_REMOVED_RA_TOGGLE_KEYS = ("enable_pst_remedial_actions", "enable_rd_remedial_actions")


def validate_ra_types(value, source) -> list:
    """The RA types of a run: a non-empty list without duplicates, values from RA_TYPE_VALUES
    (pst, rd, hvdc, countertrade). Anything else (an unknown name, a list of lists) raises
    ValueError naming `source`."""
    if not isinstance(value, (list, tuple)) or not value or not all(isinstance(v, str) for v in value):
        raise ValueError(
            f"Invalid ra_types {value!r} in {source}: must be a non-empty list with values from "
            f"{list(RA_TYPE_VALUES)}"
        )
    if len(set(value)) != len(value):
        raise ValueError(f"Invalid ra_types {list(value)!r} in {source}: duplicate value")
    unsupported = [v for v in value if v not in RA_TYPE_VALUES]
    if unsupported:
        raise ValueError(
            f"Invalid ra_types {list(value)!r} in {source}: {unsupported} not yet supported; "
            f"allowed values: {list(RA_TYPE_VALUES)}"
        )
    return list(value)


def validate_hvdc_penalty_references(config: dict, config_path) -> tuple[float, float]:
    """(preventive, curative) HVDC penalty reference in EUR/MW from penalty_costs
    (HVDC_PENALTY_REFERENCE_KEYS): each a number >= 0, otherwise ConfigKeyError (missing) or ValueError
    naming the dotted key, the value and the file. Called only when hvdc is in ra_types."""
    values = []
    for keys in HVDC_PENALTY_REFERENCE_KEYS:
        value = _config_value(config, keys, config_path)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not value >= 0:
            raise ValueError(
                f"Invalid config value '{'.'.join(keys)}' = {value!r} in {config_path}: must be a number >= 0"
            )
        values.append(float(value))
    return values[0], values[1]


def validate_ct_penalty_references(config: dict, config_path) -> tuple[float, float]:
    """(preventive, curative) countertrade penalty reference in EUR/MW from penalty_costs
    (CT_PENALTY_REFERENCE_KEYS): each a number >= 0, otherwise ConfigKeyError (missing) or ValueError
    naming the dotted key, the value and the file. Called only when countertrade is in ra_types."""
    values = []
    for keys in CT_PENALTY_REFERENCE_KEYS:
        value = _config_value(config, keys, config_path)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not value >= 0:
            raise ValueError(
                f"Invalid config value '{'.'.join(keys)}' = {value!r} in {config_path}: must be a number >= 0"
            )
        values.append(float(value))
    return values[0], values[1]


AE_CONTINGENCY_COMBINATION_VALUES = ("standard", "all_contingencies")


def validate_ae_contingency_combination(value, source) -> str:
    """workflow_toggles.ae_contingency_combination: standard or all_contingencies; anything else raises ConfigKeyError
    naming the key, the value, the file and the allowed values."""
    if not isinstance(value, str) or value not in AE_CONTINGENCY_COMBINATION_VALUES:
        raise ConfigKeyError(
            f"Invalid config value 'workflow_toggles.ae_contingency_combination' = {value!r} in {source}: "
            f"allowed values: {list(AE_CONTINGENCY_COMBINATION_VALUES)}"
        )
    return value


def ae_contingency_combination_of(config: dict, config_path) -> str:
    """The validated ae_contingency_combination of a config; a config without the key (the use-case configs of
    use_cases/ were written before it) means standard, the behaviour before the option."""
    toggles = config.get("workflow_toggles") if isinstance(config, dict) else None
    value = toggles.get("ae_contingency_combination", "standard") if isinstance(toggles, dict) else "standard"
    return validate_ae_contingency_combination(value, config_path)


def warn_ae_contingency_combination(value: str) -> None:
    """One warning per run when every assessed element is combined with all contingencies."""
    if value == "all_contingencies":
        log_summary.warn_once(
            log, "ae-contingency-combination",
            "workflow_toggles.ae_contingency_combination is all_contingencies: every assessed element is combined with all "
            "must-study contingencies (isCombinableWithContingency and the links of kind included are not applied; "
            "links of kind excluded still exclude)",
        )


def validate_calculation_mode(value, source) -> str:
    """calculation_mode: ac_iterative, ac_single_pass or dc; any other value raises ValueError
    listing the allowed values (a reserved mode would stop the run with "not yet supported")."""
    if value in CALCULATION_MODE_RESERVED:
        raise ValueError(f"calculation_mode {value} is not yet supported (in {source})")
    if value not in CALCULATION_MODE_VALUES:
        raise ValueError(
            f"Invalid calculation_mode {value!r} in {source}: allowed values: {list(CALCULATION_MODE_VALUES)}"
        )
    return value


def validate_config_ranges(config: dict, config_path: Path) -> None:
    """Thresholds must be numbers >= 0, iteration caps and lf_thread_count integers >= 1, the
    solver time limit null or > 0, the MIP gap null or >= 0 and switches booleans; otherwise
    ValueError naming the dotted key, the value and the file. The workflow options ra_types and
    calculation_mode are checked as lists / enums; the removed enable_*_remedial_actions keys are an
    error that names ra_types. The dc_mode factor table is checked only when calculation_mode is dc
    (the other modes do not use it)."""
    toggles = config.get("workflow_toggles") if isinstance(config, dict) else None
    removed = [k for k in _REMOVED_RA_TOGGLE_KEYS if isinstance(toggles, dict) and k in toggles]
    if removed:
        raise ValueError(
            f"Config key(s) {removed} in {config_path} were replaced by 'workflow_toggles.ra_types' "
            f"(a list with values from {list(RA_TYPE_VALUES)}): remove the old key(s) and set ra_types"
        )
    ra_types = validate_ra_types(_config_value(config, ("workflow_toggles", "ra_types"), config_path), config_path)
    if "hvdc" in ra_types:
        validate_hvdc_penalty_references(config, config_path)
    if "countertrade" in ra_types:
        validate_ct_penalty_references(config, config_path)
    calculation_mode = validate_calculation_mode(
        _config_value(config, ("workflow_toggles", "calculation_mode"), config_path), config_path,
    )
    ae_contingency_combination_of(config, config_path)
    if calculation_mode == "dc":
        dc_conversion.parse_current_factor_table(
            _config_value(config, ("dc_mode", "current_factor_by_nominal_voltage_kv"), config_path), config_path,
        )
    for keys in _NON_NEGATIVE_CONFIG_KEYS:
        value = _config_value(config, keys, config_path)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not value >= 0:
            raise ValueError(
                f"Invalid config value '{'.'.join(keys)}' = {value!r} in {config_path}: "
                f"must be a number >= 0"
            )
    for keys in _SHARE_CONFIG_KEYS:
        value = _config_value(config, keys, config_path)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
            raise ValueError(
                f"Invalid config value '{'.'.join(keys)}' = {value!r} in {config_path}: "
                f"must be a number from 0 to 1"
            )
    for keys in _AT_LEAST_ONE_INTEGER_CONFIG_KEYS:
        value = _config_value(config, keys, config_path)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(
                f"Invalid config value '{'.'.join(keys)}' = {value!r} in {config_path}: "
                f"must be an integer >= 1"
            )
    for keys, minimum_text, accepts in (
        *((k, "null or a number > 0", lambda v: v > 0) for k in _OPTIONAL_POSITIVE_CONFIG_KEYS),
        *((k, "null or a number >= 0", lambda v: v >= 0) for k in _OPTIONAL_NON_NEGATIVE_CONFIG_KEYS),
    ):
        value = _config_value(config, keys, config_path)
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not accepts(value):
            raise ValueError(
                f"Invalid config value '{'.'.join(keys)}' = {value!r} in {config_path}: "
                f"must be {minimum_text}"
            )
    for keys in _BOOLEAN_CONFIG_KEYS:
        value = _config_value(config, keys, config_path)
        if not isinstance(value, bool):
            raise ValueError(
                f"Invalid config value '{'.'.join(keys)}' = {value!r} in {config_path}: "
                f"must be true or false"
            )


# keys that were removed from rao_config.yaml (key path, why): a config that still sets one gets one warning, not an error
_REMOVED_CONFIG_KEYS = (
    (("ra_limits", "max_ra_count_per_timeframe_tso"),
     "the run optimises one timestamp, where it would equal ra_limits.max_ra_count_per_timestamp_tso"),
)


def warn_removed_config_keys(config: dict) -> None:
    """One warning "no longer used, ignored" per removed key that the config sets to a value (a null value is the
    documented "no limit" and stays silent), so older configs keep working."""
    for keys, reason in _REMOVED_CONFIG_KEYS:
        node = config
        for key in keys:
            node = node.get(key) if isinstance(node, dict) else None
        if node is not None:
            log.warning("Config key %s is no longer used, ignored (%s)", ".".join(keys), reason)


def resolve_tso_keyed_setting(setting: str, mapping, tsos, eic_of: dict, default_description: str):
    """A per-TSO config map with its keys turned into the TSO names of this run: a key is a TSO name or the EIC of a TSO.
    A key that is neither is reported (warn_unknown_tso_names); a TSO that no key matches gets one warning naming the
    setting, and its default applies. An unset map (None or empty) is returned as it is, without warnings."""
    if not mapping:
        return mapping
    warn_unknown_tso_names(setting, mapping, set(tsos) | set(eic_of.values()))
    resolved = {}
    for key, value in mapping.items():
        for tso in tsos:
            if key == tso or key == eic_of.get(tso):
                resolved[tso] = value
                break
    for tso in tsos:
        if tso not in resolved:
            log.warning("Config %s has no entry for TSO %s - %s applies", setting, tso, default_description)
    return resolved


def warn_unknown_tso_names(setting: str, names, known_tsos) -> None:
    """One warning per name of a per-TSO config map that is not a TSO of this run (the entry
    would otherwise be silently unused)."""
    for name in names:
        if name not in known_tsos:
            log.warning(
                "Config %s names TSO %r, which is not a TSO of this run - the entry is unused "
                "(known TSOs: %s)", setting, name, sorted(known_tsos),
            )


_rao_config = _load_yaml_config(RAO_CONFIG_PATH)
validate_config_ranges(_rao_config, RAO_CONFIG_PATH)
_load_flow_config = _load_yaml_config(LOAD_FLOW_CONFIG_PATH)


def validate_load_flow_config(config: dict, config_path: Path) -> float:
    """calculation_mode dc: dc_power_factor (the one power-factor setting; the DC mode converts MW
    to A with it) must be a number > 0, otherwise ValueError naming the key and the file. The other
    modes do not check it (pypowsybl gets it as always)."""
    if "dc_power_factor" not in config:
        raise ValueError(f"Missing config key 'dc_power_factor' in {config_path}")
    return dc_conversion.validate_dc_power_factor(config["dc_power_factor"], config_path)


def dc_factor_table(config: dict, config_path: Path):
    """The dc_mode factor table of rao_config.yaml. In calculation_mode dc the configured table
    (validate_config_ranges named a missing key); in the other modes a missing section or entry is
    the default table (every factor 1.0) and nothing is checked."""
    if config["workflow_toggles"]["calculation_mode"] == "dc":
        return _config_value(config, ("dc_mode", "current_factor_by_nominal_voltage_kv"), config_path)
    section = config.get("dc_mode")
    if not isinstance(section, dict) or "current_factor_by_nominal_voltage_kv" not in section:
        return dict(dc_conversion.DEFAULT_FACTOR_TABLE)
    return section["current_factor_by_nominal_voltage_kv"]


def _cfg(*keys):
    """Read one rao_config.yaml value by key path, e.g. _cfg("thresholds", "max_change_ra")."""
    return _config_value(_rao_config, keys, RAO_CONFIG_PATH)


# --- workflow_toggles ---
RA_TYPES = validate_ra_types(_cfg("workflow_toggles", "ra_types"), RAO_CONFIG_PATH)
CALCULATION_MODE = validate_calculation_mode(_cfg("workflow_toggles", "calculation_mode"), RAO_CONFIG_PATH)
ENABLE_NX_ANALYSIS = _cfg("workflow_toggles", "enable_nx_analysis")
AE_CONTINGENCY_COMBINATION = ae_contingency_combination_of(_rao_config, RAO_CONFIG_PATH)
PST_TAPING_DISCRETE = _cfg("workflow_toggles", "pst_taping_discrete")
RA_LIMIT_CONSTRAINT_HARD = _cfg("workflow_toggles", "ra_limit_constraint_hard")
SCANNED_ELEMENT_CONSTRAINT_HARD = _cfg("workflow_toggles", "scanned_element_constraint_hard")

# --- thresholds ---
MIN_LOADING_THRESHOLD_PCT = _cfg("thresholds", "min_loading_threshold_pct")
MIN_LOADING_THRESHOLD_PCT_NX = _cfg("thresholds", "min_loading_threshold_pct_nx")
SENSITIVITY_THRESHOLD_PCT_PST = _cfg("thresholds", "sensitivity_threshold_pct_pst")
SENSITIVITY_THRESHOLD_PCT_RD = _cfg("thresholds", "sensitivity_threshold_pct_rd")
SENSITIVITY_THRESHOLD_PCT_PST_NX = _cfg("thresholds", "sensitivity_threshold_pct_pst_nx")
SENSITIVITY_THRESHOLD_PCT_RD_NX = _cfg("thresholds", "sensitivity_threshold_pct_rd_nx")
MAX_CHANGE_RA_SUM_THRESHOLD = _cfg("thresholds", "max_change_ra_sum_threshold")
MAX_CHANGE_RA = _cfg("thresholds", "max_change_ra")
MAX_CHANGE_OVERLOAD = _cfg("thresholds", "max_change_overload")

# --- iteration_caps ---
MAX_LINEARISATION_ITERATIONS = _cfg("iteration_caps", "max_linearisation_iterations")
MAX_COMPLETENESS_ITERATIONS = _cfg("iteration_caps", "max_completeness_iterations")
INCREMENTAL_AC_FALLBACK_MAX_STEPS = _cfg("iteration_caps", "incremental_ac_fallback_max_steps")

# --- ra_limits ---
MAX_RA_COUNT_PER_TIMESTAMP_TSO = _cfg("ra_limits", "max_ra_count_per_timestamp_tso")
RA_LIMIT_PENALTY = _cfg("ra_limits", "ra_limit_penalty")
MAX_TAPS_PER_TSO = milp_builder.validate_max_taps_per_tso(_cfg("ra_limits", "max_taps_per_tso"))
# optional per-TSO overrides of the tap limit (empty = global limit for every TSO)
MAX_TAPS_PER_TSO_OVERRIDES = {
    str(tso): milp_builder.validate_max_taps_per_tso(limit, f"max_taps_per_tso_overrides[{tso!r}]")
    for tso, limit in (_cfg("ra_limits", "max_taps_per_tso_overrides") or {}).items()
}

# --- curative_ra ---
PST_GROUP_KIND_DEFAULT = validate_pst_group_kind(
    _cfg("curative_ra", "pst_group_kind_default")
)
PST_GROUP_KIND_PER_TSO = {
    str(tso): validate_pst_group_kind(kind)
    for tso, kind in (_cfg("curative_ra", "pst_group_kind_per_tso") or {}).items()
}
TATL_FALLBACK_DURATION_S = _cfg("curative_ra", "tatl_fallback_duration_s")

DEBUG_LOG_MODE = _cfg("logging_options", "debug_log_mode")

# --- performance, outputs ---
LF_THREAD_COUNT = _cfg("performance", "lf_thread_count")
SENSITIVITY_GUARD_ABSOLUTE_FLOOR_AMPERE = _cfg("sensitivity_guard", "absolute_floor_ampere")
SENSITIVITY_GUARD_MAX_DROPPED_SHARE = _cfg("sensitivity_guard", "max_dropped_share")
SOLVER_TIME_LIMIT_SECONDS = _cfg("performance", "solver_time_limit_seconds")
SOLVER_MIP_REL_GAP = _cfg("performance", "solver_mip_rel_gap")
WRITE_NX_SENSITIVITY_CSVS = _cfg("outputs", "write_nx_sensitivity_csvs")
NX_CONSTRAINT_FILTER_ENABLED = _cfg("nx_constraint_filter", "enabled")
NX_CONSTRAINT_FILTER_MARGIN_AMPERE = float(_cfg("nx_constraint_filter", "margin_ampere"))

# --- penalty_costs ---
PENALTY_REFERENCE_PST_PRA = _cfg("penalty_costs", "penalty_reference_pst_pra")
PENALTY_REFERENCE_RD_PRA = _cfg("penalty_costs", "penalty_reference_rd_pra")
PENALTY_COST_OVERLOAD_N0 = _cfg("penalty_costs", "penalty_cost_overload_n0")
PENALTY_COST_OVERLOAD_NX_PATL = _cfg("penalty_costs", "penalty_cost_overload_nx_patl")
PENALTY_COST_OVERLOAD_NX_TATL = _cfg("penalty_costs", "penalty_cost_overload_nx_tatl")
PENALTY_COST_OVERLOAD_NX_SCANNED = _cfg("penalty_costs", "penalty_cost_overload_nx_scanned")
SCANNED_ELEMENT_PENALTY_EUR_PER_KA = _cfg("penalty_costs", "scanned_element_penalty_eur_per_ka")

# --- load_flow_config.yaml: use_reactive_limits is passed on its own to every AC solve; the
# other parameters go together as EXTRA_LF_PARAMS (extra_lf_params=). ---
_extra_lf_params_full = _build_extra_lf_params(_load_flow_config)
USE_REACTIVE_LIMITS = _extra_lf_params_full.pop("use_reactive_limits")
EXTRA_LF_PARAMS = _extra_lf_params_full

# --- calculation_mode dc: the pf of the load flow config and the factor table of rao_config.yaml,
# as configured; main() validates both only in dc (the other modes ignore them) ---
DC_POWER_FACTOR = EXTRA_LF_PARAMS.get("dc_power_factor")
DC_CURRENT_FACTOR_TABLE = dc_factor_table(_rao_config, RAO_CONFIG_PATH)

# --- fixed parameters, not in the config files ---

# RAART: reference timestamp for the RD lead times. None -> radp = one hour before START_HOUR
# (milp_builder._compute_radp).
RAART = None

# Per-TSO penalty overrides: supported by the model, not configured.
PENALTY_REFERENCE_PST_PRA_PER_TSO = None  # no TSO overrides configured
PENALTY_REFERENCE_RD_PRA_PER_TSO = None   # no TSO overrides configured

# ACTIVATE_CRA_XNE_LINKAGE: fixed False. Before it can be True, parse_psts/parse_rd_units
# below must pass ae_path=profiles.get("AE") (the check needs each RA's linked XNEs).
ACTIVATE_CRA_XNE_LINKAGE = False



def _parse_each_input_file_once(run):
    """Run `run` inside ncp_parser.graph_cache(): every NCP file is parsed once for the whole run. The run time and
    START_HOUR, which the run may set from the CGM, are restored afterwards."""
    @functools.wraps(run)
    def wrapper(*args, **kwargs):
        start_hour = globals()["START_HOUR"]
        try:
            with ncp_parser.graph_cache(), run_time.run_time_scope():
                return run(*args, **kwargs)
        finally:
            globals()["START_HOUR"] = start_hour

    return wrapper


def _profiled_run(run):
    """The duration and memory of each step are logged; the summary table also when the run fails."""
    @functools.wraps(run)
    def wrapper(*args, **kwargs):
        run_profile.start("Setup and config")
        try:
            return run(*args, **kwargs)
        finally:
            run_profile.finish()
    return wrapper


@_profiled_run
@_parse_each_input_file_once
def main() -> None:
    global START_HOUR       # set from the CGM when its scenario time is inside the NCP series
    # fail fast, before any file is created or read: the options may have been set after import
    ra_types = validate_ra_types(RA_TYPES, "RA_TYPES")
    calculation_mode = validate_calculation_mode(CALCULATION_MODE, "CALCULATION_MODE")
    dc_power_factor = None
    if calculation_mode == "dc":
        # dc only (D6.3): the power factor and the factor table
        dc_power_factor = dc_conversion.validate_dc_power_factor(DC_POWER_FACTOR, LOAD_FLOW_CONFIG_PATH)
        dc_conversion.parse_current_factor_table(DC_CURRENT_FACTOR_TABLE, RAO_CONFIG_PATH)
    enable_pst = "pst" in ra_types
    enable_rd = "rd" in ra_types
    enable_hvdc = "hvdc" in ra_types
    enable_ct = "countertrade" in ra_types
    # read (and checked) only for an hvdc run: a config without the keys stays valid for the other runs
    penalty_reference_hvdc_pra, penalty_reference_hvdc_cra = (
        validate_hvdc_penalty_references(_rao_config, RAO_CONFIG_PATH) if enable_hvdc else (0.0, 0.0)
    )
    penalty_reference_ct_pra, penalty_reference_ct_cra = (
        validate_ct_penalty_references(_rao_config, RAO_CONFIG_PATH) if enable_ct else (0.0, 0.0)
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Log file: same format and filtering as the console, overwritten each run, attached here
    # so importing this module never creates output/. An explicit encoding is required: the
    # Windows default (cp1252) cannot encode characters such as "→" and logging would raise.
    _log_file_handler = logging.FileHandler(OUTPUT_DIR / "run_log.txt", mode="w", encoding=TEXT_ENCODING)
    _log_file_handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    logging.getLogger().addHandler(_log_file_handler)

    # remove result files of an earlier run (e.g. N-X files when N-X is now off): only names
    # in the output-file registry; after the log handler exists so deletions are logged
    warn_removed_config_keys(_rao_config)
    stale_files_removed = delete_registered_output_files(OUTPUT_DIR)
    if stale_files_removed:
        log.info(
            "Removed %d stale output file(s) from an earlier run: %s",
            len(stale_files_removed), ", ".join(stale_files_removed),
        )
    else:
        log.info("No stale output files from an earlier run to remove")

    log.info("calculation_mode=%s | ra_types=%s", calculation_mode, "+".join(ra_types))

    # DEBUG_LOG_MODE gates INFO verbosity only (WARNING and above always show). False: the
    # HiGHS logger is raised to hide its per-solve trace; True: the full trace prints.
    if not DEBUG_LOG_MODE:
        logging.getLogger("pyomo.contrib.appsi.solvers.highs").setLevel(logging.WARNING)
    milp_builder.DEBUG_LOG_MODE = DEBUG_LOG_MODE
    # the spec's once-per-run log lines are written again in a second run in this process
    milp_builder.reset_run_log_state()
    # run-wide settings read where the analyses and the solver are set up
    lf_parameters.LF_THREAD_COUNT = LF_THREAD_COUNT
    sensitivity_checks.ABSOLUTE_FLOOR_AMPERE = float(SENSITIVITY_GUARD_ABSOLUTE_FLOOR_AMPERE)
    sensitivity_checks.MAX_DROPPED_SHARE = float(SENSITIVITY_GUARD_MAX_DROPPED_SHARE)
    milp_solver.SOLVER_TIME_LIMIT_SECONDS = SOLVER_TIME_LIMIT_SECONDS
    milp_solver.SOLVER_MIP_REL_GAP = SOLVER_MIP_REL_GAP

    # =======================================================================
    # CGM load and base-case AC load flow
    # =======================================================================
    run_profile.mark("CGM load + base-case load flow")
    log.info("=== CGM load + base-case AC LF ===")
    cgm_zip_path = _find_unique_zip_by_prefix(INPUT_DIR, "cgm")
    log.info("Using CGM zip: %s", cgm_zip_path.name)
    network = cgmes_loader.load_cgm(cgm_zip_override=cgm_zip_path)
    # calculation_mode dc: the DC load flow and dc_converter replace the AC calculation of the
    # initial state; dc_kwargs hands the converter to every calculation function (empty in the AC
    # modes, whose calls stay as they were)
    dc_converter: Optional[dc_conversion.DcCurrentConverter] = None
    dc_kwargs: dict = {}
    if calculation_mode == "dc":
        log.info("calculation_mode dc: the initial AC load flow is not run; DC load flow, DC security analysis and "
                 "DC sensitivities give the currents (A) through dc_mode.current_factor_by_nominal_voltage_kv")
        cgmes_loader.run_initial_dc_loadflow(
            network, use_reactive_limits=USE_REACTIVE_LIMITS, extra_lf_params=EXTRA_LF_PARAMS,
        )
        dc_converter = dc_conversion.DcCurrentConverter.from_network(
            network, dc_power_factor, DC_CURRENT_FACTOR_TABLE, source=RAO_CONFIG_PATH,
        )
        dc_kwargs = {"dc_converter": dc_converter}
        converged = True
        cgmes_loader.print_network_summary(network, converged, load_flow="DC")
        lf_initial = cgmes_loader.build_lf_initial(network, **dc_kwargs)
    else:
        lf_result = cgmes_loader.run_initial_loadflow(
            network, use_reactive_limits=USE_REACTIVE_LIMITS, extra_lf_params=EXTRA_LF_PARAMS,
        )
        # run_initial_loadflow raises when the load flow does not converge, so from here on the
        # base case is converged by construction.
        converged = True
        cgmes_loader.print_network_summary(network, converged)

        lf_initial = cgmes_loader.build_lf_initial(network)
    branch_mrid_to_name = cgmes_loader.build_branch_mrid_to_name(network)
    branch_dict, tap_changer_dict, generating_unit_dict = cgmes_loader.extract_equipment_dicts(network)

    # CGMES GeneratingUnit mRID -> network generator id (the SynchronousMachine); a cost or a
    # shift key already keyed by the generator id maps to itself
    generator_id_by_unit_mrid: dict[str, str] = {}
    generators_by_unit_mrid: dict[str, list[str]] = {}
    try:
        generators_all = network.get_generators(all_attributes=True)
        if "CGMES.GeneratingUnit" in generators_all.columns:
            for generator_id, unit_mrid in generators_all["CGMES.GeneratingUnit"].items():
                if isinstance(unit_mrid, str) and unit_mrid:
                    generator_id_by_unit_mrid[ncp_parser.ncp_id(unit_mrid)] = str(generator_id)
                    generators_by_unit_mrid.setdefault(ncp_parser.ncp_id(unit_mrid), []).append(str(generator_id))
    except Exception:
        log.exception("CGMES.GeneratingUnit mapping not readable - unit costs and shift keys keyed by their own mRID")

    # equipment-mRID set for AE parsing: lines and 2-winding transformers, from the network
    # (extract_equipment_dicts().branch_dict has lines only); AE parsing stays branch-only
    cgm_equipment_mrids = set(branch_mrid_to_name.keys())

    # contingency equipment validation: 8 of the 9 CoMo equipment types
    # (cgmes_loader.build_equipment_mrid_to_type; "bus" is not mapped)
    equipment_mrid_to_type = cgmes_loader.build_equipment_mrid_to_type(network)

    # extract the CGM zip as well: the tap-changer and CurrentLimit side maps parse the per-TSO
    # EQ XML files directly
    cgm_extract_dir = INPUT_DIR / "_extracted_cgm"
    log_summary.reset_once()
    ncp_loader.clear_directory(cgm_extract_dir)
    cgmes_loader.extract_cgm_zip(cgm_zip_path, cgm_extract_dir)

    # =======================================================================
    # TSO name resolution
    # =======================================================================
    run_profile.mark("NCP discovery and parsing")
    log.info("=== TSO resolution ===")
    # the Organisation profile (Org-NineRealms_CD.xml) of the NCP zip is the only source of
    # TSO identity, so it is extracted first
    ncp_zip_path = _find_unique_zip_by_prefix(INPUT_DIR, "ncp")
    log.info("Using NCP zip: %s", ncp_zip_path.name)
    ncp_extract_dir = INPUT_DIR / "_extracted_ncp"
    ncp_loader.clear_directory(ncp_extract_dir)
    org_cd_path = ncp_loader.extract_ncp_zip(ncp_zip_path, ncp_extract_dir, require_organisation=False)
    if org_cd_path is not None:
        tso_name_map = tso_name_resolver.build_tso_name_map(org_cd_path)
        log.info("TSO_NAME_MAP (from Organisation profile): %s", sorted(tso_name_map.values()))
    else:
        tso_name_map = {}
        log.warning("No Organisation (CD) file in %s - the TSO of an NCP file is the EIC in its header (dcterms:creator) "
                    "and its name is that EIC", ncp_zip_path.name)

    # =======================================================================
    # NCP file discovery and timestamp list
    # =======================================================================
    log.info("=== NCP file discovery ===")
    # one pool of files per profile, whatever the creator of a file; an object id in two files of a profile stops the run
    pools = ncp_loader.classify_extracted_ncp_files(ncp_extract_dir, tso_name_map)
    ncp_parser.check_unique_ids(pools)
    file_groups = pools.by_label()                       # {TSO of the file: {profile: [files]}}, the TSO of an object
    operators = OperatorDirectory(                       # that names none; an operator in the content wins
        name_by_id=tso_name_map if org_cd_path is not None else {},
        label_by_eic={eic: label for label, eic in pools.label_eics().items()},
        has_organisation=org_cd_path is not None,
    )
    if org_cd_path is None:
        tso_name_map = {label: label for label in file_groups}
        log.info("TSO_NAME_MAP (creator EICs): %s", sorted(tso_name_map.values()))
    # a profile read as one: bids in one file refer to RAs in another
    pooled = {profile: ncp_parser.PooledFiles(files) for profile, files in pools.items()}
    # the run time: the scenario time of the CGM when it lies inside the NCP series (date and hour), else the start hour
    # of the pipeline with the per-reader fallbacks (pre_milp/run_time.py)
    _case_date = getattr(network, "case_date", None)
    _cgm_time = _case_date.astimezone(datetime.timezone.utc) if _case_date is not None else None
    _run_time, _why = run_time.choose_run_time(_cgm_time, run_time.ncp_series_spans(sorted(Path(ncp_extract_dir).rglob("*.xml"))))
    if _run_time is not None:
        START_HOUR = f"{_run_time.minute // 60:02d}{_run_time.minute % 60:02d}Z"
        run_time.set_run_time(_run_time)
        log.info("Run time from the CGM: %s - start hour %s, every NCP time point is read by date and time", _why, START_HOUR)
    else:
        log.warning("Run time: %s - the start hour %s of the pipeline is kept and each time-point reader falls back "
                    "(a series on one day by clock time, a series on several days by its earliest day)", _why, START_HOUR)
    log.info("NCP_FILES: %d file(s) of %d TSO(s) | run timestamp: %s", sum(len(files) for files in pools.values()),
             len(file_groups), START_HOUR)

    # =======================================================================
    # Per-TSO parsing, aggregated across the NCP-providing TSOs
    # =======================================================================
    log.info("=== NCP parsing (%d file TSO(s)) ===", len(file_groups))
    warn_ae_contingency_combination(AE_CONTINGENCY_COMBINATION)

    ae_flat_list: list[AssessedElement] = []
    co_flat_list: list[Contingency] = []
    pst_list: list[PST] = []
    rd_list: list[RDUnit] = []
    bid_list: list[PowerBid] = []
    rd_list_by_tso: dict[str, list[RDUnit]] = {}
    ct_file_labels: set = set()          # the file TSOs whose RA files hold countertrade RAs
    # HVDC remedial actions and balanced groups of all TSOs (parsed only when 'hvdc' is in ra_types)
    hvdc_units_all: list = []
    hvdc_groups_all: list = []
    # countertrade RAs and bids of all TSOs (parsed only when 'countertrade' is in ra_types)
    ct_ras_all: list = []
    ct_bids_all: list = []

    # merged over all TSOs: build_pst_tap_bounds_curative needs the full mapping after the loop
    tap_changer_to_transformer_all: dict[str, str] = {}

    # static CGM PATL (both sides), the fallback for a branch with neither an SHS entry nor a
    # BaseCaseCurrentLimit
    cgm_static_patl: dict[str, dict[int, float]] = cgmes_loader.build_cgm_static_patl(network)
    one_side_branch_mrids = cgmes_loader.one_side_branch_mrids(network)          # dangling lines: side 1 only
    phase_transformer_mrids = cgmes_loader.phase_transformer_mrids(network)

    # EQ files first: build_current_limit_side_map gives each AE's CurrentLimit side, which parse_assessed_elements
    # needs for the side-aware PATL. All EQ files of the CGM are read, whatever the name of the TSO.
    tap_changer_to_transformer: dict[str, str] = {}
    current_limit_side: dict[str, int] = {}
    eq_files = cgmes_loader.find_eq_files(cgm_extract_dir)
    if not eq_files:
        log.warning("No EQ file found in the CGM - PST TapChanger resolution and CurrentLimit side resolution will both be empty")
    for eq_file in eq_files:
        # one parse of the EQ file for both maps
        eq_graph = ncp_parser.load_eq_graph(eq_file)
        tap_changer_to_transformer.update(ncp_parser.build_tap_changer_to_transformer_map(eq_file, graph=eq_graph))
        # CurrentLimits are terminal-specific and the side varies per AE
        current_limit_side.update(ncp_parser.build_current_limit_side_map(eq_file, graph=eq_graph))
        if enable_ct:
            # the countertrade zones need the control-area units of the same file: keep them, parse it once
            ct_units.remember_control_area_units(eq_file, eq_graph)
        del eq_graph
    tap_changer_to_transformer_all.update(tap_changer_to_transformer)
    tap_changer_to_transformer_pst = ncp_parser.add_transformer_references(tap_changer_to_transformer, phase_transformer_mrids)

    # a bid or a contingency link in one file may refer to an RA in another, so the SIS, SHS and RA files are read as pools
    ncp_parser.warn_unmatched_ra_references(pooled["RA"], pooled.get("SIS"), "NCP files")
    for tso, profiles in sorted(file_groups.items()):
        log.info("--- Parsing %s ---", tso)
        for ae_file in profiles.get("AE", []):
            # assessed elements: the TSO of each is its operator in the content, else the TSO of its file
            ae_list = ncp_parser.parse_assessed_elements(
                ae_file, pooled.get("SHS"), tso, START_HOUR, cgm_static_patl,
                cgm_branch_mrids=cgm_equipment_mrids,
                current_limit_side=current_limit_side,
                sis_path=pooled.get("SIS"),
                one_side_branch_mrids=one_side_branch_mrids,
                ae_contingency_combination=AE_CONTINGENCY_COMBINATION,
                operators=operators,
            )
            ae_flat_list.extend(ncp_parser.exclude_without_usable_patl(ae_list, tso, START_HOUR))

        # contingencies
        for co_file in profiles.get("CO", []):
            co_flat_list.extend(ncp_parser.parse_contingencies(co_file, tso, START_HOUR, equipment_mrid_to_type))

        for sis_file in profiles.get("SIS", []):
            ncp_parser.warn_unsupported_sis_content(sis_file, tso)

        for ra_file in profiles.get("RA", []):
            ncp_parser.note_topology_remedial_actions(ra_file, tso)

            # PSTs: TapChanger mRIDs resolved to their PowerTransformer (pypowsybl indexes phase tap
            # changers by transformer). Not parsed at all when 'pst' is not in ra_types.
            if enable_pst:
                pst_list.extend(ncp_parser.parse_psts(
                    ra_file, tso, tap_changer_dict, tap_changer_to_transformer_pst, operators=operators))

            # RD units: not parsed at all when 'rd' is not in ra_types
            if enable_rd:
                rds, bids = ncp_parser.parse_rd_units(
                    ra_file, pooled.get("SIS"), tso, START_HOUR, generating_unit_dict,
                    warn_countertrade=not enable_ct,
                    generators_by_unit=generators_by_unit_mrid,
                    operators=operators,
                )
                rd_list.extend(rds)
                bid_list.extend(bids)

            # countertrade RAs and their bids: not parsed when 'countertrade' is not in ra_types
            if enable_ct:
                ct_ras, ct_bids = ct_units.parse_ct_remedial_actions(
                    ra_file, pooled.get("SIS"), tso, START_HOUR, operators=operators,
                )
                ct_ras_all.extend(ct_ras)
                ct_bids_all.extend(ct_bids)
                ct_file_labels.add(tso)

            # HVDC units (EquivalentInjectionAction) and balanced groups: not parsed when 'hvdc' is not in ra_types
            if enable_hvdc:
                hvdc_units_all.extend(ncp_parser.parse_hvdc_units(ra_file, tso, operators=operators))
                hvdc_groups_all.extend(ncp_parser.parse_balanced_ra_groups(ra_file, tso))

    # the TSO of every object is known now: the RD units per TSO, the TSOs of the run and the settings keyed per TSO
    for rd in rd_list:
        rd_list_by_tso.setdefault(rd.tso, []).append(rd)
    operators.log_unresolved()
    run_tsos = sorted(set(file_groups) | {unit.tso for unit in [*ae_flat_list, *pst_list, *rd_list, *ct_ras_all, *hvdc_units_all]})
    if org_cd_path is None:
        tso_name_map = {tso: tso for tso in run_tsos}
    ncp_loader.log_pool_summary(pools, {
        "AE": collections.Counter(ae.tso for ae in ae_flat_list),
        "CO": collections.Counter(co.tso for co in co_flat_list),
        "RA": collections.Counter(unit.tso for unit in [*pst_list, *rd_list, *ct_ras_all, *hvdc_units_all]),
    })
    # per-TSO settings: a key is a TSO name or an EIC; the run uses them under the TSO names
    tso_eic = pools.label_eics()
    for tso, eic in operators.eic_by_label.items():
        tso_eic.setdefault(tso, eic)
    max_taps_per_tso_overrides = resolve_tso_keyed_setting(
        "max_taps_per_tso_overrides", MAX_TAPS_PER_TSO_OVERRIDES, run_tsos, tso_eic, "max_taps_per_tso")
    max_ra_count_per_timestamp_tso = resolve_tso_keyed_setting(
        "max_ra_count_per_timestamp_tso", MAX_RA_COUNT_PER_TIMESTAMP_TSO, run_tsos, tso_eic, "no limit")
    pst_group_kind_per_tso = resolve_tso_keyed_setting(
        "pst_group_kind_per_tso", PST_GROUP_KIND_PER_TSO, run_tsos, tso_eic, "pst_group_kind_default")

    if enable_pst:
        # PSTs on a 3-winding transformer leave the list here, so no later step sees them
        pst_list[:] = exclude_three_winding_psts(network, pst_list)
        ncp_parser.check_unique_pst_rows(pst_list)
    log.info(
        "Aggregated: %d AEs, %d contingencies, %d PSTs, %d RD units, %d bids",
        len(ae_flat_list), len(co_flat_list), len(pst_list), len(rd_list), len(bid_list),
    )

    # `patl`: branch_mrid -> {ts: kA}
    patl: dict[str, dict[str, float]] = {
        ae.branch_mrid: ae.patl_ka for ae in ae_flat_list if ae.branch_mrid
    }

    # =======================================================================
    # lf_basecase.csv (needs `patl`)
    # =======================================================================
    run_profile.mark("Base-case outputs, inventory, countertrade setup")
    log.info("=== Writing lf_basecase.csv ===")
    cgmes_loader.write_lf_basecase_csv(
        network, lf_initial, patl, timestamp=START_HOUR, converged=converged,
        output_path=OUTPUT_DIR / "lf_basecase.csv",
    )

    log.info("=== writing co_flat_list.csv ===")
    ncp_parser.write_co_flat_list_csv(co_flat_list, OUTPUT_DIR / "co_flat_list.csv")

    # =======================================================================
    # Inventory summary
    # =======================================================================
    log.info("=== Inventory summary ===")
    inventory_summary.print_inventory_summary(
        tso_name_map=tso_name_map,
        tsos_with_data=run_tsos,
        ae_flat_list=ae_flat_list,
        co_flat_list=co_flat_list,
        pst_list=pst_list,
        rd_list=rd_list,
        bid_list=bid_list,
        timestamp=START_HOUR,
        hvdc_list=hvdc_units_all if enable_hvdc else None,
        node_count=len(network.get_buses()) if hasattr(network, "get_buses") else 0,
        branch_count=len(branch_mrid_to_name),
    )

    # =======================================================================
    # Countertrade: zones, units, weights (gated: 'countertrade' in ra_types)
    # =======================================================================
    ct_inputs: Optional[CtMilpInputs] = None
    ct_tsos: set = set()
    ct_ras_used: list = []              # the countertrade RAs that reached the model
    ct_bid_list: list = []
    ct_zones: dict = {}
    ct_zone_of_unit: dict = {}
    ct_tso_of_zone: dict = {}
    ct_boundaries_before: list = []     # the area boundaries of the original state (C11.3)
    if enable_ct:
        log.info("=== Countertrade zones, units and weights ===")
        ct_zones = ct_units.parse_bidding_zones(org_cd_path) if org_cd_path is not None else {}
        ct_area_units = ct_units.read_control_area_units(cgm_extract_dir)
        ct_zone_source = "the Organisation profile"
        if org_cd_path is None:
            # L6: no Organisation file - the zone EIC of an RA is the entsoe EIC of a CGM ControlArea
            ct_zones, ct_area_units = ct_units.zones_from_control_areas(ct_ras_all, cgm_extract_dir, ct_area_units)
            ct_zone_source = "the CGM ControlAreas"
            log.info(
                "Countertrade: no Organisation file - the bidding zone of an RA is the CGM ControlArea with its EIC (%d of %d zone EIC(s) found)",
                len(ct_zones), len({ra.zone_mrid for ra in ct_ras_all}),
            )
        ct_generators = ct_units.read_generator_table(network)
        ct_ras_resolved, ct_units_by_zone = ct_units.resolve_zone_units(
            ct_ras_all, ct_zones, ct_area_units, ct_generators, zone_source=ct_zone_source,
        )
        ct_weights_by_ra = ct_units.resolve_ct_weights(ct_ras_resolved, ct_units_by_zone, ct_bids_all)
        ct_rd_units, ct_bid_list, ct_bid_participation = ct_units.build_ct_milp_data(
            ct_ras_resolved, ct_bids_all, ct_units_by_zone, ct_weights_by_ra,
        )
        if ct_rd_units:
            # only with a usable countertrade RA (DECISIONS 77): the weights of every resolved RA and direction
            ct_results_export.write_ct_glsk_csv(
                ct_weights_by_ra, ct_ras_resolved, OUTPUT_DIR / "countertrade_glsk_resolved.csv", START_HOUR,
            )
            ct_ras_used = [ra for ra in ct_ras_resolved if ra.mrid in {unit.mrid for unit in ct_rd_units}]
            ct_zone_of_unit = ct_units.unit_zone_map(ct_zones, ct_area_units, ct_generators)
            ct_tso_of_zone = {zone.mrid: tso_name_map.get(zone.system_operator_mrid, "") for zone in ct_zones.values()}
            if org_cd_path is None:             # no system operator in the CGM areas: the TSO of the RAs that name the zone
                for ra in ct_ras_resolved:
                    ct_tso_of_zone[ra.zone_mrid] = ct_tso_of_zone.get(ra.zone_mrid) or ra.tso
            ct_boundaries_before = ct_results_export.read_area_boundaries(network)
            rd_list.extend(ct_rd_units)
            bid_list.extend(ct_bid_list)
            ct_tsos = {unit.tso for unit in ct_rd_units}
            ct_inputs = CtMilpInputs(
                bid_participation=ct_bid_participation,
                penalty_reference_pra=penalty_reference_ct_pra, penalty_reference_cra=penalty_reference_ct_cra,
            )
        else:
            log.info(
                "'countertrade' is in ra_types but the NCP has no usable countertrade RA - countertrade contributes "
                "nothing to this run (ct_inputs=None)"
            )

    # =======================================================================
    # Scanned element parsing
    # =======================================================================
    run_profile.mark("Scanned elements")
    log.info("=== Scanned element parsing ===")
    # max(i1, i2) is used here (no per-branch side resolution for scanned elements).
    lf_initial_ka = {b: max(sides.get(1, 0.0), sides.get(2, 0.0)) / 1000.0 for b, sides in lf_initial.items()}
    scanned_list = scanned_elements.build_scanned_element_list(
        ae_flat_list, START_HOUR, lf_initial=lf_initial_ka,
    )
    scanned_elements.write_scanned_element_csv(
        scanned_list, OUTPUT_DIR / "scanned_element_list.csv", START_HOUR,
        branch_mrid_to_name=cgmes_loader.build_branch_mrid_to_name(network),
    )

    # =======================================================================
    # PST preprocessing (gated: 'pst' in ra_types)
    # =======================================================================
    # no available PST: pst_inputs=None and the run continues (RD or N-X may still optimise)
    run_profile.mark("N-0 sensitivity and RA preprocessing")
    guard_pst_n0: dict = {}
    guard_rd_n0: dict = {}
    pst_inputs: Optional[PstMilpInputs] = None
    if enable_pst:
        available_preventive_psts = [
            pst for pst in pst_list if pst.available and pst.kind == RAKind.PREVENTIVE
        ]
        # pst_inputs is built when ANY available PST (either kind) exists
        available_curative_psts = [
            pst for pst in pst_list if pst.available and pst.kind == RAKind.CURATIVE
        ]
        if not available_preventive_psts and not available_curative_psts:
            log.warning(
                "No available preventive or curative PSTs found at timestamp %s — PST "
                "contributes nothing to this run (pst_inputs=None). Real PST "
                "kind(s) in this dataset: %s. This reflects the actual input "
                "data (every PST RA parsed is curative on this cut) — it is "
                "a genuine dataset boundary condition, not a pipeline error.",
                START_HOUR,
                {pst.mrid: pst.kind.value for pst in pst_list} if pst_list else "no PSTs parsed at all",
            )
        else:
            # one list per kind: everything preventive below reads preventive_pst_list;
            # PstMilpInputs carries pst_list (both kinds) so the AC loop updates r0 on every PST
            preventive_pst_list = [pst for pst in pst_list if pst.kind == RAKind.PREVENTIVE]

            # pypowsybl's TRANSFORMER_PHASE sensitivity is in A/degree: the builder needs degrees per
            # tap, and the PST thresholds compare the effect per tap (so it is read before them). Unique mRIDs: a PST with a preventive and a curative RA is two objects.
            pst_degrees_per_tap = cgmes_loader.build_pst_degrees_per_tap(
                network, sorted({str(pst.mrid) for pst in pst_list})
            )

            if available_preventive_psts:
                log.info("=== PST sensitivity matrix (timestamp=%s) ===", START_HOUR)
                sensitivity_pst_n0 = n0_sensitivity.compute_sensitivity_pst_n0(
                    network=network,
                    pst_list=pst_list,
                    ae_flat_list=ae_flat_list,
                    timestamp=START_HOUR,
                    lf_initial=lf_initial,
                    sensitivity_threshold_pct=SENSITIVITY_THRESHOLD_PCT_PST,
                    output_path=OUTPUT_DIR / "sensitivity_pst_n0.csv",
                    use_reactive_limits=USE_REACTIVE_LIMITS,
                    extra_lf_params=EXTRA_LF_PARAMS,
                    guard_out=guard_pst_n0,
                    **dc_kwargs,
                )
                sensitivity_pst_n0_pruned = n0_sensitivity.prune_sensitivity_by_threshold(
                    sensitivity_pst_n0, ae_flat_list, START_HOUR, lf_initial,
                    sensitivity_threshold_pct=SENSITIVITY_THRESHOLD_PCT_PST,
                    min_loading_threshold_pct=MIN_LOADING_THRESHOLD_PCT,
                    degrees_per_tap=pst_degrees_per_tap,
                )
            else:
                log.info(
                    "No available preventive PSTs — preventive PST sensitivity matrix "
                    "skipped (curative-only PST workflow)"
                )
                sensitivity_pst_n0_pruned = {}

            log.info("=== PST tap bounds ===")
            pst_tap_bounds = pst_bounds.build_pst_tap_bounds(preventive_pst_list)
            # counted per unique mRID (pst_tap_bounds is keyed by mRID)
            unique_pst_mrids = {str(pst.mrid) for pst in preventive_pst_list}
            log.info(
                "pst_tap_bounds: %d/%d unique PST(s) have valid bounds",
                len(pst_tap_bounds), len(unique_pst_mrids),
            )

            log.info("=== PST group equality pairs ===")
            pst_group_equality_pairs = pst_groups.build_pst_group_equality_pairs(preventive_pst_list, pst_tap_bounds)
            log.info("pst_group_equality_pairs: %s", pst_group_equality_pairs)

            pst_inputs = PstMilpInputs(
                pst_list=pst_list,
                pst_tap_bounds=pst_tap_bounds,
                pst_group_equality_pairs=pst_group_equality_pairs,
                sensitivity_pst_n0_pruned=sensitivity_pst_n0_pruned,
                pst_degrees_per_tap=pst_degrees_per_tap,
                penalty_reference_pst_pra=PENALTY_REFERENCE_PST_PRA,
                penalty_reference_pst_pra_per_tso=PENALTY_REFERENCE_PST_PRA_PER_TSO,
                pst_taping_discrete=PST_TAPING_DISCRETE,
                max_taps_per_tso=MAX_TAPS_PER_TSO,
                max_taps_per_tso_overrides=max_taps_per_tso_overrides,
            )
    else:
        log.info("'pst' is not in ra_types — PST preprocessing skipped entirely (parse_psts was never called above)")

    # =======================================================================
    # RD preprocessing (gated: 'rd' in ra_types)
    # =======================================================================
    rd_inputs: Optional[RdMilpInputs] = None
    if enable_rd or ct_inputs is not None:
        available_preventive_rds = [
            rd for rd in rd_list if rd.available and rd.kind == RAKind.PREVENTIVE
        ]
        # curative RD does not need a preventive RD unit (workflow_gates.rd_inputs_required)
        if not workflow_gates.rd_inputs_required(rd_list, ENABLE_NX_ANALYSIS):
            log.warning(
                "No available preventive RD units found at timestamp %s (and no "
                "available curative RD unit with N-X analysis enabled) — RD "
                "contributes nothing to this run (rd_inputs=None). Real RD "
                "kind(s) in this dataset: %s.",
                START_HOUR,
                {rd.mrid: rd.kind.value for rd in rd_list} if rd_list else "no RD units parsed at all",
            )
        else:
            if available_preventive_rds:
                log.info("=== RD sensitivity matrix (timestamp=%s) ===", START_HOUR)
                rd_n0_branch_max: dict = {}
                sensitivity_rd_n0 =n0_sensitivity.compute_sensitivity_rd_n0(
                    network=network,
                    rd_list=rd_list,
                    ae_flat_list=ae_flat_list,
                    timestamp=START_HOUR,
                    lf_initial=lf_initial,
                    sensitivity_threshold_pct=SENSITIVITY_THRESHOLD_PCT_RD,
                    output_path=OUTPUT_DIR / "sensitivity_rd_n0.csv",
                    use_reactive_limits=USE_REACTIVE_LIMITS,
                    extra_lf_params=EXTRA_LF_PARAMS,
                    guard_out=guard_rd_n0,
                    branch_max_out=rd_n0_branch_max,
                    **dc_kwargs,
                )
                sensitivity_rd_n0_pruned = n0_sensitivity.prune_sensitivity_by_threshold(
                    sensitivity_rd_n0, ae_flat_list, START_HOUR, lf_initial,
                    sensitivity_threshold_pct=SENSITIVITY_THRESHOLD_PCT_RD,
                    min_loading_threshold_pct=MIN_LOADING_THRESHOLD_PCT,
                    branch_max=rd_n0_branch_max,
                )
            else:
                # the N-0 RD sensitivity is preventive-only (compute_sensitivity_rd_n0 raises
                # without a preventive unit): empty for a curative-only workflow
                log.info(
                    "No available preventive RD units — preventive RD sensitivity "
                    "matrix skipped (curative-only RD workflow)"
                )
                sensitivity_rd_n0_pruned = {}

            log.info("=== GLSK resolution (per TSO) ===")
            glsk_all: dict[str, dict[str, float]] = {}
            glsk_tso_by_ra_mrid: dict[str, str] = {}
            for tso, rds_for_tso in sorted(rd_list_by_tso.items()):
                if not rds_for_tso:
                    continue
                # no output_path here: glsk_resolved.csv is written once for all TSOs below
                glsk_for_tso = glsk_resolver.resolve_glsk(
                    pooled.get("SIS"), rds_for_tso, tso=tso,
                    valid_generator_mrids=set(network.get_generators().index),
                    timestamp=START_HOUR,
                    connected_generator_mrids={
                        gu for gu, data in generating_unit_dict.items() if data.get("connected", True)
                    },
                    generators_by_unit=generators_by_unit_mrid,
                    ra_path=pooled.get("RA"),
                )
                glsk_all.update(glsk_for_tso)
                glsk_tso_by_ra_mrid.update({ra_mrid: tso for ra_mrid in glsk_for_tso})
            if enable_rd:
                glsk_resolver.write_glsk_csv(
                    glsk_all, OUTPUT_DIR / "glsk_resolved.csv", START_HOUR,
                    tso_by_ra_mrid=glsk_tso_by_ra_mrid,
                )
                # the curative balance is per contingency over the RAs of all TSOs, so the check runs on all keys
                if ENABLE_NX_ANALYSIS:
                    glsk_resolver.warn_curative_no_net_movement(rd_list, bid_list, glsk_all, co_flat_list, START_HOUR)

            log.info("=== Generating unit start-up / shut-down costs (SIS UnitCostSchedule, ER) ===")
            unit_costs: dict = {}
            for tso, profiles_for_tso in sorted(file_groups.items()):
                unit_costs.update(ncp_parser.parse_unit_costs(
                    ncp_parser.PooledFiles(profiles_for_tso["SIS"]) if "SIS" in profiles_for_tso else None,
                    ncp_parser.PooledFiles(profiles_for_tso["ER"]) if "ER" in profiles_for_tso else None,
                    tso, START_HOUR, generator_id_by_unit_mrid,
                ))

            log.info("=== RA group volume limits (per TSO) ===")
            ra_groups_all: dict[str, dict] = {}
            for tso in sorted((set(file_groups) if enable_rd else set()) | ct_file_labels):
                for ra_file in file_groups[tso].get("RA", []):
                    ra_groups_all.update(ncp_parser.parse_ra_groups(ra_file, pooled.get("SIS"), tso, START_HOUR))

            rd_inputs = RdMilpInputs(
                rd_list=rd_list,
                bid_list=bid_list,
                glsk=glsk_all,
                sensitivity_rd_n0_pruned=sensitivity_rd_n0_pruned,
                generator_data=generating_unit_dict,
                penalty_reference_rd_pra=PENALTY_REFERENCE_RD_PRA,
                penalty_reference_rd_pra_per_tso=PENALTY_REFERENCE_RD_PRA_PER_TSO,
                max_ra_count_per_timestamp_tso=max_ra_count_per_timestamp_tso,
                ra_limit_penalty=RA_LIMIT_PENALTY,
                ra_limit_constraint_hard=RA_LIMIT_CONSTRAINT_HARD,
                ra_groups=ra_groups_all,
                raart=RAART,
                unit_costs=unit_costs,
                bid_participation=ct_inputs.bid_participation if ct_inputs is not None else None,
                penalty_reference_ct_pra=penalty_reference_ct_pra,
                penalty_reference_ct_cra=penalty_reference_ct_cra,
            )
    else:
        log.info("'rd' is not in ra_types — RD preprocessing skipped entirely (parse_rd_units was never called above)")

    # =======================================================================
    # HVDC preprocessing (gated: 'hvdc' in ra_types; docs/specs/hvdc_ra_specs.md)
    # =======================================================================
    hvdc_inputs: Optional[HvdcMilpInputs] = None
    guard_hvdc_n0: dict = {}
    sensitivity_hvdc_n0_pruned: dict = {}
    if enable_hvdc:
        hvdc_units_usable, hvdc_groups_usable = hvdc_units.validate_hvdc_structure(hvdc_units_all, hvdc_groups_all)
        if hvdc_units_usable and not ENABLE_NX_ANALYSIS and any(u.kind == RAKind.CURATIVE for u in hvdc_units_usable):
            # curative HVDC RAs act only through the N-X constraints (H1.4, as for curative RD)
            ignored_curative = sorted(u.mrid for u in hvdc_units_usable if u.kind == RAKind.CURATIVE)
            log.info(
                "Curative HVDC remedial action(s) %s need N-X analysis, which is off - ignored", ignored_curative,
            )
            hvdc_units_usable = [u for u in hvdc_units_usable if u.kind == RAKind.PREVENTIVE]
        if not hvdc_units_usable:
            log.info(
                "'hvdc' is in ra_types but the NCP has no usable HVDC remedial action - HVDC contributes "
                "nothing to this run (hvdc_inputs=None)"
            )
        else:
            # the EI of an HVDC unit must not also be a unit of an RD GLSK (H4.6); both units must be in the
            # solved main network (H4.3); a description other than "HVDC..." is only a warning (H2.2)
            hvdc_units.validate_units_not_in_rd_glsk(hvdc_units_usable, rd_inputs.glsk if rd_inputs is not None else {})
            hvdc_network_data = hvdc_units.read_unit_network_data(network, hvdc_units_usable)
            hvdc_units.check_units_in_solved_network(hvdc_units_usable, hvdc_network_data)
            hvdc_units.warn_ei_without_hvdc_description(hvdc_units_usable, cgm_extract_dir)
            # a curative RA without a RangeConstraint of its own uses the preventive ranges of the same unit (H6.3a)
            preventive_ranges_by_ei = {u.ei_mrid: u.ranges for u in hvdc_units_usable if u.kind == RAKind.PREVENTIVE}
            hvdc_limits = {
                unit.mrid: hvdc_units.resolve_operating_limits(
                    unit, hvdc_network_data[unit.ei_mrid]["p0"],
                    hvdc_network_data[unit.ei_mrid]["min_p"], hvdc_network_data[unit.ei_mrid]["max_p"],
                    ranges=unit.ranges or (preventive_ranges_by_ei.get(unit.ei_mrid, []) if unit.kind == RAKind.CURATIVE else []),
                )
                for unit in hvdc_units_usable
            }
            log.info(
                "Considered equivalent injections to model HVDC converters: %s",
                "; ".join(
                    f"{unit.ei_mrid} ({unit.kind.value}, group {unit.group_mrid}, P0 {hvdc_network_data[unit.ei_mrid]['p0']:.3f} MW, "
                    f"limits [{hvdc_limits[unit.mrid][0]:.3f}, {hvdc_limits[unit.mrid][1]:.3f}] MW, factor {unit.penalty_factor:g})"
                    for unit in hvdc_units_usable
                ),
            )
            preventive_hvdc_eis = sorted({u.ei_mrid for u in hvdc_units_usable if u.kind == RAKind.PREVENTIVE})
            if preventive_hvdc_eis:
                log.info("=== HVDC sensitivity matrix (timestamp=%s) ===", START_HOUR)
                sensitivity_hvdc_n0 = n0_sensitivity.compute_sensitivity_hvdc_n0(
                    network, preventive_hvdc_eis, ae_flat_list, START_HOUR,
                    lf_initial=lf_initial, sensitivity_threshold_pct=SENSITIVITY_THRESHOLD_PCT_RD,
                    use_reactive_limits=USE_REACTIVE_LIMITS, extra_lf_params=EXTRA_LF_PARAMS,
                    guard_out=guard_hvdc_n0, **dc_kwargs,
                )
                sensitivity_hvdc_n0_pruned = n0_sensitivity.prune_sensitivity_by_threshold(
                    sensitivity_hvdc_n0, ae_flat_list, START_HOUR, lf_initial,
                    sensitivity_threshold_pct=SENSITIVITY_THRESHOLD_PCT_RD,
                    min_loading_threshold_pct=MIN_LOADING_THRESHOLD_PCT,
                )
            hvdc_inputs = HvdcMilpInputs(
                units=hvdc_units_usable,
                limits=hvdc_limits,
                original_p_mw={ei: info["p0"] for ei, info in hvdc_network_data.items()},
                current_p_mw={ei: info["p0"] for ei, info in hvdc_network_data.items()},
                sensitivity_hvdc_n0=sensitivity_hvdc_n0_pruned,
                penalty_reference_pra=penalty_reference_hvdc_pra,
                penalty_reference_cra=penalty_reference_hvdc_cra,
                # an EI that a contingency removes makes the curative group unavailable in it (H6.2, H11.1)
                outage_eis_by_contingency={
                    co.mrid: frozenset(set(co.equipment_mrids) & {u.ei_mrid for u in hvdc_units_usable})
                    for co in co_flat_list if set(co.equipment_mrids) & {u.ei_mrid for u in hvdc_units_usable}
                },
            )
    hvdc_nx_eis = sorted({u.ei_mrid for u in hvdc_inputs.units}) if hvdc_inputs is not None else []

    # =======================================================================
    # N-X preprocessing (gated: ENABLE_NX_ANALYSIS)
    # =======================================================================
    # runs even without dispatchable RA capacity (the slacks absorb what nothing can resolve)
    nx_inputs: Optional[NxMilpInputs] = None
    if ENABLE_NX_ANALYSIS:
        guard_pst_nx: dict = {}
        guard_rd_nx: dict = {}
        log.info("=== N-X sensitivity (lf_nx, sensitivity_pst_nx, sensitivity_rd_nx) ===")
        run_profile.mark("N-X security analysis")
        lf_nx, skipped_contingencies = compute_lf_nx(
            network, co_flat_list, ae_flat_list, START_HOUR,
            lf_initial=lf_initial,
            output_path=OUTPUT_DIR / "lf_nx_initial.csv",
            use_reactive_limits=USE_REACTIVE_LIMITS,
            extra_lf_params=EXTRA_LF_PARAMS,
            **dc_kwargs,
        )
        if skipped_contingencies:                      # the summary per reason was logged by compute_lf_nx
            log.debug("compute_lf_nx: skipped must-study contingencies: %s", skipped_contingencies)

        run_profile.mark("N-X sensitivity")
        available_nx_psts = [p for p in pst_list if p.available]
        available_nx_rds = [r for r in rd_list if r.available]
        hvdc_nx_out: dict = {}
        limit_by_branch_nx, _branch_ae_types_nx = _get_monitored_branch_limits_and_ae_types(
            ae_flat_list, START_HOUR, lf_initial,
        )
        # degrees per tap of all PSTs (the preventive block's value may not exist): the N-X PST threshold and the N-X
        # loadings export use them; unique mRIDs; {} for an empty pst_list
        pst_degrees_per_tap_for_nx = (
            cgmes_loader.build_pst_degrees_per_tap(network, sorted({str(p.mrid) for p in pst_list}))
            if pst_list else {}
        )
        # PST, RD and HVDC sensitivities from one analysis (same values as separate analyses)
        sensitivity_pst_nx, sensitivity_rd_nx = compute_sensitivities_nx(
            network, co_flat_list, pst_list, rd_list, ae_flat_list, START_HOUR,
            lf_nx=lf_nx, lf_initial=lf_initial,
            min_loading_threshold_pct_nx=MIN_LOADING_THRESHOLD_PCT_NX,
            with_pst=bool(available_nx_psts), with_rd=bool(available_nx_rds),
            pst_output_path=OUTPUT_DIR / "sensitivity_pst_nx.csv" if WRITE_NX_SENSITIVITY_CSVS else None,
            rd_output_path=OUTPUT_DIR / "sensitivity_rd_nx.csv" if WRITE_NX_SENSITIVITY_CSVS else None,
            use_reactive_limits=USE_REACTIVE_LIMITS, extra_lf_params=EXTRA_LF_PARAMS,
            pst_guard_out=guard_pst_nx, rd_guard_out=guard_rd_nx,
            ct_prune_limits=limit_by_branch_nx, ct_prune_threshold_pct=SENSITIVITY_THRESHOLD_PCT_RD_NX,
            sensitivity_threshold_pct_pst_nx=SENSITIVITY_THRESHOLD_PCT_PST_NX, degrees_per_tap=pst_degrees_per_tap_for_nx,
            **({"hvdc_ei_mrids": hvdc_nx_eis, "hvdc_out": hvdc_nx_out} if hvdc_nx_eis else {}),
            **dc_kwargs,
        )
        if not available_nx_psts:
            log.info("No available PSTs at all — sensitivity_pst_nx computed as empty ({}), not skipped")
        if not available_nx_rds:
            log.info("No available RD units at all — sensitivity_rd_nx computed as empty ({}), not skipped")
        initial_invalid_sensitivity_nx = {**guard_pst_nx, **guard_rd_nx, **hvdc_nx_out.get("guard", {})}
        log_invalid("N-X", initial_invalid_sensitivity_nx, "initial state")

        # Filter 2 (sensitivity pruning): the PST table is already pruned (done per contingency while reading); the RD
        # matrix above has Filter 1 only (CSV output) and is pruned here. limit_by_branch is shared.
        sensitivity_pst_nx_pruned = sensitivity_pst_nx
        sensitivity_rd_nx_pruned = prune_sensitivity_rd_nx(
            sensitivity_rd_nx, limit_by_branch_nx, SENSITIVITY_THRESHOLD_PCT_RD_NX,
        )
        # HVDC: the same Filter 2 as RD (the function only reads the keys); a unit with all-zero sensitivities in
        # a contingency makes its group unavailable there (curative, H8.2)
        sensitivity_hvdc_nx_pruned = prune_sensitivity_rd_nx(
            hvdc_nx_out.get("sensitivity", {}), limit_by_branch_nx, SENSITIVITY_THRESHOLD_PCT_RD_NX,
        )
        hvdc_zero_units_nx = dict(hvdc_nx_out.get("zero_units", {}))

        run_profile.mark("TATL data")
        log.info("=== TATL data ===")
        tatl_data = compute_tatl_data(
            network, ae_flat_list, START_HOUR, output_path=OUTPUT_DIR / "tatl_data.csv",
            fallback_duration_seconds=TATL_FALLBACK_DURATION_S,
        )

        run_profile.mark("Curative preprocessing")
        log.info("=== Curative RD availability per branch (reaction time vs TATL duration) ===")
        # every monitored branch (secured and scanned), with the fallback duration where there is
        # no real TATL: the branches the MILP evaluates, with the same helper (R9 rule 10)
        if enable_rd:
            compute_rd_cra_availability(
                [rd for rd in rd_list if not rd.countertrade], tatl_data, output_path=OUTPUT_DIR / "rd_cra_availability.csv",
                branch_mrids=list(limit_by_branch_nx), fallback_duration_seconds=TATL_FALLBACK_DURATION_S,
            )

        log.info("=== Curative PST availability ===")
        pst_cra_availability = compute_pst_cra_availability(
            pst_list, tatl_data, output_path=OUTPUT_DIR / "pst_cra_availability.csv",
            contingencies=co_flat_list,
        )
        # {} when no curative PST has reaction_time_minutes / time_per_stage_minutes (logged)
        if not pst_cra_availability:
            log.info(
                "pst_cra_availability computed as empty ({}) — either no curative "
                "PSTs are available, or (confirmed dataset fact on this cut) the "
                "available ones are missing reaction_time_minutes/"
                "time_per_stage_minutes. Correctly computed empty, not skipped."
            )

        log.info("=== Curative PST tap bounds (per TSO, with fallback to operating bounds) ===")
        pst_tap_bounds_curative_raw: dict[str, tuple[int, int]] = {}
        for ra_file in pools["RA"]:
            pst_tap_bounds_curative_raw.update(
                ncp_parser.build_pst_tap_bounds_curative(ra_file, tap_changer_to_transformer_all)
            )
        # fallback: each PST's own operating_tap_min/max (pst_inputs.pst_tap_bounds can be None
        # when there is no preventive PST)
        pst_tap_bounds_curative = pst_bounds.build_curative_tap_bounds_with_fallback(
            [pst for pst in pst_list if pst.kind == RAKind.CURATIVE],
            pst_tap_bounds_curative_raw,
        )

        # curative PST group constraint; the group kind is not in the NCP data ->
        # config curative_ra.pst_group_kind_default
        curative_psts_for_groups = [pst for pst in pst_list if pst.kind == RAKind.CURATIVE]
        filtered_pst_cra_availability, curative_pst_group_triples = (
            build_curative_pst_group_constraints(
                curative_psts_for_groups, pst_cra_availability, pst_tap_bounds_curative,
                co_flat_list, PST_GROUP_KIND_DEFAULT, pst_group_kind_per_tso,
            )
        )
        if filtered_pst_cra_availability != pst_cra_availability:
            pst_cra_availability = filtered_pst_cra_availability
            write_pst_cra_availability_csv(
                pst_cra_availability, OUTPUT_DIR / "pst_cra_availability.csv",
            )
        log.info(
            "Curative PST groups: %d group-equality pair(s) over contingencies "
            "(PSTGroupKind default = %s, per-TSO overrides = %s)",
            len(curative_pst_group_triples), PST_GROUP_KIND_DEFAULT, pst_group_kind_per_tso,
        )

        nx_inputs = NxMilpInputs(
            lf_nx=lf_nx,
            sensitivity_pst_nx=sensitivity_pst_nx_pruned,
            sensitivity_rd_nx=sensitivity_rd_nx_pruned,
            penalty_cost_overload_nx_patl=PENALTY_COST_OVERLOAD_NX_PATL,
            min_loading_threshold_pct_nx=MIN_LOADING_THRESHOLD_PCT_NX,
            tatl_data=tatl_data,
            penalty_cost_overload_nx_tatl=PENALTY_COST_OVERLOAD_NX_TATL,
            pst_cra_availability=pst_cra_availability,
            pst_tap_bounds_curative=pst_tap_bounds_curative,
            curative_pst_group_triples=curative_pst_group_triples,
            activate_cra_xne_linkage=ACTIVATE_CRA_XNE_LINKAGE,
            penalty_cost_overload_nx_scanned=PENALTY_COST_OVERLOAD_NX_SCANNED,
            tatl_fallback_duration_seconds=TATL_FALLBACK_DURATION_S,
            sensitivity_hvdc_nx=sensitivity_hvdc_nx_pruned,
            drop_zero_pst_terms=SENSITIVITY_THRESHOLD_PCT_PST_NX > 0.0,
            drop_zero_rd_terms=SENSITIVITY_THRESHOLD_PCT_RD_NX > 0.0,
            constraint_filter_enabled=NX_CONSTRAINT_FILTER_ENABLED,
            constraint_filter_margin_ampere=NX_CONSTRAINT_FILTER_MARGIN_AMPERE,
        )

    else:
        log.info("ENABLE_NX_ANALYSIS=False — N-X preprocessing skipped entirely (nx_inputs=None)")

    # =======================================================================
    # Data-driven "nothing to optimise" check
    # =======================================================================
    # with both flags True the data can still give pst_inputs=None and rd_inputs=None: exit
    # gracefully unless N-X is enabled (N-X alone is something to optimise)
    if pst_inputs is None and rd_inputs is None and nx_inputs is None and hvdc_inputs is None:
        log.warning(
            "Neither PST, RD, HVDC, countertrade, nor N-X has anything to optimise for this run "
            "(all resolved to None after preprocessing) — pipeline "
            "complete with data-loading/preprocessing outputs only."
        )
        log.info("=== Pipeline complete (preprocessing outputs only; PST/RD/N-X analysis skipped) ===")
        log.info("Outputs written to: %s", OUTPUT_DIR)
        for name in (
            "lf_basecase.csv", "scanned_element_list.csv",
        ):
            log.info("  %s", name)
        return

    # =======================================================================
    # MILP construction and solve
    # =======================================================================
    # one operating-point state, the linearisation loop nested inside the completeness loop;
    # PST-only, RD-only, PST+RD and N-X-only all run through it
    workflow_label = (
        "PST+RD" if pst_inputs is not None and rd_inputs is not None
        else "RD-only" if rd_inputs is not None
        else "PST-only" if pst_inputs is not None
        else "N-X-only"
    )
    if hvdc_inputs is not None or ct_inputs is not None:
        # rd_inputs also carries the countertrade RAs: RD is named only when the run has the rd type
        workflow_label = "+".join(
            name for name, inputs in (
                ("PST", pst_inputs), ("RD", rd_inputs if enable_rd else None), ("HVDC", hvdc_inputs), ("CT", ct_inputs),
            ) if inputs is not None
        )
    run_profile.mark("MILP build")
    log.info("=== MILP construction, solve, and AC verification loop (%s) ===", workflow_label)

    # Invalid sensitivities at the FIRST computation: there is no earlier valid set to reuse and no
    # model to report, so the run stops here (no silent fallback). The minimal outputs below say why.
    _initial_invalid = {
        **{f"N-0 branch {b}": r for b, r in {**guard_pst_n0, **guard_rd_n0, **guard_hvdc_n0}.items()},
        **({f"N-X contingency {c}": r for c, r in initial_invalid_sensitivity_nx.items()} if nx_inputs is not None else {}),
    }
    # the coverage line of the guard, and its stop when too large a share of the pairs was left out
    _share_error = guard_share_error(nx_inputs.lf_nx, sensitivity_checks.MAX_DROPPED_SHARE) if nx_inputs is not None else None
    if _initial_invalid or _share_error:
        import types as _types
        if _initial_invalid:
            _what, _rule = sorted(_initial_invalid.items())[0]
            _error = (
                f"sensitivities invalid at the initial computation and no earlier valid set exists: "
                f"{_what}: {_rule}" + (f" (+{len(_initial_invalid) - 1} more)" if len(_initial_invalid) > 1 else "")
            )
        else:
            _error = _share_error
        log.error("SENSITIVITY GUARD: stopping the run - %s", _error)
        _reason = SENSITIVITY_INVALID_REASON
        _status = AcVerificationStatus.from_termination_reason(_reason)
        n0_results_export.write_results_summary_csv(
            {
                "ac_verification_status": _status.value,
                "ac_verification_termination_reason": _reason,
                "sensitivity_guard_error": _error,
                "calculation_mode": calculation_mode,
                "ra_types": "+".join(ra_types),
                "max_stale_reused_sensitivity_contingencies": 0,
            },
            OUTPUT_DIR / "results_summary.csv",
        )
        n0_results_export.write_iteration_history_csv([], OUTPUT_DIR / "results_iteration_history.csv")
        if nx_inputs is not None:
            nx_results_export.export_contingency_status_csv(
                co_flat_list, START_HOUR, lf_nx_initial=lf_nx,
                skipped_contingencies=skipped_contingencies, model=_types.SimpleNamespace(),
                output_path=OUTPUT_DIR / "results_contingency_status.csv",
                invalid_sensitivity_contingencies=initial_invalid_sensitivity_nx,
            )
        raise SensitivityGuardError(_error)

    operating_point_state = OperatingPointState.at_original_network(
        pst_list=pst_inputs.pst_list if pst_inputs is not None else [],
        generator_data=generating_unit_dict,
        lf_n0=lf_initial,
        lf_nx=nx_inputs.lf_nx if nx_inputs is not None else None,
        sensitivity_pst_n0=pst_inputs.sensitivity_pst_n0_pruned if pst_inputs is not None else {},
        sensitivity_rd_n0=rd_inputs.sensitivity_rd_n0_pruned if rd_inputs is not None else {},
        sensitivity_pst_nx=nx_inputs.sensitivity_pst_nx if nx_inputs is not None else None,
        sensitivity_rd_nx=nx_inputs.sensitivity_rd_nx if nx_inputs is not None else None,
        sensitivity_hvdc_n0=sensitivity_hvdc_n0_pruned if hvdc_inputs is not None else None,
        sensitivity_hvdc_nx=(
            nx_inputs.sensitivity_hvdc_nx if hvdc_inputs is not None and nx_inputs is not None
            else ({} if hvdc_inputs is not None else None)
        ),
    )
    if hvdc_inputs is not None and nx_inputs is not None:
        operating_point_state.hvdc_zero_units_nx = hvdc_zero_units_nx

    operating_point_state.invalid_sensitivity_n0 = {**guard_pst_n0, **guard_rd_n0, **guard_hvdc_n0}
    log_invalid("N-0", operating_point_state.invalid_sensitivity_n0, "initial state")
    if nx_inputs is not None:
        operating_point_state.invalid_sensitivity_nx = dict(initial_invalid_sensitivity_nx)
        operating_point_state.invalid_sensitivity_nx_ever = dict(initial_invalid_sensitivity_nx)

    nx_refresh_skipped_log: list[dict] = []     # skipped {contingency: reason} of each N-X refresh
    nx_loop_hooks: Optional[NxLoopHooks] = None
    if nx_inputs is not None:
        def _nx_load_flow_at_current_state(network_at_state):
            refreshed_lf_nx, refreshed_skipped = compute_lf_nx(
                network_at_state, co_flat_list, ae_flat_list, START_HOUR,
                use_reactive_limits=USE_REACTIVE_LIMITS, extra_lf_params=EXTRA_LF_PARAMS,
            )
            nx_refresh_skipped_log.append(dict(refreshed_skipped))
            if refreshed_skipped:           # the summary per reason was logged by compute_lf_nx
                log.debug("N-X load flow at the applied state: skipped must-study contingencies: %s", refreshed_skipped)
            return refreshed_lf_nx

        def _nx_sensitivities_at_current_state(network_at_state, lf_nx_at_state):
            refresh_guard_pst: dict = {}
            refresh_guard_rd: dict = {}
            refresh_hvdc_out: dict = {}
            refreshed_pst_nx, refreshed_rd_nx = compute_sensitivities_nx(
                network_at_state, co_flat_list, pst_list, rd_list, ae_flat_list, START_HOUR,
                lf_nx=lf_nx_at_state, lf_initial=lf_initial,
                min_loading_threshold_pct_nx=MIN_LOADING_THRESHOLD_PCT_NX,
                with_pst=bool(available_nx_psts), with_rd=bool(available_nx_rds),
                use_reactive_limits=USE_REACTIVE_LIMITS, extra_lf_params=EXTRA_LF_PARAMS,
                pst_guard_out=refresh_guard_pst, rd_guard_out=refresh_guard_rd,
                ct_prune_limits=limit_by_branch_nx, ct_prune_threshold_pct=SENSITIVITY_THRESHOLD_PCT_RD_NX,
                sensitivity_threshold_pct_pst_nx=SENSITIVITY_THRESHOLD_PCT_PST_NX, degrees_per_tap=pst_degrees_per_tap_for_nx,
                **({"hvdc_ei_mrids": hvdc_nx_eis, "hvdc_out": refresh_hvdc_out} if hvdc_nx_eis else {}),
            )
            if hvdc_inputs is not None:
                # read by the loop (resolve_nx) after this hook returns
                operating_point_state.fresh_sensitivity_hvdc_nx = prune_sensitivity_rd_nx(
                    refresh_hvdc_out.get("sensitivity", {}), limit_by_branch_nx, SENSITIVITY_THRESHOLD_PCT_RD_NX,
                )
                operating_point_state.fresh_hvdc_zero_units_nx = dict(refresh_hvdc_out.get("zero_units", {}))
            operating_point_state.invalid_sensitivity_nx = {
                **refresh_guard_pst, **refresh_guard_rd, **refresh_hvdc_out.get("guard", {}),
            }
            for _co, _reason in operating_point_state.invalid_sensitivity_nx.items():
                operating_point_state.invalid_sensitivity_nx_ever.setdefault(_co, _reason)
            log_invalid("N-X", operating_point_state.invalid_sensitivity_nx, "refresh at the applied state")
            return (
                refreshed_pst_nx,                       # already pruned per contingency
                prune_sensitivity_rd_nx(refreshed_rd_nx, limit_by_branch_nx, SENSITIVITY_THRESHOLD_PCT_RD_NX),
            )

        def _count_new_nx_pairs(model_at_state, lf_nx_at_state):
            new_patl, new_tatl, new_scanned = post_ra_congestion_check.find_new_congested_pairs(
                lf_nx_at_state, ae_flat_list, START_HOUR,
                # the pairs the model evaluated: rows built and rows skipped as never binding
                nx_constraint_filter.pairs_with_skipped(model_at_state, "nx_patl"),
                nx_constraint_filter.pairs_with_skipped(model_at_state, "nx_tatl"),
                nx_constraint_filter.pairs_with_skipped(model_at_state, "nx_scanned"),
                nx_inputs.tatl_data, lf_initial,
                lf_nx_original=lf_nx,
            )
            return len(new_patl) + len(new_tatl) + len(new_scanned)

        nx_loop_hooks = NxLoopHooks(
            compute_lf_nx=_nx_load_flow_at_current_state,
            compute_sensitivities_nx=_nx_sensitivities_at_current_state,
            find_new_pairs=_count_new_nx_pairs,
        )

    ac_iteration_records: list[dict] = []
    # the arguments both modes accept; ac_single_pass ignores the caps and thresholds among them
    loop_arguments = dict(
        network=network,
        pst_list=pst_inputs.pst_list if pst_inputs is not None else [],
        rd_list=rd_inputs.rd_list if rd_inputs is not None else [],
        ae_flat_list=ae_flat_list,
        timestamp=START_HOUR,
        lf_initial=lf_initial,
        pst_tap_bounds=pst_inputs.pst_tap_bounds if pst_inputs is not None else {},
        sensitivity_pst_n0=pst_inputs.sensitivity_pst_n0_pruned if pst_inputs is not None else {},
        sensitivity_rd_n0=rd_inputs.sensitivity_rd_n0_pruned if rd_inputs is not None else {},
        pst_group_equality_pairs=pst_inputs.pst_group_equality_pairs if pst_inputs is not None else [],
        penalty_reference_pst_pra=pst_inputs.penalty_reference_pst_pra if pst_inputs is not None else 0.0,
        penalty_reference_rd_pra=rd_inputs.penalty_reference_rd_pra if rd_inputs is not None else 0.0,
        penalty_cost_overload_n0=PENALTY_COST_OVERLOAD_N0,
        pst_degrees_per_tap=pst_inputs.pst_degrees_per_tap if pst_inputs is not None else {},
        generator_data=generating_unit_dict,
        bid_list=rd_inputs.bid_list if rd_inputs is not None else [],
        glsk=rd_inputs.glsk if rd_inputs is not None else {},
        ra_groups=rd_inputs.ra_groups if rd_inputs is not None else None,
        raart=rd_inputs.raart if rd_inputs is not None else None,
        penalty_reference_pst_pra_per_tso=(
            pst_inputs.penalty_reference_pst_pra_per_tso if pst_inputs is not None else None
        ),
        penalty_reference_rd_pra_per_tso=(
            rd_inputs.penalty_reference_rd_pra_per_tso if rd_inputs is not None else None
        ),
        max_ra_count_per_timestamp_tso=(
            rd_inputs.max_ra_count_per_timestamp_tso if rd_inputs is not None else None
        ),
        ra_limit_penalty=rd_inputs.ra_limit_penalty if rd_inputs is not None else 0.0,
        ra_limit_constraint_hard=rd_inputs.ra_limit_constraint_hard if rd_inputs is not None else True,
        min_loading_threshold_pct=MIN_LOADING_THRESHOLD_PCT,
        pst_taping_discrete=pst_inputs.pst_taping_discrete if pst_inputs is not None else PST_TAPING_DISCRETE,
        max_taps_per_tso=pst_inputs.max_taps_per_tso if pst_inputs is not None else MAX_TAPS_PER_TSO,
        max_taps_per_tso_overrides=(
            pst_inputs.max_taps_per_tso_overrides if pst_inputs is not None else max_taps_per_tso_overrides
        ),
        max_change_ra_sum_threshold=MAX_CHANGE_RA_SUM_THRESHOLD,
        # the RD MW criterion uses the same config value
        max_rd_mw_change_threshold=MAX_CHANGE_RA_SUM_THRESHOLD,
        max_change_ra=MAX_CHANGE_RA,
        max_change_overload=MAX_CHANGE_OVERLOAD,
        max_linearisation_iterations=MAX_LINEARISATION_ITERATIONS,
        incremental_ac_fallback_max_steps=INCREMENTAL_AC_FALLBACK_MAX_STEPS,
        output_path=OUTPUT_DIR / "lf_post_n0.csv",
        sensitivity_threshold_pct_pst=SENSITIVITY_THRESHOLD_PCT_PST,
        sensitivity_threshold_pct_rd=SENSITIVITY_THRESHOLD_PCT_RD,
        nx_inputs=nx_inputs,
        use_reactive_limits=USE_REACTIVE_LIMITS,
        extra_lf_params=EXTRA_LF_PARAMS,
        scanned_element_constraint_hard=SCANNED_ELEMENT_CONSTRAINT_HARD,
        scanned_element_penalty_eur_per_ka=SCANNED_ELEMENT_PENALTY_EUR_PER_KA,
        iteration_records=ac_iteration_records,
        unit_costs=rd_inputs.unit_costs if rd_inputs is not None else None,
        **({"hvdc_inputs": hvdc_inputs} if hvdc_inputs is not None else {}),
        **({"ct_inputs": ct_inputs} if ct_inputs is not None else {}),
    )
    final_check_metrics = None
    if calculation_mode == "dc":
        (
            lf_post, final_model, final_pst_results, ac_verification_termination_reason,
            completeness_iteration_log, final_check_metrics,
        ) = dc_pass.run_dc_pass(
            state=operating_point_state, nx_hooks=nx_loop_hooks, **loop_arguments,
        )
        # summary rows of the DC mode (D7): the factors used and the basis of the original-network files
        final_check_metrics = {
            "dc_power_factor": dc_power_factor,
            "dc_voltage_levels_with_factor": dc_converter.levels_with_factor,
            "lf_basecase_basis": "dc",
            "original_network_currents_basis": "dc",
            **final_check_metrics,
        }
    elif calculation_mode == "ac_single_pass":
        (
            lf_post, final_model, final_pst_results, ac_verification_termination_reason,
            completeness_iteration_log, final_check_metrics,
        ) = single_pass.run_single_pass(
            state=operating_point_state, nx_hooks=nx_loop_hooks, **loop_arguments,
        )
    else:
        lf_post, final_model, final_pst_results, ac_verification_termination_reason, completeness_iteration_log = (
            ac_convergence_loop.run_nested_rao_loop(
                max_completeness_iterations=MAX_COMPLETENESS_ITERATIONS,
                state=operating_point_state,
                nx_hooks=nx_loop_hooks,
                **loop_arguments,
            )
        )
    log.info("AC verification loop termination reason: %s", ac_verification_termination_reason)
    run_profile.mark("Outputs")
    n0_results_export.write_iteration_history_csv(
        operating_point_state.solve_history, OUTPUT_DIR / "results_iteration_history.csv",
    )
    if nx_inputs is not None:
        write_completeness_iterations_csv(
            completeness_iteration_log, OUTPUT_DIR / "results_completeness_iterations.csv",
        )
    ac_verification_status = AcVerificationStatus.from_termination_reason(
        ac_verification_termination_reason
    )
    if final_model is None:
        # no model at all (the loop stopped before its first build): only the status is reported
        log.error("No model describes the network at the end of the loop - only the status files are written")
        n0_results_export.write_results_summary_csv(
            {
                "ac_verification_status": ac_verification_status.value,
                "ac_verification_termination_reason": ac_verification_termination_reason,
                "calculation_mode": calculation_mode,
                "ra_types": "+".join(ra_types),
            },
            OUTPUT_DIR / "results_summary.csv",
        )
        return

    # The loop applies every solution before checking convergence, so ONE final
    # model feeds both the N-0 and the N-X result files.

    if nx_inputs is not None:
        # current_lf_nx is None when the final AC check of the dc mode did not run (dc_pass): the
        # post-PRA / after columns then show the model's prediction
        log.info("=== N-X results export (%s branch) ===", workflow_label)
        nx_results_export.export_nx_loadings_csv(
            final_model, ae_flat_list, co_flat_list,
            lf_nx_initial=lf_nx, lf_initial=lf_initial,
            sensitivity_pst_nx=sensitivity_pst_nx, sensitivity_rd_nx=sensitivity_rd_nx,
            pst_degrees_per_tap=pst_degrees_per_tap_for_nx, tatl_data=tatl_data,
            timestamp=START_HOUR, output_path=OUTPUT_DIR / "results_nx_loadings.csv",
            branch_mrid_to_name=branch_mrid_to_name,
            lf_nx_final=operating_point_state.current_lf_nx,
        )
        nx_results_export.export_contingency_status_csv(
            co_flat_list, START_HOUR, lf_nx_initial=lf_nx,
            skipped_contingencies=nx_results_export.merge_skipped_contingency_reasons(
                skipped_contingencies, nx_refresh_skipped_log,
            ),
            model=final_model,
            output_path=OUTPUT_DIR / "results_contingency_status.csv",
            invalid_sensitivity_contingencies=operating_point_state.invalid_sensitivity_nx_ever,
            sensitivity_source_contingencies=operating_point_state.sensitivity_source_nx_ever,
        )
        nx_results_export.export_nx_scanned_elements_csv(
            final_model, ae_flat_list, co_flat_list,
            lf_nx_initial=lf_nx,
            sensitivity_pst_nx=sensitivity_pst_nx, sensitivity_rd_nx=sensitivity_rd_nx,
            pst_degrees_per_tap=pst_degrees_per_tap_for_nx, branch_limits=limit_by_branch_nx,
            timestamp=START_HOUR, output_path=OUTPUT_DIR / "results_nx_scanned_elements.csv",
            branch_mrid_to_name=branch_mrid_to_name,
            lf_nx_final=operating_point_state.current_lf_nx,
        )


    if pst_inputs is not None:
        n0_results_export.write_pst_results_csv(
            final_pst_results, OUTPUT_DIR / "results_pst.csv", START_HOUR,
            ac_verification_status=ac_verification_status,
            ac_verification_termination_reason=ac_verification_termination_reason,
            curative_pst_results=n0_results_export.extract_curative_pst_results(
                final_model, pst_inputs.pst_list, pst_inputs.penalty_reference_pst_pra,
                pst_inputs.penalty_reference_pst_pra_per_tso,
            ),
        )
        log.info("Final PST results: %d PSTs", len(final_pst_results))
        n0_results_export.write_tso_tap_limit_results_csv(
            n0_results_export.extract_tso_tap_limit_results(final_model, pst_inputs.pst_list),
            OUTPUT_DIR / "results_tso_tap_limits.csv", START_HOUR,
            ac_verification_status=ac_verification_status,
            ac_verification_termination_reason=ac_verification_termination_reason,
        )

    rd_results_for_summary: dict = {}
    curative_rd_rows: list[dict] = []
    if nx_inputs is not None and enable_rd:
        # curative RD volumes and cost (header-only without curative bids)
        curative_rd_rows = n0_results_export.extract_curative_rd_results(
            final_model,
            rd_list=[rd for rd in rd_inputs.rd_list if not rd.countertrade] if rd_inputs is not None else [],
            bid_list=[
                bid for bid in rd_inputs.bid_list
                if str(bid.ra_mrid) not in {str(rd.mrid) for rd in rd_inputs.rd_list if rd.countertrade}
            ] if rd_inputs is not None else [],
            penalty_reference_rd_pra=(
                rd_inputs.penalty_reference_rd_pra if rd_inputs is not None else PENALTY_REFERENCE_RD_PRA
            ),
            penalty_reference_rd_pra_per_tso=(
                rd_inputs.penalty_reference_rd_pra_per_tso if rd_inputs is not None else None
            ),
        )
        n0_results_export.write_curative_rd_results_csv(
            curative_rd_rows,
            OUTPUT_DIR / "results_curative_rd.csv", START_HOUR,
            ac_verification_status=ac_verification_status,
            ac_verification_termination_reason=ac_verification_termination_reason,
        )

    if nx_inputs is not None and enable_pst:
        n0_results_export.write_curative_pst_delta_results_csv(
            n0_results_export.extract_curative_pst_delta_results(
                final_model, pst_list, nx_inputs.pst_tap_bounds_curative, nx_inputs.pst_cra_availability,
                pst_inputs.penalty_reference_pst_pra if pst_inputs is not None else PENALTY_REFERENCE_PST_PRA,
                (
                    pst_inputs.penalty_reference_pst_pra_per_tso
                    if pst_inputs is not None else PENALTY_REFERENCE_PST_PRA_PER_TSO
                ),
            ),
            OUTPUT_DIR / "results_curative_pst.csv", START_HOUR,
            ac_verification_status=ac_verification_status,
            ac_verification_termination_reason=ac_verification_termination_reason,
        )
        n0_results_export.write_curative_pst_group_results_csv(
            n0_results_export.extract_curative_pst_group_results(
                final_model, nx_inputs.curative_pst_group_triples or [], pst_list,
            ),
            OUTPUT_DIR / "results_curative_pst_groups.csv", START_HOUR,
            ac_verification_status=ac_verification_status,
            ac_verification_termination_reason=ac_verification_termination_reason,
        )

    if hvdc_inputs is not None:
        n0_results_export.write_hvdc_results_csv(
            n0_results_export.extract_hvdc_results(final_model, hvdc_inputs.penalty_reference_pra),
            OUTPUT_DIR / "results_hvdc.csv", START_HOUR,
            ac_verification_status=ac_verification_status,
            ac_verification_termination_reason=ac_verification_termination_reason,
        )
        if nx_inputs is not None and hvdc_inputs.curative_units:
            n0_results_export.write_curative_hvdc_results_csv(
                n0_results_export.extract_curative_hvdc_results(final_model, hvdc_inputs.penalty_reference_cra),
                OUTPUT_DIR / "results_curative_hvdc.csv", START_HOUR,
                ac_verification_status=ac_verification_status,
                ac_verification_termination_reason=ac_verification_termination_reason,
            )
            n0_results_export.write_hvdc_cra_availability_csv(
                getattr(final_model, "hvdc_cra_availability_rows", []), OUTPUT_DIR / "hvdc_cra_availability.csv",
            )

    if rd_inputs is not None and enable_rd:
        # RD result extractions, from the AC-verified final_model; the countertrade RAs and bids of the merged
        # lists are left out (their files are the countertrade outputs)
        rd_export_list = [rd for rd in rd_inputs.rd_list if not rd.countertrade]
        rd_export_ras = {str(rd.mrid) for rd in rd_inputs.rd_list if rd.countertrade}
        rd_export_bids = [bid for bid in rd_inputs.bid_list if str(bid.ra_mrid) not in rd_export_ras]
        rd_results_for_summary = n0_results_export.extract_rd_results(
            final_model,
            rd_list=rd_export_list,
            bid_list=rd_export_bids,
            penalty_reference_rd_pra=rd_inputs.penalty_reference_rd_pra,
            ac_verification_status=ac_verification_status,
            ac_verification_termination_reason=ac_verification_termination_reason,
            penalty_reference_rd_pra_per_tso=rd_inputs.penalty_reference_rd_pra_per_tso,
            timestamp=START_HOUR,
            output_path=OUTPUT_DIR / "results_preventive_rd.csv",
        )
        n0_results_export.extract_results_generating_units(
            final_model,
            rd_list=rd_export_list,
            ac_verification_status=ac_verification_status,
            ac_verification_termination_reason=ac_verification_termination_reason,
            glsk=rd_inputs.glsk,
            rd_results=rd_results_for_summary,
            timestamp=START_HOUR,
            output_path=OUTPUT_DIR / "results_generating_units.csv",
            # a run with a usable countertrade RA: the countertrade part in its own column (DECISIONS 79)
            ct_bid_participation=ct_inputs.bid_participation if ct_inputs is not None else None,
            # every countertrade unit (TSO, original set point): those in no RD glsk get their own row (DECISIONS 80)
            ct_units={str(u.gu_mrid): (u.tso, float(u.initial_p)) for u in reversed(ct_rd_units)} if ct_inputs is not None else None,
        )
        n0_results_export.extract_ra_count_results(
            final_model,
            rd_list=rd_export_list,
            bid_list=rd_export_bids,
            ac_verification_status=ac_verification_status,
            ac_verification_termination_reason=ac_verification_termination_reason,
            max_ra_count_per_timestamp_tso=rd_inputs.max_ra_count_per_timestamp_tso,
            timestamp=START_HOUR,
            output_path=OUTPUT_DIR / "results_preventive_rd_ra_count.csv",
        )
        n0_results_export.extract_ra_group_results(
            final_model,
            ra_groups=rd_inputs.ra_groups,
            bid_list=rd_export_bids,
            ac_verification_status=ac_verification_status,
            ac_verification_termination_reason=ac_verification_termination_reason,
            timestamp=START_HOUR,
            output_path=OUTPUT_DIR / "results_preventive_rd_ra_groups.csv",
        )
        n0_results_export.extract_lead_time_results(
            bid_list=rd_export_bids,
            ac_verification_status=ac_verification_status,
            ac_verification_termination_reason=ac_verification_termination_reason,
            raart=rd_inputs.raart,
            timestamp=START_HOUR,
            output_path=OUTPUT_DIR / "results_lead_times.csv",
            rd_list=rd_export_list,
        )
    else:
        log.info("rd_inputs=None — RD-specific results extraction skipped")

    # countertrade result files (docs/specs/countertrade_ra_specs.md C11), only with a usable countertrade RA
    if ct_inputs is not None:
        ct_results_export.write_ct_results_csv(
            ct_results_export.extract_ct_results(final_model, ct_ras_used, ct_bid_list, penalty_reference_ct_pra),
            OUTPUT_DIR / "results_countertrade.csv", START_HOUR, ac_verification_status, ac_verification_termination_reason,
        )
        if nx_inputs is not None and any(ra.kind == RAKind.CURATIVE for ra in ct_ras_used):
            ct_results_export.write_curative_ct_results_csv(
                ct_results_export.extract_curative_ct_results(final_model, ct_ras_used, ct_bid_list, penalty_reference_ct_cra),
                OUTPUT_DIR / "results_curative_countertrade.csv", START_HOUR,
                ac_verification_status, ac_verification_termination_reason,
            )
            ct_results_export.write_ct_cra_availability_csv(
                ct_results_export.compute_ct_cra_availability(
                    ct_ras_used, nx_inputs.tatl_data or {}, list(limit_by_branch_nx), TATL_FALLBACK_DURATION_S,
                ),
                OUTPUT_DIR / "ct_cra_availability.csv",
            )
        ct_results_export.write_net_positions_csv(
            ct_results_export.extract_net_positions(final_model, rd_inputs, ct_zone_of_unit, ct_zones, ct_tso_of_zone),
            OUTPUT_DIR / "results_net_positions.csv", START_HOUR,
        )
        ct_results_export.log_units_at_limits(final_model, rd_inputs)
        # the exchange between the zones: the original state and the state after the applied results, both from the
        # same kind of load flow (AC in the AC modes, DC in the dc mode)
        try:
            if calculation_mode == "dc":
                cgmes_loader.run_initial_dc_loadflow(network, use_reactive_limits=USE_REACTIVE_LIMITS, extra_lf_params=EXTRA_LF_PARAMS)
            else:
                ct_converged, ct_status_name, _ = post_ra_congestion_check._run_ac_loadflow(
                    network, use_reactive_limits=USE_REACTIVE_LIMITS, extra_lf_params=EXTRA_LF_PARAMS,
                )
                if not ct_converged:
                    raise RuntimeError(f"the AC load flow of the final state did not converge ({ct_status_name})")
            ct_results_export.write_zone_exchange_csv(
                ct_results_export.compute_zone_exchange(
                    ct_boundaries_before, ct_results_export.read_area_boundaries(network), ct_zones, ct_tso_of_zone,
                ),
                OUTPUT_DIR / "results_zone_exchange.csv", START_HOUR, "DC" if calculation_mode == "dc" else "AC",
            )
        except Exception as exc:
            log.warning("results_zone_exchange.csv not written: %s", exc)

    # results summary: written after every other result export, from the same final_model,
    # so its components match those files
    import pyomo.environ as pyomo_environ
    # no applied solution: the model describes the original network and has no objective of its own
    objective_value = (
        None if getattr(final_model, "no_applied_solution", False)
        else float(pyomo_environ.value(final_model.objective))
    )
    n0_results_export.extract_results_summary(
        final_model,
        objective_value=objective_value,
        pst_list=pst_list,
        penalty_reference_pst_pra=PENALTY_REFERENCE_PST_PRA,
        penalty_reference_pst_pra_per_tso=PENALTY_REFERENCE_PST_PRA_PER_TSO,
        rd_results=rd_results_for_summary,
        output_path=OUTPUT_DIR / "results_summary.csv",
        curative_rd_results=curative_rd_rows,
        ac_verification_status=ac_verification_status,
        ac_verification_termination_reason=ac_verification_termination_reason,
        ac_iteration_count=len(ac_iteration_records),
        completeness_limit_reached=operating_point_state.completeness_limit_reached,
        calculation_mode=calculation_mode,
        ra_types=ra_types,
        completeness_iteration_count=max(
            [r["completeness_iteration"] for r in completeness_iteration_log], default=0,
        ),
        final_check=final_check_metrics,
        max_stale_reused_sensitivity_contingencies=max(
            [r.get("stale_reused_sensitivity_contingencies", 0) for r in operating_point_state.solve_history],
            default=0,
        ),
    )



    # lf_results_branches.csv and lf_results_buses.csv are not produced

    log.info("=== Pipeline complete ===")
    log.info("Outputs written to: %s", OUTPUT_DIR)
    for f in sorted(OUTPUT_DIR.glob("*.csv")):
        log.info("  %s", f.name)


if __name__ == "__main__":
    main()
