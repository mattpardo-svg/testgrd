"""
ct_sensitivity.py - one sensitivity variable per countertrade RA and direction (docs/DECISIONS.md 170):
sens_ct(branch, state, RA, direction) = sum over the units of weight x sens(branch, state, unit). pypowsybl gives the
sensitivity of each unit; they are read in batches and summed at once, so the per-unit table never exists in full and
the result is the per-unit one by construction.
"""
from typing import Optional

import numpy as np

from data_model import BidDirection, RAKind, RDUnit

CT_VARIABLE_PREFIX = "CT|"

# Values (units x branches x contingencies) that one batch of the N-X analysis may hold. The raw result, its copy as a
# matrix and the weighted sum need about 4 copies of 8 bytes each: 3 million values stay near 100 MB of peak memory for
# any number of units or contingencies, and a batch is large enough that the load flow of its contingencies dominates.
BATCH_BUDGET = 3_000_000


def ct_variable_id(ra_mrid, direction: BidDirection) -> str:
    """Id of the sensitivity variable of one countertrade RA and direction."""
    return f"{CT_VARIABLE_PREFIX}{ra_mrid}|{BidDirection(direction).value}"


def ct_weights(rd_list: list[RDUnit], preventive_only: bool = False) -> dict[str, dict[str, float]]:
    """{variable id: {unit: weight}} of the available countertrade RAs (preventive ones only with preventive_only);
    a direction in which no unit moves has no variable."""
    weights: dict[str, dict[str, float]] = {}
    for rd in rd_list:
        if not rd.countertrade or not rd.available or (preventive_only and rd.kind != RAKind.PREVENTIVE):
            continue
        for direction, weight in ((BidDirection.UP, rd.ct_weight_up), (BidDirection.DOWN, rd.ct_weight_down)):
            if float(weight) > 0.0:
                weights.setdefault(ct_variable_id(rd.mrid, direction), {})[str(rd.gu_mrid)] = float(weight)
    return weights


def countertrade_unit_ids(weights: dict[str, dict[str, float]]) -> list[str]:
    """Sorted units of all countertrade variables."""
    return sorted({unit for unit_weights in weights.values() for unit in unit_weights})


def individual_unit_ids(rd_list: list[RDUnit], preventive_only: bool = False) -> list[str]:
    """Sorted units that get a sensitivity of their own: those of the available redispatch RAs (a unit that serves
    countertrade only has none)."""
    return sorted({
        str(rd.gu_mrid) for rd in rd_list
        if rd.available and not rd.countertrade and (not preventive_only or rd.kind == RAKind.PREVENTIVE)
    })


def batch_size(unit_count: int, branch_count: int, budget: Optional[int] = None) -> int:
    """Contingencies per batch: at least 1, at most budget / (units x branches)."""
    return max(1, int((BATCH_BUDGET if budget is None else budget) // max(1, unit_count * branch_count)))


def aggregate_units(
    values: np.ndarray, unit_ids: list[str], weights: dict[str, dict[str, float]], thresholds: Optional[np.ndarray] = None,
) -> tuple[list[str], np.ndarray]:
    """(variable ids sorted, array (branches, variables)): the weighted sum of the per-unit values of each variable.
    values: (branches, units) in the order of unit_ids. thresholds (one per branch): a unit value with an absolute
    value below it counts as 0 (the magnitude pruning of the per-unit entries); NaN is never pruned. Only the units
    of a variable enter its sum, a NaN unit value makes the sum NaN, a unit without a value counts as 0."""
    column_of = {unit: column for column, unit in enumerate(unit_ids)}
    if thresholds is not None:
        with np.errstate(invalid="ignore"):
            values = np.where(np.abs(values) < thresholds[:, None], 0.0, values)
    variables = sorted(weights)
    aggregate = np.zeros((values.shape[0], len(variables)))
    for position, variable in enumerate(variables):
        members = [(column_of[unit], weight) for unit, weight in weights[variable].items() if unit in column_of]
        if members:
            columns, member_weights = zip(*members)
            aggregate[:, position] = values[:, list(columns)] @ np.array(member_weights)
    return variables, aggregate


def branch_maximum(values: np.ndarray) -> np.ndarray:
    """Per branch the largest absolute value over the units: NaN values are ignored, a branch whose values are all NaN
    gets NaN (it then passes any threshold, as a NaN does in the per-unit comparison), no unit gives 0."""
    if values.shape[1] == 0:
        return np.zeros(values.shape[0])
    with np.errstate(invalid="ignore"):
        return np.fmax.reduce(np.abs(values), axis=1)


def coefficient(sensitivity, branch_mrid: str, variable_id: str, weights: dict[str, float]) -> float:
    """A per MW of the volume of a countertrade bid on a branch: the entry of its RA and direction; without one (a
    sensitivity table with the units only) the weighted sum over the units."""
    value: Optional[float] = sensitivity.get((branch_mrid, variable_id))
    if value is not None:
        return float(value)
    return sum(float(weight) * float(sensitivity.get((branch_mrid, unit), 0.0)) for unit, weight in weights.items())
