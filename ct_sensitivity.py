"""
ct_sensitivity.py - one sensitivity variable per countertrade RA and direction (docs/DECISIONS.md 169):
sens_ct(branch, RA, direction) = sum over the units of weight x sens(branch, unit). pypowsybl computes it as a zone with the
weights as shift keys, so a unit that only serves countertrade has no sensitivity of its own.
"""
from typing import Optional

from data_model import BidDirection, RAKind, RDUnit

CT_VARIABLE_PREFIX = "CT|"


def ct_variable_id(ra_mrid, direction: BidDirection) -> str:
    """Id of the sensitivity variable of one countertrade RA and direction."""
    return f"{CT_VARIABLE_PREFIX}{ra_mrid}|{BidDirection(direction).value}"


def ct_zones(rd_list: list[RDUnit], preventive_only: bool = False) -> dict[str, dict[str, float]]:
    """{variable id: {unit: weight}} of the available countertrade RAs (preventive ones only with preventive_only);
    a direction in which no unit moves has no variable."""
    zones: dict[str, dict[str, float]] = {}
    for rd in rd_list:
        if not rd.countertrade or not rd.available or (preventive_only and rd.kind != RAKind.PREVENTIVE):
            continue
        for direction, weight in ((BidDirection.UP, rd.ct_weight_up), (BidDirection.DOWN, rd.ct_weight_down)):
            if float(weight) > 0.0:
                zones.setdefault(ct_variable_id(rd.mrid, direction), {})[str(rd.gu_mrid)] = float(weight)
    return zones


def individual_unit_ids(rd_list: list[RDUnit], preventive_only: bool = False, with_countertrade: bool = False) -> list[str]:
    """Sorted units that get a sensitivity of their own: those of the available redispatch RAs, not the
    countertrade-only ones (with_countertrade: those too, when the RAs have no variable of their own)."""
    return sorted({
        str(rd.gu_mrid) for rd in rd_list
        if rd.available and (with_countertrade or not rd.countertrade) and (not preventive_only or rd.kind == RAKind.PREVENTIVE)
    })


PROBE_BRANCH_COUNT = 10


def probe_branch_sides(branch_sides: dict[str, int]) -> dict[str, int]:
    """Up to PROBE_BRANCH_COUNT monitored branches, evenly spread over the sorted ids, for lost_units."""
    ids = sorted(branch_sides)
    if len(ids) > PROBE_BRANCH_COUNT:
        ids = [ids[(position * len(ids)) // PROBE_BRANCH_COUNT] for position in range(PROBE_BRANCH_COUNT)]
    return {branch: branch_sides[branch] for branch in ids}


def lost_units(unit_ids: list[str], probes: list[tuple[list[str], "np.ndarray"]]) -> set[str]:
    """The units a contingency cuts off the solved network. pypowsybl drops them from a zone and renormalises the
    weights of the others, while the sensitivity of a single unit outside the network is exactly 0, so the zone has to
    be scaled back. probes: [(variable ids, array (variables, branches))] of the unit sensitivities on a few
    branches. A unit is lost when it is exactly 0.0 on every branch that reacts at all (a column that is 0 or NaN for
    all units is a branch the contingency outages or an unrelated radial one: no information)."""
    import numpy as np
    index = {unit: [] for unit in unit_ids}
    informative = 0
    for variables, values in probes:
        rows = {str(v): position for position, v in enumerate(variables)}
        block = np.array([values[rows[unit]] if unit in rows else np.zeros(values.shape[1]) for unit in unit_ids])
        for column in range(block.shape[1]):
            data = block[:, column]
            if not np.any((data != 0.0) & np.isfinite(data)):
                continue
            informative += 1
            for unit, value in zip(unit_ids, data):
                index[unit].append(value == 0.0)
    if not informative:
        return set()
    return {unit for unit, flags in index.items() if flags and all(flags)}


def alive_weight_sums(zones: dict[str, dict[str, float]], lost: set[str]) -> dict[str, float]:
    """{variable id: sum of the weights of the units that are not lost}: the factor that gives back the weighted sum
    after pypowsybl normalised the zone over the units that remain."""
    return {zone_id: sum(w for unit, w in weights.items() if unit not in lost) for zone_id, weights in zones.items()}


RESIDUAL_TOLERANCE = 1e-7


def zone_weight_sums(
    zones: dict[str, dict[str, float]], lost: set[str], probes: list[tuple[list[str], "np.ndarray"]],
    zone_probe_values: dict[str, "np.ndarray"],
) -> tuple[dict[str, float], dict[str, str]]:
    """(sums, dropped): alive_weight_sums, and for a zone whose value on the probe branches is not the weighted
    mean of its remaining units (pypowsybl also leaves a unit out of a zone for a reason other than losing it, seen
    with a shunt outage) the sum of the weights of the units that remain: all units but the one whose removal makes
    the zone value on every probe branch equal the weighted mean. dropped: {zone: that unit}; its own sensitivity
    is needed to give back its share. zone_probe_values[zone]: the zone value (normalised, as pypowsybl returns it)
    on the probe branches, in the column order of `probes` (NaN where not read)."""
    import numpy as np
    sums = alive_weight_sums(zones, lost)
    dropped: dict[str, str] = {}
    if not probes or not zone_probe_values:
        return sums, dropped
    variables = probes[0][0]
    unit_values = np.concatenate([values for _ids, values in probes], axis=1)           # (units, probe branches)
    row_of = {str(v): position for position, v in enumerate(variables)}
    for zone_id, weights in zones.items():
        zone_values = zone_probe_values.get(zone_id)
        if zone_values is None:
            continue
        members = [u for u in weights if u in row_of]
        w = np.array([weights[u] for u in members])
        s = unit_values[[row_of[u] for u in members]]
        alive = np.array([u not in lost for u in members])
        usable = np.isfinite(zone_values) & np.any(s != 0.0, axis=0)
        if not usable.any():
            continue
        z, s = zone_values[usable], s[:, usable]
        # residual of "the zone is the weighted mean of its units that are alive" per probe branch, and its scale
        terms = w[:, None] * (z[None, :] - s) * alive[:, None]
        residual = terms.sum(axis=0)
        scale = (w[:, None] * np.abs(s) * alive[:, None]).sum(axis=0) + np.abs(z) * (w * alive).sum() + 1e-300
        if np.all(np.abs(residual) <= RESIDUAL_TOLERANCE * scale):
            continue
        fits = np.all(np.abs(residual[None, :] - terms) <= RESIDUAL_TOLERANCE * scale[None, :], axis=1) & alive
        if fits.sum() == 1:
            extra = members[int(np.flatnonzero(fits)[0])]
            sums[zone_id] = alive_weight_sums({zone_id: weights}, lost | {extra})[zone_id]
            dropped[zone_id] = extra
    return sums, dropped


def coefficient(sensitivity, branch_mrid: str, variable_id: str, weights: dict[str, float]) -> float:
    """A per MW of the volume of a countertrade bid on a branch: the entry of its RA and direction; without one (a
    sensitivity table with the units only) the weighted sum over the units."""
    value: Optional[float] = sensitivity.get((branch_mrid, variable_id))
    if value is not None:
        return float(value)
    return sum(float(weight) * float(sensitivity.get((branch_mrid, unit), 0.0)) for unit, weight in weights.items())
