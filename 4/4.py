"""glsk_resolver.py — GLSK (generation shift key) per RD remedial action.

The NC profile has no GLSKSchedule class; the chain from an RD remedial action to its
participation factors is:
  RD RA <- PowerBidSchedule.PowerRemedialAction
  PowerBidSchedule <- PowerShiftKeyDistribution.PowerBidSchedule
  PowerShiftKeyDistribution -> PowerShiftKeyDistribution.PowerShiftKeySchedule
  PowerShiftKeySchedule -> GeneratingUnit; <- ParticipationFactorTimePoint (factor per timestamp)
Distributions linked to a PowerShiftKeyStrategy instead of a bid schedule have no remedial
action and are not used. An RA without a resolvable chain gets an empty dict (logged).
"""

from __future__ import annotations

import csv
import logging
import re
from pathlib import Path
from typing import Optional

from rdflib import Namespace, RDF

from data_model import RAKind, RDUnit, effective_linked_contingencies
from pre_milp import run_time
from pre_milp.ncp_parser import (
    _OTHER_RESOURCES, _frag, _generator_of_unit, _iso_to_hhmmz, _load_graph, ncp_id, read_key_routes, warn_not_supported,
)
from text_encoding import TEXT_ENCODING

log = logging.getLogger(__name__)

# Namespaces — same values as ncp_parser.py's CIM/NC. The graph helpers
# _frag, ncp_id, _load_graph and _iso_to_hhmmz are imported from ncp_parser.
CIM = Namespace("https://cim.ucaiug.io/ns#")
NC = Namespace("https://cim4.eu/ns/nc#")


def _get_available_rd_ra_mrids(rd_list: list[RDUnit]) -> list[str]:
    """Sorted, de-duplicated mRIDs (RDUnit.mrid) of the available RD remedial actions,
    preventive and curative (curative RD needs the GLSK of its units too)."""
    return sorted({
        str(rd.mrid)
        for rd in rd_list
        if rd.available
    })


_INTERPOLATION_KINDS = ("none", "previous", "next", "linear")
_SUM_TOLERANCE = 1e-6


def _minute_of_day(hhmmz: str) -> int:
    return int(hhmmz[:2]) * 60 + int(hhmmz[2:4])


def _factor_at(points: list, minute: int, kind: str) -> Optional[float]:
    """The participation factor of one key at `minute` (of the day). points: [(minute or None, factor)];
    a point without a time is valid at every time. Otherwise an exact point counts; between points the
    interpolation kind decides (none: no value, previous / next: the neighbour, linear: no extrapolation)."""
    constant = [value for when, value in points if when is None]
    if constant:
        return constant[-1]
    timed = {when: value for when, value in points}
    if minute in timed:
        return timed[minute]
    before = max((m for m in timed if m < minute), default=None)
    after = min((m for m in timed if m > minute), default=None)
    if kind == "previous":
        return timed[before] if before is not None else None
    if kind == "next":
        return timed[after] if after is not None else None
    if kind == "linear" and before is not None and after is not None:
        weight = (minute - before) / (after - before)
        return timed[before] + weight * (timed[after] - timed[before])
    return None


def resolve_glsk(
    sis_path: Optional[Path],
    rd_list: list[RDUnit],
    tso: str,
    valid_generator_mrids: set[str],
    timestamp: str,
    output_path: Optional[Path] = None,
    connected_generator_mrids: Optional[set[str]] = None,
    *,
    generators_by_unit: Optional[dict[str, list[str]]] = None,
    ra_path: Optional[Path] = None,
) -> dict[str, dict[str, float]]:
    """Participation factors per RD remedial action of one TSO's SIS file.

    rd_list: this TSO's RDUnits. valid_generator_mrids: the generators of the network; a unit
    not in it is removed and the factors are redistributed proportionally. connected_generator_mrids
    (None: no check): a unit not connected is removed and redistributed the same way (only
    connected units take part in the redispatch, spec 2839 / 3012-3014). timestamp: HHMMZ.
    generators_by_unit: {CIM GeneratingUnit mRID: [network generator ids]}; the unit of a
    PowerShiftKeySchedule is mapped to its generator (one generator; a unit with several is dropped
    with a warning). A unit that is no key of the map (a generator id, or a unit the CGM does not have) is not
    supported: dropped, one summary warning per file. ra_path: the RA file, for the
    PowerShiftKeyStrategy -> RA links of the strategy route. Unit routes: the bid route (RA <- bid <-
    distribution -> key) wins; the strategy route (RA <- strategy <- distribution -> key) is the
    fallback of the RA. Keys whose resource is not a GeneratingUnit are dropped with a warning that
    counts them. The raw factors are relative weights, normalised to sum 1 (warning when their sum is not 1).
    Returns {ra_mrid: {gu_mrid: factor}} with every available RA as a key, an empty dict when
    nothing resolves (so "no participation" differs from "not considered").
    """
    available_ra_mrids = _get_available_rd_ra_mrids(rd_list)
    glsk: dict[str, dict[str, float]] = {ra_mrid: {} for ra_mrid in available_ra_mrids}

    if not available_ra_mrids:
        log.info("%s: no available RD RAs — resolve_glsk skipped", tso)
        return glsk

    if sis_path is None:
        log.info(
            "%s: no SIS file — GLSK cannot be resolved for %d RD RA(s): %s",
            tso, len(available_ra_mrids), available_ra_mrids,
        )
        return glsk

    sis_graph = _load_graph(sis_path)
    ra_graph = _load_graph(ra_path) if ra_path is not None else None

    # --- Steps 1-3: RA mRID -> PowerShiftKeySchedule mRIDs, per route; the bid route wins ---
    ra_to_psks_bid, ra_to_psks_strategy = read_key_routes(sis_graph, ra_graph, set(available_ra_mrids))
    # --- Step 4: PowerShiftKeySchedule mRID -> GeneratingUnit mRID(s); other resource types, interpolation ---
    psks_to_gu: dict[str, set[str]] = {}
    psks_other_resources: dict[str, list[str]] = {}
    psks_interpolation: dict[str, str] = {}
    without_interpolation: list[str] = []
    for psks_uri in sis_graph.subjects(RDF.type, NC.PowerShiftKeySchedule):
        psks_mrid_lit = sis_graph.value(psks_uri, CIM["IdentifiedObject.mRID"])
        psks_mrid = ncp_id(psks_mrid_lit) if psks_mrid_lit is not None else _frag(str(psks_uri))
        gu_mrids = {
            _frag(str(o)) for o in sis_graph.objects(psks_uri, NC["PowerShiftKeySchedule.GeneratingUnit"])
        }
        other = [name for name in _OTHER_RESOURCES if sis_graph.value(psks_uri, NC[f"PowerShiftKeySchedule.{name}"]) is not None]
        if other:
            psks_other_resources[psks_mrid] = other
        if gu_mrids:
            psks_to_gu[psks_mrid] = gu_mrids
            if len(gu_mrids) > 1:
                log.warning(
                    "%s: PowerShiftKeySchedule %s has %d GeneratingUnits — "
                    "the single participationFactor for this schedule will be "
                    "applied to all of them (not independently observed in "
                    "real data; this is an assumption for the multi-GU case)",
                    tso, psks_mrid, len(gu_mrids),
                )
        kind_uri = sis_graph.value(psks_uri, NC["BaseTimeSeries.interpolationKind"])
        if kind_uri is None:
            without_interpolation.append(psks_mrid)
            psks_interpolation[psks_mrid] = "none"
        else:
            kind = re.split(r"[#.]", str(kind_uri))[-1]
            if kind not in _INTERPOLATION_KINDS:
                log.warning("%s: PowerShiftKeySchedule %s has interpolationKind %s, which is unknown - none used", tso, psks_mrid, kind)
                kind = "none"
            psks_interpolation[psks_mrid] = kind
    if without_interpolation:
        log.warning(
            "%s: %d PowerShiftKeySchedule(s) have no BaseTimeSeries.interpolationKind - none assumed (exact time points only)",
            tso, len(without_interpolation),
        )

    # the bid route wins when it reaches an existing key; the strategy route is the RA-level fallback
    existing_keys = set(psks_to_gu) | set(psks_other_resources)
    ra_to_psks: dict[str, set[str]] = {}
    for ra_mrid in available_ra_mrids:
        bid_keys = ra_to_psks_bid[ra_mrid] & existing_keys
        ra_to_psks[ra_mrid] = bid_keys if bid_keys else ra_to_psks_strategy[ra_mrid]

    # --- Step 5: PowerShiftKeySchedule mRID -> participationFactor at this timestamp ---
    # a point without atTime is valid at every timestamp (the schedule follows the bid time)
    psks_points: dict[str, list] = {}
    factor_points = list(sis_graph.subjects(RDF.type, NC.ParticipationFactorTimePoint))
    required_day = run_time.required_day(
        (run_time.iso_day(str(sis_graph.value(p, NC["ParticipationFactorTimePoint.atTime"]))) for p in factor_points),
        f"{tso}: participation factors", sis_path.name,
    )
    for timepoint_uri in factor_points:
        psks_ref = sis_graph.value(timepoint_uri, NC["ParticipationFactorTimePoint.PowerShiftKeySchedule"])
        at_time = sis_graph.value(timepoint_uri, NC["ParticipationFactorTimePoint.atTime"])
        factor_lit = sis_graph.value(timepoint_uri, NC["ParticipationFactorTimePoint.participationFactor"])

        if psks_ref is None or factor_lit is None:
            continue
        resolved_timestamp = _iso_to_hhmmz(str(at_time)) if at_time is not None else None
        if at_time is not None and resolved_timestamp is None:
            continue
        if at_time is not None and not run_time.day_matches(
            run_time.iso_day(str(at_time)), required_day, _minute_of_day(resolved_timestamp), _minute_of_day(timestamp),
        ):
            continue

        psks_mrid = _frag(str(psks_ref))
        try:
            factor = float(str(factor_lit))
        except (TypeError, ValueError):
            log.warning(
                "%s: ParticipationFactorTimePoint for PowerShiftKeySchedule %s "
                "has a non-numeric participationFactor %r — skipped",
                tso, psks_mrid, factor_lit,
            )
            continue
        psks_points.setdefault(psks_mrid, []).append(
            (_minute_of_day(resolved_timestamp) if resolved_timestamp is not None else None, factor)
        )
    run_minute = _minute_of_day(timestamp)
    psks_to_participation_factor: dict[str, float] = {}
    for psks_mrid, points in psks_points.items():
        value = _factor_at(points, run_minute, psks_interpolation.get(psks_mrid, "none"))
        if value is not None:
            psks_to_participation_factor[psks_mrid] = value

    # --- Step 6: assemble raw (pre-normalisation) participation factors per RA ---
    raw_participation_by_ra: dict[str, dict[str, float]] = {ra_mrid: {} for ra_mrid in available_ra_mrids}
    not_supported_units: dict[str, int] = {}
    for ra_mrid, psks_mrids in ra_to_psks.items():
        dropped_types: list[str] = []
        for psks_mrid in sorted(psks_mrids):
            if psks_mrid in psks_other_resources and not psks_to_gu.get(psks_mrid):
                dropped_types.extend(psks_other_resources[psks_mrid])
                continue
            gu_mrids = psks_to_gu.get(psks_mrid)
            participation_factor = psks_to_participation_factor.get(psks_mrid)
            if not gu_mrids or participation_factor is None:
                continue
            for unit_mrid in gu_mrids:
                generator = _generator_of_unit(unit_mrid, generators_by_unit, not_supported_units, tso)
                if generator is None:
                    continue
                raw_participation_by_ra[ra_mrid][generator] = (
                    raw_participation_by_ra[ra_mrid].get(generator, 0.0) + participation_factor
                )
        if dropped_types:
            log.warning(
                "%s: RD RA %s — %d PowerShiftKeySchedule(s) have a resource that is not a GeneratingUnit (%s) — "
                "not read, dropped",
                tso, ra_mrid, len(dropped_types), ", ".join(sorted(set(dropped_types))),
            )

    warn_not_supported(sis_path, tso, "unit-ids", not_supported_units)

    # --- Step 7: normalise — drop GUs absent from the network, redistribute proportionally ---
    for ra_mrid, raw_participation in raw_participation_by_ra.items():
        if not raw_participation:
            log.warning(
                "%s: RD RA %s — no PowerShiftKeySchedule/participationFactor "
                "resolved for timestamp %s (0 generating units) — GLSK is empty "
                "for this RA (confirmed dataset boundary condition on Jotunheim "
                "for the real Belgovia RD RA; a genuine parsing gap on other "
                "dataset cuts would look identical, so check the RA<->SIS "
                "linkage directly if this is unexpected)",
                tso, ra_mrid, timestamp,
            )
            glsk[ra_mrid] = {}
            continue

        raw_sum = sum(raw_participation.values())
        if abs(raw_sum - 1.0) > _SUM_TOLERANCE:
            log.warning(
                "%s: RD RA %s — the participation factors sum to %s, not 1 — they are relative weights and are normalised",
                tso, ra_mrid, f"{raw_sum:g}",
            )

        valid_participation = {
            gu_mrid: factor
            for gu_mrid, factor in raw_participation.items()
            if gu_mrid in valid_generator_mrids
        }
        removed_gu_mrids = set(raw_participation) - set(valid_participation)
        if removed_gu_mrids:
            log.warning(
                "%s: RD RA %s — %d generating unit(s) not found in "
                "network.get_generators().index, removed and redistributed: %s",
                tso, ra_mrid, len(removed_gu_mrids), sorted(removed_gu_mrids),
            )
        if connected_generator_mrids is not None:
            unconnected = sorted(g for g in valid_participation if g not in connected_generator_mrids)
            if unconnected:
                log.warning(
                    "%s: RD RA %s — %d generating unit(s) not connected, excluded from the "
                    "redispatch and redistributed: %s", tso, ra_mrid, len(unconnected), unconnected,
                )
                valid_participation = {g: f for g, f in valid_participation.items() if g not in unconnected}

        total_valid_factor = sum(valid_participation.values())
        if total_valid_factor <= 0.0:
            if valid_participation:
                log.warning(
                    "%s: RD RA %s — remaining generating unit(s) after removal "
                    "have zero total participation factor (%s) — cannot "
                    "redistribute proportionally, GLSK is empty for this RA",
                    tso, ra_mrid, sorted(valid_participation.keys()),
                )
            glsk[ra_mrid] = {}
            continue

        glsk[ra_mrid] = {
            gu_mrid: factor / total_valid_factor
            for gu_mrid, factor in valid_participation.items()
        }

    if output_path is not None:
        write_glsk_csv(glsk, output_path, timestamp, tso_by_ra_mrid={ra: tso for ra in glsk})

    total_ra_with_data = sum(1 for factors in glsk.values() if factors)
    log.info(
        "%s: resolve_glsk — %d / %d RD RA(s) have non-empty GLSK for timestamp %s",
        tso, total_ra_with_data, len(available_ra_mrids), timestamp,
    )

    return glsk


def warn_curative_no_net_movement(
    rd_list: list[RDUnit],
    bid_list: list,
    glsk: dict[str, dict[str, float]],
    contingencies: list,
    timestamp: str,
) -> list[dict]:
    """Warn once per contingency in which curative redispatch cannot move at all. The curative balance of a
    contingency sums to 0 over every curative RD RA acting there (any TSO), so there is no net movement when only
    one direction (up or down) exists among those RAs, or when every up RA and every down RA has the same
    normalised key (their volumes must be equal and cancel on every unit). An RA acts in the contingencies it is
    linked to, or in all of `contingencies` when it has no link, minus its excluded ones; only available curative RAs
    with a key and a bid of positive size at `timestamp` count (countertrade RAs have their own weights and are left
    out). contingencies: objects with mrid, name and tso. Returns [{contingency, name, reason, ras}] for the
    contingencies warned about."""
    direction_of: dict[str, set[str]] = {}
    for bid in bid_list:
        if float(bid.available_mw.get(timestamp, 0.0)) > 0.0:
            direction_of.setdefault(str(bid.ra_mrid), set()).add(bid.direction.value)
    curative: dict[str, RDUnit] = {}
    for rd in rd_list:
        if rd.available and rd.kind == RAKind.CURATIVE and not rd.countertrade and glsk.get(str(rd.mrid)):
            curative.setdefault(str(rd.mrid), rd)
    all_mrids = [str(c.mrid) for c in contingencies]
    found: list[dict] = []
    for contingency in contingencies:
        acting = {
            mrid: rd for mrid, rd in curative.items()
            if mrid in direction_of
            and str(contingency.mrid) in effective_linked_contingencies(
                rd.linked_contingency_mrids, rd.excluded_contingency_mrids, all_mrids)
        }
        ups = sorted(m for m in acting if "up" in direction_of[m])
        downs = sorted(m for m in acting if "down" in direction_of[m])
        if not acting:
            continue
        if not downs:
            reason = "only up RAs act there"
        elif not ups:
            reason = "only down RAs act there"
        else:
            keys = {tuple(sorted((unit, round(float(f), 9)) for unit, f in glsk[m].items())) for m in ups + downs}
            if len(keys) != 1:
                continue
            reason = "every up and every down RA has the same key"
        names = sorted(acting[m].name for m in acting)
        found.append({"contingency": str(contingency.mrid), "name": contingency.name, "reason": reason, "ras": names})
        log.warning(
            "curative redispatch in contingency %s (%s, TSO %s): no net movement is possible - %s. RAs acting there: %s. "
            "The curative redispatch balance of a contingency sums to 0, so these RAs can never move; a curative RA "
            "of the other direction with another key is needed",
            contingency.mrid, contingency.name, getattr(contingency, "tso", ""), reason, ", ".join(names),
        )
    return found


def write_glsk_csv(
    glsk: dict[str, dict[str, float]],
    output_path: Path,
    timestamp: str,
    tso_by_ra_mrid: Optional[dict[str, str]] = None,
) -> None:
    """Write glsk_resolved.csv once for all TSOs (with a tso column from tso_by_ra_mrid, blank
    when unknown). An RA with an empty factor dict still gets one row with gu_mrid and
    participation_factor blank, so every considered RA is listed."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    row_count = 0
    with open(output_path, "w", newline="", encoding=TEXT_ENCODING) as f:
        writer = csv.writer(f)
        writer.writerow(["ra_mrid", "gu_mrid", "participation_factor", "tso", "timestamp"])
        tso_by_ra_mrid = tso_by_ra_mrid or {}
        for ra_mrid in sorted(glsk.keys()):
            gu_factors = glsk[ra_mrid]
            tso = tso_by_ra_mrid.get(ra_mrid, "")
            if not gu_factors:
                writer.writerow([ra_mrid, "", "", tso, timestamp])
                row_count += 1
                continue
            for gu_mrid, factor in sorted(gu_factors.items()):
                writer.writerow([ra_mrid, gu_mrid, factor, tso, timestamp])
                row_count += 1

    log.info("Written GLSK CSV: %s (%d rows)", output_path, row_count)
