"""ct_units.py — countertrade remedial actions without a MILP (docs/specs/countertrade_ra_specs.md C2, C3).

- parse_ct_remedial_actions: nc:CountertradeRemedialAction of one TSO's RA file with their power
  bids from the SIS file. Only the strategy proportionalForRemainingCapacity is implemented. The
  profile has no strategy attribute, so an RA without one gets it by default (warning); the old
  any gLSKStrategy value is ignored with a warning and the RA is kept (the remaining capacity is always used).
- parse_bidding_zones, read_control_area_units, read_generator_table, resolve_zone_units: the zone
  of an RA to its generating units (C2.5, C2.6): BiddingZone <- SchedulingArea (Organisation
  profile) -> ControlArea <- ControlAreaGeneratingUnit (CGM EQ files) -> GeneratingUnit -> the
  network generator. Units without a unit id or an area are dropped with a log line.
- resolve_ct_weights: the weights of every RA and direction from the original set points (C3).
"""

from __future__ import annotations

import dataclasses
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from rdflib import RDF

from data_model import BidDirection, CtRa, CtZoneUnit, PowerBid, RAKind, RDUnit
from pre_milp import cgmes_loader
from pre_milp.ct_weights import CtWeights, compute_ct_weights
from pre_milp.hvdc_units import MAIN_SYNCHRONOUS_COMPONENT, UNBOUNDED_MW
from pre_milp.ncp_parser import (
    CIM, NC, _enum_token, _eq_subjects_of_type, _eq_value, _frag, _load_graph, _parse_reaction_time_minutes,
    load_eq_graph, ncp_id, parse_power_bids, read_contingency_links, run_cached, warn_not_supported,
)

log = logging.getLogger(__name__)

STRATEGY_REMAINING_CAPACITY = "proportionalForRemainingCapacity"

# CGMES classes behind a network generator that are never countertrade units (C2.6)
EXCLUDED_GENERATOR_CLASSES = ("EquivalentInjection", "ExternalNetworkInjection")


# ---------------------------------------------------------------------------
# The remedial actions and their bids (RA and SIS files)
# ---------------------------------------------------------------------------

def _mrid(graph, subject) -> str:
    literal = graph.value(subject, CIM["IdentifiedObject.mRID"])
    return ncp_id(literal) if literal is not None else _frag(str(subject))


def _warn_shift_keys_of_countertrade(ra_graph, sis_graph, ct_subjects, ra_path: Path, tso: str) -> tuple[int, int]:
    """One counting warning per RA file when a countertrade RA has a PowerShiftKeyStrategy or one of its
    bids a PowerShiftKeyDistribution: countertrade takes its units and weights from the bidding zone
    (docs/specs/countertrade_ra_specs.md C3), so such a key is not read. Returns (strategies, distributions)."""
    ct_ids = {_frag(str(uri)) for uri in ct_subjects} | {_mrid(ra_graph, uri) for uri in ct_subjects}
    strategies = {
        _frag(str(strategy)) for strategy in ra_graph.subjects(RDF.type, NC.PowerShiftKeyStrategy)
        if (ref := ra_graph.value(strategy, NC["PowerShiftKeyStrategy.PowerRemedialAction"])) is not None
        and _frag(str(ref)) in ct_ids
    }
    distributions = 0
    if sis_graph is not None:
        ct_bids = {
            _frag(str(pbs)) for pbs in sis_graph.subjects(RDF.type, NC.PowerBidSchedule)
            if (ref := sis_graph.value(pbs, NC["PowerBidSchedule.PowerRemedialAction"])) is not None
            and _frag(str(ref)) in ct_ids
        }
        for distribution in sis_graph.subjects(RDF.type, NC.PowerShiftKeyDistribution):
            bids = {_frag(str(o)) for o in sis_graph.objects(distribution, NC["PowerShiftKeyDistribution.PowerBidSchedule"])}
            keys = {_frag(str(o)) for o in sis_graph.objects(distribution, NC["PowerShiftKeyDistribution.PowerShiftKeyStrategy"])}
            if bids & ct_bids or keys & strategies:
                distributions += 1
    if strategies or distributions:
        log.warning(
            "%s: %s has %d PowerShiftKeyStrategy element(s) of countertrade RAs and %d PowerShiftKeyDistribution "
            "element(s) of countertrade RAs or their bids - not read, the units and weights of a countertrade RA come "
            "from its bidding zone", tso, ra_path.name, len(strategies), distributions,
        )
    return len(strategies), distributions


def parse_ct_remedial_actions(
    ra_path: Path,
    sis_path: Optional[Path],
    tso: str,
    timestamp: str,
    operators=None,  # OperatorDirectory: the TSO of an RA is its RemedialActionSystemOperator (else `tso`, the file's)
) -> tuple[list[CtRa], list[PowerBid]]:
    """The usable countertrade RAs of {TSO}_RA.xml and their power bids from {TSO}_SIS.xml.
    A usable RA has a bidding zone. Its strategy is always proportionalForRemainingCapacity: a RA that declares nothing
    gets it with a warning; any strategy value (nc:CountertradeRemedialAction.gLSKStrategy / nc:PowerBidSchedule.gLSKStrategy,
    nc:CountertradeRemedialAction.shiftMethod) is ignored with one warning per RA (L5). An RA without a
    bidding zone is ignored with a warning. powerShiftKey and method are not supported: ignored, one summary warning
    per file. The zone itself is checked
    later, against the Organisation profile (resolve_zone_units). Missing RemedialAction.penaltyFactor
    is 1.0 with a log line (as RD); a bid without volume at the run timestamp keeps volume 0 with a
    warning. RAs without bids are kept (logged). Bids are PowerBid objects with ra_mrid = the RA."""
    ra_graph = _load_graph(ra_path)
    ct_subjects = sorted(ra_graph.subjects(RDF.type, NC.CountertradeRemedialAction), key=str)
    if not ct_subjects:
        return [], []

    sis_graph = _load_graph(sis_path) if sis_path is not None else None
    run_timestamp = timestamp

    links, excluded_links = read_contingency_links(ra_graph, ra_path, tso)
    _warn_shift_keys_of_countertrade(ra_graph, sis_graph, ct_subjects, ra_path, tso)

    # strategies declared on the bids, per RA
    bid_strategies: dict[str, set[str]] = {}
    if sis_graph is not None:
        for pbs_uri in sis_graph.subjects(RDF.type, NC.PowerBidSchedule):
            ra_uri = sis_graph.value(pbs_uri, NC["PowerBidSchedule.PowerRemedialAction"])
            token = _enum_token(sis_graph.value(pbs_uri, NC["PowerBidSchedule.gLSKStrategy"]))
            if ra_uri is not None and token:
                bid_strategies.setdefault(_frag(str(ra_uri)), set()).add(token)

    ras: list[CtRa] = []
    not_supported_attributes: dict[str, int] = {}
    for ra_uri in ct_subjects:
        mrid = _mrid(ra_graph, ra_uri)
        name_literal = ra_graph.value(ra_uri, CIM["IdentifiedObject.name"])
        name = str(name_literal) if name_literal is not None else mrid

        declared = set(bid_strategies.get(_frag(str(ra_uri)), set()) | bid_strategies.get(mrid, set()))
        ra_token = _enum_token(ra_graph.value(ra_uri, NC["CountertradeRemedialAction.gLSKStrategy"]))
        if ra_token:
            declared.add(ra_token)
        for attribute in ("powerShiftKey", "method"):
            if ra_graph.value(ra_uri, NC[f"CountertradeRemedialAction.{attribute}"]) is not None:
                key = f"CountertradeRemedialAction.{attribute}"
                not_supported_attributes[key] = not_supported_attributes.get(key, 0) + 1
        # L5 (DECISIONS 167): any strategy value (gLSKStrategy on the RA or its bids, shiftMethod) is ignored with one warning
        # per RA; the remaining capacity is always used
        shift_method = ra_graph.value(ra_uri, NC["CountertradeRemedialAction.shiftMethod"])
        ignored_values = [f"gLSKStrategy {token}" for token in sorted(declared)]
        if shift_method is not None:
            ignored_values.append(f"shiftMethod {shift_method}")
        strategy = STRATEGY_REMAINING_CAPACITY
        if ignored_values:
            log.warning(
                "%s: countertrade RA %s (%s): %s ignored - the remaining capacity (%s) is always used",
                tso, mrid, name, ", ".join(ignored_values), STRATEGY_REMAINING_CAPACITY,
            )
        if not declared and not ignored_values:
            log.warning(
                "%s: countertrade RA %s (%s) has no strategy (the profile defines none) - %s assumed",
                tso, mrid, name, STRATEGY_REMAINING_CAPACITY,
            )
        zone_uri = ra_graph.value(ra_uri, NC["PowerRemedialAction.BiddingZone"])
        if zone_uri is None:
            log.warning("%s: countertrade RA %s (%s) is IGNORED - it has no bidding zone", tso, mrid, name)
            continue

        available_literal = ra_graph.value(ra_uri, NC["RemedialAction.normalAvailable"])
        kind_uri = ra_graph.value(ra_uri, NC["RemedialAction.kind"])
        kind = RAKind.CURATIVE if kind_uri is not None and "curative" in _frag(str(kind_uri)).lower() else RAKind.PREVENTIVE
        penalty_literal = ra_graph.value(ra_uri, NC["RemedialAction.penaltyFactor"])
        if penalty_literal is None:
            log.warning(
                "%s: countertrade RA %s (%s) has no RemedialAction.penaltyFactor - defaulted to 1.0", tso, mrid, name,
            )
            penalty_factor = 1.0
        else:
            penalty_factor = float(str(penalty_literal))
        ras.append(CtRa(
            mrid=mrid, name=name,
            tso=(operators.tso_of(ra_graph, ra_uri, "RemedialAction.RemedialActionSystemOperator", tso, ra_path.name, mrid)
                 if operators is not None else tso),
            zone_mrid=_frag(str(zone_uri)),
            available=str(available_literal).lower() == "true" if available_literal is not None else False,
            kind=kind, penalty_factor=penalty_factor, strategy=strategy,
            reaction_time_minutes=_parse_reaction_time_minutes(
                ra_graph.value(ra_uri, NC["RemedialAction.timeToImplement"]), tso, mrid,
            ),
            linked_contingency_mrids=list(links.get(mrid, links.get(_frag(str(ra_uri)), []))),
            excluded_contingency_mrids=list(excluded_links.get(mrid, excluded_links.get(_frag(str(ra_uri)), []))),
        ))

    warn_not_supported(ra_path, tso, "countertrade-attributes", not_supported_attributes, log)
    bids: list[PowerBid] = []
    if ras and sis_graph is not None:
        bids = parse_power_bids(sis_graph, {ra.mrid for ra in ras}, tso, timestamp)
    for ra in ras:
        own = [bid for bid in bids if bid.ra_mrid == ra.mrid]
        if not own:
            log.warning("%s: countertrade RA %s (%s) has no power bid - it can do nothing", tso, ra.mrid, ra.name)
        for bid in own:
            if float(bid.available_mw.get(run_timestamp, 0.0)) <= 0.0:
                log.warning(
                    "%s: countertrade bid %s (RA %s) has no volume at %s - volume 0", tso, bid.mrid, ra.mrid, run_timestamp,
                )
    log.info("%s: %d countertrade RA(s) and %d bid(s) parsed", tso, len(ras), len(bids))
    return ras, bids


# ---------------------------------------------------------------------------
# Bidding zones (Organisation profile) and control-area units (CGM EQ files)
# ---------------------------------------------------------------------------

@dataclass
class CtZone:
    mrid: str
    name: str
    control_area_mrids: list[str] = field(default_factory=list)
    system_operator_mrid: str = ""      # SchedulingArea.SystemOperator, "" when the profile gives none


def parse_bidding_zones(cd_path: Path) -> dict[str, CtZone]:
    """{BiddingZone mRID: CtZone} from Org-NineRealms_CD.xml: the zone with its name and the
    ControlAreas of the SchedulingAreas that point to it."""
    graph = _load_graph(cd_path)
    zones: dict[str, CtZone] = {}
    for zone_uri in graph.subjects(RDF.type, NC.BiddingZone):
        mrid = _mrid(graph, zone_uri)
        name = graph.value(zone_uri, CIM["IdentifiedObject.name"])
        zones[_frag(str(zone_uri))] = CtZone(mrid=mrid, name=str(name) if name is not None else mrid)
    for area_uri in graph.subjects(RDF.type, NC.SchedulingArea):
        zone_uri = graph.value(area_uri, NC["SchedulingArea.BiddingZone"])
        control_area_uri = graph.value(area_uri, NC["SchedulingArea.ControlArea"])
        if zone_uri is None or control_area_uri is None:
            continue
        zone = zones.get(_frag(str(zone_uri)))
        if zone is not None and _frag(str(control_area_uri)) not in zone.control_area_mrids:
            zone.control_area_mrids.append(_frag(str(control_area_uri)))
        operator_uri = graph.value(area_uri, NC["SchedulingArea.SystemOperator"])
        if zone is not None and operator_uri is not None and not zone.system_operator_mrid:
            zone.system_operator_mrid = _frag(str(operator_uri))
    return zones


def unit_zone_map(zones: dict[str, CtZone], area_units: dict[str, set[str]], generators: dict[str, "CtGenerator"]) -> dict[str, str]:
    """{generator id: bidding zone mRID} of every generator whose GeneratingUnit is in a control area of a zone
    (net positions of the zones, C11.3)."""
    zone_of_unit_mrid: dict[str, str] = {}
    for zone in zones.values():
        for area in zone.control_area_mrids:
            for unit_mrid in area_units.get(area, set()):
                zone_of_unit_mrid.setdefault(unit_mrid, zone.mrid)
    return {
        gu_mrid: zone_of_unit_mrid[generator.unit_mrid]
        for gu_mrid, generator in generators.items() if generator.unit_mrid in zone_of_unit_mrid
    }


def _control_area_unit_rows(graph) -> list[tuple[str, str]]:
    """[(ControlArea mRID, GeneratingUnit mRID)] of the ControlAreaGeneratingUnit objects of one EQ graph, in the
    order of the graph."""
    rows = []
    for row in _eq_subjects_of_type(graph, "ControlAreaGeneratingUnit"):
        area = _eq_value(graph, row, "ControlAreaGeneratingUnit.ControlArea")
        unit = _eq_value(graph, row, "ControlAreaGeneratingUnit.GeneratingUnit")
        if area is not None and unit is not None:
            rows.append((_frag(str(area)), _frag(str(unit))))
    return rows


@dataclass
class CtControlArea:
    """A ControlArea of the CGM with an EIC (entsoe energyIdentCodeEic): its mRID, name and every GeneratingUnit of the EQ
    file it is in."""
    mrid: str
    name: str
    eic: str
    file_units: set[str] = field(default_factory=set)


_RDF = "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}"


def read_control_areas(eq_dir: Path) -> dict[str, CtControlArea]:
    """{EIC: CtControlArea} of the ControlAreas of the CGM EQ files that have an energyIdentCodeEic (any namespace of
    the entsoe extension: CGMES 3.0 or 2.4.15; the classes are read by local name, in any CIM namespace). One streaming
    pass per file that holds the attribute; the first ControlArea with an EIC wins."""
    import xml.etree.ElementTree as ET

    found: dict[str, CtControlArea] = {}
    for path in cgmes_loader.find_eq_files(Path(eq_dir)):
        if b"energyIdentCodeEic" not in path.read_bytes():
            continue
        areas: list[tuple[str, str, str]] = []
        units: set[str] = set()
        depth = 0
        root = None
        for event, element in ET.iterparse(str(path), events=("start", "end")):
            if event == "start":
                depth += 1
                if root is None:
                    root = element
                continue
            depth -= 1
            if depth != 1:
                continue                                       # not an object directly under rdf:RDF
            local = element.tag.rsplit("}", 1)[-1]
            ident = ncp_id(element.get(f"{_RDF}ID") or element.get(f"{_RDF}about") or "")
            if local == "ControlArea":
                eic = next((c.text.strip() for c in element if c.tag.rsplit("}", 1)[-1] == "IdentifiedObject.energyIdentCodeEic" and c.text), "")
                name = next((c.text.strip() for c in element if c.tag.rsplit("}", 1)[-1] == "IdentifiedObject.name" and c.text), "")
                if eic and ident:
                    areas.append((ident, name, eic))
            elif local.endswith("GeneratingUnit") and local != "ControlAreaGeneratingUnit" and ident:
                units.add(ident)
            root.clear()
        for ident, name, eic in areas:
            found.setdefault(eic, CtControlArea(mrid=ident, name=name, eic=eic, file_units=units))
    return found


def zones_from_control_areas(
    ras: list[CtRa], eq_dir: Path, area_units: dict[str, set[str]],
) -> tuple[dict[str, CtZone], dict[str, set[str]]]:
    """Without an Organisation file (L6): ({zone EIC: CtZone}, {ControlArea mRID: units}) for the zones of `ras`. The zone
    of an RA is the EIC in its PowerRemedialAction.BiddingZone; it is the zone of the CGM ControlArea with that EIC. The
    units of the area are those of ControlAreaGeneratingUnit (area_units) when it has any, else every GeneratingUnit of the
    EQ file the ControlArea is in. A zone EIC that no ControlArea carries has no zone (resolve_zone_units drops its RAs)."""
    areas = read_control_areas(eq_dir)
    zones: dict[str, CtZone] = {}
    units = {area: set(members) for area, members in area_units.items()}
    for eic in sorted({ra.zone_mrid for ra in ras}):
        area = areas.get(eic)
        if area is None:
            continue
        zones[eic] = CtZone(mrid=eic, name=area.name or eic, control_area_mrids=[area.mrid])
        if not units.get(area.mrid):
            units[area.mrid] = set(area.file_units)
    return zones, units


def remember_control_area_units(eq_path: Path, graph) -> None:
    """Keep the control-area units of an EQ file that is parsed anyway, for read_control_area_units later in the
    same run (inside ncp_parser.graph_cache(); otherwise nothing is kept)."""
    run_cached("area-units", eq_path, lambda: _control_area_unit_rows(graph))


def read_control_area_units(eq_dir: Path) -> dict[str, set[str]]:
    """{ControlArea mRID: {GeneratingUnit mRIDs}} from the ControlAreaGeneratingUnit objects of
    the CGM EQ files in eq_dir (only the files that contain such objects are parsed; a file whose units were
    kept by remember_control_area_units is not parsed again)."""
    units: dict[str, set[str]] = {}
    for path in cgmes_loader.find_eq_files(Path(eq_dir)):
        if b"ControlAreaGeneratingUnit" not in path.read_bytes():
            continue
        rows = run_cached("area-units", path, lambda path=path: _control_area_unit_rows(load_eq_graph(path)))
        for area, unit in rows:
            units.setdefault(area, set()).add(unit)
    return units


# ---------------------------------------------------------------------------
# The generators of the network
# ---------------------------------------------------------------------------

@dataclass
class CtGenerator:
    """One network generator: its id, the CGMES GeneratingUnit mRID behind it ("" = none), the
    CGMES class, the original set point and limits (MW, generation sign), whether it is
    connected and in the main synchronous component."""
    gu_mrid: str
    unit_mrid: str
    name: str
    original_class: str
    prod: float
    pmin: float
    pmax: float
    connected: bool
    in_main_component: bool


def _text(value) -> str:
    return "" if value is None or (isinstance(value, float) and math.isnan(value)) else str(value)


def read_generator_table(network) -> dict[str, CtGenerator]:
    """{generator id: CtGenerator} of every generator of the network (including the
    EquivalentInjections, which pypowsybl lists as generators), from the loaded network."""
    generators = network.get_generators(all_attributes=True)
    buses = network.get_buses(all_attributes=True)
    table: dict[str, CtGenerator] = {}
    for gu_mrid, row in generators.iterrows():
        bus_id = row.get("bus_id")
        connected = bool(row.get("connected", True)) and isinstance(bus_id, str) and bus_id in buses.index
        component = buses.loc[bus_id, "synchronous_component"] if connected else None
        table[str(gu_mrid)] = CtGenerator(
            gu_mrid=str(gu_mrid),
            unit_mrid=ncp_id(_text(row.get("CGMES.GeneratingUnit"))),
            name=_text(row.get("name")) or str(gu_mrid),
            original_class=_text(row.get("CGMES.originalClass")),
            prod=float(row["target_p"]), pmin=float(row["min_p"]), pmax=float(row["max_p"]),
            connected=connected,
            in_main_component=bool(connected and component == MAIN_SYNCHRONOUS_COMPONENT),
        )
    return table


def _limits_unbounded(generator: CtGenerator) -> bool:
    return not all(math.isfinite(v) and abs(v) < UNBOUNDED_MW for v in (generator.pmin, generator.pmax))


def _first_ids(label: str, ids: list[str], shown: int = 5) -> str:
    """"; <label>: a, b, c" with the first `shown` ids and the number of the others; "" for none."""
    if not ids:
        return ""
    more = f" and {len(ids) - shown} more" if len(ids) > shown else ""
    return f"; {label}: {", ".join(ids[:shown])}{more}"


def _log_unassigned_generators(generators: dict[str, CtGenerator], area_units: dict[str, set[str]]) -> None:
    """One log line for the generators no zone can reach: no unit id, or a unit in no control area
    (the EquivalentInjections and external grids are not units and are only counted)."""
    in_an_area = set().union(*area_units.values()) if area_units else set()
    not_units = [g for g in generators.values() if g.original_class in EXCLUDED_GENERATOR_CLASSES]
    machines = [g for g in generators.values() if g.original_class not in EXCLUDED_GENERATOR_CLASSES]
    no_id = sorted(g.gu_mrid for g in machines if not g.unit_mrid)
    no_area = sorted(g.gu_mrid for g in machines if g.unit_mrid and g.unit_mrid not in in_an_area)
    log.info(
        "countertrade units: %d generator(s) are EquivalentInjections or external grids (never units); "
        "%d without a GeneratingUnit id and %d in no control area are dropped from every zone%s%s",
        len(not_units), len(no_id), len(no_area),
        _first_ids("no unit id", no_id), _first_ids("no area", no_area),
    )
    log.debug("countertrade units: no unit id: %s; no area: %s", no_id, no_area)


def resolve_zone_units(
    ras: list[CtRa],
    zones: dict[str, CtZone],
    area_units: dict[str, set[str]],
    generators: dict[str, CtGenerator],
    zone_source: str = "the Organisation profile",
) -> tuple[list[CtRa], dict[str, list[CtZoneUnit]]]:
    """(the RAs whose zone resolves, {zone mRID: its candidate units}). An RA whose zone is not in
    zone_source (the Organisation profile, or the CGM ControlAreas without one), or whose zone has no candidate unit, is dropped with a warning.
    Candidate units (C2.6): generators of the zone's control areas that are not an
    EquivalentInjection or external grid, are connected, in the main synchronous component and
    have bounded limits; each removal is counted in one log line per zone. Units of a zone are
    ordered by generator id. The returned RAs carry zone_name."""
    by_unit: dict[str, list[CtGenerator]] = {}
    for generator in generators.values():
        if generator.unit_mrid:
            by_unit.setdefault(generator.unit_mrid, []).append(generator)
    _log_unassigned_generators(generators, area_units)

    units_by_zone: dict[str, list[CtZoneUnit]] = {}
    for zone_mrid in sorted({ra.zone_mrid for ra in ras}):
        zone = zones.get(zone_mrid)
        if zone is None:
            continue
        dropped = {"not a unit class": 0, "not connected": 0, "outside the main component": 0, "limits unbounded": 0}
        candidates: list[CtZoneUnit] = []
        unit_mrids = sorted(set().union(*(area_units.get(a, set()) for a in zone.control_area_mrids)))
        for unit_mrid in unit_mrids:
            for generator in by_unit.get(unit_mrid, []):
                if generator.original_class in EXCLUDED_GENERATOR_CLASSES:
                    dropped["not a unit class"] += 1
                elif not generator.connected:
                    dropped["not connected"] += 1
                elif not generator.in_main_component:
                    dropped["outside the main component"] += 1
                elif _limits_unbounded(generator):
                    dropped["limits unbounded"] += 1
                else:
                    candidates.append(CtZoneUnit(
                        gu_mrid=generator.gu_mrid, unit_mrid=unit_mrid, name=generator.name, prod=generator.prod,
                        pmin=generator.pmin, pmax=generator.pmax, connected=True,
                    ))
        candidates.sort(key=lambda unit: unit.gu_mrid)
        units_by_zone[zone_mrid] = candidates
        log.info(
            "countertrade zone %s (%s): %d candidate unit(s) of %d in its control area(s); dropped: %s",
            zone_mrid, zone.name, len(candidates), len(unit_mrids),
            ", ".join(f"{count} {reason}" for reason, count in dropped.items() if count) or "none",
        )

    kept: list[CtRa] = []
    for ra in ras:
        zone = zones.get(ra.zone_mrid)
        if zone is None:
            log.warning(
                "%s: countertrade RA %s (%s) is DROPPED - bidding zone %s is not in %s",
                ra.tso, ra.mrid, ra.name, ra.zone_mrid, zone_source,
            )
        elif not units_by_zone.get(ra.zone_mrid):
            log.warning(
                "%s: countertrade RA %s (%s) is DROPPED - bidding zone %s (%s) has no generating unit that can take part",
                ra.tso, ra.mrid, ra.name, ra.zone_mrid, zone.name,
            )
        else:
            kept.append(dataclasses.replace(ra, zone_name=zone.name))
    return kept, units_by_zone


def build_ct_milp_data(
    ras: list[CtRa],
    bids: list[PowerBid],
    units_by_zone: dict[str, list[CtZoneUnit]],
    weights: dict[tuple[str, BidDirection], CtWeights],
) -> tuple[list[RDUnit], list[PowerBid], dict[str, dict[str, float]]]:
    """The countertrade RAs in the shape the RD part of the MILP reads (C4.1): (RDUnit objects with
    countertrade=True, one per RA and unit that can move in at least one direction; the bids of those
    RAs; {bid mRID: {unit: weight}} of the direction of each bid, empty for a direction without room,
    which fixes that bid to volume 0). An RA with no unit that can move at all is left out (logged)."""
    zone_units = {zone: {unit.gu_mrid: unit for unit in units} for zone, units in units_by_zone.items()}
    rd_units: list[RDUnit] = []
    kept_bids: list[PowerBid] = []
    participation: dict[str, dict[str, float]] = {}
    for ra in ras:
        movable = sorted(
            set(weights[(ra.mrid, BidDirection.UP)].pf) | set(weights[(ra.mrid, BidDirection.DOWN)].pf)
        )
        if not movable:
            log.warning(
                "%s: countertrade RA %s (%s) is DROPPED - no unit of its zone can move in either direction",
                ra.tso, ra.mrid, ra.name,
            )
            continue
        for gu_mrid in movable:
            unit = zone_units[ra.zone_mrid][gu_mrid]
            rd_units.append(RDUnit(
                ct_weight_up=float(weights[(ra.mrid, BidDirection.UP)].pf.get(gu_mrid, 0.0)),
                ct_weight_down=float(weights[(ra.mrid, BidDirection.DOWN)].pf.get(gu_mrid, 0.0)),
                mrid=ra.mrid, name=ra.name, tso=ra.tso, gu_mrid=gu_mrid, gu_name=unit.name, available=ra.available,
                kind=ra.kind, initial_p=unit.prod, max_p=unit.pmax, min_p=unit.pmin,
                raise_ramp_rate=None, lower_ramp_rate=None, penalty_factor=ra.penalty_factor,
                linked_contingency_mrids=list(ra.linked_contingency_mrids),
                excluded_contingency_mrids=list(ra.excluded_contingency_mrids),
                linked_xne_mrids=list(ra.linked_xne_mrids), reaction_time_minutes=ra.reaction_time_minutes,
                countertrade=True,
            ))
        for bid in bids:
            if bid.ra_mrid == ra.mrid:
                kept_bids.append(bid)
                participation[str(bid.mrid)] = dict(weights[(ra.mrid, bid.direction)].pf)
    return rd_units, kept_bids, participation


def resolve_ct_weights(
    ras: list[CtRa],
    units_by_zone: dict[str, list[CtZoneUnit]],
    bids: Optional[list[PowerBid]] = None,
) -> dict[tuple[str, BidDirection], CtWeights]:
    """{(RA mRID, direction): CtWeights} for both directions of every RA, from the original set
    points of the zone's units (the same in every mode and iteration, C3.5). With bids, the log
    lines (an empty direction is a warning) are written only for the directions the RA has a bid in."""
    weights: dict[tuple[str, BidDirection], CtWeights] = {}
    for ra in ras:
        bid_directions = {bid.direction for bid in bids if bid.ra_mrid == ra.mrid} if bids is not None else None
        for direction in (BidDirection.UP, BidDirection.DOWN):
            has_bid = bid_directions is None or direction in bid_directions
            weights[(ra.mrid, direction)] = compute_ct_weights(
                units_by_zone.get(ra.zone_mrid, []), direction,
                label=f"countertrade RA {ra.mrid}" if has_bid else "",
            )
    return weights
