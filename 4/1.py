"""ae_switches.py - the NCP switches of the assessed elements, read from the AE and SIS graphs.

An absent switch means enabled. parse_assessed_elements (ncp_parser) applies what this module reads:
- AssessedElement.normalEnabled=false, or an AssessedElementTimePoint with enabled=false at the run time, makes the
  element EXCLUDED (not monitored); the time point overrides normalEnabled;
- inBaseCase=false: not monitored in the N state, only after contingencies;
- AssessedElementWithContingency links: kind excluded = not monitored after that contingency; included (and, with a
  monitored only after the linked contingencies; considered, a missing and an unknown kind are not allowed:
  counting warning, the link is treated as absent; a link with normalEnabled=false is ignored (counted);
- AssessedElement.isCombinableWithContingency=false: monitored after the contingencies with an included link only (none:
  after no contingency); true or absent: the links above decide (AssessedElement Profile Specification v2.4.1).
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

from rdflib import RDF, Graph

from pre_milp import run_time

log = logging.getLogger(__name__)


def boolean(literal) -> Optional[bool]:
    """True / False of a boolean literal, None when absent (an absent switch means enabled)."""
    return None if literal is None else str(literal).strip().lower() != "false"


def read_element_contingency_links(ae_graph: Graph, ae_path: Path, tso: str) -> dict[str, tuple[set[str], set[str]]]:
    """{assessed element fragment: (contingency fragments it is monitored for, contingency fragments it is not monitored
    for)} from the AssessedElementWithContingency links."""
    from pre_milp.ncp_parser import NC, _frag

    result: dict[str, tuple[set[str], set[str]]] = {}
    disabled = unusual = considered = 0
    for link in ae_graph.subjects(RDF.type, NC.AssessedElementWithContingency):
        element = ae_graph.value(link, NC["AssessedElementWithContingency.AssessedElement"])
        contingency = ae_graph.value(link, NC["AssessedElementWithContingency.Contingency"])
        if element is None or contingency is None:
            continue
        if boolean(ae_graph.value(link, NC["AssessedElementWithContingency.normalEnabled"])) is False:
            disabled += 1
            continue
        kind_literal = ae_graph.value(link, NC["AssessedElementWithContingency.combinationConstraintKind"])
        kind = re.split(r"[#.]", str(kind_literal))[-1] if kind_literal is not None else ""
        if kind == "considered":
            considered += 1          # not allowed on this link class (AssessedElement profile): the link is absent
            continue
        if kind not in ("included", "excluded"):
            unusual += 1             # kind is mandatory (1..1) and has two allowed values: the link is absent
            continue
        included, excluded = result.setdefault(_frag(str(element)), (set(), set()))
        (excluded if kind == "excluded" else included).add(_frag(str(contingency)))
    if disabled:
        log.warning("%s: %s has %d AssessedElementWithContingency link(s) with normalEnabled=false - ignored",
                    tso, ae_path.name, disabled)
    if considered:
        log.warning("%s: %s has %d AssessedElementWithContingency link(s) with combinationConstraintKind considered - "
                    "not allowed for this class (AssessedElement Profile Specification v2.4.1), the link is treated as absent",
                    tso, ae_path.name, considered)
    if unusual:
        log.warning("%s: %s has %d AssessedElementWithContingency link(s) with a missing or unknown "
                    "combinationConstraintKind - the link is treated as absent", tso, ae_path.name, unusual)
    return result


def read_element_schedule_enabled(sis_path: Optional[Path], timestamp: str, tso: str) -> dict[str, bool]:
    """{assessed element fragment: enabled} at the run time from the AssessedElementSchedule / AssessedElementTimePoint
    of the SIS file (points of the run's date, see run_time.py). An exact point at the run's hour counts; between points
    the interpolation kind decides (previous / linear: the earlier point, next: the later one, none: no value). An
    element without a value is absent from the result (then AssessedElement.normalEnabled, absent = enabled, decides)."""
    from pre_milp.ncp_parser import NC, _frag, _iso_to_hhmmz, _load_graph, _minute_of_day

    if sis_path is None:
        return {}
    graph = _load_graph(sis_path)
    schedule_element: dict[str, str] = {}
    schedule_kind: dict[str, str] = {}
    for schedule in graph.subjects(RDF.type, NC.AssessedElementSchedule):
        element = graph.value(schedule, NC["AssessedElementSchedule.AssessedElement"])
        if element is not None:
            schedule_element[_frag(str(schedule))] = _frag(str(element))
            kind = graph.value(schedule, NC["BaseTimeSeries.interpolationKind"])
            schedule_kind[_frag(str(schedule))] = re.split(r"[#.]", str(kind))[-1] if kind is not None else "none"
    points = list(graph.subjects(RDF.type, NC.AssessedElementTimePoint))
    if not points:
        return {}
    required = run_time.required_day(
        (run_time.iso_day(str(graph.value(p, NC["AssessedElementTimePoint.atTime"]))) for p in points),
        f"{tso}: assessed element schedules", sis_path.name,
    )
    run_minute = _minute_of_day(timestamp)
    series: dict[str, dict[int, bool]] = {}
    for point in points:
        schedule = graph.value(point, NC["AssessedElementTimePoint.AssessedElementSchedule"])
        at_time = graph.value(point, NC["AssessedElementTimePoint.atTime"])
        enabled = boolean(graph.value(point, NC["AssessedElementTimePoint.enabled"]))
        clock = _iso_to_hhmmz(str(at_time)) if at_time is not None else None
        if schedule is None or clock is None or enabled is None or _frag(str(schedule)) not in schedule_element:
            continue
        if not run_time.day_matches(run_time.iso_day(str(at_time)), required, _minute_of_day(clock), run_minute):
            continue
        series.setdefault(_frag(str(schedule)), {})[_minute_of_day(clock)] = enabled
    result: dict[str, bool] = {}
    for schedule, values in series.items():
        kind = schedule_kind.get(schedule, "none")
        if run_minute in values:
            value = values[run_minute]
        elif kind in ("previous", "linear") and any(m < run_minute for m in values):
            value = values[max(m for m in values if m < run_minute)]
        elif kind == "next" and any(m > run_minute for m in values):
            value = values[min(m for m in values if m > run_minute)]
        else:
            continue
        result[schedule_element[schedule]] = value
    return result
