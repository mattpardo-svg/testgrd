"""
ncp_parser.py — NCP XML parsing per TSO: assessed elements + PATL (AE, SHS), contingencies
(CO), PST and RD remedial actions with power bids (RA, SIS), RA groups, and the EQ-file maps
(TapChanger -> PowerTransformer, CurrentLimit -> side). Returns data_model dataclasses.

SHS atTime "2023-07-22T10:30:00Z" is matched to the timestamp "1030Z" via "THHMM" -> "HHMM".
"""

from __future__ import annotations

import contextlib
import contextvars
import csv
import logging
import re
import sys
from datetime import date
from pathlib import Path
from typing import Optional

from rdflib import Graph, Namespace, RDF, URIRef

# Ensure src/ is on the path when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_model import (
    AEType, AssessedElement,
    ContingencyType, Contingency,
    RAKind, PST,
    BidDirection, RDUnit, PowerBid, UnitCost,
    HVDC_DIRECTIONS, HVDC_VALUE_KINDS, HvdcGroup, HvdcRange, HvdcUnit,
)
from pre_milp import ae_switches, run_time
from pre_milp.eq_stream import stream_eq
from text_encoding import TEXT_ENCODING

log = logging.getLogger(__name__)

# Namespaces
CIM = Namespace("https://cim.ucaiug.io/ns#")
NC  = Namespace("https://cim4.eu/ns/nc#")
# Grid/EQ CGMES files may use the IEC CIM100 namespace or the CGMES 2.4.15 one (cim16) for the "cim:" prefix instead
# of cim.ucaiug.io; tried as fallbacks below.
CIM100 = Namespace("http://iec.ch/TC57/CIM100#")
CIM16 = Namespace("http://iec.ch/TC57/2013/CIM-schema-cim16#")
# The nc and cim namespaces of the NCP 2.2 legacy form; a file that declares one is read as if it used the current URI.
LEGACY_NCP_NAMESPACES = {"http://entsoe.eu/ns/nc#": str(NC), "http://iec.ch/TC57/CIM100#": str(CIM)}

# Exclusion reason URI fragments that mean "excluded" (not secured, not scanned): the complete
# set of SecuredExclusionReasonKind literals. Any other value is fatal (see below).
_NO_REGION_CODE_REASON = "no secured or scanned region code"

_EXCLUSION_REASON_VALID = {
    "SecuredExclusionReasonKind.systemOperator",
    "SecuredExclusionReasonKind.capacityCalculationRegion",
    "SecuredExclusionReasonKind.nonNativeCapacityCalculationRegion",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DEPRECATION_WARNED: set = set()


def warn_deprecated_form_once(path: Path, key: str, message: str) -> None:
    """Log `message` (already formatted) as a warning once per file and kind of old-form data
    (the standard form is read first; the old mock form stays a fallback until it is removed)."""
    token = (str(path), key)
    if token in _DEPRECATION_WARNED:
        return
    _DEPRECATION_WARNED.add(token)
    log.warning(message)


def read_contingency_links(graph: Graph, path: Path, tso: str) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """({RA fragment: contingency fragments}, {RA fragment: excluded contingency fragments}) from the
    ContingencyWithRemedialAction elements of an RA graph. combinationConstraintKind (last word):
    excluded -> the second map (the RA is not used in that contingency); included -> the first;
    considered -> the first, read as included until its meaning is settled (warning once per file);
    missing or unknown -> the first, with a warning once per file."""
    included: dict[str, list[str]] = {}
    excluded: dict[str, list[str]] = {}
    for cwra_uri in graph.subjects(RDF.type, NC.ContingencyWithRemedialAction):
        ra_uri = graph.value(cwra_uri, NC["ContingencyWithRemedialAction.RemedialAction"])
        co_uri = graph.value(cwra_uri, NC["ContingencyWithRemedialAction.Contingency"])
        if ra_uri is None or co_uri is None:
            continue
        kind_lit = graph.value(cwra_uri, NC["ContingencyWithRemedialAction.combinationConstraintKind"])
        kind = re.split(r"[#.]", str(kind_lit))[-1] if kind_lit is not None else ""
        if kind == "excluded":
            excluded.setdefault(_frag(str(ra_uri)), []).append(_frag(str(co_uri)))
            continue
        if kind == "considered":
            warn_deprecated_form_once(
                path, "combination-considered",
                "%s: %s has ContingencyWithRemedialAction links with combinationConstraintKind considered - read as included "
                "until the meaning is settled" % (tso, path.name),
            )
        elif kind != "included":
            warn_deprecated_form_once(
                path, f"combination-kind-{kind}",
                "%s: %s has ContingencyWithRemedialAction links with a missing or unknown combinationConstraintKind (%s) - "
                "read as included" % (tso, path.name, kind or "none"),
            )
        included.setdefault(_frag(str(ra_uri)), []).append(_frag(str(co_uri)))
    return included, excluded


def _float_or_none(literal) -> Optional[float]:
    """float of an RDF literal; None when absent or not a number."""
    try:
        return float(str(literal)) if literal is not None else None
    except (TypeError, ValueError):
        return None


def ncp_id(text) -> str:
    """The id in `text`, the one place where an NCP id is cleaned (DECISIONS 146): an IdentifiedObject.mRID literal or an
    rdf:ID / rdf:about / rdf:resource value without its "#" or "urn:uuid:" wrapper. The first leading "_" is removed
    (some exports write "_" before the id, in the id, in the references or in the mRID, and the CGM ids have none);
    nothing else changes ("RA_1" and "a_b" stay as they are, "__x" becomes "_x")."""
    text = str(text).strip().rsplit("#", 1)[-1]
    if text.startswith("urn:uuid:"):
        text = text[len("urn:uuid:"):]
    return text[1:] if text.startswith("_") else text


def _frag(uri: str) -> str:
    """Local fragment of a URI ("#_x" and ".../x" give x), through ncp_id."""
    return ncp_id(uri.split("#")[-1] if "#" in uri else uri.split("/")[-1])


def _iso_to_hhmmz(iso: str) -> Optional[str]:
    """Extract HHMMZ from an ISO datetime string.

    "2023-07-22T10:30:00Z" → "1030Z"
    Returns None if the string doesn't match.
    """
    m = re.search(r"T(\d{2}):(\d{2}):", iso)
    if not m:
        return None
    return f"{m.group(1)}{m.group(2)}Z"


_ISO_DURATION = re.compile(
    r"P(?:(\d+)Y)?(?:(\d+)M)?(?:(\d+(?:\.\d+)?)D)?"
    r"(?:T(?:(\d+(?:\.\d+)?)H)?(?:(\d+(?:\.\d+)?)M)?(?:(\d+(?:\.\d+)?)S)?)?"
)


def _parse_iso_duration_hours_or_none(duration: str) -> Optional[float]:
    """ISO 8601 duration -> fractional hours, days included ("P1D" -> 24.0, "P1DT2H" -> 26.0,
    "P0Y0M0DT0H30M0S" -> 0.5). None when the text is not a duration: no match, no component at
    all ("P", "PT"), or a non-zero year or month (no fixed length in hours)."""
    m = _ISO_DURATION.fullmatch(str(duration).strip())
    if not m or all(group is None for group in m.groups()):
        return None
    years, months, days, hours, minutes, seconds = m.groups()
    if float(years or 0) or float(months or 0):
        return None
    return float(days or 0) * 24 + float(hours or 0) + float(minutes or 0) / 60 + float(seconds or 0) / 3600


def _parse_iso_duration_hours(duration: str) -> float:
    """_parse_iso_duration_hours_or_none, with 0.0 on parse failure."""
    hours = _parse_iso_duration_hours_or_none(duration)
    return 0.0 if hours is None else hours


def _parse_reaction_time_minutes(time_to_implement, tso: str, ra_mrid: str) -> Optional[float]:
    """RemedialAction.timeToImplement of an RD RA in minutes (CORNET-613): None when absent (the
    reaction time is then 0 without a warning); a value that cannot be read counts as 0 with a
    warning (R9 rule 9)."""
    if time_to_implement is None:
        return None
    hours = _parse_iso_duration_hours_or_none(str(time_to_implement))
    if hours is None:
        log.warning(
            "%s: RD remedial action %s has an unreadable RemedialAction.timeToImplement %r - "
            "reaction time 0 used (available on every branch)", tso, ra_mrid, str(time_to_implement),
        )
        return 0.0
    return hours * 60.0


# One run parses each NCP file once, not once per reader: inside `with graph_cache():` _load_graph keeps the parsed
# graph of a file (and run_cached keeps what a reader derived from a CGM file) until the block ends. The key is the
# file's resolved path, modification time and size; nothing is kept outside the block, and a thread or an
# asynchronous task has its own cache. The graphs are only read, never changed.
_RUN_CACHE: contextvars.ContextVar = contextvars.ContextVar("rao_run_cache", default=None)


@contextlib.contextmanager
def graph_cache():
    """Parse every NCP file once while the block runs (see above)."""
    token = _RUN_CACHE.set({})
    try:
        yield
    finally:
        _RUN_CACHE.reset(token)


def run_cached(kind: str, path: Path, build):
    """build() once per (kind, file) inside graph_cache(); build() every time outside it."""
    cache = _RUN_CACHE.get()
    if cache is None:
        return build()
    try:
        stat = Path(path).stat()
        key = (kind, str(Path(path).resolve()), stat.st_mtime_ns, stat.st_size)
    except OSError:
        return build()
    if key not in cache:
        cache[key] = build()
    return cache[key]


def _use_current_namespaces(g: Graph) -> Graph:
    """Rewrite the URIs in a legacy-form namespace that the file itself declares (nc, cim) to the current namespace, so
    every reader finds the terms and the enumeration values under NC / CIM (same triple order). Ids are not touched in
    substance: they are read after the "#" (ncp_id), so an xml:base does not matter. A file that declares none gives g
    itself."""
    declared = {str(namespace) for _, namespace in g.namespaces()}
    legacy = {old: new for old, new in LEGACY_NCP_NAMESPACES.items() if old in declared}
    if not legacy:
        return g

    def current(term):
        if isinstance(term, URIRef):
            text = str(term)
            for old, new in legacy.items():
                if text.startswith(old):
                    return URIRef(new + text[len(old):])
        return term

    result = Graph()
    for prefix, namespace in g.namespaces():
        result.bind(prefix, namespace, override=True)
    for triple in g:
        result.add(tuple(current(term) for term in triple))
    return result


def _load_graph(path: Path, cache: bool = True) -> Graph:
    """The parsed graph of an RDF/XML file; cache=False for the big CGM EQ files, which are not kept and whose
    namespaces are not rewritten (an NCP file in the legacy namespaces is, see _use_current_namespaces)."""
    def parse() -> Graph:
        g = Graph()
        g.parse(str(path), format="xml")
        return _use_current_namespaces(g) if cache else g

    return run_cached("graph", path, parse) if cache else parse()


def load_eq_graph(eq_path: Path):
    """Read one TSO's EQ file once, to pass as `graph` to build_current_limit_side_map and
    build_tap_changer_to_transformer_map (both read the same file). Not kept in the run cache (EQ files are big).
    The file is streamed (eq_stream.StreamedEq, answers the calls of those readers); a file the stream cannot
    answer exactly like rdflib is parsed with rdflib."""
    streamed = stream_eq(eq_path)
    return streamed if streamed is not None else _load_graph(eq_path, cache=False)


# ---------------------------------------------------------------------------
# Assessed elements + PATL from SHS
# ---------------------------------------------------------------------------

def _false_count(graph: Graph, class_uri, predicate: str) -> int:
    """Number of objects of class_uri whose boolean `predicate` is false."""
    return sum(
        1 for subject in set(graph.subjects(RDF.type, class_uri))
        if str(graph.value(subject, NC[predicate])).strip().lower() == "false"
    )


def warn_unread_assessed_element_switches(ae_graph: Graph, ae_path: Path, tso: str) -> dict[str, int]:
    """Warn, one line per kind with its count, about the assessed-element switches that are defined by the standard but
    not applied: isCombinableWithRemedialAction=false and the links AssessedElementWithRemedialAction (the effect of one
    remedial action cannot be removed from one assessed element's constraints in a contained way, ASSUMPTIONS OPEN-33).
    normalEnabled, inBaseCase, isCombinableWithContingency, the time-dependent enabled and the links
    AssessedElementWithContingency are applied by parse_assessed_elements. Returns {kind: count} of the lines."""
    counts: dict[str, int] = {}
    for predicate in ("isCombinableWithRemedialAction",):
        count = _false_count(ae_graph, NC.AssessedElement, f"AssessedElement.{predicate}")
        if count:
            counts[f"AssessedElement.{predicate}=false"] = count
            log.warning(
                "%s: %s has %d AssessedElement(s) with %s=false - NOT applied (ASSUMPTIONS OPEN-33): by the standard only "
                "the remedial actions with an included link would act on the element, but the model cannot take one "
                "remedial action out of one element's constraints yet; the effect of every remedial action on the element's loading is NOT removed (remedial actions are applied to it as usual)",
                tso, ae_path.name, count, predicate,
            )
    for link_class in ("AssessedElementWithRemedialAction",):
        links = set(ae_graph.subjects(RDF.type, NC[link_class]))
        if not links:
            continue
        excluded = sum(
            1 for link in links
            if str(ae_graph.value(link, NC[f"{link_class}.combinationConstraintKind"])).endswith("excluded")
        )
        disabled = _false_count(ae_graph, NC[link_class], f"{link_class}.normalEnabled")
        counts[link_class] = len(links)
        log.warning(
            "%s: %s has %d %s link(s) (%d excluded, %d with normalEnabled=false) - NOT applied (ASSUMPTIONS OPEN-33), the "
            "effect of the remedial actions on the assessed elements is NOT removed or restricted to the linked remedial actions", tso, ae_path.name,
            len(links), link_class, excluded, disabled,
        )
    return counts


def parse_assessed_elements(
    ae_path: Path,
    shs_path: Optional[Path],
    tso: str,
    timestamp_list: list[str],
    cgm_static_patl: dict[str, dict[int, float]],   # branch_mrid → {side: static PATL in A}
    cgm_branch_mrids: Optional[set[str]] = None,
    current_limit_side: Optional[dict[str, int]] = None,   # ol_mrid → resolved side (1 or 2)
    sis_path: Optional[Path] = None,   # AssessedElementSchedule / AssessedElementTimePoint (enabled)
) -> list[AssessedElement]:
    """Parse {TSO}_AE.xml and {TSO}_SHS.xml (shs_path None if absent) into AssessedElements.
    cgm_static_patl: {branch: {side: A}}, the last PATL fallback. current_limit_side:
    {OperationalLimit mRID: side} from build_current_limit_side_map (missing: side 1, logged).
    cgm_branch_mrids: valid branch mRIDs; an AE pointing at something else (e.g. a
    BusbarSection) gets branch_mrid None (logged). None: no validation.
    Raises ValueError for an exclusionReason outside _EXCLUSION_REASON_VALID (fatal).
    NCP switches (an absent switch means enabled; ae_switches.py): AssessedElement.normalEnabled=false or an
    AssessedElementTimePoint with enabled=false at the run time (sis_path) make the element EXCLUDED (not monitored; the
    time point overrides normalEnabled); inBaseCase=false = not monitored in the N state; the AssessedElementWithContingency
    links restrict it to (included) or take it out of (excluded) the linked contingencies. Every element switched off or
    restricted is logged. isCombinableWithContingency=false restricts the element to the contingencies with an included
    link (none: after no contingency); isCombinableWithRemedialAction=false and the links to remedial actions are only
    warned about (OPEN-33).
    """
    ae_graph = _load_graph(ae_path)
    warn_unread_assessed_element_switches(ae_graph, ae_path, tso)
    contingency_links = ae_switches.read_element_contingency_links(ae_graph, ae_path, tso)
    schedule_enabled = ae_switches.read_element_schedule_enabled(sis_path, timestamp_list, tso)

    # ------------------------------------------------------------------
    # Build PATL map from SHS: operational_limit_mrid → {timestamp → A}
    # ------------------------------------------------------------------
    patl_by_ol: dict[str, dict[str, float]] = {}   # op-limit mRID → {ts → A}

    if shs_path is not None:
        shs_graph = _load_graph(shs_path)

        # Map CurrentLimitSchedule mRID → OperationalLimit mRID
        cls_to_ol: dict[str, str] = {}
        for cls_uri in shs_graph.subjects(RDF.type, NC.CurrentLimitSchedule):
            ol_uri = shs_graph.value(cls_uri, NC["CurrentLimitSchedule.CurrentLimit"])
            if ol_uri is None:
                continue
            cls_to_ol[_frag(str(cls_uri))] = _frag(str(ol_uri))

        # Collect time points (of the run's date when the run has one, see run_time.py)
        limit_points = list(shs_graph.subjects(RDF.type, NC.CurrentLimitTimePoint))
        required_day = run_time.required_day(
            (run_time.iso_day(str(shs_graph.value(tp, NC["CurrentLimitTimePoint.atTime"]))) for tp in limit_points),
            f"{tso}: PATL", shs_path.name,
        )
        first_minute = run_time.first_minute_of(timestamp_list)
        for tp_uri in limit_points:
            cls_uri = shs_graph.value(tp_uri, NC["CurrentLimitTimePoint.CurrentLimitSchedule"])
            at_time = shs_graph.value(tp_uri, NC["CurrentLimitTimePoint.atTime"])
            value   = shs_graph.value(tp_uri, NC["CurrentLimitTimePoint.value"])

            if cls_uri is None or at_time is None or value is None:
                continue

            ts = _iso_to_hhmmz(str(at_time))
            if ts not in timestamp_list:
                continue
            if not run_time.day_matches(run_time.iso_day(str(at_time)), required_day, _minute_of_day(ts), first_minute):
                continue

            ol_mrid = cls_to_ol.get(_frag(str(cls_uri)))
            if ol_mrid is None:
                continue

            patl_by_ol.setdefault(ol_mrid, {})[ts] = float(str(value))

    # ------------------------------------------------------------------
    # Parse AssessedElement entries (including subtypes)
    # ------------------------------------------------------------------
    ae_types_to_parse = [
        NC.AssessedElement,
        NC.AssessedElementWithContingency,
        NC.AssessedElementWithRemedialAction,
    ]

    result: list[AssessedElement] = []
    base_case_conflicts = 0

    for ae_type_uri in ae_types_to_parse:
        # AssessedElementWithContingency / ...WithRemedialAction are association classes: an object
        # without its own branch is a link, not a malformed element (the links are counted by
        # warn_unread_assessed_element_switches)
        is_link_class = ae_type_uri != NC.AssessedElement
        for ae_uri in ae_graph.subjects(RDF.type, ae_type_uri):
            mrid = ae_graph.value(ae_uri, CIM["IdentifiedObject.mRID"])
            name = ae_graph.value(ae_uri, CIM["IdentifiedObject.name"])
            mrid = ncp_id(mrid) if mrid is not None else _frag(str(ae_uri))

            # Branch mRID: an AE is valid with at least one of ConductingEquipment,
            # OperationalLimit, DCTieCorridor, AssessedPowerTransferCorridor, Line.
            # ConductingEquipment and Line give the branch; the others (HVDC tie, corridor)
            # are out of scope and logged as such.
            eq_uri   = ae_graph.value(ae_uri, NC["AssessedElement.ConductingEquipment"])
            line_uri = ae_graph.value(ae_uri, NC["AssessedElement.Line"])
            branch_uri = eq_uri if eq_uri is not None else line_uri
            branch_mrid = _frag(str(branch_uri)) if branch_uri is not None else None

            if branch_mrid is None:
                dctie_uri = ae_graph.value(ae_uri, NC["AssessedElement.DCTieCorridor"])
                ptc_uri = ae_graph.value(ae_uri, NC["AssessedElement.AssessedPowerTransferCorridor"])
                ol_uri_check = ae_graph.value(ae_uri, NC["AssessedElement.OperationalLimit"])
                if dctie_uri is not None or ptc_uri is not None or ol_uri_check is not None:
                    log.info(
                        "%s: AE %s has no ConductingEquipment/Line (valid via "
                        "DCTieCorridor/PowerTransferCorridor/OperationalLimit "
                        "instead) — a different assessment kind, out of scope "
                        "for this branch-loading-only prototype — skipped",
                        tso, mrid,
                    )
                elif not is_link_class:
                    log.warning(
                        "%s: AE %s has none of ConductingEquipment, Line, "
                        "OperationalLimit, DCTieCorridor, or "
                        "AssessedPowerTransferCorridor — spec violation "
                        "(C:NC:AE:AssessedElement:associations requires at "
                        "least one) — skipped",
                        tso, mrid,
                    )
                continue

            if cgm_branch_mrids is not None and branch_mrid not in cgm_branch_mrids:
                log.info(
                    "%s: AE %s branch reference %s is not a Line or "
                    "2W-Transformer in the CGM — valid CIM data (e.g. a "
                    "BusbarSection is a legitimate AssessedElement.ConductingEquipment "
                    "target per spec) but out of scope for this "
                    "branch-loading-only prototype — skipped",
                    tso, mrid, branch_mrid,
                )
                continue

            # OperationalLimit mRID (PATL lookup in SHS); a BaseCaseCurrentLimit's normalValue
            # is a static PATL (A), used before cgm_static_patl
            ol_uri = ae_graph.value(ae_uri, NC["AssessedElement.OperationalLimit"])
            ol_mrid = _frag(str(ol_uri)) if ol_uri is not None else None

            # side of THIS AE's OperationalLimit (varies per AE); 1 when unresolvable (logged)
            branch_side = 1
            if current_limit_side is not None and ol_mrid is not None:
                resolved_side = current_limit_side.get(ol_mrid)
                if resolved_side is not None:
                    branch_side = resolved_side
                else:
                    log.debug(
                        "%s: AE %s OperationalLimit %s not found in "
                        "current_limit_side map — defaulting to side 1",
                        tso, mrid, ol_mrid,
                    )

            base_case_current_limit_a: Optional[float] = None
            has_base_case_limit = False
            if ol_uri is not None:
                ol_types = set(ae_graph.objects(ol_uri, RDF.type))
                if NC.BaseCaseCurrentLimit in ol_types or CIM.BaseCaseCurrentLimit in ol_types:
                    has_base_case_limit = True
                    normal_value_lit = ae_graph.value(ol_uri, NC["BaseCaseCurrentLimit.normalValue"])
                    if normal_value_lit is not None:
                        try:
                            base_case_current_limit_a = float(str(normal_value_lit))
                        except ValueError:
                            log.warning(
                                "%s: AE %s BaseCaseCurrentLimit.normalValue %r "
                                "is not a valid float — ignored",
                                tso, mrid, str(normal_value_lit),
                            )

            # exclusion classification
            secured_uri  = ae_graph.value(ae_uri, NC["AssessedElement.SecuredForRegion"])
            scanned_uri  = ae_graph.value(ae_uri, NC["AssessedElement.ScannedForRegion"])
            excl_uri     = ae_graph.value(ae_uri, NC["AssessedElement.exclusionReason"])

            secured_for  = _frag(str(secured_uri))  if secured_uri  is not None else None
            scanned_for  = _frag(str(scanned_uri))  if scanned_uri  is not None else None
            excl_reason  = _frag(str(excl_uri))     if excl_uri     is not None else None

            # an invalid exclusionReason is fatal: the parse for this TSO is cancelled
            if excl_reason is not None and excl_reason not in _EXCLUSION_REASON_VALID:
                log.fatal(
                    "%s: AE %s has invalid exclusionReason %r — cancelling the AE parse",
                    tso, mrid, excl_reason,
                )
                raise ValueError(
                    f"{tso}: AE {mrid} has invalid exclusionReason {excl_reason!r}"
                )

            if excl_reason is None and secured_for is not None:
                ae_type = AEType.SECURED
            elif excl_reason is not None and scanned_for is not None:
                ae_type = AEType.SCANNED
            elif excl_reason is not None and scanned_for is None:
                ae_type = AEType.EXCLUDED
            elif secured_for is None and scanned_for is None:
                # Neither a secured nor a scanned region code and no exclusion
                # reason: the element is excluded (not passed to the optimisation).
                excl_reason = _NO_REGION_CODE_REASON
                ae_type = AEType.EXCLUDED
                log.info(
                    "%s: AE %s excluded - %s", tso, mrid, _NO_REGION_CODE_REASON,
                )
            else:
                # ScannedForRegion only, no SecuredForRegion, no exclusionReason: stays
                # SECURED (the dataset relies on this)
                log.debug(
                    "%s: AE %s has ScannedForRegion but neither SecuredForRegion nor "
                    "exclusionReason — defaulting to SECURED (undocumented case)",
                    tso, mrid,
                )
                ae_type = AEType.SECURED

            # per-timestamp PATL (A -> kA), first available of: (1) SHS CurrentLimitTimePoint,
            # (2) BaseCaseCurrentLimit.normalValue, (3) cgm_static_patl, (4) 0.0 (warning)
            shs_values = patl_by_ol.get(ol_mrid, {}) if ol_mrid else {}
            static_a   = cgm_static_patl.get(branch_mrid, {}).get(branch_side) if branch_mrid else None
            patl_ka: dict[str, float] = {}

            for ts in timestamp_list:
                if ts in shs_values:
                    val_ka = shs_values[ts] / 1000.0
                    if static_a is not None:
                        static_ka = static_a / 1000.0
                        if static_ka > 0 and abs(val_ka - static_ka) / static_ka > 0.10:
                            log.warning(
                                "%s: AE %s ts=%s SHS PATL %.3f kA differs >10%% from CGM static %.3f kA",
                                tso, mrid, ts, val_ka, static_ka,
                            )
                    patl_ka[ts] = val_ka
                elif base_case_current_limit_a is not None:
                    patl_ka[ts] = base_case_current_limit_a / 1000.0
                elif static_a is not None:
                    # falling back to the CGM static PATL is logged: NCP is the priority
                    # PATL source
                    log.warning(
                        "%s: AE %s ts=%s no PATL in NCP (neither SHS nor "
                        "BaseCaseCurrentLimit) — using CGM static PATL "
                        "%.3f kA instead",
                        tso, mrid, ts, static_a / 1000.0,
                    )
                    patl_ka[ts] = static_a / 1000.0
                else:
                    log.warning(
                        "%s: AE %s ts=%s no SHS entry, no BaseCaseCurrentLimit, "
                        "and no CGM static PATL — defaulting to 0",
                        tso, mrid, ts,
                    )
                    patl_ka[ts] = 0.0

            # NCP switches
            element_key = _frag(str(ae_uri))
            notes: list[str] = []
            enabled = ae_switches.boolean(ae_graph.value(ae_uri, NC["AssessedElement.normalEnabled"]))
            if enabled is False:
                notes.append("AssessedElement.normalEnabled=false")
            at_run_time = schedule_enabled.get(element_key, schedule_enabled.get(str(mrid)))
            if at_run_time is not None:
                if at_run_time != (enabled is not False):
                    notes.append(f"AssessedElementTimePoint.enabled={str(at_run_time).lower()} at the run time")
                enabled = at_run_time
            if enabled is False:
                ae_type = AEType.EXCLUDED
                excl_reason = "disabled by the NCP"
            in_base_case = ae_switches.boolean(ae_graph.value(ae_uri, NC["AssessedElement.inBaseCase"])) is not False
            if not in_base_case and has_base_case_limit:
                # the standard requires inBaseCase=true for a BaseCaseCurrentLimit: scan it in the base case (conservative)
                base_case_conflicts += 1
                in_base_case = True
                notes.append("inBaseCase=false with a BaseCaseCurrentLimit (not allowed): scanned in the base case")
            elif not in_base_case:
                notes.append("inBaseCase=false (monitored after contingencies only)")
            included, excluded = contingency_links.get(element_key, contingency_links.get(str(mrid), (set(), set())))
            combinable = ae_switches.boolean(ae_graph.value(ae_uri, NC["AssessedElement.isCombinableWithContingency"])) is not False
            if not combinable:
                notes.append(f"isCombinableWithContingency=false (monitored only after the {len(included)} contingency(ies) "
                             f"with an included link {sorted(included)})")
            elif included:
                notes.append(f"monitored only after the {len(included)} linked contingency(ies) {sorted(included)}")
            if excluded:
                notes.append(f"not monitored after the {len(excluded)} excluded contingency(ies) {sorted(excluded)}")

            result.append(AssessedElement(
                mrid=str(mrid),
                name=str(name) if name is not None else str(mrid),
                tso=tso,
                ae_type=ae_type,
                branch_mrid=branch_mrid,
                secured_for_region=secured_for,
                scanned_for_region=scanned_for,
                exclusion_reason=excl_reason,
                patl_ka=patl_ka,
                branch_side=branch_side,
                in_base_case=in_base_case,
                included_contingencies=tuple(sorted(included)) if (included or not combinable) else None,
                excluded_contingencies=tuple(sorted(excluded)),
                switch_notes=tuple(notes),
            ))

    if base_case_conflicts:
        log.warning(
            "%s: %s has %d AssessedElement(s) with a BaseCaseCurrentLimit and inBaseCase=false - the standard requires "
            "inBaseCase=true; the element is scanned in the base case", tso, ae_path.name, base_case_conflicts)
    switched = [ae for ae in result if ae.switch_notes]
    for ae in switched:
        log.info("%s: assessed element %s (branch %s): %s", tso, ae.mrid, ae.branch_mrid, "; ".join(ae.switch_notes))
    if switched:
        log.warning(
            "%s: %d assessed element(s) switched off or restricted by NCP switches: %s",
            tso, len(switched), ", ".join(sorted(ae.mrid for ae in switched)),
        )
    log.info(
        "%s: %d AEs parsed (secured=%d, scanned=%d, excluded=%d)",
        tso, len(result),
        sum(1 for a in result if a.ae_type == AEType.SECURED),
        sum(1 for a in result if a.ae_type == AEType.SCANNED),
        sum(1 for a in result if a.ae_type == AEType.EXCLUDED),
    )
    return result


# ---------------------------------------------------------------------------
# Contingencies
# ---------------------------------------------------------------------------

def parse_contingencies(
    co_path: Path,
    tso: str,
    timestamp_list: list[str],
    equipment_mrid_to_type: dict[str, str],
) -> list[Contingency]:
    """Parse {TSO}_CO.xml. must_study = nc:Contingency.normalMustStudy for every timestamp
    (one flag in the file). Ordinary, Exceptional and OutOfRange contingencies are read.
    equipment_mrid_to_type: {mRID: CoMo type} from build_equipment_mrid_to_type; a contingency
    with an equipment mRID not in it is excluded (logged "not found in CGM").
    """
    g = _load_graph(co_path)

    # Build equipment map: contingency mRID → list of equipment mRIDs
    co_equipment: dict[str, list[str]] = {}
    for ce_uri in g.subjects(RDF.type, CIM.ContingencyEquipment):
        co_uri  = g.value(ce_uri, CIM["ContingencyElement.Contingency"])
        eq_uri  = g.value(ce_uri, CIM["ContingencyEquipment.Equipment"])
        if co_uri is None or eq_uri is None:
            continue
        co_mrid = _frag(str(co_uri))
        eq_mrid = _frag(str(eq_uri))
        co_equipment.setdefault(co_mrid, []).append(eq_mrid)

    result: list[Contingency] = []
    without_must_study = 0

    for co_type, ct_enum in [
        (NC.OrdinaryContingency,    ContingencyType.ORDINARY),
        (NC.ExceptionalContingency, ContingencyType.EXCEPTIONAL),
        (NC.OutOfRangeContingency,  ContingencyType.OUT_OF_RANGE),
    ]:
        for co_uri in g.subjects(RDF.type, co_type):
            mrid = g.value(co_uri, CIM["IdentifiedObject.mRID"])
            name = g.value(co_uri, CIM["IdentifiedObject.name"])
            mrid = ncp_id(mrid) if mrid is not None else _frag(str(co_uri))

            must_study_lit = g.value(co_uri, NC["Contingency.normalMustStudy"])
            must_study_val = str(must_study_lit).lower() == "true" if must_study_lit else False
            if must_study_lit is None:
                without_must_study += 1

            eq_mrids = co_equipment.get(str(mrid), co_equipment.get(_frag(str(co_uri)), []))

            # a KEY in the mapping is one of the 8 covered CoMo types; its VALUE is that type
            valid_eq: list[str] = []
            valid_types: list[str] = []
            for eq in eq_mrids:
                eq_type = equipment_mrid_to_type.get(eq)
                if eq_type is not None:
                    valid_eq.append(eq)
                    valid_types.append(eq_type)
                else:
                    log.warning(
                        "%s: contingency %s equipment %s not found in CGM — skipped",
                        tso, mrid, eq,
                    )

            if not valid_eq and eq_mrids:
                log.warning(
                    "%s: contingency %s has no valid equipment after CGM check — excluded",
                    tso, mrid,
                )
                continue

            result.append(Contingency(
                mrid=str(mrid),
                name=str(name) if name is not None else str(mrid),
                tso=tso,
                contingency_type=ct_enum,
                must_study={ts: must_study_val for ts in timestamp_list},
                equipment_mrids=valid_eq,
                equipment_types=valid_types,
            ))

    if without_must_study:
        log.warning(
            "%s: %s has %d contingenc(ies) without Contingency.normalMustStudy - read as not must-study, "
            "so they are not analysed", tso, co_path.name, without_must_study,
        )
    log.info(
        "%s: %d contingencies parsed (%d must-study)",
        tso, len(result),
        sum(1 for c in result if any(c.must_study.values())),
    )
    return result


def write_co_flat_list_csv(co_flat_list: list[Contingency], output_path: Path) -> None:
    """co_flat_list.csv, one row per (contingency, timestamp): [contingency_mrid, name,
    contingency_type, must_study, equipment_mrids, equipment_type, timestamp].
    equipment_mrids/equipment_type are ';'-joined when a contingency trips several elements."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding=TEXT_ENCODING) as f:
        writer = csv.writer(f)
        writer.writerow([
            "contingency_mrid", "name", "contingency_type", "must_study",
            "equipment_mrids", "equipment_type", "timestamp",
        ])
        for co in co_flat_list:
            for timestamp, must_study_val in sorted(co.must_study.items()):
                writer.writerow([
                    co.mrid, co.name, co.contingency_type.value, must_study_val,
                    ";".join(co.equipment_mrids), ";".join(co.equipment_types),
                    timestamp,
                ])

    log.info("Written co_flat_list CSV: %s", output_path)


# ---------------------------------------------------------------------------
# TapChanger -> parent PowerTransformer resolution
# ---------------------------------------------------------------------------

def _eq_terms(local_name: str):
    """Return both candidate namespace terms for a Grid/EQ-file class or
    predicate local name, mirroring eic_resolver._cim_terms()."""
    return [CIM[local_name], CIM100[local_name], CIM16[local_name]]


def _eq_value(g: Graph, subject, local_name: str):
    """g.value() that tries both candidate EQ-file namespaces."""
    for term in _eq_terms(local_name):
        value = g.value(subject, term)
        if value is not None:
            return value
    return None


def _eq_subjects_of_type(g: Graph, local_name: str):
    """g.subjects(RDF.type, ...) that tries both candidate EQ-file namespaces."""
    subjects = set()
    for term in _eq_terms(local_name):
        for subject in g.subjects(RDF.type, term):
            subjects.add(subject)
    return subjects


# CIM classes of a phase shifting transformer. RatioTapChanger is excluded: a plain
# voltage-regulating transformer with no phase-angle sensitivity (pypowsybl rejects it).
_TAP_CHANGER_CLASSES_AND_TE_PREDICATES = [
    ("PhaseTapChangerSymmetrical", ["PhaseTapChanger.TransformerEnd", "TapChanger.TransformerEnd"]),
    ("PhaseTapChangerAsymmetrical", ["PhaseTapChanger.TransformerEnd", "TapChanger.TransformerEnd"]),
    ("PhaseTapChangerTabular", ["PhaseTapChanger.TransformerEnd", "TapChanger.TransformerEnd"]),
    ("PhaseTapChangerLinear", ["PhaseTapChanger.TransformerEnd", "TapChanger.TransformerEnd"]),
]


def build_current_limit_side_map(eq_path: Path, graph: Optional[Graph] = None) -> dict[str, int]:
    """{CurrentLimit mRID: side (1 or 2)} from one TSO's EQ file, via
        CurrentLimit.OperationalLimitSet -> OperationalLimitSet.Terminal ->
        ACDCTerminal.sequenceNumber (1 or 2).
    Assumes sequenceNumber 1 == pypowsybl side ONE (i1), 2 == TWO (i2). A CurrentLimit without
    this chain is omitted; parse_assessed_elements applies side 1 (logged).
    graph: the already parsed EQ file (load_eq_graph); None parses eq_path.
    """
    g = graph if graph is not None else load_eq_graph(eq_path)
    result: dict[str, int] = {}

    for cl_uri in _eq_subjects_of_type(g, "CurrentLimit"):
        cl_mrid_lit = _eq_value(g, cl_uri, "IdentifiedObject.mRID")
        cl_mrid = str(cl_mrid_lit) if cl_mrid_lit is not None else _frag(str(cl_uri))

        ols_uri = _eq_value(g, cl_uri, "OperationalLimit.OperationalLimitSet")
        if ols_uri is None:
            continue

        term_uri = _eq_value(g, ols_uri, "OperationalLimitSet.Terminal")
        if term_uri is None:
            # OperationalLimitSet.Equipment (no side) is valid CIM: nothing to resolve
            continue

        seq_lit = _eq_value(g, term_uri, "ACDCTerminal.sequenceNumber")
        if seq_lit is None:
            continue

        try:
            side = int(str(seq_lit))
        except ValueError:
            log.warning(
                "build_current_limit_side_map: CurrentLimit %s has non-integer "
                "sequenceNumber %r — skipped",
                cl_mrid, str(seq_lit),
            )
            continue

        if side not in (1, 2):
            log.warning(
                "build_current_limit_side_map: CurrentLimit %s has "
                "sequenceNumber %d outside {1,2} — skipped",
                cl_mrid, side,
            )
            continue

        result[cl_mrid] = side

    log.info(
        "build_current_limit_side_map: resolved side for %d CurrentLimit "
        "objects from %s",
        len(result), eq_path.name,
    )
    return result


def build_tap_changer_to_transformer_map(eq_path: Path, graph: Optional[Graph] = None) -> dict[str, str]:
    """{TapChanger mRID: parent PowerTransformer mRID} from one TSO's EQ file, via
    TapChanger.TransformerEnd -> PowerTransformerEnd.PowerTransformer -> mRID.
    Needed because pypowsybl indexes tap changers by transformer while
    TapPositionAction.TapChanger references the TapChanger. Lookups try the
    cim.ucaiug.io, CIM100 and CGMES 2.4.15 namespaces. graph: as in build_current_limit_side_map.
    """
    g = graph if graph is not None else load_eq_graph(eq_path)
    tc_to_transformer: dict[str, str] = {}

    for class_name, te_predicate_names in _TAP_CHANGER_CLASSES_AND_TE_PREDICATES:
        for tc_uri in _eq_subjects_of_type(g, class_name):
            tc_mrid_lit = _eq_value(g, tc_uri, "IdentifiedObject.mRID")
            tc_mrid = str(tc_mrid_lit) if tc_mrid_lit is not None else _frag(str(tc_uri))

            te_uri = None
            for pred_name in te_predicate_names:
                te_uri = _eq_value(g, tc_uri, pred_name)
                if te_uri is not None:
                    break
            if te_uri is None:
                log.warning(
                    "TapChanger %s (%s) has no TransformerEnd reference — "
                    "cannot resolve parent transformer",
                    tc_mrid, class_name,
                )
                continue

            transformer_uri = _eq_value(g, te_uri, "PowerTransformerEnd.PowerTransformer")
            if transformer_uri is None:
                log.warning(
                    "TapChanger %s TransformerEnd %s has no PowerTransformerEnd.PowerTransformer "
                    "reference — cannot resolve parent transformer",
                    tc_mrid, _frag(str(te_uri)),
                )
                continue

            transformer_mrid_lit = _eq_value(g, transformer_uri, "IdentifiedObject.mRID")
            transformer_mrid = (
                str(transformer_mrid_lit) if transformer_mrid_lit is not None
                else _frag(str(transformer_uri))
            )

            if tc_mrid in tc_to_transformer and tc_to_transformer[tc_mrid] != transformer_mrid:
                log.warning(
                    "TapChanger %s resolves to multiple distinct transformers "
                    "(%s and %s) — keeping first match",
                    tc_mrid, tc_to_transformer[tc_mrid], transformer_mrid,
                )
                continue

            tc_to_transformer[tc_mrid] = transformer_mrid

    log.info(
        "build_tap_changer_to_transformer_map: resolved %d TapChanger -> "
        "PowerTransformer mappings from %s",
        len(tc_to_transformer), eq_path.name,
    )
    return tc_to_transformer


# ---------------------------------------------------------------------------
# PST remedial actions
# ---------------------------------------------------------------------------

def parse_psts(
    ra_path: Path,
    tso: str,
    cgm_tap_changers: dict[str, dict],  # PowerTransformer mrid -> {low_tap, high_tap, tap}
    tap_changer_to_transformer: dict[str, str],  # TapChanger mrid -> PowerTransformer mrid
    ae_path: Optional[Path] = None,  # {TSO}_AE.xml, for linked_xne_mrids resolution
) -> list[PST]:
    """Parse PST RAs from {TSO}_RA.xml: a GridStateAlterationRemedialAction (GSARA) with at
    least one TapPositionAction. Several TapPositionActions under one GSARA share
    group_id = GSARA mRID; a single one has group_id None. PST.mrid is the parent
    PowerTransformer mRID (the pypowsybl ID), via tap_changer_to_transformer.
    Curative fields: reaction_time_minutes (GSARA timeToImplement), time_per_stage_minutes
    (per TapPositionAction timePerStage), linked_contingency_mrids, linked_xne_mrids
    (from ae_path; [] when ae_path is None). Durations are ISO 8601 ("PT20M0S").
    """
    g = _load_graph(ra_path)

    # GSARA mRID -> TapChanger mRIDs, and (GSARA, TapChanger) -> its own TapPositionAction URI:
    # keyed by the pair because two GSARAs can reference the same TapChanger
    gsara_to_tc: dict[str, list[str]] = {}
    tc_to_tpa_uri: dict[tuple[str, str], object] = {}
    for tpa_uri in g.subjects(RDF.type, NC.TapPositionAction):
        gsara_uri = g.value(tpa_uri, NC["GridStateAlteration.GridStateAlterationRemedialAction"])
        tc_uri    = g.value(tpa_uri, NC["TapPositionAction.TapChanger"])
        if gsara_uri is None or tc_uri is None:
            continue
        gsara_mrid = _frag(str(gsara_uri))
        tc_mrid    = _frag(str(tc_uri))
        gsara_to_tc.setdefault(gsara_mrid, []).append(tc_mrid)
        tc_to_tpa_uri[(gsara_mrid, tc_mrid)] = tpa_uri

    # GSARA mRID -> linked / excluded contingency mRIDs (ContingencyWithRemedialAction)
    gsara_to_linked_contingencies, gsara_to_excluded_contingencies = read_contingency_links(g, ra_path, tso)

    # GSARA mRID -> linked XNE (AssessedElement) mRIDs, from {TSO}_AE.xml (only with ae_path)
    gsara_to_linked_xnes: dict[str, list[str]] = {}
    if ae_path is not None and ae_path.exists():
        ae_graph = _load_graph(ae_path)
        for aewra_uri in ae_graph.subjects(RDF.type, NC.AssessedElementWithRemedialAction):
            ra_uri = ae_graph.value(aewra_uri, NC["AssessedElementWithRemedialAction.RemedialAction"])
            ae_uri = ae_graph.value(aewra_uri, NC["AssessedElementWithRemedialAction.AssessedElement"])
            if ra_uri is None or ae_uri is None:
                continue
            gsara_to_linked_xnes.setdefault(_frag(str(ra_uri)), []).append(_frag(str(ae_uri)))

    result: list[PST] = []

    for gsara_uri in g.subjects(RDF.type, NC.GridStateAlterationRemedialAction):
        mrid = g.value(gsara_uri, CIM["IdentifiedObject.mRID"])
        mrid = ncp_id(mrid) if mrid is not None else _frag(str(gsara_uri))
        gsara_mrid = str(mrid)

        tc_mrids = gsara_to_tc.get(gsara_mrid, gsara_to_tc.get(_frag(str(gsara_uri)), []))
        if not tc_mrids:
            continue  # Not a PST RA

        # the name is the GSARA's, so the PSTs of one group share the same display name
        name = g.value(gsara_uri, CIM["IdentifiedObject.name"])
        available_lit = g.value(gsara_uri, NC["RemedialAction.normalAvailable"])
        available = str(available_lit).lower() == "true" if available_lit else False

        kind_uri = g.value(gsara_uri, NC["RemedialAction.kind"])
        kind_str = _frag(str(kind_uri)) if kind_uri is not None else ""
        kind = RAKind.CURATIVE if "curative" in kind_str.lower() else RAKind.PREVENTIVE

        penalty_lit = g.value(gsara_uri, NC["RemedialAction.penaltyFactor"])
        if penalty_lit is None:
            log.warning(
                "%s: PST remedial action %s has no RemedialAction.penaltyFactor - defaulted to 1.0",
                tso, gsara_mrid,
            )
            penalty = 1.0
        else:
            penalty = float(str(penalty_lit))

        # reaction_time_minutes: GSARA level, the same for each of its TapPositionActions
        time_to_implement_lit = g.value(gsara_uri, NC["RemedialAction.timeToImplement"])
        reaction_time_minutes = (
            _parse_iso_duration_hours(str(time_to_implement_lit)) * 60.0
            if time_to_implement_lit is not None else None
        )

        linked_contingency_mrids = list(gsara_to_linked_contingencies.get(gsara_mrid, []))
        linked_xne_mrids = list(gsara_to_linked_xnes.get(gsara_mrid, []))

        # Group: if GSARA has multiple TapPositionActions → group_id = GSARA mRID
        group_id = gsara_mrid if len(tc_mrids) > 1 else None

        for tc_mrid in tc_mrids:
            transformer_mrid = tap_changer_to_transformer.get(tc_mrid)
            if transformer_mrid is None:
                log.info(
                    "%s: PST %s TapChanger %s has no resolved parent Phase "
                    "Shifting Transformer — likely a RatioTapChanger (plain "
                    "LTC, not a PST) referenced by this TapPositionAction, "
                    "out of PST PRA scope — skipped",
                    tso, mrid, tc_mrid,
                )
                continue

            tc_data = cgm_tap_changers.get(transformer_mrid)
            if tc_data is None:
                log.warning(
                    "%s: PST %s TapChanger %s (transformer %s) not found in CGM — skipped",
                    tso, mrid, tc_mrid, transformer_mrid,
                )
                continue

            low_tap  = tc_data.get("low_tap") or tc_data.get("low_step") or 0
            high_tap = tc_data.get("high_tap") or tc_data.get("high_step") or 0
            tap      = tc_data.get("tap") or tc_data.get("r0") or 0

            # operating bounds default to the hardware limits
            operating_tap_min = int(low_tap)
            operating_tap_max = int(high_tap)

            # Clamp tap to hardware bounds
            tap = max(operating_tap_min, min(operating_tap_max, int(tap)))

            # time_per_stage_minutes: from THIS GSARA's own TapPositionAction for tc_mrid
            tpa_uri_for_tc = tc_to_tpa_uri.get((gsara_mrid, tc_mrid))
            time_per_stage_lit = (
                g.value(tpa_uri_for_tc, NC["GridStateAlteration.timePerStage"])
                if tpa_uri_for_tc is not None else None
            )
            # a zero or unreadable timePerStage stays unset (as a missing one), with a warning: it is a
            # divisor of the curative tap count
            time_per_stage_minutes = None
            if time_per_stage_lit is not None:
                time_per_stage_hours = _parse_iso_duration_hours_or_none(str(time_per_stage_lit))
                if time_per_stage_hours is None or time_per_stage_hours <= 0.0:
                    log.warning(
                        "%s: PST %s (remedial action %s) has a zero or unreadable "
                        "GridStateAlteration.timePerStage %r - left unset, as a missing one",
                        tso, transformer_mrid, gsara_mrid, str(time_per_stage_lit),
                    )
                else:
                    time_per_stage_minutes = time_per_stage_hours * 60.0

            if kind == RAKind.CURATIVE and (reaction_time_minutes is None or time_per_stage_minutes is None):
                log.warning(
                    "%s: curative PST %s missing reaction_time_minutes and/or "
                    "time_per_stage_minutes (RemedialAction.timeToImplement / "
                    "GridStateAlteration.timePerStage not set in the source data) — "
                    "pst_cra_availability cannot be computed for this PST until this "
                    "dataset cut provides that timing data. This is a genuine dataset "
                    "boundary condition, not a parsing gap (the CIM predicates "
                    "themselves are confirmed present and correctly parsed elsewhere "
                    "in this same dataset cut, just not set for this RA).",
                    tso, transformer_mrid,
                )

            result.append(PST(
                mrid=transformer_mrid,
                name=str(name) if name is not None else gsara_mrid,
                tso=tso,
                available=available,
                kind=kind,
                penalty_factor=penalty,
                low_step=int(low_tap),
                high_step=int(high_tap),
                r0=int(tap),
                operating_tap_min=operating_tap_min,
                operating_tap_max=operating_tap_max,
                group_id=group_id,
                reaction_time_minutes=reaction_time_minutes,
                time_per_stage_minutes=time_per_stage_minutes,
                linked_contingency_mrids=linked_contingency_mrids,
                excluded_contingency_mrids=list(gsara_to_excluded_contingencies.get(gsara_mrid, [])),
                linked_xne_mrids=linked_xne_mrids,
                ra_mrid=gsara_mrid,
            ))

    log.info("%s: %d PSTs parsed", tso, len(result))
    return result


def check_unique_pst_rows(pst_list: list[PST]) -> None:
    """Raise ValueError when two available PST rows of the same kind (preventive or curative) act on one transformer;
    one preventive and one curative row on the same PST are allowed. The message names the PST and both remedial actions."""
    seen: dict[tuple[str, RAKind], PST] = {}
    for pst in pst_list:
        if not pst.available:
            continue
        key = (str(pst.mrid), pst.kind)
        first = seen.setdefault(key, pst)
        if first is not pst:
            raise ValueError(
                f"PST {pst.mrid} (transformer) is acted on by two available {pst.kind.value} remedial actions: "
                f"{first.tso} {first.ra_mrid} ({first.name}) and {pst.tso} {pst.ra_mrid} ({pst.name}). A PST may have one preventive "
                f"and one curative remedial action, not two of the same kind - correct the RA file."
            )


def build_pst_tap_bounds_curative(
    ra_path: Path,
    tap_changer_to_transformer: dict[str, str],
) -> dict[str, tuple[int, int]]:
    """Curative tap bounds {transformer_mrid: (min, max)} from the nc:StaticPropertyRange
    entries attached to a PST's TapPositionAction (RangeConstraint.GridStateAlteration).
    Only PSTs with both bounds and lower <= upper are returned (lower > upper: warning); the
    caller falls back to the preventive bounds.
    RelativeDirectionKind containing "up" is the upper bound, "down" the lower; "none" is
    skipped (warning). This mapping is not verified on a real PST range (none in the data).
    """
    g = _load_graph(ra_path)

    # TapChanger mrid -> its TapPositionAction URI (same traversal as parse_psts, standalone)
    tc_mrid_to_tpa_uri: dict[str, object] = {}
    for tpa_uri in g.subjects(RDF.type, NC.TapPositionAction):
        tc_uri = g.value(tpa_uri, NC["TapPositionAction.TapChanger"])
        if tc_uri is not None:
            tc_mrid_to_tpa_uri[_frag(str(tc_uri))] = tpa_uri

    # TapPositionAction URI -> [(direction, value)] of the StaticPropertyRanges pointing at it
    tpa_uri_to_ranges: dict[object, list[tuple[str, float]]] = {}
    for spr_uri in g.subjects(RDF.type, NC.StaticPropertyRange):
        gsa_uri = g.value(spr_uri, NC["RangeConstraint.GridStateAlteration"])
        value_lit = g.value(spr_uri, NC["RangeConstraint.normalValue"])
        if gsa_uri is None or value_lit is None:
            continue
        direction_uri = g.value(spr_uri, NC["RangeConstraint.direction"])
        direction_str = _frag(str(direction_uri)) if direction_uri is not None else "none"
        try:
            value = float(str(value_lit))
        except ValueError:
            continue
        tpa_uri_to_ranges.setdefault(gsa_uri, []).append((direction_str, value))

    curative_tap_bounds: dict[str, tuple[int, int]] = {}

    for tc_mrid, transformer_mrid in tap_changer_to_transformer.items():
        tpa_uri = tc_mrid_to_tpa_uri.get(tc_mrid)
        if tpa_uri is None:
            continue
        ranges = tpa_uri_to_ranges.get(tpa_uri)
        if not ranges:
            continue  # no curative RangeConstraint for this PST — caller falls back

        upper = None
        lower = None
        for direction_str, value in ranges:
            direction_lower = direction_str.lower()
            if "up" in direction_lower:
                upper = value
            elif "down" in direction_lower:
                lower = value
            else:
                log.warning(
                    "build_pst_tap_bounds_curative: PST %s has a StaticPropertyRange "
                    "with unrecognized/ambiguous direction %r (value=%s) — cannot "
                    "determine whether this is the upper or lower curative bound, "
                    "skipped (see this function's own docstring — this mapping is "
                    "unverified against real curative PST data on this dataset cut)",
                    transformer_mrid, direction_str, value,
                )

        if upper is not None and lower is not None:
            # lower above upper: no tap position fits (an infeasible curative row); lower = upper fixes the PST
            if int(lower) > int(upper):
                log.warning(
                    "build_pst_tap_bounds_curative: PST %s has its curative lower bound %d above its "
                    "upper bound %d — curative range ignored; caller should fall back to this PST's "
                    "operating tap range",
                    transformer_mrid, int(lower), int(upper),
                )
                continue
            curative_tap_bounds[transformer_mrid] = (int(lower), int(upper))
        else:
            log.warning(
                "build_pst_tap_bounds_curative: PST %s has %d StaticPropertyRange "
                "entries but could not resolve both an upper AND a lower curative "
                "bound (upper=%s, lower=%s) — no curative bounds entry added; "
                "caller should fall back to this PST's preventive bounds",
                transformer_mrid, len(ranges), upper, lower,
            )

    log.info(
        "build_pst_tap_bounds_curative: resolved curative tap bounds for %d PST(s)",
        len(curative_tap_bounds),
    )
    return curative_tap_bounds


# ---------------------------------------------------------------------------
# RD units + power bids
# ---------------------------------------------------------------------------

def _is_remedial_action_class(class_uri) -> bool:
    """True for an nc: class of a remedial action itself (GridStateAlterationRemedialAction,
    CountertradeRemedialAction, RedispatchRemedialAction, ...), not an association such as
    ContingencyWithRemedialAction."""
    text = str(class_uri)
    if not text.startswith(str(NC)):
        return False
    local_name = text[len(str(NC)):]
    return local_name.endswith("RemedialAction") and "With" not in local_name


def warn_unsupported_sis_content(sis_path: Optional[Path], tso: str) -> None:
    """Warn once per SIS file about standard content that is not read, so that it is never silently lost:
    PowerRemedialActionSchedule / PowerRemedialActionTimePoint (the per-RA volume cap
    maxRegulatingUp/Down), PowerBidDependency (PeerTemporalDependency), GenericAvailableSchedule
    (availability of remedial actions) and AssessedElementTimePoint with enabled=false are ignored."""
    if sis_path is None or not sis_path.exists():
        return
    graph = _load_graph(sis_path)
    availability = len(set(graph.subjects(RDF.type, NC.GenericAvailableSchedule)))
    if availability:
        log.warning(
            "%s: %s has %d GenericAvailableSchedule element(s) (availability of remedial actions over time) - not read, "
            "a remedial action scheduled unavailable is still used", tso, sis_path.name, availability,
        )
    disabled = sum(
        1 for point in set(graph.subjects(RDF.type, NC.AssessedElementTimePoint))
        if str(graph.value(point, NC["AssessedElementTimePoint.enabled"])).lower() == "false"
    )
    if disabled:
        log.warning(
            "%s: %s has %d AssessedElementTimePoint element(s) with enabled=false - not read, the assessed element "
            "stays monitored", tso, sis_path.name, disabled,
        )
    disabled_alterations = _false_count(graph, NC.GridStateAlterationTimePoint, "GridStateAlterationTimePoint.enabled")
    if disabled_alterations:
        log.warning(
            "%s: %s has %d GridStateAlterationTimePoint element(s) with enabled=false - not read, the grid state "
            "alteration stays in force", tso, sis_path.name, disabled_alterations,
        )
    disabled_schedules = _false_count(graph, NC.GridStateAlterationSchedule, "GridStateAlterationSchedule.enabled")
    if disabled_schedules:
        log.warning(
            "%s: %s has %d GridStateAlterationSchedule element(s) with enabled=false - not read, the grid state "
            "alteration stays in force", tso, sis_path.name, disabled_schedules,
        )
    dependencies = len(set(graph.subjects(RDF.type, NC.PowerBidDependency)))
    if dependencies:
        log.warning(
            "%s: %s has %d PowerBidDependency element(s) (PeerTemporalDependency constraints between bids) - not read, "
            "the bids are optimised independently", tso, sis_path.name, dependencies,
        )
    schedules = len(set(graph.subjects(RDF.type, NC.PowerRemedialActionSchedule)))
    points = len(set(graph.subjects(RDF.type, NC.PowerRemedialActionTimePoint)))
    if schedules or points:
        log.warning(
            "%s: %s has %d PowerRemedialActionSchedule and %d PowerRemedialActionTimePoint element(s) "
            "(per-RA maxRegulatingUp/Down volume caps) - not read, the caps are ignored",
            tso, sis_path.name, schedules, points,
        )


def warn_unmatched_ra_references(ra_path: Path, sis_path: Optional[Path], tso: str) -> list[dict]:
    """Warn about every nc:ContingencyWithRemedialAction (ra_path) and nc:PowerBidSchedule (sis_path)
    whose RemedialAction / PowerRemedialAction refers to an mRID that is the mRID of no remedial action
    of ra_path (e.g. an RA whose mRID differs from its rdf:ID). Diagnostic
    only: what the parsers match is unchanged. Returns one dict per warning: object_class, object_mrid,
    ra_mrid."""
    ra_graph = _load_graph(ra_path)
    ra_mrids: set[str] = set()
    for subject, class_uri in ra_graph.subject_objects(RDF.type):
        if _is_remedial_action_class(class_uri):
            mrid_lit = ra_graph.value(subject, CIM["IdentifiedObject.mRID"])
            ra_mrids.add(ncp_id(mrid_lit) if mrid_lit is not None else _frag(str(subject)))

    references = [
        (ra_graph, NC.ContingencyWithRemedialAction, "ContingencyWithRemedialAction.RemedialAction", ra_path.name),
    ]
    if sis_path is not None:
        references.append(
            (_load_graph(sis_path), NC.PowerBidSchedule, "PowerBidSchedule.PowerRemedialAction", sis_path.name),
        )

    unmatched: list[dict] = []
    for graph, class_uri, predicate, file_name in references:
        for subject in sorted(graph.subjects(RDF.type, class_uri), key=str):
            ra_uri = graph.value(subject, NC[predicate])
            if ra_uri is None or _frag(str(ra_uri)) in ra_mrids:
                continue
            mrid_lit = graph.value(subject, CIM["IdentifiedObject.mRID"])
            entry = {
                "object_class": _frag(str(class_uri)),
                "object_mrid": ncp_id(mrid_lit) if mrid_lit is not None else _frag(str(subject)),
                "ra_mrid": _frag(str(ra_uri)),
            }
            unmatched.append(entry)
            log.warning(
                "%s: nc:%s %s (%s) refers to remedial action %s, which is the mRID of no remedial action "
                "of %s - not matched by the PST and RD links or by any bid (they match the RA's mRID; "
                "HVDC and countertrade contingency links also accept its rdf:ID)",
                tso, entry["object_class"], entry["object_mrid"], file_name, entry["ra_mrid"], ra_path.name,
            )
    return unmatched


def warn_unsupported_power_remedial_actions(
    ra_graph: Graph, sis_graph: Optional[Graph], tso: str, include_countertrade: bool = True,
    parsed_mrids: Optional[set[str]] = None,
) -> list[dict]:
    """Warn about nc:RedispatchRemedialAction and nc:CountertradeRemedialAction that are not used: a
    redispatch RA needs a generating unit in its shift keys, a countertrade RA is parsed by ct_units;
    they and their PowerBidSchedules are ignored. One warning per RA with its bid count. Returns one dict per
    RA: class_name, mrid, name, bid_schedule_count, time_point_count. include_countertrade False
    (a run with countertrade in ra_types, which parses those RAs itself) leaves the CT RAs out;
    parsed_mrids: RAs already parsed as RD units are left out."""
    schedules_by_ra: dict[str, list] = {}
    time_points_by_schedule: dict[str, int] = {}
    if sis_graph is not None:
        for pbs_uri in sis_graph.subjects(RDF.type, NC.PowerBidSchedule):
            ra_uri = sis_graph.value(pbs_uri, NC["PowerBidSchedule.PowerRemedialAction"])
            if ra_uri is not None:
                schedules_by_ra.setdefault(_frag(str(ra_uri)), []).append(_frag(str(pbs_uri)))
        for tp_uri in sis_graph.subjects(RDF.type, NC.PowerBidScheduleTimePoint):
            pbs_uri = sis_graph.value(tp_uri, NC["PowerBidScheduleTimePoint.PowerBidSchedule"])
            if pbs_uri is not None:
                key = _frag(str(pbs_uri))
                time_points_by_schedule[key] = time_points_by_schedule.get(key, 0) + 1

    ignored: list[dict] = []
    ignored_classes = ("RedispatchRemedialAction", "CountertradeRemedialAction") if include_countertrade else (
        "RedispatchRemedialAction",
    )
    for class_name in ignored_classes:
        for ra_uri in sorted(ra_graph.subjects(RDF.type, NC[class_name])):
            mrid_lit = ra_graph.value(ra_uri, CIM["IdentifiedObject.mRID"])
            mrid = ncp_id(mrid_lit) if mrid_lit is not None else _frag(str(ra_uri))
            name_lit = ra_graph.value(ra_uri, CIM["IdentifiedObject.name"])
            if parsed_mrids and mrid in parsed_mrids:
                continue
            schedules = sorted(set(schedules_by_ra.get(mrid, [])))
            entry = {
                "class_name": class_name, "mrid": mrid,
                "name": str(name_lit) if name_lit is not None else "",
                "bid_schedule_count": len(schedules),
                "time_point_count": sum(time_points_by_schedule.get(sch, 0) for sch in schedules),
            }
            ignored.append(entry)
            reason = (
                "countertrade is not in ra_types (countertrade RAs are read only when it is listed)"
                if class_name == "CountertradeRemedialAction" else
                "it has no generating unit in a shift key (bid route or strategy route)"
            )
            log.warning(
                "%s: nc:%s %s (%s) is not used and is IGNORED — %s; its %d PowerBidSchedule(s) with %d time point(s) are ignored too",
                tso, class_name, mrid, entry["name"], reason, entry["bid_schedule_count"], entry["time_point_count"],
            )
    return ignored


_INTERPOLATION_KINDS = ("none", "previous", "next", "linear")
# bid attributes and time-point attributes of the profile that the model does not use (warned about)
_UNREAD_BID_ATTRIBUTES = ("isFixed", "totalMinimumEnergy", "totalMaximumEnergy")
_UNREAD_BID_POINT_ATTRIBUTES = ("minimumActivationP", "stepIncrementP")
# attributes of the old mock form that the profile does not have (still read, deprecated)
_NON_STANDARD_BID_ATTRIBUTES = ("penaltyFactor", "minEnergyBid", "maxEnergyBid", "stepSizeCTRD")


def _minute_of_day(hhmmz: str) -> int:
    return int(hhmmz[:2]) * 60 + int(hhmmz[2:4])


_ISO_DATE = re.compile(r"(\d{4}-\d{2}-\d{2})T\d{2}:\d{2}:")


def _iso_day_number(iso: str) -> Optional[int]:
    """Day number (date.toordinal) of an ISO datetime string; None when it carries no date."""
    m = _ISO_DATE.search(iso)
    if not m:
        return None
    try:
        return date.fromisoformat(m.group(1)).toordinal()
    except ValueError:
        return None


def _bid_value_at(points: dict, minute: int, kind: str):
    """The time point that governs `minute` for a bid with the interpolation kind `kind`, as
    (signed p, price, activation cost, reserve price), or None. A point at exactly `minute` always
    counts. previous: the latest point before, next: the earliest point after, linear: p and price
    interpolated between the two neighbours (no extrapolation; the costs of the earlier point);
    none: no value between points."""
    if minute in points:
        return points[minute]
    if kind == "none":
        return None
    before = max((m for m in points if m < minute), default=None)
    after = min((m for m in points if m > minute), default=None)
    if kind == "previous":
        return points[before] if before is not None else None
    if kind == "next":
        return points[after] if after is not None else None
    if before is None or after is None:
        return None
    weight = (minute - before) / (after - before)
    (p0, price0, act0, res0), (p1, price1, _, _) = points[before], points[after]
    price = None if price0 is None or price1 is None else price0 + weight * (price1 - price0)
    return (p0 + weight * (p1 - p0), price, act0, res0)


def parse_power_bids(
    sis_graph: Graph,
    ra_mrids: set[str],
    tso: str,
    timestamp_list: list[str],
    price_timestamp: Optional[str] = None,
    run_date: Optional[str] = None,
) -> list[PowerBid]:
    """The power bids (PowerBidSchedule with their time points) of the power remedial actions
    ra_mrids from one TSO's SIS graph; the size is |p| in MW, price in EUR/MW at the run timestamp
    (price_timestamp, default the first of timestamp_list). Used for RD and CT RAs.
    Time points: a schedule whose points all lie on one day is read by clock time (the day is not
    compared). A schedule with points on several days is read by date and time: the run's datetime is
    run_date (ISO date, "YYYY-MM-DD") at the timestamp, and a timestamp earlier on the clock than the first
    of timestamp_list belongs to the next day; run_date None = the earliest date of any bid time point of
    the file (warning). Where the run's datetime has no point, the interpolation kind decides as for a
    single day (none: no value, so volume 0), with a warning that counts such bids.
    Direction: the PowerBidSchedule.direction literal (last word up or down) is authoritative; a bid
    without a direction, or with another one (upAndDown, stable), is skipped with a warning and the
    direction is never taken from p. The sign of p is a cross-check only (up p > 0, down p < 0 as in
    the official sample): a disagreement is a warning and |p| is still used. Between time points the
    value follows BaseTimeSeries.interpolationKind (none, previous, next, linear; missing = none, with
    one warning for all such bids). Attributes the model does not use are named in a warning."""
    power_bids: list[PowerBid] = []

    # pbs_mrid -> {minute of day -> (signed p, price, time-point activation cost, reserve price)}, plus the
    # time-point attributes that are not read. Every point is kept: previous / next / linear need
    # the neighbours of a requested timestamp.
    pbs_dated_points: dict[str, dict[tuple, tuple]] = {}   # keyed by (day number or None, minute of day)
    pbs_unread_point_attributes: dict[str, set[str]] = {}
    for tp_uri in sis_graph.subjects(RDF.type, NC.PowerBidScheduleTimePoint):
        pbs_uri  = sis_graph.value(tp_uri, NC["PowerBidScheduleTimePoint.PowerBidSchedule"])
        at_time  = sis_graph.value(tp_uri, NC["PowerBidScheduleTimePoint.atTime"])
        p_val    = sis_graph.value(tp_uri, NC["PowerBidScheduleTimePoint.p"])
        price    = sis_graph.value(tp_uri, NC["PowerBidScheduleTimePoint.price"])
        tp_activation = sis_graph.value(tp_uri, NC["PowerBidScheduleTimePoint.activationCost"])
        tp_reserve    = sis_graph.value(tp_uri, NC["PowerBidScheduleTimePoint.reservePrice"])

        if pbs_uri is None or at_time is None or p_val is None:
            continue

        ts = _iso_to_hhmmz(str(at_time))
        if ts is None:
            continue

        pbs_mrid = _frag(str(pbs_uri))
        pbs_dated_points.setdefault(pbs_mrid, {})[(_iso_day_number(str(at_time)), _minute_of_day(ts))] = (
            float(str(p_val)),
            float(str(price)) if price is not None else None,
            float(str(tp_activation)) if tp_activation is not None else None,
            float(str(tp_reserve)) if tp_reserve is not None else None,
        )
        for attribute in _UNREAD_BID_POINT_ATTRIBUTES:
            if sis_graph.value(tp_uri, NC[f"PowerBidScheduleTimePoint.{attribute}"]) is not None:
                pbs_unread_point_attributes.setdefault(pbs_mrid, set()).add(attribute)

    # the run's day: given, else the earliest day of any bid time point of the file
    all_days = {day for points in pbs_dated_points.values() for day, _ in points if day is not None}
    run_day_given = run_date is not None
    run_day = date.fromisoformat(run_date).toordinal() if run_day_given else (min(all_days) if all_days else None)
    first_minute = _minute_of_day(timestamp_list[0]) if timestamp_list else 0
    # the run time of the CGM, when the pipeline found it inside the NCP series (run_time.py): its date is the run day
    # of every dated schedule, even one that lies on a single day
    cgm_run_time = run_time.current() if run_date is None else None
    if cgm_run_time is not None:
        run_day, run_day_given, first_minute = cgm_run_time.day, True, cgm_run_time.minute

    def requested_minute(timestamp: str, several_days: bool) -> int:
        """The key of the run's timestamp in the points of a schedule (clock minute, or day number x 1440 + minute)."""
        minute = _minute_of_day(timestamp)
        if not several_days:
            return minute
        return ((run_day or 0) + (1 if minute < first_minute else 0)) * 1440 + minute

    several_days_bids: list[str] = []
    several_days_without_point: list[str] = []
    without_interpolation: list[str] = []
    non_standard: dict[str, set[str]] = {}
    for pbs_uri in sis_graph.subjects(RDF.type, NC.PowerBidSchedule):
        pbs_mrid_lit = sis_graph.value(pbs_uri, CIM["IdentifiedObject.mRID"])
        pbs_mrid = ncp_id(pbs_mrid_lit) if pbs_mrid_lit is not None else _frag(str(pbs_uri))
        # time points name their schedule by its rdf:ID (the URI fragment), which can differ from the mRID literal
        pbs_fragment = _frag(str(pbs_uri))

        ra_uri   = sis_graph.value(pbs_uri, NC["PowerBidSchedule.PowerRemedialAction"])
        if ra_uri is None:
            continue
        ra_mrid = _frag(str(ra_uri))

        if ra_mrid not in ra_mrids:
            log.debug("%s: PowerBidSchedule %s RA %s is not a wanted RA — skipped", tso, pbs_mrid, ra_mrid)
            continue

        dir_uri = sis_graph.value(pbs_uri, NC["PowerBidSchedule.direction"])
        if dir_uri is None:
            log.warning("%s: bid %s has no direction - skipped (the direction is never taken from the sign of p)", tso, pbs_mrid)
            continue
        dir_word = re.split(r"[#.]", str(dir_uri))[-1]
        if dir_word.lower() not in ("up", "down"):
            log.warning("%s: bid %s has direction %s, which is neither up nor down - skipped", tso, pbs_mrid, dir_word)
            continue
        direction = BidDirection.DOWN if dir_word.lower() == "down" else BidDirection.UP

        kind_uri = sis_graph.value(pbs_uri, NC["BaseTimeSeries.interpolationKind"])
        if kind_uri is None:
            without_interpolation.append(pbs_mrid)
            interpolation = "none"
        else:
            interpolation = re.split(r"[#.]", str(kind_uri))[-1]
            if interpolation not in _INTERPOLATION_KINDS:
                log.warning("%s: bid %s has interpolationKind %s, which is unknown - none used", tso, pbs_mrid, interpolation)
                interpolation = "none"

        unread = [a for a in _UNREAD_BID_ATTRIBUTES
                  if sis_graph.value(pbs_uri, NC[f"PowerBidSchedule.{a}"]) is not None
                  and not (a == "isFixed" and str(sis_graph.value(pbs_uri, NC["PowerBidSchedule.isFixed"])).lower() == "false")]
        unread += sorted(pbs_unread_point_attributes.get(pbs_fragment, ()))
        if unread:
            log.warning("%s: bid %s has %s - not read (the bid is optimised as a continuous bid)", tso, pbs_mrid, ", ".join(unread))

        lead_lit = sis_graph.value(pbs_uri, NC["PowerBidSchedule.leadTime"])
        lead_time_h = _parse_iso_duration_hours(str(lead_lit)) if lead_lit is not None else 0.0

        act_cost_lit = sis_graph.value(pbs_uri, NC["PowerBidSchedule.activationCost"])
        act_cost = float(str(act_cost_lit)) if act_cost_lit is not None else None

        penalty_lit = sis_graph.value(pbs_uri, NC["PowerBidSchedule.penaltyFactor"])
        penalty = float(str(penalty_lit)) if penalty_lit is not None else 0.0

        min_e_lit = sis_graph.value(pbs_uri, NC["PowerBidSchedule.minEnergyBid"])
        max_e_lit = sis_graph.value(pbs_uri, NC["PowerBidSchedule.maxEnergyBid"])
        step_lit  = sis_graph.value(pbs_uri, NC["PowerBidSchedule.stepSizeCTRD"])
        for attribute, literal in zip(_NON_STANDARD_BID_ATTRIBUTES, (penalty_lit, min_e_lit, max_e_lit, step_lit)):
            if literal is not None:
                non_standard.setdefault(pbs_mrid, set()).add(attribute)

        dated = pbs_dated_points.get(pbs_fragment, {})
        days = {day for day, _ in dated}
        several_days = (
            len(days - {None}) > 1 or (len(days) > 1 and run_day is not None)
            or (cgm_run_time is not None and bool(days - {None}))
        )
        if several_days:
            # a point without a date belongs to the run's day
            points = {((run_day if day is None else day) or 0) * 1440 + minute: value for (day, minute), value in dated.items()}
            several_days_bids.append(pbs_mrid)
        else:
            points = {minute: value for (_, minute), value in dated.items()}
        run_timestamp = price_timestamp or timestamp_list[0]
        if several_days and requested_minute(run_timestamp, True) not in points:
            several_days_without_point.append(pbs_mrid)
        available_mw: dict[str, float] = {}
        for ts in timestamp_list:
            value = _bid_value_at(points, requested_minute(ts, several_days), interpolation)
            available_mw[ts] = abs(value[0]) if value is not None else 0.0

        # the price of the run's timestamp; a missing or negative price is 0 (with a warning)
        run_point = _bid_value_at(points, requested_minute(run_timestamp, several_days), interpolation)
        run_price = run_point[1] if run_point is not None else None
        if run_price is None:
            log.warning("%s: bid %s has no price at %s - price set to 0", tso, pbs_mrid, run_timestamp)
            price_eur_mw = 0.0
        elif run_price < 0:
            log.warning(
                "%s: bid %s has a negative price (%s) at %s - price set to 0", tso, pbs_mrid, run_price, run_timestamp,
            )
            price_eur_mw = 0.0
        else:
            price_eur_mw = float(run_price)

        # the sign of p is a cross-check of the direction only (up p > 0, down p < 0)
        if run_point is not None and run_point[0] != 0.0 and (run_point[0] > 0) != (direction == BidDirection.UP):
            log.warning(
                "%s: bid %s has direction %s but the sign of p disagrees (p = %s at %s; up bids are expected to have p > 0 "
                "and down bids p < 0) - the direction is used and |p| is the size", tso, pbs_mrid, dir_word, run_point[0], run_timestamp,
            )

        # the activation cost of the run's time point overrides the schedule-level one
        # (BUC 14#102); the reserve price exists per time point only
        tp_activation, reserve_price = (run_point[2], run_point[3]) if run_point is not None else (None, None)
        if tp_activation is not None:
            act_cost = tp_activation
        # a negative reservation price is handled like a negative energy price (O1 rule 4): a
        # negative cost would make the solver reserve curative volume just to earn it
        if reserve_price is not None and reserve_price < 0:
            log.warning(
                "%s: bid %s has a negative reservePrice (%s) at %s - reservePrice set to 0",
                tso, pbs_mrid, reserve_price, run_timestamp,
            )
            reserve_price = 0.0

        currency_uri = sis_graph.value(pbs_uri, NC["PowerBidSchedule.currency"])
        warn_if_not_eur(currency_uri, f"{tso}: bid {pbs_mrid}")

        power_bids.append(PowerBid(
            mrid=pbs_mrid,
            ra_mrid=ra_mrid,
            direction=direction,
            available_mw=available_mw,
            price_eur_mw=price_eur_mw,
            penalty_factor=penalty,
            lead_time_h=lead_time_h,
            activation_cost_eur=act_cost,
            min_energy_mwh=float(str(min_e_lit)) if min_e_lit is not None else None,
            max_energy_mwh=float(str(max_e_lit)) if max_e_lit is not None else None,
            step_size_mw=float(str(step_lit)) if step_lit is not None else None,
            reserve_price_eur_mw=reserve_price,
        ))

    if several_days_bids:
        log.warning(
            "%s: %d bid(s) have time points on several days - read by date and time, run day %s%s: %s",
            tso, len(several_days_bids), date.fromordinal(run_day).isoformat() if run_day else "unknown",
            "" if run_day_given else " (the earliest day of the bid time points of the file)", ", ".join(several_days_bids),
        )
    if several_days_without_point:
        log.warning(
            "%s: %d bid(s) with time points on several days have no time point at the run's date and time (%s) - "
            "the interpolation kind decides (none: volume 0): %s",
            tso, len(several_days_without_point), price_timestamp or timestamp_list[0], ", ".join(several_days_without_point),
        )
    if without_interpolation:
        log.warning(
            "%s: %d bid(s) have no BaseTimeSeries.interpolationKind - none assumed (exact time points only): %s",
            tso, len(without_interpolation), ", ".join(without_interpolation),
        )
    if non_standard:
        log.warning(
            "%s: %d bid(s) use attributes that are not in the profile (deprecated, still read): %s",
            tso, len(non_standard),
            "; ".join(f"{mrid}: {', '.join(sorted(names))}" for mrid, names in sorted(non_standard.items())),
        )

    return power_bids


# PowerShiftKeySchedule resources other than GeneratingUnit (SIS profile): not read, dropped with a warning
_OTHER_RESOURCES = (
    "PowerElectronicsUnit", "HydroPump", "EnergyConsumer", "EnergySource", "EquivalentInjection",
    "ExternalNetworkInjection", "DCEquivalentInjection", "DCPole", "EnergyBlockOrder",
    "EnergyResourceGroupCapacity", "ScheduleResource", "SchedulingArea",
)


def read_key_routes(
    sis_graph: Graph, ra_graph: Optional[Graph], ra_mrids: set[str],
) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """The PowerShiftKeySchedule mRIDs of each RA in ra_mrids by its two unit routes: the bid route
    (RA <- PowerBidSchedule <- PowerShiftKeyDistribution -> PowerShiftKeySchedule) and the strategy
    route (RA <- PowerShiftKeyStrategy <- PowerShiftKeyDistribution -> PowerShiftKeySchedule; the
    strategy -> RA links are read from ra_graph, None = no strategy route). Returns the two maps
    {RA mRID: set of key mRIDs}, every RA of ra_mrids a key of both."""
    pbs_to_ra: dict[str, str] = {}
    for pbs_uri in sis_graph.subjects(RDF.type, NC.PowerBidSchedule):
        # the distributions name the schedule by its rdf:ID (the URI fragment), not by its mRID literal
        pbs_fragment = _frag(str(pbs_uri))
        ra_uri = sis_graph.value(pbs_uri, NC["PowerBidSchedule.PowerRemedialAction"])
        if ra_uri is not None and _frag(str(ra_uri)) in ra_mrids:
            pbs_to_ra[pbs_fragment] = _frag(str(ra_uri))

    strategy_to_ra: dict[str, str] = {}
    if ra_graph is not None:
        for strategy_uri in ra_graph.subjects(RDF.type, NC.PowerShiftKeyStrategy):
            ra_uri = ra_graph.value(strategy_uri, NC["PowerShiftKeyStrategy.PowerRemedialAction"])
            if ra_uri is not None:
                strategy_to_ra[_frag(str(strategy_uri))] = _frag(str(ra_uri))

    by_bid: dict[str, set[str]] = {ra_mrid: set() for ra_mrid in ra_mrids}
    by_strategy: dict[str, set[str]] = {ra_mrid: set() for ra_mrid in ra_mrids}
    # .objects(), not .value(): RDF allows several triples with the same subject and predicate
    for distribution_uri in sis_graph.subjects(RDF.type, NC.PowerShiftKeyDistribution):
        psks_mrids = {_frag(str(o)) for o in sis_graph.objects(distribution_uri, NC["PowerShiftKeyDistribution.PowerShiftKeySchedule"])}
        for o in sis_graph.objects(distribution_uri, NC["PowerShiftKeyDistribution.PowerBidSchedule"]):
            ra_mrid = pbs_to_ra.get(_frag(str(o)))
            if ra_mrid is not None:
                by_bid[ra_mrid].update(psks_mrids)
        for o in sis_graph.objects(distribution_uri, NC["PowerShiftKeyDistribution.PowerShiftKeyStrategy"]):
            ra_mrid = strategy_to_ra.get(_frag(str(o)))
            if ra_mrid in by_strategy:
                by_strategy[ra_mrid].update(psks_mrids)
    return by_bid, by_strategy


def read_redispatch_units(sis_graph: Graph, ra_graph: Optional[Graph], ra_mrids: set[str]) -> dict[str, list[str]]:
    """{RA mRID: sorted CIM GeneratingUnit mRIDs} of the redispatch RAs ra_mrids: the units of the shift
    keys of the RA (a key with a resource other than a GeneratingUnit has no unit here). The bid route
    wins when it reaches an existing key, otherwise the strategy route is used (as in resolve_glsk)."""
    by_bid, by_strategy = read_key_routes(sis_graph, ra_graph, ra_mrids)
    units_of_key: dict[str, set[str]] = {}
    existing_keys: set[str] = set()
    for psks_uri in sis_graph.subjects(RDF.type, NC.PowerShiftKeySchedule):
        mrid_lit = sis_graph.value(psks_uri, CIM["IdentifiedObject.mRID"])
        psks_mrid = ncp_id(mrid_lit) if mrid_lit is not None else _frag(str(psks_uri))
        units = {_frag(str(o)) for o in sis_graph.objects(psks_uri, NC["PowerShiftKeySchedule.GeneratingUnit"])}
        if units:
            units_of_key[psks_mrid] = units
        if units or any(sis_graph.value(psks_uri, NC[f"PowerShiftKeySchedule.{name}"]) is not None for name in _OTHER_RESOURCES):
            existing_keys.add(psks_mrid)
    result: dict[str, list[str]] = {}
    for ra_mrid in ra_mrids:
        keys = (by_bid[ra_mrid] & existing_keys) or by_strategy[ra_mrid]
        result[ra_mrid] = sorted({unit for key in keys for unit in units_of_key.get(key, ())})
    return result


def _generator_of_unit(
    unit_mrid: str,
    generators_by_unit: Optional[dict[str, list[str]]],
    valid_generator_mrids: set[str],
    sis_path: Path,
    tso: str,
) -> Optional[str]:
    """The network generator of a PowerShiftKeySchedule unit id: through generators_by_unit (a CIM
    GeneratingUnit), else the id itself (the old form, a generator id: deprecation warning once per
    file), else the id unchanged so that the later check reports it as not in the network."""
    if generators_by_unit and unit_mrid in generators_by_unit:
        generators = generators_by_unit[unit_mrid]
        if len(generators) == 1:
            return generators[0]
        log.warning(
            "%s: GeneratingUnit %s is the unit of %d generators %s — dropped (the unit cannot be split)",
            tso, unit_mrid, len(generators) if generators else 0, sorted(generators or []),
        )
        return None
    if unit_mrid in valid_generator_mrids and generators_by_unit:
        warn_deprecated_form_once(
            sis_path, "glsk-generator-id",
            "%s: %s names network generators in PowerShiftKeySchedule.GeneratingUnit (deprecated non-standard form; "
            "the profile has the CIM GeneratingUnit)" % (tso, sis_path.name),
        )
    return unit_mrid


def parse_rd_units(
    ra_path: Path,
    sis_path: Optional[Path],
    tso: str,
    timestamp_list: list[str],
    cgm_generating_units: dict[str, dict],  # gu_mrid → {name, target_p, max_p, min_p, ...}
    ae_path: Optional[Path] = None,  # {TSO}_AE.xml, for linked_xne_mrids resolution
    price_timestamp: Optional[str] = None,  # timestamp of the run; None = timestamp_list[0]
    warn_countertrade: bool = True,  # False: the CT RAs are parsed by the caller, not reported as ignored
    generators_by_unit: Optional[dict[str, list[str]]] = None,  # CIM GeneratingUnit mRID -> network generator ids
) -> tuple[list[RDUnit], list[PowerBid]]:
    """Parse RD RAs from {TSO}_RA.xml and power bids from {TSO}_SIS.xml (PowerBidScheduleTimePoint:
    p in MW, price in EUR/MW; leadTime ISO 8601 -> hours). An RD RA is a RedispatchRemedialAction whose
    units are the GeneratingUnits of its shift keys (bid route or strategy route, mapped to network
    generators through generators_by_unit), or in the old form a GSARA with at least one
    RotatingMachineAction. Curative linkage as in parse_psts; RDUnit.mrid is the RA mRID, so the
    lookup is direct. ae_path None: linked_xne_mrids = [].
    """
    ra_graph = _load_graph(ra_path)
    sis_graph = _load_graph(sis_path) if sis_path is not None else None

    # GSARA mRID -> linked / excluded contingency mRIDs
    gsara_to_linked_contingencies, gsara_to_excluded_contingencies = read_contingency_links(ra_graph, ra_path, tso)

    # GSARA mRID -> linked XNE mRIDs, from {TSO}_AE.xml (only with ae_path)
    gsara_to_linked_xnes: dict[str, list[str]] = {}
    if ae_path is not None and ae_path.exists():
        ae_graph = _load_graph(ae_path)
        for aewra_uri in ae_graph.subjects(RDF.type, NC.AssessedElementWithRemedialAction):
            ra_uri = ae_graph.value(aewra_uri, NC["AssessedElementWithRemedialAction.RemedialAction"])
            ae_uri = ae_graph.value(aewra_uri, NC["AssessedElementWithRemedialAction.AssessedElement"])
            if ra_uri is None or ae_uri is None:
                continue
            gsara_to_linked_xnes.setdefault(_frag(str(ra_uri)), []).append(_frag(str(ae_uri)))

    # Map GSARA mRID → list of RotatingMachine mRIDs
    gsara_to_gu: dict[str, list[str]] = {}
    # Also collect ramp rates per (gsara, gu)
    ramp_rates: dict[tuple[str, str], tuple[Optional[float], Optional[float]]] = {}
    for rma_uri in ra_graph.subjects(RDF.type, NC.RotatingMachineAction):
        gsara_uri = ra_graph.value(rma_uri, NC["GridStateAlteration.GridStateAlterationRemedialAction"])
        rm_uri    = ra_graph.value(rma_uri, NC["RotatingMachineAction.RotatingMachine"])
        if gsara_uri is None or rm_uri is None:
            continue
        gsara_mrid = _frag(str(gsara_uri))
        gu_mrid    = _frag(str(rm_uri))
        gsara_to_gu.setdefault(gsara_mrid, []).append(gu_mrid)

        # ramp rates from their documented predicate names
        raise_rate_lit = ra_graph.value(rma_uri, NC["RotatingMachineAction.raiseRampRate"])
        lower_rate_lit = ra_graph.value(rma_uri, NC["RotatingMachineAction.lowerRampRate"])
        try:
            raise_rate = float(str(raise_rate_lit)) if raise_rate_lit is not None else None
        except Exception:
            raise_rate = None
        try:
            lower_rate = float(str(lower_rate_lit)) if lower_rate_lit is not None else None
        except Exception:
            lower_rate = None

        ramp_rates[(gsara_mrid, gu_mrid)] = (raise_rate, lower_rate)

    # RedispatchRemedialAction: its units are the generating units of its shift keys (SIS)
    standard_ras = sorted(ra_graph.subjects(RDF.type, NC.RedispatchRemedialAction), key=str)
    standard_units: dict[str, list[str]] = {}
    if standard_ras and sis_graph is not None:
        unit_ids = read_redispatch_units(sis_graph, ra_graph, {_frag(str(uri)) for uri in standard_ras})
        for ra_fragment, units in unit_ids.items():
            generators = (_generator_of_unit(unit, generators_by_unit, set(cgm_generating_units), sis_path, tso) for unit in units)
            standard_units[ra_fragment] = list(dict.fromkeys(g for g in generators if g is not None))

    rd_units: list[RDUnit] = []
    rd_mrid_set: set[str] = set()
    warned_default_limit_units: set[str] = set()
    standard_with_old_actions: set[str] = set()

    for gsara_uri in list(ra_graph.subjects(RDF.type, NC.GridStateAlterationRemedialAction)) + standard_ras:
        mrid = ra_graph.value(gsara_uri, CIM["IdentifiedObject.mRID"])
        mrid = ncp_id(mrid) if mrid is not None else _frag(str(gsara_uri))
        gsara_mrid = str(mrid)

        if (gsara_uri, RDF.type, NC.RedispatchRemedialAction) in ra_graph:
            gu_mrids = standard_units.get(_frag(str(gsara_uri)), [])
            if gsara_to_gu.get(gsara_mrid, gsara_to_gu.get(_frag(str(gsara_uri)))):
                standard_with_old_actions.add(gsara_mrid)
        else:
            gu_mrids = gsara_to_gu.get(gsara_mrid, gsara_to_gu.get(_frag(str(gsara_uri)), []))
            if gu_mrids:
                warn_deprecated_form_once(
                    ra_path, "rotating-machine-action",
                    "%s: %s describes redispatch with RotatingMachineAction (deprecated non-standard form; the profile "
                    "has RedispatchRemedialAction with its units in the shift keys)" % (tso, ra_path.name),
                )
        if not gu_mrids:
            continue  # Not an RD RA

        name = ra_graph.value(gsara_uri, CIM["IdentifiedObject.name"])
        available_lit = ra_graph.value(gsara_uri, NC["RemedialAction.normalAvailable"])
        available = str(available_lit).lower() == "true" if available_lit else False

        kind_uri = ra_graph.value(gsara_uri, NC["RemedialAction.kind"])
        kind_str = _frag(str(kind_uri)) if kind_uri is not None else ""
        kind = RAKind.CURATIVE if "curative" in kind_str.lower() else RAKind.PREVENTIVE

        # RA-level penaltyFactor (RemedialAction), not PowerBidSchedule.penaltyFactor (bid level,
        # PowerBid.penalty_factor). Default 1.0 when absent.
        penalty_lit = ra_graph.value(gsara_uri, NC["RemedialAction.penaltyFactor"])
        if penalty_lit is None:
            log.warning(
                "%s: RD remedial action %s has no RemedialAction.penaltyFactor - defaulted to 1.0",
                tso, gsara_mrid,
            )
            penalty_factor = 1.0
        else:
            penalty_factor = float(str(penalty_lit))

        # reaction time (RA level), compared with the TATL durations for curative RD (CORNET-613)
        reaction_time_minutes = _parse_reaction_time_minutes(
            ra_graph.value(gsara_uri, NC["RemedialAction.timeToImplement"]), tso, gsara_mrid,
        )

        for gu_mrid in gu_mrids:
            gu_data = cgm_generating_units.get(gu_mrid)
            if gu_data is None:
                log.warning(
                    "%s: RD RA %s GeneratingUnit %s not found in CGM — skipped",
                    tso, mrid, gu_mrid,
                )
                continue

            # initial_p is the generator's current setpoint in the CGM (target_p)
            initial_p = float(gu_data.get("target_p", gu_data.get("initial_p", 0.0)))
            max_p = float(gu_data.get("max_p", 0.0))
            min_p = float(gu_data.get("min_p", 0.0))
            defaulted_limits = [key for key in ("max_p", "min_p") if key not in gu_data]
            if defaulted_limits and gu_mrid not in warned_default_limit_units:
                warned_default_limit_units.add(gu_mrid)
                log.warning(
                    "%s: generating unit %s (%s) has no %s in the CGM - defaulted to 0.0 "
                    "(its Pmax/Pmin redispatch limits are then 0.0)",
                    tso, gu_mrid, gu_data.get("name", gu_mrid), " / ".join(defaulted_limits),
                )

            # Ramp rates should be sourced from the RA RotatingMachineAction
            raise_rate, lower_rate = ramp_rates.get((gsara_mrid, gu_mrid), (None, None))

            rd_units.append(RDUnit(
                mrid=gsara_mrid,
                name=str(name) if name is not None else gsara_mrid,
                tso=tso,
                gu_mrid=gu_mrid,
                gu_name=gu_data.get("name", gu_mrid),
                available=available,
                kind=kind,
                initial_p=initial_p,
                max_p=max_p,
                min_p=min_p,
                raise_ramp_rate=raise_rate,
                lower_ramp_rate=lower_rate,
                penalty_factor=penalty_factor,
                linked_contingency_mrids=list(gsara_to_linked_contingencies.get(gsara_mrid, [])),
                excluded_contingency_mrids=list(gsara_to_excluded_contingencies.get(gsara_mrid, [])),
                linked_xne_mrids=list(gsara_to_linked_xnes.get(gsara_mrid, [])),
                reaction_time_minutes=reaction_time_minutes,
            ))
            rd_mrid_set.add(gsara_mrid)

    if standard_with_old_actions:
        log.warning(
            "%s: %d RedispatchRemedialAction(s) also have RotatingMachineAction(s) (%s) - the standard form wins, "
            "the old actions are ignored", tso, len(standard_with_old_actions), ", ".join(sorted(standard_with_old_actions)),
        )
    log.info("%s: %d RD units parsed", tso, len(rd_units))

    # ------------------------------------------------------------------
    # Power bids from SIS
    # ------------------------------------------------------------------
    power_bids: list[PowerBid] = []

    if sis_graph is None:
        log.info("%s: no SIS file — no power bids", tso)
        warn_unsupported_power_remedial_actions(
            ra_graph, None, tso, include_countertrade=warn_countertrade, parsed_mrids=rd_mrid_set,
        )
        return rd_units, power_bids

    warn_unsupported_power_remedial_actions(
        ra_graph, sis_graph, tso, include_countertrade=warn_countertrade, parsed_mrids=rd_mrid_set,
    )

    power_bids = parse_power_bids(sis_graph, rd_mrid_set, tso, timestamp_list, price_timestamp)

    log.info("%s: %d power bids parsed", tso, len(power_bids))
    return rd_units, power_bids


def warn_if_not_eur(currency, what: str) -> None:
    """All costs are taken as EUR (no conversion): a currency other than EUR (a URI or literal
    ending in EUR, e.g. Currency.EUR) is logged as a warning. No currency given: nothing."""
    if currency is None:
        return
    code = str(currency).rsplit("#", 1)[-1].rsplit(".", 1)[-1]
    if code != "EUR":
        log.warning("%s: currency %s is not EUR - its costs are used as EUR (no conversion)", what, code)


def parse_unit_costs(
    sis_path: Optional[Path],
    er_path: Optional[Path],
    tso: str,
    timestamp: str,
    generator_id_by_unit_mrid: Optional[dict[str, str]] = None,
) -> dict[str, UnitCost]:
    """{generator id: UnitCost} of one TSO at `timestamp` (HHMMZ). Start-up cost:
    UnitCostTimePoint.warmStartupCost, else UnitCostTimePoint.startupCost (SIS, the time point
    at `timestamp` of the unit's UnitCostSchedule), else GeneratingUnit.normalWarmStartupCost
    (ER). Shut-down cost: UnitCostTimePoint.shutdownCost, else GeneratingUnit.normalShutdownCost.
    A missing value stays None (that state change is not allowed). generator_id_by_unit_mrid maps
    a CGMES GeneratingUnit mRID to the network generator id (a generator id maps to itself).
    A negative start-up or shut-down cost is fatal (BUC 14#103: the RAO cancels itself)."""
    to_generator = generator_id_by_unit_mrid or {}
    sis_values: dict[str, dict[str, float]] = {}
    er_values: dict[str, dict[str, float]] = {}

    if sis_path is not None:
        g = _load_graph(sis_path)
        schedule_unit: dict[str, str] = {}
        for schedule_uri in g.subjects(RDF.type, NC.UnitCostSchedule):
            unit_ref = g.value(schedule_uri, NC["UnitCostSchedule.GeneratingUnit"])
            if unit_ref is None:
                continue
            schedule_mrid = _frag(str(schedule_uri))
            schedule_unit[schedule_mrid] = _frag(str(unit_ref))
            warn_if_not_eur(g.value(schedule_uri, NC["UnitCostSchedule.currency"]),
                            f"{tso}: UnitCostSchedule {schedule_mrid}")
        cost_points = list(g.subjects(RDF.type, NC.UnitCostTimePoint))
        required_day = run_time.required_day(
            (run_time.iso_day(str(g.value(p, NC["UnitCostTimePoint.atTime"]))) for p in cost_points),
            f"{tso}: unit costs", sis_path.name,
        )
        for point_uri in cost_points:
            schedule_ref = g.value(point_uri, NC["UnitCostTimePoint.UnitCostSchedule"])
            at_time = g.value(point_uri, NC["UnitCostTimePoint.atTime"])
            if schedule_ref is None or at_time is None or _iso_to_hhmmz(str(at_time)) != timestamp:
                continue
            if not run_time.day_matches(run_time.iso_day(str(at_time)), required_day, _minute_of_day(timestamp), _minute_of_day(timestamp)):
                continue
            unit_mrid = schedule_unit.get(_frag(str(schedule_ref)))
            if unit_mrid is None:
                continue
            values = sis_values.setdefault(unit_mrid, {})
            for attribute in ("warmStartupCost", "startupCost", "shutdownCost"):
                literal = g.value(point_uri, NC[f"UnitCostTimePoint.{attribute}"])
                if literal is not None:
                    values[attribute] = float(str(literal))

    if er_path is not None:
        g = _load_graph(er_path)
        for attribute in ("normalWarmStartupCost", "normalShutdownCost"):
            for unit_uri, literal in g.subject_objects(NC[f"GeneratingUnit.{attribute}"]):
                er_values.setdefault(_frag(str(unit_uri)), {})[attribute] = float(str(literal))
        for unit_uri, currency in g.subject_objects(NC["GeneratingUnit.costCurrency"]):
            warn_if_not_eur(currency, f"{tso}: GeneratingUnit {_frag(str(unit_uri))}")

    unit_costs: dict[str, UnitCost] = {}
    for unit_mrid in sorted(set(sis_values) | set(er_values)):
        sis, er = sis_values.get(unit_mrid, {}), er_values.get(unit_mrid, {})
        startup = next((v for v in (sis.get("warmStartupCost"), sis.get("startupCost"),
                                    er.get("normalWarmStartupCost")) if v is not None), None)
        shutdown = next((v for v in (sis.get("shutdownCost"), er.get("normalShutdownCost")) if v is not None), None)
        for label, value in (("start-up", startup), ("shut-down", shutdown)):
            if value is not None and value < 0:
                raise ValueError(f"{tso}: generating unit {unit_mrid} has a negative {label} cost ({value})")
        generator_id = to_generator.get(unit_mrid, unit_mrid)
        unit_costs[generator_id] = UnitCost(startup_cost_eur=startup, shutdown_cost_eur=shutdown)
    log.info("%s: start-up/shut-down costs for %d generating unit(s) at %s", tso, len(unit_costs), timestamp)
    return unit_costs


# ---------------------------------------------------------------------------
# RA group volume limits
# ---------------------------------------------------------------------------

# The group predicate of a RemedialActionDependency: the unmodified NCP_sample_generic_2_4 fixture uses
# DependingRemedialActionGroup, the earlier code and its tests the plain RemedialActionGroup.
_DEPENDENCY_GROUP_PREDICATES = (
    "RemedialActionDependency.DependingRemedialActionGroup",
    "RemedialActionDependency.RemedialActionGroup",
)


def _dependency_group_uri(graph: Graph, dependency_uri):
    """The RemedialActionGroup a RemedialActionDependency points to, under either predicate name
    (None when it has neither)."""
    for predicate in _DEPENDENCY_GROUP_PREDICATES:
        group_uri = graph.value(dependency_uri, NC[predicate])
        if group_uri is not None:
            return group_uri
    return None


def parse_ra_groups(
    ra_path: Path,
    sis_path: Optional[Path],
    tso: str,
    timestamp: str,
) -> dict[str, dict]:
    """Restrictive RA groups from {TSO}_RA.xml and their volume limits from {TSO}_SIS.xml.
    Returns {group_mrid: {"name", "ra_mrids" (restrictive members), "max_regulating_up",
    "max_regulating_down" (MW, None if not resolved for this timestamp)}}. Only restrictive
    dependencies count; a group without a resolved limit is kept with None limits.
    Limits: standard chain RemedialActionGroupTimePoint -> RemedialActionGroupSchedule ->
    RemedialActionGroup (limits declared by the time point); the old mock form (time point linked to
    the group directly) is a fallback with a deprecation warning. No official sample has groups.
    """
    ra_graph = _load_graph(ra_path)

    # RemedialActionDependency links a RemedialAction to a RemedialActionGroup with a .kind;
    # inferred predicates .RemedialActionGroup / .RemedialAction
    group_to_ra_mrids: dict[str, set[str]] = {}
    group_mrid_to_name: dict[str, str] = {}

    not_read_kinds: dict[str, int] = {}
    for dependency_uri in ra_graph.subjects(RDF.type, NC.RemedialActionDependency):
        kind_uri = ra_graph.value(dependency_uri, NC["RemedialActionDependency.kind"])
        kind_str = _frag(str(kind_uri)) if kind_uri is not None else ""
        if "restrictive" not in kind_str.lower():
            # restrictive limits volumes here, balanced is read for HVDC (parse_balanced_ra_groups); any
            # other kind (exclusive, ...) is not read
            if "balanced" not in kind_str.lower():
                kind_name = re.split(r"[#.]", kind_str)[-1] or "none"
                not_read_kinds[kind_name] = not_read_kinds.get(kind_name, 0) + 1
            continue  # spec: skip groups without restrictive dependency

        group_uri = _dependency_group_uri(ra_graph, dependency_uri)
        ra_uri = ra_graph.value(dependency_uri, NC["RemedialActionDependency.RemedialAction"])
        if group_uri is None or ra_uri is None:
            log.warning(
                "%s: RemedialActionDependency %s has kind=restrictive but "
                "missing RemedialActionGroup or RemedialAction link — skipped",
                tso, _frag(str(dependency_uri)),
            )
            continue

        group_mrid_lit = ra_graph.value(group_uri, CIM["IdentifiedObject.mRID"])
        group_mrid = ncp_id(group_mrid_lit) if group_mrid_lit is not None else _frag(str(group_uri))
        ra_mrid = _frag(str(ra_uri))

        group_to_ra_mrids.setdefault(group_mrid, set()).add(ra_mrid)

        if group_mrid not in group_mrid_to_name:
            name_lit = ra_graph.value(group_uri, CIM["IdentifiedObject.name"])
            group_mrid_to_name[group_mrid] = str(name_lit) if name_lit is not None else group_mrid

    if not_read_kinds:
        log.warning(
            "%s: %s has RemedialActionDependency element(s) of a kind that is not read (%s) - ignored, the "
            "dependency does not restrict the remedial actions", tso, ra_path.name,
            ", ".join(f"{name}: {count}" for name, count in sorted(not_read_kinds.items())),
        )
    if not group_to_ra_mrids:
        log.info("%s: no restrictive RemedialActionGroup found in %s", tso, ra_path.name)
        return {}

    # SIS volume limits at the timestamp. Standard form: RemedialActionGroupTimePoint ->
    # RemedialActionGroupSchedule -> RemedialActionGroup, limits declared by the time point. Old
    # mock form: the time point names the group directly and the limits are declared by
    # RemedialActionGroup; read only for a group with no standard time point at this timestamp.
    group_limits: dict[str, tuple[Optional[float], Optional[float]]] = {}
    if sis_path is not None:
        sis_graph = _load_graph(sis_path)
        standard: dict[str, tuple[Optional[float], Optional[float]]] = {}
        old_form: dict[str, tuple[Optional[float], Optional[float]]] = {}
        group_points = list(sis_graph.subjects(RDF.type, NC.RemedialActionGroupTimePoint))
        required_day = run_time.required_day(
            (run_time.iso_day(str(sis_graph.value(p, NC["RemedialActionGroupTimePoint.atTime"]))) for p in group_points),
            f"{tso}: RA group limits", sis_path.name,
        )
        for timepoint_uri in group_points:
            schedule_uri = sis_graph.value(timepoint_uri, NC["RemedialActionGroupTimePoint.RemedialActionGroupSchedule"])
            direct_uri = sis_graph.value(timepoint_uri, NC["RemedialActionGroupTimePoint.RemedialActionGroup"])
            at_time = sis_graph.value(timepoint_uri, NC["RemedialActionGroupTimePoint.atTime"])
            if at_time is None:
                continue
            if schedule_uri is not None:
                group_uri = sis_graph.value(schedule_uri, NC["RemedialActionGroupSchedule.RemedialActionGroup"])
                up_name, down_name, target = "RemedialActionGroupTimePoint.maxRegulatingUp", "RemedialActionGroupTimePoint.maxRegulatingDown", standard
            else:
                group_uri = direct_uri
                up_name, down_name, target = "RemedialActionGroup.maxRegulatingUp", "RemedialActionGroup.maxRegulatingDown", old_form
            if group_uri is None or _iso_to_hhmmz(str(at_time)) != timestamp:
                continue
            if not run_time.day_matches(run_time.iso_day(str(at_time)), required_day, _minute_of_day(timestamp), _minute_of_day(timestamp)):
                continue
            max_up = _float_or_none(sis_graph.value(timepoint_uri, NC[up_name]))
            max_down = _float_or_none(sis_graph.value(timepoint_uri, NC[down_name]))
            target[_frag(str(group_uri))] = (max_up, max_down)
        if old_form:
            warn_deprecated_form_once(
                sis_path, "group-limits",
                "%s: %s uses the deprecated non-standard group limit form (RemedialActionGroupTimePoint."
                "RemedialActionGroup, limits declared by RemedialActionGroup); the standard chain is "
                "RemedialActionGroupTimePoint -> RemedialActionGroupSchedule -> RemedialActionGroup" % (tso, sis_path.name),
            )
        both = sorted(set(standard) & set(old_form))
        if both:
            log.warning(
                "%s: %s gives the limits of group(s) %s in both the standard and the deprecated form - "
                "the standard form is used", tso, sis_path.name, both,
            )
        group_limits = {**old_form, **standard}
    else:
        log.info("%s: no SIS file — RA group volume limits will be unresolved (None)", tso)
    ra_groups: dict[str, dict] = {}
    for group_mrid, ra_mrids in group_to_ra_mrids.items():
        max_up, max_down = group_limits.get(group_mrid, (None, None))
        if max_up is None and max_down is None:
            log.warning(
                "%s: restrictive RemedialActionGroup %s has no resolved "
                "maxRegulatingUp/maxRegulatingDown for timestamp %s — group "
                "volume constraint will be skipped for both directions "
                "(confirmed dataset boundary condition on Jotunheim if this "
                "is unexpected, check the SIS file directly before assuming a bug)",
                tso, group_mrid, timestamp,
            )
        ra_groups[group_mrid] = {
            "name": group_mrid_to_name.get(group_mrid, group_mrid),
            "ra_mrids": sorted(ra_mrids),
            "max_regulating_up": max_up,
            "max_regulating_down": max_down,
        }

    log.info(
        "%s: %d restrictive RA group(s) parsed, %d with a resolved volume limit for timestamp %s",
        tso, len(ra_groups),
        sum(1 for g in ra_groups.values() if g["max_regulating_up"] is not None or g["max_regulating_down"] is not None),
        timestamp,
    )
    return ra_groups


# ---------------------------------------------------------------------------
# HVDC remedial actions (docs/specs/hvdc_ra_specs.md, H2-H4)
# ---------------------------------------------------------------------------

def _enum_token(value) -> str:
    """The enumeration literal of an rdf:resource, i.e. the text after its last "."
    ("nc:RelativeDirectionKind.upAndDown", "https://cim4.eu/ns/nc#ValueOffsetKind.absolute"); "" when absent."""
    return "" if value is None else str(value).rsplit(".", 1)[-1]


def parse_balanced_ra_groups(ra_path: Path, tso: str) -> list[HvdcGroup]:
    """The RemedialActionGroups of {TSO}_RA.xml that have a RemedialActionDependency of kind
    balanced (H2.4): one HvdcGroup per group with every remedial action that depends on it, sorted
    by group mRID. The group predicate may be either name (see _DEPENDENCY_GROUP_PREDICATES)."""
    graph = _load_graph(ra_path)
    members: dict[str, set[str]] = {}
    names: dict[str, str] = {}
    for dependency_uri in graph.subjects(RDF.type, NC.RemedialActionDependency):
        if _enum_token(graph.value(dependency_uri, NC["RemedialActionDependency.kind"])) != "balanced":
            continue
        group_uri = _dependency_group_uri(graph, dependency_uri)
        ra_uri = graph.value(dependency_uri, NC["RemedialActionDependency.RemedialAction"])
        if group_uri is None or ra_uri is None:
            log.warning(
                "%s: RemedialActionDependency %s has kind=balanced but a missing RemedialActionGroup or "
                "RemedialAction link - skipped", tso, _frag(str(dependency_uri)),
            )
            continue
        group_mrid_lit = graph.value(group_uri, CIM["IdentifiedObject.mRID"])
        group_mrid = ncp_id(group_mrid_lit) if group_mrid_lit is not None else _frag(str(group_uri))
        members.setdefault(group_mrid, set()).add(_frag(str(ra_uri)))
        if group_mrid not in names:
            name_lit = graph.value(group_uri, CIM["IdentifiedObject.name"])
            names[group_mrid] = str(name_lit) if name_lit is not None else group_mrid
    groups = [HvdcGroup(mrid=m, name=names[m], tso=tso, ra_mrids=sorted(members[m])) for m in sorted(members)]
    log.info("%s: %d balanced RA group(s) parsed", tso, len(groups))
    return groups


def parse_hvdc_units(ra_path: Path, tso: str) -> list[HvdcUnit]:
    """HVDC remedial actions of {TSO}_RA.xml (H2.1-H2.3): every GridStateAlterationRemedialAction
    that has an EquivalentInjectionAction (nc:EquivalentInjectionAction.EquivalentInjection) is one
    unit, identified by this link only. An EquivalentInjectionAction without a
    GridStateAlteration.GridStateAlterationRemedialAction belongs to a remedial action scheme and is
    ignored. penaltyFactor missing = 1.0, the profile default (warning). Ranges are the static RangeConstraints of the
    action, kept as read (direction, valueKind, value); the operating limits are resolved later
    (hvdc_units.resolve_operating_limits). More than one EI action in one RA is an error (H4.6).
    Units are sorted by RA mRID."""
    g = _load_graph(ra_path)

    # RA key -> [(action URI, EI mRID)]; keyed by the RA's URI fragment as parse_psts does
    ra_to_actions: dict[str, list] = {}
    for action_uri in g.subjects(RDF.type, NC.EquivalentInjectionAction):
        ra_uri = g.value(action_uri, NC["GridStateAlteration.GridStateAlterationRemedialAction"])
        ei_uri = g.value(action_uri, NC["EquivalentInjectionAction.EquivalentInjection"])
        if ra_uri is None:
            log.info(
                "%s: EquivalentInjectionAction %s has no GridStateAlterationRemedialAction (part of a remedial "
                "action scheme) - not an HVDC remedial action, ignored", tso, _frag(str(action_uri)),
            )
            continue
        if ei_uri is None:
            log.warning("%s: EquivalentInjectionAction %s has no EquivalentInjection - ignored", tso, _frag(str(action_uri)))
            continue
        ra_to_actions.setdefault(_frag(str(ra_uri)), []).append((action_uri, _frag(str(ei_uri))))

    # contingency links, as for PST and RD
    ra_to_contingencies, ra_to_excluded_contingencies = read_contingency_links(g, ra_path, tso)

    # the static ranges of each action URI of an HVDC remedial action (the ranges of other actions, e.g. the
    # EquivalentInjectionActions of a remedial action scheme, are not read and not warned about)
    hvdc_action_uris = {action_uri for actions in ra_to_actions.values() for action_uri, _ei in actions}
    action_ranges: dict[object, list[HvdcRange]] = {}
    for range_uri in g.subjects(RDF.type, NC.StaticPropertyRange):
        action_uri = g.value(range_uri, NC["RangeConstraint.GridStateAlteration"])
        if action_uri is None or action_uri not in hvdc_action_uris:
            continue
        label = f"{tso}: StaticPropertyRange {_frag(str(range_uri))}"
        reference = g.value(range_uri, NC["StaticPropertyRange.PropertyReference"])
        if reference is not None and not str(reference).endswith("EquivalentInjection.p"):
            continue  # a range of another property of a (non-HVDC) action
        direction = _enum_token(g.value(range_uri, NC["RangeConstraint.direction"]))
        value_kind = _enum_token(g.value(range_uri, NC["RangeConstraint.valueKind"]))
        value_lit = g.value(range_uri, NC["RangeConstraint.normalValue"])
        if direction not in HVDC_DIRECTIONS or value_kind not in HVDC_VALUE_KINDS or value_lit is None:
            log.warning(
                "%s has direction %r, valueKind %r, normalValue %r - not a usable HVDC range, ignored",
                label, direction, value_kind, None if value_lit is None else str(value_lit),
            )
            continue
        if direction == "none":
            log.warning("%s has direction none (a fixed set point, not a range) - ignored", label)
            continue
        try:
            value = float(str(value_lit))
        except ValueError:
            log.warning("%s has a non-numeric normalValue %r - ignored", label, str(value_lit))
            continue
        action_ranges.setdefault(action_uri, []).append(HvdcRange(direction, value_kind, value))

    units: list[HvdcUnit] = []
    for gsara_uri in g.subjects(RDF.type, NC.GridStateAlterationRemedialAction):
        mrid_lit = g.value(gsara_uri, CIM["IdentifiedObject.mRID"])
        mrid = ncp_id(mrid_lit) if mrid_lit is not None else _frag(str(gsara_uri))
        key = _frag(str(gsara_uri))
        actions = ra_to_actions.get(key) or ra_to_actions.get(mrid)
        if not actions:
            continue
        if len(actions) > 1:
            raise ValueError(
                f"{tso}: HVDC remedial action {mrid} has {len(actions)} EquivalentInjectionActions "
                f"({sorted(ei for _a, ei in actions)}); an HVDC unit is one remedial action with exactly one"
            )
        action_uri, ei_mrid = actions[0]

        name = g.value(gsara_uri, CIM["IdentifiedObject.name"])
        available_lit = g.value(gsara_uri, NC["RemedialAction.normalAvailable"])
        kind_str = _enum_token(g.value(gsara_uri, NC["RemedialAction.kind"]))
        penalty_lit = g.value(gsara_uri, NC["RemedialAction.penaltyFactor"])
        if penalty_lit is None:
            log.warning(
                "%s: HVDC remedial action %s has no RemedialAction.penaltyFactor - defaulted to 1.0 (the profile default)", tso, mrid,
            )
            penalty = 1.0
        else:
            penalty = float(str(penalty_lit))
        units.append(HvdcUnit(
            mrid=mrid,
            name=str(name) if name is not None else mrid,
            tso=tso,
            ei_mrid=ei_mrid,
            available=str(available_lit).lower() == "true" if available_lit else False,
            kind=RAKind.CURATIVE if "curative" in kind_str.lower() else RAKind.PREVENTIVE,
            penalty_factor=penalty,
            reaction_time_minutes=_parse_reaction_time_minutes(
                g.value(gsara_uri, NC["RemedialAction.timeToImplement"]), tso, mrid,
            ),
            linked_contingency_mrids=list(ra_to_contingencies.get(key) or ra_to_contingencies.get(mrid) or []),
            excluded_contingency_mrids=list(ra_to_excluded_contingencies.get(key) or ra_to_excluded_contingencies.get(mrid) or []),
            ranges=list(action_ranges.get(action_uri, [])),
        ))
    units.sort(key=lambda u: u.mrid)
    log.info("%s: %d HVDC remedial action(s) parsed", tso, len(units))
    return units
