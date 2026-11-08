"""inventory_summary.py — console table of the parsed AE, contingency, PST, RD and bid lists."""

from __future__ import annotations

import logging
from typing import Iterable

from data_model import AEType, AssessedElement, Contingency, PST, RDUnit, PowerBid

log = logging.getLogger(__name__)


def print_inventory_summary(
    tso_name_map:   dict[str, str],          # mrid → display name
    tsos_with_data: Iterable[str],           # the TSOs of the parsed objects (an element counts for its operator)
    ae_flat_list:   list[AssessedElement],
    co_flat_list:   list[Contingency],
    pst_list:       list[PST],
    rd_list:        list[RDUnit],
    bid_list:       list[PowerBid],
    timestamp:      str,
    node_count:     int = 0,
    branch_count:   int = 0,
    hvdc_list:      list | None = None,   # HVDC remedial actions; the column appears only when given
) -> None:
    """Log the inventory table as one multi-line log.info message (one call keeps the
    table aligned in the console and in run_log.txt)."""

    tso_names = sorted(set(tso_name_map.values()))
    tsos_with_data = sorted(set(tsos_with_data))

    lines = [
        "",
        "=" * 72,
        "  INVENTORY SUMMARY",
        "=" * 72,
        f"  Network : {node_count} nodes, {branch_count} branches",
        f"  TSOs in CGM          : {len(tso_names)}",
        f"  TSOs with NCP data   : {len(tsos_with_data)}",
        f"  Timestamp            : {timestamp}",
        "-" * 72,
    ]

    # Per-TSO breakdown
    header = f"  {'TSO':<20} {'AE-sec':>6} {'AE-scn':>6} {'AE-exc':>6} {'CO':>5} {'CO-must':>7} {'PST':>5} {'RD':>5} {'Bids':>5}"
    if hvdc_list is not None:
        header += f" {'HVDC':>5}"
    lines.append(header)
    lines.append("  " + "-" * 68)

    for tso in tsos_with_data:
        aes   = [a for a in ae_flat_list  if a.tso == tso]
        cos   = [c for c in co_flat_list  if c.tso == tso]
        psts  = [p for p in pst_list      if p.tso == tso]
        rds   = [r for r in rd_list       if r.tso == tso]
        bids  = [b for b in bid_list      if b.ra_mrid in {r.mrid for r in rds}]

        sec = sum(1 for a in aes if a.ae_type == AEType.SECURED)
        scn = sum(1 for a in aes if a.ae_type == AEType.SCANNED)
        exc = sum(1 for a in aes if a.ae_type == AEType.EXCLUDED)
        must = sum(1 for c in cos if any(c.must_study.values()))

        lines.append(f"  {tso:<20} {sec:>6} {scn:>6} {exc:>6} {len(cos):>5} {must:>7} {len(psts):>5} {len(rds):>5} {len(bids):>5}" + (f" {sum(1 for h in hvdc_list if h.tso == tso):>5}" if hvdc_list is not None else ""))

    lines.append("-" * 72)
    sec_total = sum(1 for a in ae_flat_list if a.ae_type == AEType.SECURED)
    scn_total = sum(1 for a in ae_flat_list if a.ae_type == AEType.SCANNED)
    exc_total = sum(1 for a in ae_flat_list if a.ae_type == AEType.EXCLUDED)
    must_total = sum(1 for c in co_flat_list if any(c.must_study.values()))
    lines.append(f"  {'TOTAL':<20} {sec_total:>6} {scn_total:>6} {exc_total:>6} {len(co_flat_list):>5} {must_total:>7} {len(pst_list):>5} {len(rd_list):>5} {len(bid_list):>5}" + (f" {len(hvdc_list):>5}" if hvdc_list is not None else ""))
    lines.append("=" * 72)

    log.info("\n".join(lines))
