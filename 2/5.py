"""nx_constraint_filter.py - N-X flow rows that can never bind are not built (exact model reduction).

For a row  I + sum(coefficient x variable) <= limit + slack  the largest possible |flow| over all feasible
variable values follows from the variable bounds:  |I| + sum |coefficient| x (largest change the bounds allow).
When that bound is at most limit - margin the row is slack at every feasible point, so the row and its slack
variable leave the optimum, the objective and every variable unchanged. A variable without a finite bound
(None here) disables the skip for the row. A term with coefficient 0 is not a term.
"""

import logging
import math
from typing import Iterable, Optional

log = logging.getLogger(__name__)

KINDS = ("nx_patl", "nx_tatl", "nx_scanned")


def empty_skipped_pairs() -> dict[str, set]:
    return {kind: set() for kind in KINDS}


def flow_upper_bound_ampere(current_ampere: float, terms: Iterable[tuple[float, Optional[float]]]) -> Optional[float]:
    """|current| + sum |coefficient| x max_change over (coefficient, max_change) terms; None when a term with a
    non-zero coefficient has no finite max_change (variable without a finite bound)."""
    bound = abs(float(current_ampere))
    for coefficient, max_change in terms:
        if coefficient == 0.0:
            continue
        if max_change is None or not math.isfinite(max_change):
            return None
        bound += abs(float(coefficient)) * float(max_change)
    return bound


def can_never_bind(bound_ampere: Optional[float], limit_ampere: float, margin_ampere: float) -> bool:
    return bound_ampere is not None and bound_ampere <= float(limit_ampere) - float(margin_ampere)


def pairs_with_skipped(model, kind: str) -> set:
    """The pairs of a kind that were evaluated: the ones with a row plus the ones skipped as never binding."""
    names = {
        "nx_patl": "nx_branch_contingency_pairs",
        "nx_tatl": "nx_tatl_branch_contingency_pairs",
        "nx_scanned": "nx_scanned_branch_contingency_pairs",
    }
    pairs = set(getattr(model, names[kind], ()))
    pairs |= getattr(model, "nx_skipped_pairs", {}).get(kind, set())
    return pairs


def log_filter_summary(considered: int, added: int, enabled: bool, margin_ampere: float) -> None:
    skipped = considered - added
    share = (100.0 * skipped / considered) if considered else 0.0
    log.info(
        "N-X constraint filter (%s, margin %.1f A): considered %d, added %d, skipped %d (%.1f%%) N-X rows that can never bind",
        "on" if enabled else "off", margin_ampere, considered, added, skipped, share,
    )
