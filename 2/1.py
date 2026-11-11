"""
pst_support.py - the PSTs the analyses can handle. pypowsybl's sensitivity variable and tap changer update work on
2-winding transformers only, so a PST on a 3-winding transformer is left out of the PST list once, before every step.
"""

import dataclasses
import logging
from collections import defaultdict

from data_model import PST
from pre_milp import log_summary

log = logging.getLogger(__name__)


def _index_ids(read_table) -> set[str]:
    try:
        return {str(mrid) for mrid in read_table().index}
    except Exception:
        log.debug("exclude_three_winding_psts: a transformer table could not be read", exc_info=True)
        return set()


def three_winding_pst_ids(network, mrids) -> set[str]:
    """The mRIDs of `mrids` that are not 2-winding transformers but are 3-winding ones or have a phase tap changer.
    A mRID that is nowhere in the network is not one of them (it is reported where it is used)."""
    two_winding = _index_ids(network.get_2_windings_transformers)
    known = _index_ids(network.get_3_windings_transformers) | _index_ids(network.get_phase_tap_changers)
    return {str(mrid) for mrid in mrids if str(mrid) not in two_winding and str(mrid) in known}


def exclude_three_winding_psts(network, pst_list: list[PST]) -> list[PST]:
    """A new PST list without the PSTs on a 3-winding transformer (both rows of a transformer with a preventive and a
    curative remedial action). An excluded member leaves its group; a group left with fewer than 2 transformers is
    dissolved (its remaining PST gets no group). One summary warning, one warning per affected group."""
    excluded_ids = three_winding_pst_ids(network, {pst.mrid for pst in pst_list})
    if not excluded_ids:
        return list(pst_list)

    members_by_group: dict[str, set[str]] = defaultdict(set)
    for pst in pst_list:
        if pst.group_id:
            members_by_group[str(pst.group_id)].add(str(pst.mrid))

    kept = [pst for pst in pst_list if str(pst.mrid) not in excluded_ids]
    log_summary.warn_ids(
        log,
        "%d PST(s) on a 3-winding transformer excluded from every step (sensitivities, MILP, groups, apply, "
        "AC verification, outputs); the analyses support 2-winding transformers only",
        sorted(excluded_ids),
    )

    dissolved = set()
    for group_id, members in sorted(members_by_group.items()):
        lost = sorted(members & excluded_ids)
        if not lost:
            continue
        left = sorted(members - excluded_ids)
        if len(left) < 2:
            dissolved.add(group_id)
        log.warning(
            "PST group %s: member(s) %s excluded (3-winding transformer); %s",
            group_id, ", ".join(lost),
            f"{len(left)} member(s) left, group dissolved" if len(left) < 2 else f"{len(left)} members left",
        )
    return [
        dataclasses.replace(pst, group_id=None) if pst.group_id and str(pst.group_id) in dissolved else pst
        for pst in kept
    ]
