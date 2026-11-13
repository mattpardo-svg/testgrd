"""nx_results_export.py

N-X result files from the final model: results_nx_loadings.csv, results_contingency_status.csv
and results_nx_scanned_elements.csv. The linearised post-PRA / post-PRA+CRA loadings use the
model's variables and the N-X sensitivities with the same formula as the N-X constraints; the
actual post-PRA loading comes from the N-X load flow of the final applied state when given.
Corridor (PowerTransferCorridor) results are not built: no corridor data is parsed.
"""

import csv
import logging
from pathlib import Path
from typing import Optional

from data_model import AssessedElement, Contingency, select_by_side
from milp import nx_constraint_filter
from pre_milp import ct_sensitivity
from text_encoding import TEXT_ENCODING

log = logging.getLogger(__name__)


class _SolvedValues:
    """The solved values of the final model that the linearised effect reads, each evaluated once on first use
    (a Pyomo evaluation per branch and contingency row is the slow part of the export). The model is not changed
    while an export runs. Same arithmetic as reading the variables in the formula: up - down for a tap change, the
    value of the unit's net injection expression."""

    def __init__(self, model):
        self._model = model
        self._tap_change: dict[str, float] = {}
        self._rd_net_injection: dict[tuple[str, bool], float] = {}
        self._bid_volume: dict[str, float] = {}

    def bid_volume(self, bid_mrid: str) -> float:
        """up - down of a preventive bid of this solve."""
        if bid_mrid not in self._bid_volume:
            import pyomo.environ as pyo
            self._bid_volume[bid_mrid] = (
                pyo.value(self._model.rd_volume_upward[bid_mrid]) - pyo.value(self._model.rd_volume_downward[bid_mrid])
            )
        return self._bid_volume[bid_mrid]

    def tap_change(self, pst_mrid: str) -> float:
        if pst_mrid not in self._tap_change:
            import pyomo.environ as pyo
            self._tap_change[pst_mrid] = (
                pyo.value(self._model.tap_change_upward[pst_mrid]) - pyo.value(self._model.tap_change_downward[pst_mrid])
            )
        return self._tap_change[pst_mrid]

    def rd_net_injection(self, gu_mrid: str, rd_only: bool = False) -> float:
        """The unit's net injection; rd_only: its RD part, without the countertrade volumes."""
        if (gu_mrid, rd_only) not in self._rd_net_injection:
            import pyomo.environ as pyo
            expression = self._model.rd_only_net_injection_g if rd_only else self._model.rd_net_injection_g
            self._rd_net_injection[(gu_mrid, rd_only)] = pyo.value(expression[gu_mrid])
        return self._rd_net_injection[(gu_mrid, rd_only)]


def _linearised_pst_rd_effect(
    model,
    branch_mrid: str,
    contingency_mrid: str,
    sensitivity_pst_nx: dict,
    sensitivity_rd_nx: dict,
    pst_degrees_per_tap: dict,
    include_curative: bool,
    solved: Optional["_SolvedValues"] = None,
) -> float:
    """Evaluate the SAME linear PST+RD effect formula the N-X PATL/TATL
    constraints use (see milp_builder._nx_pst_and_rd_effect), reading
    SOLVED variable values from the final model — used to derive
    loading_pct_nx_post_pra (include_curative=False) and
    loading_pct_nx_post_pra_cra (include_curative=True).
    solved: the solved values read once per export (_SolvedValues); None reads them from the model each time.
    """
    solved = solved if solved is not None else _SolvedValues(model)

    sensitivity_for_pst = sensitivity_pst_nx.get(contingency_mrid, {})
    sensitivity_for_rd = sensitivity_rd_nx.get(contingency_mrid, {})

    effect = 0.0
    if hasattr(model, "tap_change_upward"):
        for pst_mrid in model.pst_ids:
            if pst_mrid not in pst_degrees_per_tap:
                continue
            sensitivity = float(sensitivity_for_pst.get((branch_mrid, pst_mrid), 0.0))
            if sensitivity == 0.0:
                continue
            effect += sensitivity * float(pst_degrees_per_tap[pst_mrid]) * solved.tap_change(pst_mrid)

    # curative terms from the builder's own evaluator (model.nx_curative_effect_ampere), so the
    # export applies the constraints' pair gating (curative magnitude triples, xne linkage)
    if include_curative and hasattr(model, "nx_curative_effect_ampere"):
        effect += model.nx_curative_effect_ampere(branch_mrid, contingency_mrid)

    if hasattr(model, "nx_hvdc_preventive_effect_ampere"):
        effect += model.nx_hvdc_preventive_effect_ampere(branch_mrid, contingency_mrid)

    if hasattr(model, "rd_net_injection_g"):
        # with countertrade bids: the RD part of the unit totals through the units, the CT volumes through the
        # sensitivity of their RA and direction (the same split as the constraints)
        ct_bid_ids = getattr(model, "ct_bid_ids", frozenset())
        for gu_mrid in (model.rd_effect_unit_ids if ct_bid_ids else model.gu_ids):
            sensitivity = float(sensitivity_for_rd.get((branch_mrid, gu_mrid), 0.0))
            if sensitivity == 0.0:
                continue
            effect += sensitivity * solved.rd_net_injection(gu_mrid, rd_only=bool(ct_bid_ids))
        for bid_mrid in sorted(ct_bid_ids):
            variable_id, weights = model.ct_bid_variable[bid_mrid]
            coefficient = ct_sensitivity.coefficient(sensitivity_for_rd, branch_mrid, variable_id, weights)
            if coefficient != 0.0:
                effect += coefficient * solved.bid_volume(bid_mrid)

    return effect


def export_nx_loadings_csv(
    model,
    ae_flat_list: list[AssessedElement],
    co_flat_list: list[Contingency],
    lf_nx_initial: dict[str, dict[str, dict[int, float]]],
    lf_initial: dict[str, dict[int, float]],
    sensitivity_pst_nx: dict[str, dict[tuple[str, str], float]],
    sensitivity_rd_nx: dict[str, dict[tuple[str, str], float]],
    pst_degrees_per_tap: dict[str, float],
    tatl_data: dict[str, dict],
    timestamp: str,
    output_path: Path,
    branch_mrid_to_name: Optional[dict[str, str]] = None,
    lf_nx_final: Optional[dict[str, dict[str, dict[int, float]]]] = None,
) -> None:
    """results_nx_loadings.csv — one row per (contingency, branch) in lf_nx_initial (the
    ORIGINAL pre-RA N-X currents). With lf_nx_final (the N-X load flow of the final applied
    state): post-PRA loading = that actual value, post-PRA+CRA = actual + curative effect (curative
    actions are never applied); the model's prediction (build baseline + its change) goes into
    loading_pct_nx_post_pra_model and the difference (actual - model, pp) into ..._model_diff_pct.
    Without lf_nx_final both loadings are linearised from lf_nx_initial and the difference is 0.
    """
    branch_mrid_to_name = branch_mrid_to_name or {}
    build_baseline = getattr(model, "nx_build_baseline", None)
    model_lf_nx = build_baseline["lf_nx"] if build_baseline else lf_nx_initial
    model_sensitivity_pst_nx = build_baseline["sensitivity_pst_nx"] if build_baseline else sensitivity_pst_nx
    model_sensitivity_rd_nx = build_baseline["sensitivity_rd_nx"] if build_baseline else sensitivity_rd_nx
    contingency_name_by_mrid = {c.mrid: c.name for c in co_flat_list}
    solved = _SolvedValues(model)
    skipped_pairs = getattr(model, "nx_skipped_pairs", {"nx_patl": set(), "nx_tatl": set()})
    branch_sides: dict[str, int] = {}
    branch_patl_ampere: dict[str, float] = {}
    branch_tso: dict[str, str] = {}
    for ae in ae_flat_list:
        if not ae.branch_mrid:
            continue
        branch_mrid = str(ae.branch_mrid)
        branch_sides[branch_mrid] = int(getattr(ae, "branch_side", 1))
        patl_ka = ae.patl_ka.get(timestamp)
        if patl_ka is not None:
            branch_patl_ampere[branch_mrid] = float(patl_ka) * 1000.0
        branch_tso[branch_mrid] = ae.tso

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding=TEXT_ENCODING) as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp", "contingency_mrid", "contingency_name", "branch_mrid", "branch_name",
            "tso", "side", "i_ka_n0", "i_ka_nx_initial", "loading_pct_nx_initial",
            "loading_pct_nx_post_pra", "loading_pct_nx_post_pra_cra",
            "patl_ka", "tatl_ka", "slack_patl", "slack_tatl",
            "loading_pct_nx_post_pra_model", "loading_pct_nx_post_pra_model_diff_pct",
        ])

        for contingency_mrid in sorted(lf_nx_initial):
            per_branch_currents = lf_nx_initial[contingency_mrid]
            for branch_mrid in sorted(per_branch_currents):
                side = branch_sides.get(branch_mrid, 1)
                patl_ampere = branch_patl_ampere.get(branch_mrid)

                i_ampere_n0 = select_by_side(lf_initial.get(branch_mrid, {}), side, branch_mrid)
                i_ka_n0 = i_ampere_n0 / 1000.0 if i_ampere_n0 is not None else ""

                i_ampere_nx_initial = select_by_side(per_branch_currents[branch_mrid], side, branch_mrid)
                if i_ampere_nx_initial is None:
                    continue
                i_ka_nx_initial = i_ampere_nx_initial / 1000.0

                loading_pct_nx_initial = (
                    (i_ampere_nx_initial / patl_ampere) * 100.0 if patl_ampere else ""
                )

                post_pra_effect = _linearised_pst_rd_effect(
                    model, branch_mrid, contingency_mrid,
                    model_sensitivity_pst_nx, model_sensitivity_rd_nx, pst_degrees_per_tap,
                    include_curative=False, solved=solved,
                )
                post_pra_cra_effect = _linearised_pst_rd_effect(
                    model, branch_mrid, contingency_mrid,
                    model_sensitivity_pst_nx, model_sensitivity_rd_nx, pst_degrees_per_tap,
                    include_curative=True, solved=solved,
                )
                # the model's prediction: the baseline it was built on + its own change
                model_baseline_ampere = select_by_side(
                    model_lf_nx.get(contingency_mrid, {}).get(branch_mrid, {}), side, branch_mrid,
                )
                if model_baseline_ampere is None:
                    model_baseline_ampere = i_ampere_nx_initial
                i_ampere_post_pra_model = model_baseline_ampere + post_pra_effect
                loading_pct_nx_post_pra_model = (
                    (i_ampere_post_pra_model / patl_ampere) * 100.0 if patl_ampere else ""
                )

                actual_ampere = None
                if lf_nx_final is not None:
                    actual_ampere = select_by_side(
                        lf_nx_final.get(contingency_mrid, {}).get(branch_mrid, {}), side, branch_mrid,
                    )
                if actual_ampere is not None:
                    i_ampere_post_pra = actual_ampere
                    i_ampere_post_pra_cra = actual_ampere + (post_pra_cra_effect - post_pra_effect)
                else:
                    i_ampere_post_pra = i_ampere_post_pra_model
                    i_ampere_post_pra_cra = model_baseline_ampere + post_pra_cra_effect

                loading_pct_nx_post_pra = (i_ampere_post_pra / patl_ampere) * 100.0 if patl_ampere else ""
                loading_pct_nx_post_pra_cra = (
                    (i_ampere_post_pra_cra / patl_ampere) * 100.0 if patl_ampere else ""
                )
                loading_pct_nx_post_pra_model_diff_pct = (
                    loading_pct_nx_post_pra - loading_pct_nx_post_pra_model if patl_ampere else ""
                )

                tatl_entry = tatl_data.get(branch_mrid)
                tatl_ka = float(tatl_entry["tatl_limit_ampere"]) / 1000.0 if tatl_entry else ""

                slack_patl = ""
                if hasattr(model, "nx_patl_overload_slack") and (branch_mrid, contingency_mrid) in model.nx_branch_contingency_pairs:
                    import pyomo.environ as pyo
                    slack_patl = pyo.value(model.nx_patl_overload_slack[branch_mrid, contingency_mrid])
                elif (branch_mrid, contingency_mrid) in skipped_pairs["nx_patl"]:
                    slack_patl = 0.0        # no row: it can never bind, so its slack is 0

                slack_tatl = ""
                if (
                    hasattr(model, "nx_tatl_overload_slack")
                    and (branch_mrid, contingency_mrid) in model.nx_tatl_branch_contingency_pairs
                ):
                    import pyomo.environ as pyo
                    slack_tatl = pyo.value(model.nx_tatl_overload_slack[branch_mrid, contingency_mrid])
                elif (branch_mrid, contingency_mrid) in skipped_pairs["nx_tatl"]:
                    slack_tatl = 0.0

                writer.writerow([
                    timestamp, contingency_mrid, contingency_name_by_mrid.get(contingency_mrid, contingency_mrid),
                    branch_mrid, branch_mrid_to_name.get(branch_mrid, branch_mrid),
                    branch_tso.get(branch_mrid, ""), side,
                    i_ka_n0, i_ka_nx_initial, loading_pct_nx_initial,
                    loading_pct_nx_post_pra, loading_pct_nx_post_pra_cra,
                    (patl_ampere / 1000.0) if patl_ampere else "", tatl_ka,
                    slack_patl, slack_tatl,
                    loading_pct_nx_post_pra_model, loading_pct_nx_post_pra_model_diff_pct,
                ])

    log.info("Written N-X loadings results CSV: %s", output_path)


def merge_skipped_contingency_reasons(
    initial_skipped: dict[str, str], refresh_skipped: list[dict[str, str]],
) -> dict[str, str]:
    """{contingency_mrid: reason} for results_contingency_status.csv: the reason from the first
    N-X computation, else the first reason a later N-X refresh (1-based) gave."""
    merged = dict(initial_skipped)
    for refresh_number, skipped in enumerate(refresh_skipped, start=1):
        for contingency_mrid, reason in skipped.items():
            merged.setdefault(contingency_mrid, f"N-X refresh {refresh_number}: {reason}")
    return merged


def export_contingency_status_csv(
    co_flat_list: list[Contingency],
    timestamp: str,
    lf_nx_initial: dict[str, dict],
    skipped_contingencies: dict[str, str],
    model,
    output_path: Path,
    invalid_sensitivity_contingencies: Optional[dict] = None,
    sensitivity_source_contingencies: Optional[dict] = None,
) -> None:
    """results_contingency_status.csv — one row per must-study contingency. converged: the
    contingency is in lf_nx_initial; skipped_reason from compute_lf_nx; included_in_milp: a PATL or
    TATL pair of it is in the final model; sensitivity_source "stale_reused" if a solve used its
    last valid set, else "fresh"; sensitivity_valid False if the guard ever flagged it.
    """
    invalid_sensitivity_contingencies = invalid_sensitivity_contingencies or {}
    sensitivity_source_contingencies = sensitivity_source_contingencies or {}
    must_study_contingencies = [
        c for c in co_flat_list if c.must_study.get(timestamp, False)
    ]

    patl_pairs = nx_constraint_filter.pairs_with_skipped(model, "nx_patl")
    tatl_pairs = nx_constraint_filter.pairs_with_skipped(model, "nx_tatl")
    contingencies_in_milp = {co_mrid for _branch, co_mrid in patl_pairs} | {co_mrid for _branch, co_mrid in tatl_pairs}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding=TEXT_ENCODING) as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "contingency_mrid", "converged", "skipped_reason", "included_in_milp", "sensitivity_valid", "sensitivity_source"])
        for contingency in sorted(must_study_contingencies, key=lambda c: c.mrid):
            converged = contingency.mrid in lf_nx_initial
            skipped_reason = skipped_contingencies.get(contingency.mrid, "")
            if not converged and not skipped_reason:
                skipped_reason = "no load flow result (reason not recorded)"
            dropped = getattr(lf_nx_initial, "dropped", {}).get(contingency.mrid, [])
            if converged and not skipped_reason and dropped:
                # evaluated, without the monitored branches that have no result after it (the header stays as it is)
                skipped_reason = f"evaluated; {len(dropped)} monitored branch(es) dropped (no result after the contingency)"
            included_in_milp = contingency.mrid in contingencies_in_milp
            writer.writerow([
                timestamp, contingency.mrid, converged, skipped_reason, included_in_milp,
                contingency.mrid not in invalid_sensitivity_contingencies,
                sensitivity_source_contingencies.get(contingency.mrid, "fresh"),
            ])

    log.info("Written contingency status results CSV: %s", output_path)


def export_nx_scanned_elements_csv(
    model,
    ae_flat_list: list[AssessedElement],
    co_flat_list: list[Contingency],
    lf_nx_initial: dict[str, dict[str, dict[int, float]]],
    sensitivity_pst_nx: dict[str, dict[tuple[str, str], float]],
    sensitivity_rd_nx: dict[str, dict[tuple[str, str], float]],
    pst_degrees_per_tap: dict[str, float],
    branch_limits: dict[str, float],
    timestamp: str,
    output_path: Path,
    branch_mrid_to_name: Optional[dict[str, str]] = None,
    lf_nx_final: Optional[dict[str, dict[str, dict[int, float]]]] = None,
) -> None:
    """results_nx_scanned_elements.csv — one row per pair in nx_scanned_branch_contingency_pairs
    (header only when there is none): [timestamp, contingency_mrid, branch_mrid, branch_name, tso,
    side, effective_limit_ka, loading_before_rao_ka, loading_after_rao_ka, slack_ka,
    loading_after_model_ka]. before = ORIGINAL lf_nx_initial; after = actual flow at the final
    state (lf_nx_final) + curative effect, else build baseline + full linearised effect;
    model = build baseline + full linearised effect. branch_limits holds the effective limits.
    """
    import pyomo.environ as pyo

    branch_mrid_to_name = branch_mrid_to_name or {}
    build_baseline = getattr(model, "nx_build_baseline", None)
    model_lf_nx = build_baseline["lf_nx"] if build_baseline else lf_nx_initial
    model_sensitivity_pst_nx = build_baseline["sensitivity_pst_nx"] if build_baseline else sensitivity_pst_nx
    model_sensitivity_rd_nx = build_baseline["sensitivity_rd_nx"] if build_baseline else sensitivity_rd_nx
    branch_sides: dict[str, int] = {}
    branch_tso: dict[str, str] = {}
    for ae in ae_flat_list:
        if not ae.branch_mrid:
            continue
        branch_mrid = str(ae.branch_mrid)
        branch_sides[branch_mrid] = int(getattr(ae, "branch_side", 1))
        branch_tso[branch_mrid] = ae.tso

    solved = _SolvedValues(model)
    scanned_pairs = nx_constraint_filter.pairs_with_skipped(model, "nx_scanned")
    skipped_scanned_pairs = getattr(model, "nx_skipped_pairs", {}).get("nx_scanned", set())
    pair_limits = getattr(model, "nx_scanned_pair_limit_ampere", {})

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding=TEXT_ENCODING) as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp", "contingency_mrid", "branch_mrid", "branch_name", "tso", "side",
            "effective_limit_ka", "loading_before_rao_ka", "loading_after_rao_ka", "slack_ka",
            "loading_after_model_ka",
        ])

        for branch_mrid, contingency_mrid in sorted(scanned_pairs):
            per_branch_currents = lf_nx_initial.get(contingency_mrid, {}).get(branch_mrid)
            if not per_branch_currents:
                continue

            side = branch_sides.get(branch_mrid, 1)
            i_ampere_before = select_by_side(per_branch_currents, side, branch_mrid)
            if i_ampere_before is None:
                continue
            loading_before_rao_ka = i_ampere_before / 1000.0

            # the N-X limit of the pair: max(PATL, its original post-contingency loading)
            effective_limit_ampere = pair_limits.get((branch_mrid, contingency_mrid), branch_limits.get(branch_mrid))

            full_effect = _linearised_pst_rd_effect(
                model, branch_mrid, contingency_mrid,
                model_sensitivity_pst_nx, model_sensitivity_rd_nx, pst_degrees_per_tap,
                include_curative=True, solved=solved,
            )
            preventive_effect = _linearised_pst_rd_effect(
                model, branch_mrid, contingency_mrid,
                model_sensitivity_pst_nx, model_sensitivity_rd_nx, pst_degrees_per_tap,
                include_curative=False, solved=solved,
            )
            model_baseline_ampere = select_by_side(
                model_lf_nx.get(contingency_mrid, {}).get(branch_mrid, {}), side, branch_mrid,
            )
            if model_baseline_ampere is None:
                model_baseline_ampere = i_ampere_before
            loading_after_model_ka = (model_baseline_ampere + full_effect) / 1000.0

            actual_ampere = None
            if lf_nx_final is not None:
                actual_ampere = select_by_side(
                    lf_nx_final.get(contingency_mrid, {}).get(branch_mrid, {}), side, branch_mrid,
                )
            if actual_ampere is not None:
                loading_after_rao_ka = (actual_ampere + (full_effect - preventive_effect)) / 1000.0
            else:
                loading_after_rao_ka = loading_after_model_ka

            slack_ka = ""
            if hasattr(model, "nx_scanned_overload_slack"):
                if (branch_mrid, contingency_mrid) in skipped_scanned_pairs:
                    slack_ka = 0.0        # no row: it can never bind, so its slack is 0
                else:
                    slack_ka = pyo.value(model.nx_scanned_overload_slack[branch_mrid, contingency_mrid]) / 1000.0

            writer.writerow([
                timestamp, contingency_mrid, branch_mrid,
                branch_mrid_to_name.get(branch_mrid, branch_mrid),
                branch_tso.get(branch_mrid, ""), side,
                effective_limit_ampere / 1000.0 if effective_limit_ampere else "",
                loading_before_rao_ka, loading_after_rao_ka, slack_ka,
                loading_after_model_ka,
            ])

    log.info("Written scanned-element N-X results CSV: %s", output_path)
