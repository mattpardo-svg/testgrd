"""lf_parameters.py

The one source of the pypowsybl load-flow parameters: every AC load flow and every
sensitivity analysis (N-0 and N-X, PST and RD) builds them from use_reactive_limits and
extra_lf_params (load_flow_config.yaml). A sensitivity analysis on other parameters would
linearise a different load flow and can return all-zero sensitivities.
"""

# OpenLoadFlow threadCount of the security and sensitivity analyses; set once per run by
# run_pipeline from performance.lf_thread_count. 1 passes nothing (the provider default).
LF_THREAD_COUNT = 1


def _thread_provider_parameters() -> dict:
    return {} if LF_THREAD_COUNT == 1 else {"threadCount": str(LF_THREAD_COUNT)}


def build_loadflow_parameters(use_reactive_limits: bool = False, extra_lf_params: dict = None, **overrides):
    """pp.loadflow.Parameters(use_reactive_limits, **extra_lf_params), with `overrides`
    (e.g. voltage_init_mode=PREVIOUS_VALUES for a warm start) applied on top.

    pypowsybl accepts a loadflow Parameters object directly wherever a sensitivity or
    security analysis takes parameters, so the same object type serves all of them."""
    try:
        import pypowsybl as pp
    except Exception as exc:
        raise ImportError("pypowsybl is required to build load-flow parameters") from exc
    return pp.loadflow.Parameters(
        use_reactive_limits=use_reactive_limits, **{**(extra_lf_params or {}), **overrides},
    )


def _analysis_provider_parameters() -> dict:
    """The provider parameters of the security and of the sensitivity analysis, the same for both: a contingency trips
    exactly the equipment its list names (contingencyPropagation=false; with propagation the trip also opens the bays,
    lines and loads behind the same bus, so lf_nx and the sensitivity reference would differ), and the thread count."""
    return {"contingencyPropagation": "false", **_thread_provider_parameters()}


def build_security_analysis_parameters(use_reactive_limits: bool = False, extra_lf_params: dict = None):
    """pp.security.Parameters for every security analysis (compute_lf_nx)."""
    try:
        import pypowsybl as pp
    except Exception as exc:
        raise ImportError("pypowsybl is required to build security-analysis parameters") from exc
    return pp.security.Parameters(
        load_flow_parameters=build_loadflow_parameters(use_reactive_limits, extra_lf_params),
        provider_parameters=_analysis_provider_parameters(),
    )


def build_sensitivity_parameters(use_reactive_limits: bool = False, extra_lf_params: dict = None, **overrides):
    """Parameters of every sensitivity analysis: the load-flow parameters of build_loadflow_parameters in
    pp.sensitivity.Parameters, with the same provider parameters as the security analysis."""
    import pypowsybl as pp
    return pp.sensitivity.Parameters(
        load_flow_parameters=build_loadflow_parameters(use_reactive_limits, extra_lf_params, **overrides),
        provider_parameters=_analysis_provider_parameters(),
    )
