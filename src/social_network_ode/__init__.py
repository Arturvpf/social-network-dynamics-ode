"""Nonlinear ODE framework for modeling user behavior dynamics in social networks."""

from .model import (
    DEFAULT_INITIAL_STATE,
    DEFAULT_PARAMS,
    PRESET_REGIMES,
    ModelParameters,
    SocialNetworkODE,
)
from .stability import (
    EquilibriumReport,
    RouthHurwitzResult,
    activation_number,
    analyze_equilibria,
    critical_parameters,
    find_nontrivial_equilibrium,
    routh_hurwitz_3d,
)
from .solvers import (
    ConvergenceRow,
    SimulationResult,
    benchmark_convergence,
    dopri45_adaptive,
    euler_explicit,
    heun_rk2,
    rk4_classic,
    simulate,
)
from .sensitivity import (
    BifurcationSweepResult,
    SensitivitySweepResult,
    bifurcation_sweep,
    compute_elasticities,
    initial_condition_ensemble,
    parameter_trajectory_sweep,
)
from .calibration import (
    CalibrationResult,
    calibrate_parameters,
    generate_synthetic_observations,
)
from .extensions import (
    FatigueModelParameters,
    NetworkODEModel,
    PolicyEvaluation,
    SaturationVariant,
    evaluate_recommendation_policies,
    simulate_fatigue_model,
    simulate_saturation_variant,
)

__version__ = "1.0.0"

__all__ = [
    "__version__",
    "DEFAULT_INITIAL_STATE",
    "DEFAULT_PARAMS",
    "PRESET_REGIMES",
    "ModelParameters",
    "SocialNetworkODE",
    "EquilibriumReport",
    "RouthHurwitzResult",
    "activation_number",
    "analyze_equilibria",
    "critical_parameters",
    "find_nontrivial_equilibrium",
    "routh_hurwitz_3d",
    "ConvergenceRow",
    "SimulationResult",
    "benchmark_convergence",
    "dopri45_adaptive",
    "euler_explicit",
    "heun_rk2",
    "rk4_classic",
    "simulate",
    "BifurcationSweepResult",
    "SensitivitySweepResult",
    "bifurcation_sweep",
    "compute_elasticities",
    "initial_condition_ensemble",
    "parameter_trajectory_sweep",
    "CalibrationResult",
    "calibrate_parameters",
    "generate_synthetic_observations",
    "FatigueModelParameters",
    "NetworkODEModel",
    "PolicyEvaluation",
    "SaturationVariant",
    "evaluate_recommendation_policies",
    "simulate_fatigue_model",
    "simulate_saturation_variant",
]
