"""Policy-aligned prediction-level auditing for frozen DTI predictors."""

from .components import (
    EVIDENCE_DIAGNOSTIC_PROFILES,
    PRIMARY_PROFILES,
    ProfileSpec,
    audit_capabilities,
    calibrated_error_risk,
    engineer_features,
    profile_manifest,
    profile_requirements_met,
)
from .metrics import calibration_metrics
from .modeling import (
    CalibrationResult,
    ProfileResult,
    cross_fitted_calibration,
    cross_fitted_profile,
    fit_calibrator,
    fit_profile_for_test,
)
from .policy import (
    apply_uncertainty_policy,
    evaluate_ranking,
    select_one_standard_error,
)

__all__ = [
    "EVIDENCE_DIAGNOSTIC_PROFILES",
    "PRIMARY_PROFILES",
    "CalibrationResult",
    "ProfileResult",
    "ProfileSpec",
    "apply_uncertainty_policy",
    "audit_capabilities",
    "calibrated_error_risk",
    "calibration_metrics",
    "cross_fitted_calibration",
    "cross_fitted_profile",
    "engineer_features",
    "evaluate_ranking",
    "fit_calibrator",
    "fit_profile_for_test",
    "profile_manifest",
    "profile_requirements_met",
    "select_one_standard_error",
]
