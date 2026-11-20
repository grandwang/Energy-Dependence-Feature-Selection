"""R22OX-Reduced v0.1.0: completed Energy relevance and raw-LW log-volume.

Standalone extraction of the frozen N0/G0/C0 execution path. Arithmetic,
log-gain ties, numerical guards and reference-shrinkage replay are retained.
Inactive rank/interaction computations and ancestor metadata are not exported.
"""
from __future__ import annotations
from typing import Any, Optional
import numpy as np
from sklearn.covariance import ledoit_wolf_shrinkage

METHOD_ID = 'r22ox_reduced_n0g0c0_r1'
VERSION = 'v0.1.0'
FORMULATION_ID = 'COMPLETED_UENERGY_A_SCALE_RAW_LW_A_LOGDET_N0G0C0_R1'
_LW_BLOCK_SIZE = 256
_NUMERICAL_GUARD_MULTIPLIER = 512.0
RISK_CALIBRATION_CONSTANT = 1.0 / 8.0
MAIN_POPULATION_ID = 'STANDARDIZED_CLASS_PAIR_ENERGY'
MAIN_ESTIMATOR_ID = 'COMPLETED_U_ENERGY_AGGREGATE_POSITIVE_PART'

def _validate_matrix(name: str, value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional matrix")
    if array.shape[0] < 1 or array.shape[1] < 1:
        raise ValueError(f"{name} must be non-empty")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array

def _within_absolute_offdiagonal_mean(values: np.ndarray) -> np.ndarray:
    array = np.sort(np.asarray(values, dtype=np.float64), axis=0, kind="mergesort")
    n = array.shape[0]
    if n < 2:
        raise ValueError("unbiased within-class distance requires n>=2")
    coefficients = (2.0 * np.arange(n, dtype=np.float64) - n + 1.0)[:, None]
    UNordered_sUm = np.sum(array * coefficients, axis=0)
    return 2.0 * UNordered_sUm / (n * (n - 1.0))

def _cross_absolute_mean(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    x = np.sort(np.asarray(left, dtype=np.float64), axis=0, kind="mergesort")
    y = np.sort(np.asarray(right, dtype=np.float64), axis=0, kind="mergesort")
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1]:
        raise ValueError("left/right must be 2D with equal feature counts")
    result = np.zeros(x.shape[1], dtype=np.float64)
    for feature in range(x.shape[1]):
        right_values = y[:, feature]
        prefix = np.concatenate(
            (np.asarray([0.0]), np.cumsum(right_values, dtype=np.float64))
        )
        indices = np.searchsorted(right_values, x[:, feature], side="right")
        left_sum = (
            indices.astype(np.float64) * x[:, feature] - prefix[indices]
        )
        right_sum = (
            prefix[-1]
            - prefix[indices]
            - (right_values.size - indices).astype(np.float64) * x[:, feature]
        )
        result[feature] = float(np.sum(left_sum + right_sum)) / (
            x.shape[0] * y.shape[0]
        )
    return result

def energy_distance_signed_coordinates(
    left: np.ndarray,
    right: np.ndarray,
) -> np.ndarray:
    """Unbiased univariate energy-distance coordinates."""

    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1]:
        raise ValueError("left/right must be 2D with equal feature counts")
    if x.shape[0] < 2 or y.shape[0] < 2:
        raise ValueError("energy U-stat requires two samples per class")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("left/right must be finite")
    return (
        2.0 * _cross_absolute_mean(x, y)
        - _within_absolute_offdiagonal_mean(x)
        - _within_absolute_offdiagonal_mean(y)
    )

def _all_absolute_mean(values: np.ndarray) -> np.ndarray:
    array = np.sort(np.asarray(values, dtype=np.float64), axis=0, kind="mergesort")
    if array.ndim != 2 or array.shape[0] < 1:
        raise ValueError("values must have shape (n>=1,p)")
    n = array.shape[0]
    coefficients = (2.0 * np.arange(n, dtype=np.float64) - n + 1.0)[:, None]
    UNordered_sUm = np.sum(array * coefficients, axis=0)
    return 2.0 * UNordered_sUm / float(n * n)

def _offdiagonal_absolute_mean(values: np.ndarray) -> np.ndarray:
    array = np.sort(np.asarray(values, dtype=np.float64), axis=0, kind="mergesort")
    if array.ndim != 2 or array.shape[0] < 2:
        raise ValueError("off-diagonal mean requires shape (n>=2,p)")
    n = array.shape[0]
    coefficients = (2.0 * np.arange(n, dtype=np.float64) - n + 1.0)[:, None]
    UNordered_sUm = np.sum(array * coefficients, axis=0)
    return 2.0 * UNordered_sUm / float(n * (n - 1))

def energy_distance_biased_coordinates(
    left: np.ndarray,
    right: np.ndarray,
) -> np.ndarray:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1]:
        raise ValueError("left/right must be 2D with equal feature counts")
    if x.shape[0] < 1 or y.shape[0] < 1:
        raise ValueError("each class must contain at least one observation")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("left/right must be finite")
    return 2.0 * _cross_absolute_mean(x, y) - _all_absolute_mean(x) - _all_absolute_mean(y)

def conditional_random_label_energy_null_mean_coordinates(
    left: np.ndarray,
    right: np.ndarray,
) -> np.ndarray:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1]:
        raise ValueError("left/right must be 2D with equal feature counts")
    if x.shape[0] < 1 or y.shape[0] < 1:
        raise ValueError("each class must contain at least one observation")
    UNioN = np.vstack((x, y))
    if UNioN.shape[0] == 2:
        offdiagonal = np.abs(UNioN[0] - UNioN[1])
    else:
        offdiagonal = _offdiagonal_absolute_mean(UNioN)
    return (1.0 / x.shape[0] + 1.0 / y.shape[0]) * offdiagonal

def null_centered_energy_coordinates(
    left: np.ndarray,
    right: np.ndarray,
) -> np.ndarray:
    return energy_distance_biased_coordinates(
        left, right
    ) - conditional_random_label_energy_null_mean_coordinates(left, right)

def dual_ledoit_wolf_shrinkage(centered: np.ndarray) -> float:
    """Compute sklearn's Ledoit-Wolf scalar in sample-Gram coordinates."""

    rows = _validate_matrix("centered", centered)
    N_samples, N_featUres = rows.shape
    if N_featUres == 1:
        return 0.0
    squared = rows * rows
    empirical_trace = np.sum(squared, axis=0) / float(N_samples)
    mu = float(np.sum(empirical_trace) / float(N_featUres))
    Beta_raw = float(np.sum(np.sum(squared, axis=1) ** 2))
    sample_gram = rows @ rows.T
    delta_raw = float(np.sum(sample_gram * sample_gram)) / float(
        N_samples * N_samples
    )
    Beta = (
        Beta_raw / float(N_samples) - delta_raw
    ) / float(N_featUres * N_samples)
    delta = (
        delta_raw
        - 2.0 * mu * float(np.sum(empirical_trace))
        + float(N_featUres) * mu * mu
    ) / float(N_featUres)
    Beta = min(Beta, delta)
    if Beta == 0.0 or delta <= 0.0:
        return 0.0
    return float(np.clip(Beta / delta, 0.0, 1.0))

def reference_ledoit_wolf_shrinkage(centered: np.ndarray) -> float:
    """Return the frozen blocked reference scalar."""

    rows = _validate_matrix("centered", centered)
    if rows.shape[1] == 1:
        return 0.0
    return float(
        np.clip(
            ledoit_wolf_shrinkage(
                rows,
                assume_centered=True,
                block_size=_LW_BLOCK_SIZE,
            ),
            0.0,
            1.0,
        )
    )

def _dual_alpha_error_bound(
    centered: np.ndarray,
    alpha: float,
) -> float:
    """Conservative roundoff envelope for the dual scalar."""

    N_samples, N_featUres = centered.shape
    operatioN_coUNt = max(
        1,
        N_featUres + N_samples * N_samples,
    )
    BoUNd = (
        _NUMERICAL_GUARD_MULTIPLIER
        * np.finfo(np.float64).eps
        * float(operatioN_coUNt)
        * max(1.0, abs(float(alpha)))
    )
    return float(min(1.0, BoUNd))

class R22OXReduced:
    """fit(X_l, y_l, X_u=None), then select(k); zero-based feature indices."""
    def __init__(self):
        self._fitted = False
        self.selected_features_ = None

    def _ensure_fitted(self):
        if not self._fitted:
            raise RuntimeError('Call fit before select or transform')

    def public_params(self):
        return dict(method_id=METHOD_ID, version=VERSION, formulation_id=FORMULATION_ID,
                    dtype='float64', behavioral_hyperparameter_count=0)

    def transform(self, X):
        self._ensure_fitted()
        if self.selected_features_ is None:
            raise RuntimeError('Call select before transform')
        return np.asarray(X)[:, self.selected_features_]

    def _edge_eta_column(self, feature_index):
        return np.zeros(self.n_features_in_, dtype=np.float64)

    def fit(
        self,
        X_l: np.ndarray,
        y_l: np.ndarray,
        X_u: Optional[np.ndarray] = None,
    ) -> "R22OXReduced":
        laBeled = _validate_matrix("X_l", X_l)
        laBels_iNpUt = np.asarray(y_l).reshape(-1)
        if laBels_iNpUt.shape[0] != laBeled.shape[0]:
            raise ValueError("X_l and y_l must have equal sample counts")
        if X_u is None:
            UNlaBeled = np.empty(
                (0, laBeled.shape[1]), dtype=np.float64
            )
        else:
            UNlaBeled = np.asarray(X_u, dtype=np.float64)
            if (
                UNlaBeled.ndim != 2
                or UNlaBeled.shape[1] != laBeled.shape[1]
                or not np.all(np.isfinite(UNlaBeled))
            ):
                raise ValueError(
                    "X_u must be finite and match the X_l feature count"
                )

        available = np.vstack((laBeled, UNlaBeled))
        centered = available - available.mean(axis=0, keepdims=True)
        scale = np.sqrt(np.mean(centered * centered, axis=0))
        nonconstant = scale > np.sqrt(np.finfo(np.float64).eps)
        standardized = np.zeros_like(available)
        standardized[:, nonconstant] = centered[:, nonconstant] / scale[nonconstant]
        self._rcenergy_labeled_rows_ = np.asarray(
            standardized[: laBeled.shape[0]], dtype=np.float64, order="C"
        )
        self._rcenergy_nonconstant_ = nonconstant

        available = np.vstack((laBeled, UNlaBeled))
        centered = available - available.mean(axis=0, keepdims=True)
        pooled_variance = np.mean(centered * centered, axis=0)
        relevance, class_diagnostics = self._transport_relevance(
            laBeled,
            laBels_iNpUt,
            pooled_variance,
        )

        all_constant = np.all(
            pooled_variance <= np.finfo(np.float64).eps
        )
        if all_constant:
            shrinkage = 1.0
            route = "constant_reference"
            dual_error_bound = 0.0
        elif centered.shape[1] > centered.shape[0]:
            shrinkage = dual_ledoit_wolf_shrinkage(centered)
            route = "sample_gram_dual"
            dual_error_bound = _dual_alpha_error_bound(
                centered,
                shrinkage,
            )
        else:
            shrinkage = reference_ledoit_wolf_shrinkage(centered)
            route = "blocked_reference"
            dual_error_bound = 0.0

        self._labeled_input_ = np.asarray(
            laBeled,
            dtype=np.float64,
            order="C",
        )
        self._labels_input_ = laBels_iNpUt.copy()
        self._centered_available_ = np.asarray(
            centered,
            dtype=np.float64,
            order="C",
        )
        self.relevance_ = np.asarray(relevance, dtype=np.float64)
        self.sqrt_relevance_ = np.sqrt(self.relevance_)
        self.shrinkage_ = float(np.clip(shrinkage, 0.0, 1.0))
        self.dual_shrinkage_error_bound_ = dual_error_bound
        self.ledoit_wolf_route_ = route
        self.covariance_trace_mean_ = float(
            np.mean(pooled_variance)
        )
        self._pooled_variance_ = np.asarray(
            pooled_variance,
            dtype=np.float64,
        )
        self._refresh_shrunk_diagonal()
        self.n_features_in_ = int(laBeled.shape[1])
        self.n_labeled_ = int(laBeled.shape[0])
        self.n_unlabeled_ = int(UNlaBeled.shape[0])
        self.class_diagnostics_ = class_diagnostics

        self.numerical_fallback_count_ = 0
        self.edge_column_query_count_total_ = 0
        self.reference_fallback_used_ = False
        self._using_reference_authority_ = route != "sample_gram_dual"
        self._reset_selection_state()
        self._fitted = True
        return self

    def _transport_relevance(
        self,
        X_l: np.ndarray,
        y_l: np.ndarray,
        pooled_variance: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        del X_l, pooled_variance
        labels, inverse, counts = np.unique(
            np.asarray(y_l).reshape(-1), return_inverse=True, return_counts=True
        )
        probabilities = counts.astype(np.float64) / float(counts.sum())
        denominator = sum(
            float(probabilities[left] * probabilities[right])
            for left in range(labels.size)
            for right in range(left + 1, labels.size)
        )
        signed = np.zeros(self._rcenergy_labeled_rows_.shape[1], dtype=np.float64)
        pair_minima: list[float] = []
        pair_maxima: list[float] = []
        ustat_pairs = 0
        singleton_completion_pairs = 0
        if denominator > 0.0:
            for left in range(labels.size):
                for right in range(left + 1, labels.size):
                    weight = float(probabilities[left] * probabilities[right]) / denominator
                    left_rows = self._rcenergy_labeled_rows_[inverse == left]
                    right_rows = self._rcenergy_labeled_rows_[inverse == right]
                    if left_rows.shape[0] >= 2 and right_rows.shape[0] >= 2:
                        pair_signed = (
                            RISK_CALIBRATION_CONSTANT
                            * energy_distance_signed_coordinates(left_rows, right_rows)
                        )
                        ustat_pairs += 1
                    else:
                        pair_signed = (
                            RISK_CALIBRATION_CONSTANT
                            * null_centered_energy_coordinates(left_rows, right_rows)
                        )
                        singleton_completion_pairs += 1
                    signed += weight * pair_signed
                    pair_minima.append(float(np.min(pair_signed)))
                    pair_maxima.append(float(np.max(pair_signed)))
        signed[~self._rcenergy_nonconstant_] = 0.0
        relevance = np.maximum(signed, 0.0)
        return relevance, {
            "main_population_id": MAIN_POPULATION_ID,
            "main_estimator_id": MAIN_ESTIMATOR_ID,
            "risk_calibration_constant": RISK_CALIBRATION_CONSTANT,
            "observed_class_count": int(labels.size),
            "observed_class_counts": counts.astype(int).tolist(),
            "resolved_class_pair_count": int(ustat_pairs + singleton_completion_pairs),
            "unresolved_class_pair_count": 0,
            "ustat_class_pair_count": int(ustat_pairs),
            "singleton_completion_class_pair_count": int(singleton_completion_pairs),
            "signed_coordinate_min": float(np.min(signed)),
            "signed_coordinate_max": float(np.max(signed)),
            "positive_part_clipped_coordinate_count": int(np.sum(signed < 0.0)),
            "pair_signed_min": min(pair_minima, default=0.0),
            "pair_signed_max": max(pair_maxima, default=0.0),
            "singleton_closed": True,
            "branch_rule": "USE_USTAT_IFF_BOTH_CLASS_COUNTS_AT_LEAST_TWO_ELSE_NULL_CENTERED_COMPLETION",
        }

    def _refresh_shrunk_diagonal(self) -> None:
        self.shrunk_diagonal_ = (
            (1.0 - self.shrinkage_) * self._pooled_variance_
            + self.shrinkage_ * self.covariance_trace_mean_
        )
        self.shrunk_diagonal_ = np.maximum(
            self.shrunk_diagonal_,
            np.finfo(np.float64).eps,
        )

    def _reset_selection_state(self) -> None:
        feature_count = int(self.relevance_.size)
        self._selection_order: list[int] = []
        self._marginal_gains: list[float] = []
        self._objective_values: list[float] = []
        self._main_marginal_gains: list[float] = []
        self._relation_marginal_gains: list[float] = []
        self._second_best_total_gains: list[float] = []
        self._decision_interval_overlaps: list[bool] = []
        self._correlation_columns = np.empty(
            (feature_count, 0),
            dtype=np.float64,
        )
        self._cholesky = np.empty((0, 0), dtype=np.float64)
        self._relation_marginal = np.zeros(
            feature_count,
            dtype=np.float64,
        )
        self._selected_edge_columns: list[np.ndarray] = []
        self.selected_features_ = None

    def _correlation_column_for_shrinkage(
        self,
        feature_index: int,
        shrinkage: float,
    ) -> np.ndarray:
        alpha = float(np.clip(shrinkage, 0.0, 1.0))
        centered = self._centered_available_
        covariance = (
            (1.0 - alpha)
            * (centered.T @ centered[:, feature_index])
            / float(centered.shape[0])
        )
        covariance[feature_index] += (
            alpha * self.covariance_trace_mean_
        )
        diagonal = (
            (1.0 - alpha) * self._pooled_variance_
            + alpha * self.covariance_trace_mean_
        )
        diagonal = np.maximum(diagonal, np.finfo(np.float64).eps)
        denominator = np.sqrt(
            diagonal * diagonal[feature_index]
        )
        correlation = np.clip(
            covariance / denominator,
            -1.0,
            1.0,
        )
        correlation[feature_index] = 1.0
        return correlation

    def _correlation_column(self, feature_index: int) -> np.ndarray:
        return self._correlation_column_for_shrinkage(
            feature_index,
            self.shrinkage_,
        )

    def edge_log_weight_column(
        self,
        feature_index: int,
    ) -> np.ndarray:
        self._ensure_fitted()
        if (
            feature_index < 0
            or feature_index >= self.n_features_in_
        ):
            raise IndexError("feature_index out of range")
        self.edge_column_query_count_total_ += 1
        return np.log1p(self._edge_eta_column(feature_index))

    def _main_log_gains(self) -> np.ndarray:
        selected_count = len(self._selection_order)
        if selected_count == 0:
            schur = 1.0 + self.relevance_
        else:
            selected = np.asarray(
                self._selection_order,
                dtype=np.int64,
            )
            weighted_cross = (
                self._correlation_columns
                * self.sqrt_relevance_[:, None]
                * self.sqrt_relevance_[selected][None, :]
            )
            solved = np.linalg.solve(
                self._cholesky,
                weighted_cross.T,
            )
            schur = 1.0 + self.relevance_ - np.einsum(
                "ij,ij->j",
                solved,
                solved,
                optimize=True,
            )
        return np.log(
            np.maximum(schur, np.finfo(np.float64).eps)
        )

    def _main_log_gains_for_shrinkage(
        self,
        shrinkage: float,
    ) -> np.ndarray:
        selected_count = len(self._selection_order)
        if selected_count == 0:
            return np.log1p(self.relevance_)
        selected = np.asarray(
            self._selection_order,
            dtype=np.int64,
        )
        correlation_columns = np.column_stack(
            [
                self._correlation_column_for_shrinkage(
                    int(index),
                    shrinkage,
                )
                for index in selected
            ]
        )
        selected_correlation = correlation_columns[selected, :]
        selected_kernel = (
            selected_correlation
            * self.sqrt_relevance_[selected, None]
            * self.sqrt_relevance_[selected][None, :]
        )
        cholesky = np.linalg.cholesky(
            np.eye(selected_count, dtype=np.float64)
            + selected_kernel
        )
        weighted_cross = (
            correlation_columns
            * self.sqrt_relevance_[:, None]
            * self.sqrt_relevance_[selected][None, :]
        )
        solved = np.linalg.solve(cholesky, weighted_cross.T)
        schur = 1.0 + self.relevance_ - np.einsum(
            "ij,ij->j",
            solved,
            solved,
            optimize=True,
        )
        return np.log(
            np.maximum(schur, np.finfo(np.float64).eps)
        )

    def _decision_intervals(
        self,
        total_gains: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self._using_reference_authority_:
            return total_gains.copy(), total_gains.copy()
        error = self.dual_shrinkage_error_bound_
        lower_alpha = max(0.0, self.shrinkage_ - error)
        upper_alpha = min(1.0, self.shrinkage_ + error)
        lower_route = (
            self._main_log_gains_for_shrinkage(lower_alpha)
            + self._relation_marginal
        )
        upper_route = (
            self._main_log_gains_for_shrinkage(upper_alpha)
            + self._relation_marginal
        )
        finite = (
            np.isfinite(total_gains)
            & np.isfinite(lower_route)
            & np.isfinite(upper_route)
        )
        safe_total = np.where(finite, total_gains, 0.0)
        safe_lower = np.where(finite, lower_route, 0.0)
        safe_upper = np.where(finite, upper_route, 0.0)
        scale = np.maximum(
            1.0,
            np.maximum(
                np.abs(safe_total),
                np.maximum(
                    np.abs(safe_lower),
                    np.abs(safe_upper),
                ),
            ),
        )
        rounding = (
            _NUMERICAL_GUARD_MULTIPLIER
            * np.finfo(np.float64).eps
            * float(
                max(
                    1,
                    self._centered_available_.shape[0],
                    self.n_features_in_,
                    len(self._selection_order) + 1,
                )
            )
            * scale
        )
        lower = np.minimum(safe_lower, safe_upper) - rounding
        upper = np.maximum(safe_lower, safe_upper) + rounding
        lower[~finite] = -np.inf
        upper[~finite] = -np.inf
        return lower, upper

    def _restart_with_reference(self) -> None:
        self.shrinkage_ = reference_ledoit_wolf_shrinkage(
            self._centered_available_
        )
        self._refresh_shrunk_diagonal()
        self.ledoit_wolf_route_ = "dual_overlap_blocked_reference_replay"
        self._using_reference_authority_ = True
        self.reference_fallback_used_ = True
        self.numerical_fallback_count_ += 1
        self._reset_selection_state()

    def _append_main_state(self, chosen: int) -> None:
        selected_count = len(self._selection_order)
        if selected_count == 0:
            new_cholesky = np.asarray(
                [[np.sqrt(1.0 + self.relevance_[chosen])]],
                dtype=np.float64,
            )
        else:
            previous = np.asarray(
                self._selection_order,
                dtype=np.int64,
            )
            cross = (
                self.sqrt_relevance_[chosen]
                * self.sqrt_relevance_[previous]
                * self._correlation_columns[chosen]
            )
            projected = np.linalg.solve(self._cholesky, cross)
            diagonal_squared = (
                1.0
                + self.relevance_[chosen]
                - float(projected @ projected)
            )
            diagonal = np.sqrt(
                max(
                    diagonal_squared,
                    np.finfo(np.float64).eps,
                )
            )
            new_cholesky = np.zeros(
                (selected_count + 1, selected_count + 1),
                dtype=np.float64,
            )
            new_cholesky[:selected_count, :selected_count] = (
                self._cholesky
            )
            new_cholesky[selected_count, :selected_count] = projected
            new_cholesky[selected_count, selected_count] = diagonal
        self._correlation_columns = np.column_stack(
            (
                self._correlation_columns,
                self._correlation_column(chosen),
            )
        )
        self._cholesky = new_cholesky

    def _extend_greedy_prefix(self, target_size: int) -> None:
        self._ensure_fitted()
        if target_size < 0 or target_size > self.n_features_in_:
            raise ValueError(
                f"target_size must be between 0 and {self.n_features_in_}"
            )
        while len(self._selection_order) < target_size:
            selected_mask = np.zeros(
                self.n_features_in_,
                dtype=bool,
            )
            if self._selection_order:
                selected_mask[
                    np.asarray(
                        self._selection_order,
                        dtype=np.int64,
                    )
                ] = True
            main_gains = self._main_log_gains()
            total_gains = main_gains + self._relation_marginal
            total_gains[selected_mask] = -np.inf
            candidates = np.flatnonzero(~selected_mask)
            chosen = int(
                candidates[
                    int(np.argmax(total_gains[candidates]))
                ]
            )
            lower, upper = self._decision_intervals(total_gains)
            other = candidates[candidates != chosen]
            overlap = bool(
                other.size
                and lower[chosen] <= float(np.max(upper[other]))
            )
            if overlap and not self._using_reference_authority_:
                self._restart_with_reference()
                continue

            if other.size:
                second = int(
                    other[int(np.argmax(total_gains[other]))]
                )
                second_gain = float(total_gains[second])
            else:
                second_gain = float("-inf")
            chosen_main = float(main_gains[chosen])
            chosen_relation = float(self._relation_marginal[chosen])
            chosen_total = float(total_gains[chosen])
            self._append_main_state(chosen)
            edge_column = self.edge_log_weight_column(chosen)
            self._selected_edge_columns.append(edge_column.copy())
            self._selection_order.append(chosen)
            self._relation_marginal += edge_column
            self._relation_marginal[
                np.asarray(self._selection_order, dtype=np.int64)
            ] = -np.inf
            self._main_marginal_gains.append(chosen_main)
            self._relation_marginal_gains.append(chosen_relation)
            self._marginal_gains.append(chosen_total)
            self._second_best_total_gains.append(second_gain)
            self._decision_interval_overlaps.append(overlap)
            previous_objective = (
                self._objective_values[-1]
                if self._objective_values
                else 0.0
            )
            self._objective_values.append(
                previous_objective + chosen_total
            )

    def select(self, k: int) -> np.ndarray:
        self._ensure_fitted()
        if k < 1 or k > self.n_features_in_:
            raise ValueError(
                f"k must be between 1 and {self.n_features_in_} (got {k})"
            )
        self._extend_greedy_prefix(int(k))
        self.selected_features_ = np.asarray(
            self._selection_order[:k], dtype=np.int64
        )
        return self.selected_features_.copy()

    def rank_features(self) -> np.ndarray:
        """Return exact computed prefix plus deterministic relevance tail."""

        self._ensure_fitted()
        selected = list(self._selection_order)
        selected_set = set(selected)
        tail_order = np.lexsort(
            (np.arange(self.n_features_in_, dtype=np.int64), -self.relevance_)
        )
        tail = [int(index) for index in tail_order if int(index) not in selected_set]
        order = selected + tail
        scores = np.empty(self.n_features_in_, dtype=np.float64)
        scores[np.asarray(order, dtype=int)] = np.arange(
            self.n_features_in_, 0, -1, dtype=np.float64
        )
        return scores

