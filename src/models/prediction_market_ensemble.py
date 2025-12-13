"""
Prediction Market-Inspired Ensemble
Based on "Prediction Markets and the Wisdom of Imperfect Crowds" by Benjamin Kolicic

Key Insight:
E[p̂] = E[1{X>0}X] / E[|X|]

Expected implied probability depends only on information distribution,
not on individual rationality or wealth (model complexity or training size).

This module treats each model as a "market participant" with:
- Information score (X): How much useful information the model has
- Rationality (Y): How well-calibrated the model's predictions are
- Capital (M): The model's confidence in its predictions

Following the paper: We aggregate predictions weighted by information quality,
making the ensemble robust to individual model failures.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import log_loss, brier_score_loss
from sklearn.calibration import calibration_curve
import warnings


class PredictionMarketEnsemble:
    """
    Ensemble model using prediction market aggregation logic.

    Each model is a market participant that votes with weight proportional
    to its information quality, not arbitrary fixed weights.

    Core Principles from paper:
    1. Information aggregation: Better-informed models get more weight
    2. Robustness: Independent of model complexity or training set size
    3. Calibration: Output probabilities represent true likelihoods
    """

    def __init__(self, models: Dict[str, any], calibrate: bool = True):
        """
        Initialize prediction market ensemble.

        Args:
            models: Dictionary of {model_name: model_object}
            calibrate: Whether to apply probability calibration
        """
        self.models = models
        self.calibrate = calibrate

        # Track model performance (information scores)
        self.model_info_scores = {name: 1.0 for name in models.keys()}
        self.model_calibration = {name: 1.0 for name in models.keys()}
        self.model_recent_accuracy = {name: 0.5 for name in models.keys()}

        # Historical performance tracking
        self.performance_history = {name: [] for name in models.keys()}

        # Regime-specific performance
        self.regime_performance = {}

        print(f"[PredictionMarketEnsemble] Initialized with {len(models)} participants")
        print(f"  Models: {list(models.keys())}")

    def calculate_information_score(
        self,
        model_name: str,
        recent_accuracy: float,
        calibration_score: float,
        prediction_confidence: float,
        regime: Optional[str] = None
    ) -> float:
        """
        Calculate model's information score (analogous to X in paper).

        Information score represents how much useful information
        the model has about the outcome.

        Components:
        1. Recent accuracy: Has the model been right recently?
        2. Calibration: Are the model's probabilities well-calibrated?
        3. Confidence: How certain is the model about this prediction?
        4. Regime match: Does the model perform well in current market regime?

        Args:
            model_name: Name of the model
            recent_accuracy: Recent prediction accuracy (0 to 1)
            calibration_score: How well-calibrated (0 to 1, higher better)
            prediction_confidence: Model's confidence (0 to 1)
            regime: Current market regime (optional)

        Returns:
            Information score (can be positive or negative)
        """
        # Base information from accuracy and calibration
        base_info = recent_accuracy * calibration_score

        # Adjust by confidence (models that are uncertain have less information)
        # Following paper: rationality (y) modulates information usage
        confidence_adjusted = base_info * (0.5 + 0.5 * prediction_confidence)

        # Regime adjustment if available
        if regime and regime in self.regime_performance.get(model_name, {}):
            regime_accuracy = self.regime_performance[model_name][regime]
            regime_weight = 0.3  # 30% weight to regime-specific performance
            confidence_adjusted = (
                (1 - regime_weight) * confidence_adjusted +
                regime_weight * regime_accuracy
            )

        # Convert to information score (centered at 0.5)
        # Scores > 0.5 suggest upward movement, < 0.5 suggest downward
        # Following paper's formulation: information supports event A or not-A
        information_score = 2 * (confidence_adjusted - 0.5)

        return information_score

    def predict_proba(
        self,
        X: np.ndarray,
        regime: Optional[str] = None,
        return_weights: bool = False
    ) -> np.ndarray:
        """
        Generate probabilistic predictions using prediction market aggregation.

        Following the paper's formula:
        E[p̂] = E[1{X>0}X] / E[|X|]

        Where:
        - E[1{X>0}X] = Sum of information supporting bullish outcome
        - E[|X|] = Total information from all models

        Args:
            X: Input features
            regime: Current market regime (optional)
            return_weights: If True, also return model weights used

        Returns:
            Probability predictions (and optionally weights)
        """
        predictions = []
        information_scores = []
        confidences = []

        # Get predictions from each model (market participant)
        for model_name, model in self.models.items():
            try:
                # Get probability prediction
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(X)
                    # Handle different output formats
                    if len(proba.shape) > 1:
                        pred_proba = proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]
                    else:
                        pred_proba = proba
                else:
                    # Model doesn't have predict_proba, use predict and convert
                    pred = model.predict(X)
                    pred_proba = self._convert_to_probability(pred)

                # Calculate prediction confidence
                # For binary: distance from 0.5 indicates confidence
                confidence = np.abs(pred_proba - 0.5) * 2  # Scale to [0, 1]

                # Get model's information score
                info_score = self.calculate_information_score(
                    model_name=model_name,
                    recent_accuracy=self.model_recent_accuracy[model_name],
                    calibration_score=self.model_calibration[model_name],
                    prediction_confidence=np.mean(confidence),
                    regime=regime
                )

                predictions.append(pred_proba)
                information_scores.append(info_score)
                confidences.append(confidence)

            except Exception as e:
                warnings.warn(f"Model {model_name} prediction failed: {e}")
                continue

        if len(predictions) == 0:
            raise ValueError("No models successfully generated predictions")

        # Convert to numpy arrays
        predictions = np.array(predictions)  # Shape: (n_models, n_samples)
        information_scores = np.array(information_scores)  # Shape: (n_models,)

        # Aggregate using prediction market logic
        # Following paper: E[p̂] = E[1{X>0}X] / E[|X|]

        # Calculate weights from information scores
        # Positive info scores support bullish (prob > 0.5)
        # Negative info scores support bearish (prob < 0.5)
        abs_info_scores = np.abs(information_scores)

        # Avoid division by zero
        total_info = np.sum(abs_info_scores)
        if total_info == 0:
            weights = np.ones(len(information_scores)) / len(information_scores)
        else:
            weights = abs_info_scores / total_info

        # Aggregate predictions
        # Each model votes weighted by its information content
        aggregated_proba = np.average(predictions, axis=0, weights=weights)

        # Store weights for analysis
        self.last_weights = dict(zip(self.models.keys(), weights))

        if return_weights:
            return aggregated_proba, self.last_weights

        return aggregated_proba

    def predict(self, X: np.ndarray, threshold: float = 0.5, regime: Optional[str] = None) -> np.ndarray:
        """
        Generate binary predictions.

        Args:
            X: Input features
            threshold: Probability threshold for positive class
            regime: Current market regime

        Returns:
            Binary predictions
        """
        proba = self.predict_proba(X, regime=regime)
        return (proba >= threshold).astype(int)

    def update_model_performance(
        self,
        model_name: str,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        regime: Optional[str] = None
    ):
        """
        Update model's performance metrics after observing outcomes.

        This is critical for maintaining accurate information scores.

        Args:
            model_name: Name of model to update
            y_true: True binary outcomes
            y_pred_proba: Predicted probabilities
            regime: Market regime during predictions
        """
        if model_name not in self.models:
            return

        # Calculate recent accuracy
        y_pred = (y_pred_proba >= 0.5).astype(int)
        accuracy = np.mean(y_pred == y_true)

        # Calculate calibration (Brier score - lower is better, so invert)
        try:
            brier = brier_score_loss(y_true, y_pred_proba)
            calibration = 1 - brier  # Convert to "higher is better"
        except:
            calibration = 0.5

        # Update stored metrics with exponential moving average
        alpha = 0.3  # Learning rate
        self.model_recent_accuracy[model_name] = (
            alpha * accuracy + (1 - alpha) * self.model_recent_accuracy[model_name]
        )
        self.model_calibration[model_name] = (
            alpha * calibration + (1 - alpha) * self.model_calibration[model_name]
        )

        # Update information score
        self.model_info_scores[model_name] = (
            self.model_recent_accuracy[model_name] *
            self.model_calibration[model_name]
        )

        # Track regime-specific performance
        if regime:
            if model_name not in self.regime_performance:
                self.regime_performance[model_name] = {}
            if regime not in self.regime_performance[model_name]:
                self.regime_performance[model_name][regime] = accuracy
            else:
                self.regime_performance[model_name][regime] = (
                    alpha * accuracy +
                    (1 - alpha) * self.regime_performance[model_name][regime]
                )

        # Store in history
        self.performance_history[model_name].append({
            'accuracy': accuracy,
            'calibration': calibration,
            'info_score': self.model_info_scores[model_name],
            'regime': regime
        })

    def get_model_rankings(self) -> pd.DataFrame:
        """
        Get current model rankings by information score.

        Returns:
            DataFrame with model performance metrics
        """
        rankings = []
        for model_name in self.models.keys():
            rankings.append({
                'model': model_name,
                'information_score': self.model_info_scores[model_name],
                'recent_accuracy': self.model_recent_accuracy[model_name],
                'calibration': self.model_calibration[model_name],
                'n_predictions': len(self.performance_history[model_name])
            })

        df = pd.DataFrame(rankings)
        df = df.sort_values('information_score', ascending=False)
        return df

    def _convert_to_probability(self, predictions: np.ndarray) -> np.ndarray:
        """Convert raw predictions to probabilities."""
        # Assume predictions are in [0, 1] or need sigmoid
        if predictions.min() >= 0 and predictions.max() <= 1:
            return predictions
        else:
            # Apply sigmoid
            return 1 / (1 + np.exp(-predictions))

    def get_market_consensus(self, X: np.ndarray) -> Dict[str, float]:
        """
        Get market consensus metrics.

        Returns:
            Dictionary with consensus statistics
        """
        proba, weights = self.predict_proba(X, return_weights=True)

        # Calculate consensus strength (how concentrated are the weights?)
        weight_concentration = np.max(list(weights.values()))

        # Calculate prediction dispersion
        all_preds = []
        for model in self.models.values():
            if hasattr(model, 'predict_proba'):
                pred = model.predict_proba(X)
                if len(pred.shape) > 1:
                    pred = pred[:, 1] if pred.shape[1] > 1 else pred[:, 0]
                all_preds.append(pred)

        prediction_std = np.std(all_preds, axis=0).mean() if all_preds else 0

        return {
            'consensus_probability': float(np.mean(proba)),
            'weight_concentration': float(weight_concentration),
            'prediction_dispersion': float(prediction_std),
            'n_participants': len(self.models),
            'top_model': max(weights, key=weights.get),
            'top_model_weight': float(weights[max(weights, key=weights.get)])
        }


class ModelInformationTracker:
    """
    Track and analyze information distribution across models over time.

    Following the paper's insight that f_X(x) (information distribution)
    evolves as more information becomes available.
    """

    def __init__(self):
        self.history = []

    def record_snapshot(
        self,
        timestamp: pd.Timestamp,
        model_scores: Dict[str, float],
        regime: str,
        market_state: Dict
    ):
        """Record a snapshot of information distribution."""
        self.history.append({
            'timestamp': timestamp,
            'model_scores': model_scores.copy(),
            'regime': regime,
            'market_state': market_state
        })

    def get_information_evolution(self) -> pd.DataFrame:
        """
        Analyze how information distribution evolved over time.

        Returns DataFrame showing information concentration over time.
        """
        if not self.history:
            return pd.DataFrame()

        df = pd.DataFrame(self.history)

        # Calculate information concentration (analogous to market convergence)
        df['info_concentration'] = df['model_scores'].apply(
            lambda x: max(x.values()) if x else 0
        )

        # Calculate total information
        df['total_information'] = df['model_scores'].apply(
            lambda x: sum(abs(v) for v in x.values()) if x else 0
        )

        return df


def create_prediction_market_ensemble(
    models: Dict[str, any],
    X_train,  # Can be np.ndarray or pd.DataFrame
    y_train: np.ndarray,
    X_val,  # Can be np.ndarray or pd.DataFrame
    y_val: np.ndarray
) -> PredictionMarketEnsemble:
    """
    Factory function to create and initialize a prediction market ensemble.

    Args:
        models: Dictionary of trained models
        X_train: Training features (for calibration)
        y_train: Training labels
        X_val: Validation features (for initial performance assessment)
        y_val: Validation labels

    Returns:
        Initialized PredictionMarketEnsemble
    """
    ensemble = PredictionMarketEnsemble(models)

    # Initialize performance metrics using validation set
    print("\n[Initialization] Assessing model information scores...")

    for model_name, model in models.items():
        try:
            # Get predictions - handle different model types
            # Determine if model needs numpy or DataFrame input
            needs_numpy = hasattr(model, 'lookback')  # Neural models have lookback attribute

            # Prepare input based on model type
            if needs_numpy:
                X_val_for_model = X_val.values if hasattr(X_val, 'values') else X_val
            else:
                X_val_for_model = X_val  # Keep as DataFrame for tree models

            if hasattr(model, 'predict_proba'):
                # Sklearn-style model with probability output
                y_pred_proba = model.predict_proba(X_val_for_model)
                if len(y_pred_proba.shape) > 1:
                    y_pred_proba = y_pred_proba[:, 1]
            elif hasattr(model, 'predict'):
                # Check model type and prepare appropriate input
                try:
                    import xgboost as xgb
                    if isinstance(model, (xgb.XGBClassifier, xgb.XGBRegressor)):
                        # XGBoost sklearn interface - can handle DataFrame
                        y_pred = model.predict(X_val_for_model)
                    elif isinstance(model, xgb.Booster):
                        # XGBoost native booster - needs DMatrix
                        dmatrix = xgb.DMatrix(X_val_for_model)
                        y_pred = model.predict(dmatrix)
                    else:
                        # Standard predict (input already converted above based on needs_numpy)
                        y_pred = model.predict(X_val_for_model)
                except ImportError:
                    # XGBoost not available, use standard predict
                    y_pred = model.predict(X_val_for_model)

                # Ensure y_pred is numpy array (handle any remaining DataFrame/Series cases)
                if hasattr(y_pred, 'values'):
                    y_pred = y_pred.values
                # Also ensure it's a proper numpy array, not a 0-d array or scalar
                y_pred = np.asarray(y_pred).flatten()

                y_pred_proba = ensemble._convert_to_probability(y_pred)
            else:
                # Skip models without predict method (e.g., raw PyTorch models)
                raise AttributeError(f"Model {model_name} has no predict or predict_proba method")

            # Update performance - handle length mismatch
            # Some models (like neural networks) may produce shorter predictions
            min_len = min(len(y_val), len(y_pred_proba))
            ensemble.update_model_performance(model_name, y_val[:min_len], y_pred_proba[:min_len])

            print(f"  {model_name}:")
            print(f"    Accuracy: {ensemble.model_recent_accuracy[model_name]:.3f}")
            print(f"    Calibration: {ensemble.model_calibration[model_name]:.3f}")
            print(f"    Information Score: {ensemble.model_info_scores[model_name]:.3f}")

        except Exception as e:
            print(f"  {model_name}: Failed initialization - {e}")

    return ensemble
