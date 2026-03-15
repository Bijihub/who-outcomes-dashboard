from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split


@dataclass
class RiskModelArtifacts:
    model: LinearRegression
    r2: float
    mae: float
    feature_means: np.ndarray


FEATURE_COLUMNS = ["Life_Expectancy", "NCD_Mortality", "Admission_Rate", "Year"]
TARGET_COLUMN = "Risk_Score"


def train_risk_model(master_df: pd.DataFrame) -> RiskModelArtifacts:
    """
    Train a linear regression model to predict Risk_Score.

    Features: Life_Expectancy, NCD_Mortality, Admission_Rate, Year
    Target: Risk_Score

    Train on all years except last 2, test on last 2 years (if possible).
    """
    # Determine which feature columns are actually available (handle missing Admission_Rate)
    available_features = [c for c in FEATURE_COLUMNS if c in master_df.columns]
    if "Admission_Rate" not in available_features:
        print("Training risk model without Admission_Rate feature.")
    # Drop rows where any feature or target is NaN
    df = master_df.dropna(subset=available_features + [TARGET_COLUMN]).copy()
    if df.empty:
        # Degenerate placeholder model
        dummy_model = LinearRegression()
        dummy_model.fit(np.zeros((1, len(available_features))), np.array([0.0]))
        return RiskModelArtifacts(model=dummy_model, r2=0.0, mae=0.0, feature_means=np.zeros(len(available_features)))

    max_year = int(df["Year"].max())
    cutoff_year = max_year - 2

    train_df = df[df["Year"] <= cutoff_year]
    test_df = df[df["Year"] > cutoff_year]

    # If not enough data after split, fall back to random split.
    if len(train_df) < 10 or len(test_df) < 5:
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    # Compute feature means on training data for NaN imputation during prediction
    feature_means = train_df[available_features].mean().values

    X_train = train_df[available_features].values
    y_train = train_df[TARGET_COLUMN].values
    X_test = test_df[available_features].values
    y_test = test_df[TARGET_COLUMN].values

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    r2 = float(r2_score(y_test, y_pred))
    mae = float(mean_absolute_error(y_test, y_pred))

    print("Risk model R^2:", r2)
    print("Risk model MAE:", mae)

    return RiskModelArtifacts(model=model, r2=r2, mae=mae, feature_means=feature_means)


def predict_risk_score(
    artifacts: RiskModelArtifacts,
    life_expectancy: float,
    ncd_mortality: float,
    admission_rate: float | None,
    year: int,
) -> float:
    # Build feature vector and replace NaNs with training feature means.
    # If Admission_Rate was not in the training data, drop it from the input as well.
    # Infer feature dimension from artifacts.feature_means.
    if artifacts.feature_means.shape[0] == 3:
        feats = [life_expectancy, ncd_mortality, year]
    else:
        feats = [life_expectancy, ncd_mortality, admission_rate if admission_rate is not None else np.nan, year]

    X = np.array([feats], dtype=float)
    # Broadcast means if any NaN present
    if np.isnan(X).any():
        means = artifacts.feature_means.reshape(1, -1)
        # Where X is NaN, replace with corresponding mean
        X = np.where(np.isnan(X), means, X)
    return float(artifacts.model.predict(X)[0])


def actual_vs_predicted_df(master_df: pd.DataFrame, model: LinearRegression) -> pd.DataFrame:
    """
    Build a DataFrame of actual vs predicted Risk_Score for scatter plotting.
    """
    feature_cols = [c for c in FEATURE_COLUMNS if c in master_df.columns]
    df = master_df.dropna(subset=feature_cols + [TARGET_COLUMN]).copy()
    if df.empty:
        return pd.DataFrame(columns=["Actual", "Predicted", "Country_Code", "Region"])

    X = df[feature_cols].values
    y_true = df[TARGET_COLUMN].values
    y_pred = model.predict(X)

    out = pd.DataFrame(
        {
            "Actual": y_true,
            "Predicted": y_pred,
            "Country_Code": df["Country_Code"].values,
            "Region": df["Region"].values,
        }
    )
    return out


if __name__ == "__main__":
    # Simple synthetic test
    years = np.arange(2010, 2021)
    n = len(years)
    df = pd.DataFrame(
        {
            "Country_Code": ["XXX"] * n,
            "Region": ["Test"] * n,
            "Year": years,
            "Life_Expectancy": 60 + 0.4 * (years - 2010),
            "NCD_Mortality": 20 - 0.5 * (years - 2010),
            "Admission_Rate": 10 + 0.1 * (years - 2010),
        }
    )
    df["Risk_Score"] = 100 - df["Life_Expectancy"] + 2 * df["NCD_Mortality"] + df["Admission_Rate"]

    artifacts = train_risk_model(df)
    print("Artifacts:", artifacts)

