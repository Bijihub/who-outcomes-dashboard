from __future__ import annotations

from typing import Literal

import pandas as pd


RiskTier = Literal["High Risk", "Medium Risk", "Low Risk"]


def compute_risk_tier(life_expectancy: float | int | None) -> RiskTier | None:
    """
    Map life expectancy to a risk tier.

    High Risk: Life_Expectancy < 65
    Medium Risk: 65–75
    Low Risk: > 75
    """
    if life_expectancy is None:
        return None
    try:
        le = float(life_expectancy)
    except (TypeError, ValueError):
        return None

    if le < 65:
        return "High Risk"
    if 65 <= le <= 75:
        return "Medium Risk"
    return "Low Risk"


def compute_risk_score(
    life_expectancy: float | int | None,
    ncd_mortality: float | int | None,
    admission_rate: float | int | None,
) -> float | None:
    """
    Risk_Score formula:
    100 - Life_Expectancy + (NCD_Mortality * 2) + Admission_Rate
    Admission_Rate is optional — defaults to 0 if not available.
    """
    try:
        le = float(life_expectancy) if life_expectancy is not None else None
        ncd = float(ncd_mortality) if ncd_mortality is not None else None
        adm = float(admission_rate) if admission_rate is not None else 0.0
    except (TypeError, ValueError):
        return None

    if le is None or ncd is None:
        return None
    return 100.0 - le + (ncd * 2.0) + adm


def apply_risk_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add Risk_Tier and Risk_Score columns to a DataFrame in-place and return it."""
    if df.empty:
        df["Risk_Tier"] = []
        df["Risk_Score"] = []
        return df

    df["Risk_Tier"] = df["Life_Expectancy"].apply(compute_risk_tier)
    df["Risk_Score"] = df.apply(
        lambda row: compute_risk_score(
            row.get("Life_Expectancy"),
            row.get("NCD_Mortality"),
            row.get("Admission_Rate"),
        ),
        axis=1,
    )
    return df


if __name__ == "__main__":
    sample = pd.DataFrame(
        {
            "Life_Expectancy": [60, 70, 80],
            "NCD_Mortality": [15, 10, 8],
            "Admission_Rate": [10, 12, 9],
        }
    )
    print(apply_risk_columns(sample))

