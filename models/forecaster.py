from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA


@dataclass
class ForecastResult:
    df: pd.DataFrame
    last_history_year: int


def _linear_trend_forecast(series: pd.Series, steps: int) -> ForecastResult:
    """Fallback simple linear forecast when ARIMA fails."""
    if series.empty:
        return ForecastResult(pd.DataFrame(columns=["Year", "Forecast", "Lower_CI", "Upper_CI"]), last_history_year=0)

    y = series.values.astype(float)
    x = np.arange(len(y))
    coef = np.polyfit(x, y, deg=1)
    trend = np.poly1d(coef)

    last_year = int(series.index.max())
    future_x = np.arange(len(y), len(y) + steps)
    forecast_vals = trend(future_x)

    years = np.arange(last_year + 1, last_year + 1 + steps)
    # Simple +/- 2 std dev band as a crude CI
    std = float(np.std(y)) if len(y) > 1 else 0.0
    lower = forecast_vals - 2 * std
    upper = forecast_vals + 2 * std

    df = pd.DataFrame(
        {
            "Year": years,
            "Forecast": forecast_vals,
            "Lower_CI": lower,
            "Upper_CI": upper,
        }
    )
    return ForecastResult(df=df, last_history_year=last_year)


def forecast_life_expectancy(country_df: pd.DataFrame, steps: int = 5) -> ForecastResult:
    """
    Forecast life expectancy using ARIMA(1,1,1).

    Expects columns: Year, Life_Expectancy.
    Returns DataFrame with columns: Year, Forecast, Lower_CI, Upper_CI.
    """
    if country_df.empty or "Life_Expectancy" not in country_df.columns:
        return ForecastResult(pd.DataFrame(columns=["Year", "Forecast", "Lower_CI", "Upper_CI"]), last_history_year=0)

    df = country_df.sort_values("Year")
    df = df.dropna(subset=["Life_Expectancy"])
    if df.empty:
        return ForecastResult(pd.DataFrame(columns=["Year", "Forecast", "Lower_CI", "Upper_CI"]), last_history_year=0)

    series = pd.Series(df["Life_Expectancy"].values, index=df["Year"].astype(int))

    # Not enough points for ARIMA? Use linear trend directly.
    if series.shape[0] < 5:
        return _linear_trend_forecast(series, steps)

    try:
        model = ARIMA(series, order=(1, 1, 1))
        fitted = model.fit()
        forecast_res = fitted.get_forecast(steps=steps)
        forecast_vals = forecast_res.predicted_mean
        conf_int = forecast_res.conf_int(alpha=0.05)

        last_year = int(series.index.max())
        years = np.arange(last_year + 1, last_year + 1 + steps)

        # conf_int columns may be named like 'lower y', 'upper y'
        lower = conf_int.iloc[:, 0].values
        upper = conf_int.iloc[:, 1].values

        df_out = pd.DataFrame(
            {
                "Year": years,
                "Forecast": forecast_vals.values,
                "Lower_CI": lower,
                "Upper_CI": upper,
            }
        )
        return ForecastResult(df=df_out, last_history_year=last_year)
    except Exception:  # noqa: BLE001
        # Fallback to linear trend
        return _linear_trend_forecast(series, steps)


def forecast_region_average(master_df: pd.DataFrame, region: str, steps: int = 5) -> Optional[ForecastResult]:
    """Run forecast for regional average life expectancy."""
    if master_df.empty:
        return None
    region_df = (
        master_df[master_df["Region"] == region]
        .groupby("Year", as_index=False)["Life_Expectancy"]
        .mean()
        .rename(columns={"Life_Expectancy": "Life_Expectancy"})
    )
    if region_df.empty:
        return None
    return forecast_life_expectancy(region_df, steps=steps)


if __name__ == "__main__":
    # Basic smoke test with synthetic data
    years = np.arange(2000, 2021)
    le = 60 + 0.5 * (years - 2000)
    sample = pd.DataFrame({"Year": years, "Life_Expectancy": le})
    res = forecast_life_expectancy(sample, steps=5)
    print(res.df)

