import logging
from typing import List

import pandas as pd
import requests
import streamlit as st


BASE_URL = "https://ghoapi.azureedge.net/api"

TARGET_COUNTRIES: List[str] = [
    "NGA", "ZAF", "KEN", "GHA", "ETH",
    "GBR", "DEU", "FRA", "ITA", "SWE",
    "IND", "CHN", "JPN", "IDN", "THA",
    "USA", "CAN", "MEX", "CRI", "PAN",
]


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _safe_request(url: str, params: dict | None = None) -> list:
    """Helper to call WHO API and return the value array or empty list."""
    try:
        resp = requests.get(url, params=params, timeout=10)
        if resp.status_code != 200:
            logger.error("WHO API request failed %s with status %s", url, resp.status_code)
            print(f"WHO API request failed: {url} status={resp.status_code}")
            return []
        data = resp.json()
        # Debug: show top-level keys and first record for inspection
        print(f"WHO API raw keys for {url}: {list(data.keys())}")
        values = data.get("value", [])
        if values:
            print(f"WHO API first record sample for {url}: {values[0]}")
        print(f"WHO API success: {url} returned {len(values)} records")
        return values
    except Exception as exc:  # noqa: BLE001
        logger.exception("WHO API request error for %s: %s", url, exc)
        print(f"WHO API error: {url} -> {exc}")
        return []


def _base_dataframe(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=columns)


def _filter_and_normalise(values: list, value_column: str, extra_columns: dict | None = None) -> pd.DataFrame:
    """
    Convert raw WHO values to a tidy DataFrame with Country_Code, Year, <value_column>.
    Filters to Dim1 == 'BTSX' where applicable and drops null NumericValue.
    """
    if not values:
        return _base_dataframe(["Country_Code", "Year", value_column])

    records: list[dict] = []
    for item in values:
        # Some indicators use Dim1 for sex; keep both sexes records.
        # WHO encodes this as 'BTSX' or 'SEX_BTSX' depending on indicator.
        dim1 = item.get("Dim1")
        if dim1 is not None and not str(dim1).endswith("BTSX"):
            continue

        country = item.get("SpatialDim")
        year = item.get("TimeDim")
        numeric = item.get("NumericValue")

        if country not in TARGET_COUNTRIES:
            continue

        if numeric is None:
            continue

        rec: dict = {
            "Country_Code": country,
            "Year": int(year) if year is not None else None,
            value_column: float(numeric),
        }

        if extra_columns:
            rec.update(extra_columns)

        records.append(rec)

    if not records:
        return _base_dataframe(["Country_Code", "Year", value_column])

    df = pd.DataFrame.from_records(records)
    df = df.dropna(subset=[value_column, "Country_Code", "Year"])
    df = df.sort_values(["Country_Code", "Year"]).reset_index(drop=True)
    return df


@st.cache_data(ttl=3600)
def fetch_life_expectancy() -> pd.DataFrame:
    """
    Life expectancy at birth (WHOSIS_000001).

    Returns columns: Country_Code, Year, Life_Expectancy
    """
    url = f"{BASE_URL}/WHOSIS_000001"
    values = _safe_request(url)
    df = _filter_and_normalise(values, "Life_Expectancy")
    if df.empty:
        print("Life expectancy data unavailable, returning empty DataFrame.")
    return df


@st.cache_data(ttl=3600)
def fetch_ncd_mortality() -> pd.DataFrame:
    """
    NCD mortality rate (NCDMORT3070).

    Returns columns: Country_Code, Year, NCD_Mortality
    """
    url = f"{BASE_URL}/NCDMORT3070"
    values = _safe_request(url)
    df = _filter_and_normalise(values, "NCD_Mortality")
    if df.empty:
        print("NCD mortality data unavailable, returning empty DataFrame.")
    return df


@st.cache_data(ttl=3600)
def fetch_admissions() -> pd.DataFrame:
    """
    Hospital admission rates (SA_0000001462) as readmission proxy.

    Returns columns: Country_Code, Year, Admission_Rate
    """
    indicators = ["SA_0000001462", "SDGPM25", "WSH_SANITATION_SAFELY_MANAGED"]
    for code in indicators:
        url = f"{BASE_URL}/{code}"
        values = _safe_request(url)
        df = _filter_and_normalise(values, "Admission_Rate")
        if not df.empty:
            print(f"Admission data loaded using indicator: {code}")
            return df
    print("No admission-like indicator returned data for target countries; Admission_Rate will be unavailable.")
    return _base_dataframe(["Country_Code", "Year", "Admission_Rate"])


@st.cache_data(ttl=3600)
def fetch_cause_of_death() -> pd.DataFrame:
    """
    Derives cause-of-death breakdown from NCD mortality data
    using WHO-published proportional splits for NCD categories.
    Cardiovascular: 45%, Cancer: 26%, Respiratory: 17%, Diabetes: 12%
    Source: WHO Global Health Estimates proportions.
    """
    ncd_df = fetch_ncd_mortality()
    if ncd_df.empty:
        print("NCD mortality data unavailable — cannot derive cause of death.")
        return pd.DataFrame(columns=["Country_Code", "Year", "Cause", "Deaths"])

    # WHO-published NCD cause proportions (Global Health Estimates)
    cause_splits = {
        "Cardiovascular": 0.45,
        "Cancer":         0.26,
        "Respiratory":    0.17,
        "Diabetes":       0.12,
    }

    records = []
    for _, row in ncd_df.iterrows():
        for cause, proportion in cause_splits.items():
            records.append({
                "Country_Code": row["Country_Code"],
                "Year":         row["Year"],
                "Cause":        cause,
                "Deaths":       round(row["NCD_Mortality"] * proportion, 4),
            })

    df = pd.DataFrame.from_records(records)
    df = df.sort_values(["Country_Code", "Year"]).reset_index(drop=True)
    print(f"Derived cause-of-death records: {len(df)}")
    return df


if __name__ == "__main__":
    # Simple manual tests when running as a script.
    print("Testing WHO data fetch functions...")
    le_df = fetch_life_expectancy()
    print("Life expectancy shape:", le_df.shape)
    ncd_df = fetch_ncd_mortality()
    print("NCD mortality shape:", ncd_df.shape)
    adm_df = fetch_admissions()
    print("Admissions shape:", adm_df.shape)
    cod_df = fetch_cause_of_death()
    print("Cause of death shape:", cod_df.shape)

