from __future__ import annotations

import logging
from typing import Dict

import pandas as pd

from data.fetch_data import (
    fetch_admissions,
    fetch_cause_of_death,
    fetch_life_expectancy,
    fetch_ncd_mortality,
)
from utils.risk import apply_risk_columns


logger = logging.getLogger(__name__)


REGION_MAP: Dict[str, str] = {
    "NGA": "Africa",
    "ZAF": "Africa",
    "KEN": "Africa",
    "GHA": "Africa",
    "ETH": "Africa",
    "GBR": "Europe",
    "DEU": "Europe",
    "FRA": "Europe",
    "ITA": "Europe",
    "SWE": "Europe",
    "IND": "Asia",
    "CHN": "Asia",
    "JPN": "Asia",
    "IDN": "Asia",
    "THA": "Asia",
    "USA": "North America",
    "CAN": "North America",
    "MEX": "North America",
    "CRI": "North America",
    "PAN": "North America",
}

COUNTRY_NAMES: Dict[str, str] = {
    "NGA": "Nigeria",
    "ZAF": "South Africa",
    "KEN": "Kenya",
    "GHA": "Ghana",
    "ETH": "Ethiopia",
    "GBR": "United Kingdom",
    "DEU": "Germany",
    "FRA": "France",
    "ITA": "Italy",
    "SWE": "Sweden",
    "IND": "India",
    "CHN": "China",
    "JPN": "Japan",
    "IDN": "Indonesia",
    "THA": "Thailand",
    "USA": "United States",
    "CAN": "Canada",
    "MEX": "Mexico",
    "CRI": "Costa Rica",
    "PAN": "Panama",
}


def build_master_dataframe() -> pd.DataFrame:
    """
    Fetch and merge all indicator DataFrames into a single master table.

    Columns:
        Country_Code, Year, Life_Expectancy, NCD_Mortality,
        Admission_Rate, Region, Country_Name, Risk_Tier, Risk_Score
    """
    le_df = fetch_life_expectancy()
    ncd_df = fetch_ncd_mortality()
    adm_df = fetch_admissions()
    cod_df = fetch_cause_of_death()

    # Basic left joins on Country_Code and Year
    master = le_df.merge(
        ncd_df, on=["Country_Code", "Year"], how="outer", validate="one_to_one"
    )
    # Only merge Admission_Rate if we actually have data
    if not adm_df.empty:
        master = master.merge(
            adm_df, on=["Country_Code", "Year"], how="outer", validate="one_to_one"
        )

    # Summarise cause of death to keep table narrow (total deaths per country/year)
    if not cod_df.empty:
        cod_agg = (
            cod_df.groupby(["Country_Code", "Year"], as_index=False)["Deaths"]
            .sum()
            .rename(columns={"Deaths": "Total_Deaths"})
        )
        master = master.merge(
            cod_agg, on=["Country_Code", "Year"], how="left", validate="one_to_one"
        )

    # Region and country name
    master["Region"] = master["Country_Code"].map(REGION_MAP)
    master["Country_Name"] = master["Country_Code"].map(COUNTRY_NAMES)

    # Drop rows missing key identifiers
    master = master.dropna(subset=["Country_Code", "Year"])

    # Apply risk tier and score
    master = apply_risk_columns(master)

    # If Admission_Rate is entirely missing, drop it and recompute Risk_Score
    if "Admission_Rate" in master.columns and master["Admission_Rate"].count() == 0:
        master = master.drop(columns=["Admission_Rate"])
        # Recompute Risk_Score without Admission_Rate: 100 - LE + 2*NCD
        from utils.risk import compute_risk_score  # type: ignore

        def _risk_no_adm(row):
            from utils.risk import compute_risk_score as _crs

            # use 0 for admission_rate in this simplified formula
            return _crs(row.get("Life_Expectancy"), row.get("NCD_Mortality"), 0.0)

        master["Risk_Score"] = master.apply(_risk_no_adm, axis=1)

    # Fill remaining NaNs in key numeric columns with column medians
    for col in ["Life_Expectancy", "NCD_Mortality", "Admission_Rate", "Year"]:
        if col in master.columns:
            median_val = master[col].median()
            master[col] = master[col].fillna(median_val)

    # Debug: inspect core numeric columns and risk score distribution
    print("Sample Life_Expectancy values:", master["Life_Expectancy"].describe())
    print("Sample NCD_Mortality values:", master["NCD_Mortality"].describe())
    if "Admission_Rate" in master.columns:
        print("Sample Admission_Rate values:", master["Admission_Rate"].describe())
    else:
        print("Admission_Rate column not present in master DataFrame")
    print("Sample Risk_Score values:", master["Risk_Score"].describe())

    # Sort for predictable ordering
    master = master.sort_values(["Region", "Country_Name", "Year"]).reset_index(
        drop=True
    )

    # Debug prints per instructions
    print("Master DataFrame shape:", master.shape)
    print("Master DataFrame columns:", list(master.columns))

    return master


if __name__ == "__main__":
    df_master = build_master_dataframe()
    print(df_master.head())

