from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st


def country_spotlight_page(
    master_df: pd.DataFrame,
    region_colors: dict[str, str],
    plotly_layout: dict,
) -> None:
    st.header("Country Spotlight")
    st.caption("Drill into a single country and compare it to its regional peers.")

    if master_df.empty:
        st.error("No data available to display country spotlight.")
        return

    countries = master_df["Country_Name"].dropna().unique()
    countries = sorted(countries)
    selected_country = st.selectbox("Select Country", countries)

    country_df = master_df[master_df["Country_Name"] == selected_country]
    if country_df.empty:
        st.warning("No data available for the selected country.")
        return

    latest_year = int(country_df["Year"].max())
    latest_row = country_df[country_df["Year"] == latest_year].iloc[0]
    region = latest_row["Region"]

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Life Expectancy", f"{latest_row['Life_Expectancy']:.1f} years")
    with col2:
        st.metric("NCD Mortality", f"{latest_row['NCD_Mortality']:.1f}")
    with col3:
        tier = latest_row.get("Risk_Tier", "N/A")
        st.metric("Risk Tier", tier)

    # Trend vs regional average
    region_df = master_df[master_df["Region"] == region]
    region_avg = (
        region_df.groupby("Year", as_index=False)["Life_Expectancy"]
        .mean()
        .rename(columns={"Life_Expectancy": "Region_Avg_LE"})
    )
    trend_df = (
        country_df[["Year", "Life_Expectancy"]]
        .merge(region_avg, on="Year", how="left")
        .sort_values("Year")
    )
    fig_trend = px.line(
        trend_df,
        x="Year",
        y="Life_Expectancy",
        labels={"Life_Expectancy": "Life Expectancy"},
    )
    fig_trend.add_scatter(
        x=trend_df["Year"],
        y=trend_df["Region_Avg_LE"],
        mode="lines",
        name=f"{region} Average",
        line=dict(color=region_colors.get(region, "#6B7280"), dash="dash"),
    )
    fig_trend.update_layout(
        **plotly_layout,
        xaxis_title="Year",
        yaxis_title="Life Expectancy",
    )
    st.subheader("Life Expectancy — Country vs Regional Average")
    st.plotly_chart(fig_trend, use_container_width=True)

    # Bar chart: country vs regional peers across key metrics
    latest_region = region_df[region_df["Year"] == latest_year]
    metric_cols = ["Country_Name", "Life_Expectancy", "NCD_Mortality"]
    if "Admission_Rate" in latest_region.columns:
        metric_cols.append("Admission_Rate")
    metrics_df = latest_region[metric_cols]
    metrics_melted = metrics_df.melt(
        id_vars=["Country_Name"],
        var_name="Metric",
        value_name="Value",
    )
    fig_bar = px.bar(
        metrics_melted,
        x="Country_Name",
        y="Value",
        color="Metric",
        barmode="group",
    )
    fig_bar.update_layout(
        **plotly_layout,
        xaxis_title="Country",
        yaxis_title="Value",
    )
    st.subheader(f"{region} Peer Comparison — Latest Year ({latest_year})")
    st.plotly_chart(fig_bar, use_container_width=True)

    # Region summary panel
    st.subheader(f"{region} Region Summary")
    mini_cols = [
        "Country_Name",
        "Year",
        "Life_Expectancy",
        "NCD_Mortality",
        "Admission_Rate",
        "Risk_Tier",
        "Risk_Score",
    ]
    available_cols = [c for c in mini_cols if c in region_df.columns]
    mini_df = region_df[available_cols]
    latest_mini = mini_df[mini_df["Year"] == latest_year]
    if "Risk_Score" in latest_mini.columns:
        latest_mini = latest_mini.sort_values("Risk_Score", ascending=False)
    st.dataframe(
        latest_mini.drop(columns=["Year"]),
        use_container_width=True,
    )

