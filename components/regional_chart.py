from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


def regional_overview_charts(
    master_df: pd.DataFrame,
    region_filter: str | None,
    country_filter: list[str] | None,
    year_range: tuple[int, int],
    region_colors: dict[str, str],
    plotly_layout: dict,
) -> None:
    """Grouped bar, regional trend line, and risk tier bar charts."""
    if master_df.empty:
        st.error("No data available to display charts.")
        return

    df = master_df.copy()
    df = df[(df["Year"] >= year_range[0]) & (df["Year"] <= year_range[1])]

    if region_filter and region_filter != "All":
        df = df[df["Region"] == region_filter]
    if country_filter:
        df = df[df["Country_Code"].isin(country_filter)]

    if df.empty:
        st.warning("No records match the selected filters.")
        return

    latest_year = int(df["Year"].max())
    latest_df = df[df["Year"] == latest_year]

    # Grouped bar: Life Expectancy vs NCD Mortality by country
    bar_df = latest_df.melt(
        id_vars=["Country_Name", "Region"],
        value_vars=["Life_Expectancy", "NCD_Mortality"],
        var_name="Metric",
        value_name="Value",
    )
    fig_bar = px.bar(
        bar_df,
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
    # Choropleth world map — countries colored by Risk Tier
    st.subheader("Global Risk Map")
    st.caption("Countries colored by Risk Tier based on life expectancy thresholds")

    map_df = latest_df[["Country_Code", "Country_Name", "Region", "Risk_Tier", "Risk_Score", "Life_Expectancy"]].drop_duplicates()

    risk_order = {"High Risk": 1, "Medium Risk": 2, "Low Risk": 3}
    map_df = map_df.copy()
    map_df["Risk_Rank"] = map_df["Risk_Tier"].map(risk_order)

    fig_map = px.choropleth(
        map_df,
        locations="Country_Code",
        color="Risk_Tier",
        hover_name="Country_Name",
        hover_data={
            "Country_Code": False,
            "Region": True,
            "Life_Expectancy": ":.1f",
            "Risk_Score": ":.1f",
            "Risk_Tier": True,
        },
        color_discrete_map={
            "High Risk":   "#EF4444",
            "Medium Risk": "#F59E0B",
            "Low Risk":    "#10B981",
        },
        category_orders={"Risk_Tier": ["High Risk", "Medium Risk", "Low Risk"]},
    )
    fig_map.update_layout(
        **plotly_layout,
        geo=dict(
            showframe=False,
            showcoastlines=True,
            coastlinecolor="#334155",
            showland=True,
            landcolor="#1E293B",
            showocean=True,
            oceancolor="#0F172A",
            showlakes=False,
            showcountries=True,
            countrycolor="#334155",
            bgcolor="#0F172A",
        ),
        height=450,
        margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            font=dict(size=12, color="#F1F5F9"),
            bgcolor="rgba(0,0,0,0)",
        ),
    )
    st.plotly_chart(fig_map, use_container_width=True)

    st.subheader("Life Expectancy vs NCD Mortality by Country")
    st.plotly_chart(fig_bar, use_container_width=True)

    # Line chart: life expectancy trend by region
    region_trend = (
        df.groupby(["Region", "Year"], as_index=False)["Life_Expectancy"]
        .mean()
        .dropna()
    )
    fig_line = px.line(
        region_trend,
        x="Year",
        y="Life_Expectancy",
        color="Region",
        color_discrete_map=region_colors,
    )
    fig_line.update_layout(
        **plotly_layout,
        xaxis_title="Year",
        yaxis_title="Avg Life Expectancy",
    )
    st.subheader("Life Expectancy Trend by Region")
    st.plotly_chart(fig_line, use_container_width=True)

    # Risk tier horizontal bar chart for all 20 countries
    risk_df = latest_df.copy()
    risk_df = risk_df.sort_values("Risk_Score", ascending=False)

    color_map = {
        "High Risk": "#DC2626",  # red
        "Medium Risk": "#F59E0B",  # amber
        "Low Risk": "#22C55E",  # green
    }
    bar_colors = [color_map.get(tier, "#6B7280") for tier in risk_df["Risk_Tier"]]

    fig_risk = go.Figure(
        data=go.Bar(
            x=risk_df["Risk_Score"],
            y=risk_df["Country_Name"],
            orientation="h",
            marker_color=bar_colors,
            hovertemplate="Country: %{y}<br>Risk Score: %{x:.1f}<br>Tier: %{customdata}",
            customdata=risk_df["Risk_Tier"],
        )
    )
    fig_risk.update_layout(
        **plotly_layout,
        xaxis_title="Risk Score",
        yaxis_title="Country",
    )
    st.subheader("Risk Tier by Country")
    st.plotly_chart(fig_risk, use_container_width=True)

