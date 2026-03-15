from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from data.fetch_data import fetch_cause_of_death


def disease_burden_page(
    master_df: pd.DataFrame,
    region_colors: dict[str, str],
    plotly_layout: dict,
) -> None:
    st.header("Disease Burden Story")
    st.caption("Countries ranked by cause of death — identifying where each disease hits hardest.")

    cod_df = fetch_cause_of_death()
    if cod_df.empty:
        st.error("Cause-of-death data is not available at the moment.")
        return

    # Merge region info
    cod_merged = cod_df.merge(
        master_df[["Country_Code", "Country_Name", "Region"]].drop_duplicates(),
        on="Country_Code",
        how="left",
    )

    # Use latest available year
    latest_year = int(cod_merged["Year"].max())
    latest_df = cod_merged[cod_merged["Year"] == latest_year]

    # Aggregate by country and cause for latest year
    country_cause = (
        latest_df.groupby(["Country_Name", "Region", "Cause"], as_index=False)["Deaths"]
        .sum()
    )

    st.markdown(
        f'<p style="font-size:13px;color:#64748B;margin-bottom:20px;">'
        f'Showing latest available year: <strong style="color:#0D9488">{latest_year}</strong> '
        f'· Data derived from WHO NCD mortality using published disease proportions</p>',
        unsafe_allow_html=True,
    )

    # One horizontal bar chart per disease
    causes = ["Cardiovascular", "Cancer", "Respiratory", "Diabetes"]
    cause_colors = {
        "Cardiovascular": "#EF4444",
        "Cancer":         "#F59E0B",
        "Respiratory":    "#6366F1",
        "Diabetes":       "#0D9488",
    }

    # Single stacked horizontal bar chart — all diseases per country
    st.subheader(f"Disease Burden by Country — {latest_year}")
    st.caption("Each bar shows the full NCD mortality breakdown per country, colored by disease")

    stacked_df = (
        country_cause.groupby(["Country_Name", "Region", "Cause"], as_index=False)["Deaths"]
        .sum()
        .sort_values("Deaths", ascending=False)
    )

    # Sort countries by total burden
    country_order = (
        stacked_df.groupby("Country_Name")["Deaths"]
        .sum()
        .sort_values(ascending=True)
        .index.tolist()
    )

    fig_stacked = px.bar(
        stacked_df,
        x="Deaths",
        y="Country_Name",
        color="Cause",
        orientation="h",
        color_discrete_map=cause_colors,
        category_orders={"Country_Name": country_order},
        barmode="stack",
    )
    fig_stacked.update_layout(
        **plotly_layout,
        xaxis_title="Mortality Index",
        yaxis_title="",
        height=600,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            font=dict(size=12),
        ),
        margin=dict(l=10, r=20, t=60, b=40),
        yaxis=dict(tickfont=dict(size=11)),
    )
    st.plotly_chart(fig_stacked, use_container_width=True)

    # Donut chart + regional comparison side by side
    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader(f"Share by Cause — {latest_year}")
        st.caption("Proportion of NCD mortality attributed to each disease category")
        global_totals = (
            country_cause.groupby("Cause", as_index=False)["Deaths"].sum()
        )
        fig_donut = px.pie(
            global_totals,
            names="Cause",
            values="Deaths",
            hole=0.55,
            color="Cause",
            color_discrete_map=cause_colors,
        )
        fig_donut.update_layout(**plotly_layout, showlegend=True)
        fig_donut.update_traces(textfont_size=13)
        st.plotly_chart(fig_donut, use_container_width=True)

    with col2:
        st.subheader("Regional Disease Burden")
        st.caption("Total NCD mortality index by region and disease category")
        region_cause = (
            latest_df.groupby(["Region", "Cause"], as_index=False)["Deaths"].sum()
        )
        fig_region = px.bar(
            region_cause,
            x="Deaths",
            y="Region",
            color="Cause",
            orientation="h",
            color_discrete_map=cause_colors,
            barmode="group",
        )
        fig_region.update_layout(
            **plotly_layout,
            xaxis_title="Mortality Index",
            yaxis_title="",
            height=380,
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig_region, use_container_width=True)

    # Key insights
    st.markdown("---")
    st.subheader("Key Insights")
    col1, col2, col3 = st.columns(3)

    top_cvd_country = (
        country_cause[country_cause["Cause"] == "Cardiovascular"]
        .sort_values("Deaths", ascending=False)
        .iloc[0]
    )
    top_cancer_country = (
        country_cause[country_cause["Cause"] == "Cancer"]
        .sort_values("Deaths", ascending=False)
        .iloc[0]
    )
    highest_burden_region = (
        latest_df.groupby("Region")["Deaths"].sum().idxmax()
    )

    for card, (icon, title, text) in enumerate([
        ("🫀", "Cardiovascular Leader",
         f"<strong>{top_cvd_country['Country_Name']}</strong> ({top_cvd_country['Region']}) "
         f"carries the highest cardiovascular disease burden across all 20 tracked countries."),
        ("🎗️", "Cancer Burden",
         f"<strong>{top_cancer_country['Country_Name']}</strong> ({top_cancer_country['Region']}) "
         f"shows the highest cancer mortality index among tracked countries."),
        ("🌍", "Highest Overall Region",
         f"<strong>{highest_burden_region}</strong> carries the highest total NCD disease burden "
         f"across all four disease categories combined."),
    ]):
        with [col1, col2, col3][card]:
            st.markdown(
                f"""
                <div style="
                    background: linear-gradient(135deg, #1E293B, #0F1F35);
                    border-radius: 12px;
                    padding: 18px 20px;
                    border: 1px solid #334155;
                    border-top: 3px solid #0D9488;
                    min-height: 130px;
                ">
                    <div style="font-size:22px;margin-bottom:8px;">{icon}</div>
                    <div style="font-size:13px;font-weight:700;color:#F1F5F9;margin-bottom:6px;">{title}</div>
                    <div style="font-size:13px;color:#94A3B8;line-height:1.6;">{text}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )