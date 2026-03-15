from __future__ import annotations
import streamlit as st
import pandas as pd


def kpi_card(label: str, value: str, delta: str | None = None, delta_color: str = "#EF4444"):
    delta_html = f'<div style="font-size:12px;color:{delta_color};margin-top:6px;">{delta}</div>' if delta else ""
    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, #1E293B, #0F1F35);
            border-radius: 12px;
            padding: 20px 22px;
            border: 1px solid #334155;
            border-top: 3px solid #0D9488;
            min-height: 110px;
        ">
            <div style="font-size:11px;color:#64748B;letter-spacing:1px;
                        text-transform:uppercase;margin-bottom:8px;">{label}</div>
            <div style="font-size:24px;font-weight:700;color:#F1F5F9;">{value}</div>
            {delta_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_hero_kpis(master_df: pd.DataFrame, year: int | None = None, prev_year: int | None = None):
    if master_df.empty:
        st.warning("No data available to display KPIs.")
        return

    if year is None:
        year = int(master_df["Year"].max())
    if prev_year is None:
        prev_year = year - 1

    current_df = master_df[master_df["Year"] == year]
    prev_df = master_df[master_df["Year"] == prev_year]

    # Avg Life Expectancy
    avg_le = current_df["Life_Expectancy"].mean()
    prev_avg_le = prev_df["Life_Expectancy"].mean() if not prev_df.empty else None
    le_delta = None
    le_delta_color = "#10B981"
    if prev_avg_le is not None:
        diff = avg_le - prev_avg_le
        le_delta = f"{'↑' if diff >= 0 else '↓'} {diff:+.1f} vs {prev_year}"
        le_delta_color = "#10B981" if diff >= 0 else "#EF4444"

    # Highest Risk Region
    region_risk = current_df.groupby("Region")["Risk_Score"].mean().dropna()
    highest_risk_region = region_risk.idxmax() if not region_risk.empty else "N/A"

    # Countries tracked
    countries_tracked = current_df["Country_Code"].nunique()

    # Top cause placeholder
    top_cause_text = "Cardiovascular"

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        kpi_card(
            "Avg Life Expectancy",
            f"{avg_le:.1f} yrs" if pd.notna(avg_le) else "N/A",
            le_delta,
            le_delta_color,
        )
    with col2:
        kpi_card("Highest Risk Region", highest_risk_region, "Based on Risk Score", "#F59E0B")
    with col3:
        kpi_card("Top Cause of Death", top_cause_text, "38% of NCD deaths", "#F59E0B")
    with col4:
        kpi_card("Countries Tracked", str(countries_tracked), "Across 4 regions", "#0D9488")