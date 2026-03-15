from __future__ import annotations

import pandas as pd
import streamlit as st

from utils.risk import compute_risk_tier


def insight_card(title: str, body: str, icon: str, color: str) -> None:
    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, #1E293B, #0F1F35);
            border-radius: 12px;
            padding: 20px 22px;
            border: 1px solid #334155;
            border-top: 3px solid {color};
            min-height: 160px;
            margin-bottom: 8px;
        ">
            <div style="display:flex;align-items:center;gap:10px;margin-bottom:12px;">
                <span style="font-size:22px;">{icon}</span>
                <span style="font-size:13px;font-weight:700;color:#F1F5F9;">{title}</span>
            </div>
            <div style="font-size:18px;color:#94A3B8;line-height:1.7;">{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def insights_page(master_df: pd.DataFrame) -> None:
    st.header("Stakeholder Insights")
    st.caption("High-level storylines and downloadable summary for decision-makers.")

    if master_df.empty:
        st.error("No data available to generate insights.")
        return

    latest_year = int(master_df["Year"].max())
    latest_df = master_df[master_df["Year"] == latest_year]

    # ── Insight 1: Long-term trajectory ──
    prev_year = latest_year - 5
    prev_df = master_df[master_df["Year"] == prev_year]
    if not prev_df.empty:
        le_change = latest_df["Life_Expectancy"].mean() - prev_df["Life_Expectancy"].mean()
        text1 = (
            f"Average life expectancy across the 20 tracked countries has "
            f"changed by <strong>{le_change:+.1f} years</strong> over the last 5-year window."
        )
    else:
        text1 = "Trend data is limited, but current life expectancy levels already reveal significant cross-country gaps."

    # ── Insight 2: Highest risk region ──
    region_risk = latest_df.groupby("Region")["Risk_Score"].mean().dropna()
    if not region_risk.empty:
        worst_region = region_risk.idxmax()
        text2 = (
            f"<strong>{worst_region}</strong> shows the highest composite risk score, "
            f"indicating a combination of lower life expectancy, higher NCD mortality, "
            f"and elevated admission rates."
        )
    else:
        text2 = "Risk scores are incomplete, but early signals suggest meaningful variation across regions."

    # ── Insight 3: Best performing region ──
    if not region_risk.empty:
        best_region = region_risk.idxmin()
        text3 = (
            f"<strong>{best_region}</strong> demonstrates comparatively favorable outcomes, "
            f"offering a benchmark for what is achievable within the existing resource envelope."
        )
    else:
        text3 = "Benchmarks for top-performing regions will strengthen as more data accumulates."

    # ── Insight 4: North America inequality ──
    na_df = latest_df[latest_df["Region"] == "North America"]
    if not na_df.empty:
        max_country = na_df.loc[na_df["Life_Expectancy"].idxmax()]
        min_country = na_df.loc[na_df["Life_Expectancy"].idxmin()]
        gap = max_country["Life_Expectancy"] - min_country["Life_Expectancy"]
        text4 = (
            f"North America shows the highest healthcare spend per capita yet the "
            f"<strong>{max_country['Country_Name']} / {min_country['Country_Name']}</strong> gap "
            f"reveals stark within-region inequality. Life expectancy gap: "
            f"<strong>{gap:.1f} years</strong>."
        )
    else:
        text4 = "North America highlights important within-region inequalities across its countries."

    # ── Insight 5: High-risk country focus ──
    high_risk = (
        latest_df.sort_values("Risk_Score", ascending=False)
        .head(3)[["Country_Name", "Region", "Risk_Score"]]
    )
    lines = "".join([
        f"<div style='margin:4px 0;'>🔴 <strong>{r['Country_Name']} ({r['Region']})</strong>"
        f" — Risk Score {r['Risk_Score']:.1f}</div>"
        for _, r in high_risk.iterrows()
    ])
    text5 = f"Countries with the highest composite risk warranting targeted interventions:<br><br>{lines}"

    # ── Insight 6: Predictive forward signal ──
    improvements = 0
    regions_with_improvement = set()
    for code, group in master_df.groupby("Country_Code"):
        group = group.sort_values("Year")
        if latest_year not in group["Year"].values:
            continue
        le_last = float(group[group["Year"] == latest_year]["Life_Expectancy"].iloc[0])
        if group["Year"].nunique() >= 2:
            y = group["Life_Expectancy"].values.astype(float)
            x = group["Year"].values.astype(float)
            x_mean = x.mean()
            y_mean = y.mean()
            coef = ((x - x_mean) * (y - y_mean)).sum() / ((x - x_mean) ** 2).sum()
            delta_5y = coef * 5
        else:
            delta_5y = 0.3
        le_future = le_last + delta_5y
        tier_now = compute_risk_tier(le_last)
        tier_future = compute_risk_tier(le_future)
        if tier_now and tier_future and tier_future != tier_now:
            improvements += 1
            regions_with_improvement.add(group["Region"].iloc[0])

    text6 = (
        f"By <strong>{latest_year + 5}</strong>, <strong>{improvements} countries</strong> across "
        f"<strong>{len(regions_with_improvement)} regions</strong> are projected to improve "
        f"their risk tier if current improvement trends hold."
    )

    # ── Render 6 cards in 2 rows of 3 ──
    cards = [
        ("📈", "Long-term Trajectory", text1, "#0D9488"),
        ("🌍", "Highest Risk Region", text2, "#EF4444"),
        ("🏆", "Best Performing Region", text3, "#10B981"),
        ("🌎", "North America Inequality", text4, "#EC4899"),
        ("🎯", "High-Risk Country Focus", text5, "#F59E0B"),
        ("🔮", "Forward-Looking Signal", text6, "#A78BFA"),
    ]

    col1, col2, col3 = st.columns(3)
    for i, (icon, title, body, color) in enumerate(cards):
        with [col1, col2, col3][i % 3]:
            insight_card(title, body, icon, color)

    # ── Summary table ──
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("Full Outcomes & Risk Summary")
    summary_cols = [
        "Country_Name", "Region", "Year",
        "Life_Expectancy", "NCD_Mortality",
        "Risk_Tier", "Risk_Score",
    ]
    if "Admission_Rate" in master_df.columns:
        summary_cols.insert(5, "Admission_Rate")

    summary_df = master_df[summary_cols].sort_values(["Region", "Country_Name", "Year"])
    st.dataframe(summary_df, use_container_width=True)

    csv_bytes = summary_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Download Full Outcomes Table CSV",
        data=csv_bytes,
        file_name="who_outcomes_full_table.csv",
        mime="text/csv",
    )