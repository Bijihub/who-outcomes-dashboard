from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from models.forecaster import (
    ForecastResult,
    forecast_life_expectancy,
    forecast_region_average,
)
from models.risk_model import (
    RiskModelArtifacts,
    actual_vs_predicted_df,
    predict_risk_score,
    train_risk_model,
)
from utils.risk import compute_risk_tier, compute_risk_score


def _forecast_figure(
    history_df: pd.DataFrame,
    forecast: ForecastResult,
    region_colors: dict[str, str],
    plotly_layout: dict,
) -> go.Figure:
    hist_trace = go.Scatter(
        x=history_df["Year"],
        y=history_df["Life_Expectancy"],
        mode="lines+markers",
        name="Historical",
        line=dict(color="#0D9488"),  # teal solid
    )

    fc = forecast.df
    fc_trace = go.Scatter(
        x=fc["Year"],
        y=fc["Forecast"],
        mode="lines+markers",
        name="Forecast",
        line=dict(color="#A855F7", dash="dash"),  # purple dashed
    )

    ci_trace = go.Scatter(
        x=list(fc["Year"]) + list(fc["Year"][::-1]),
        y=list(fc["Upper_CI"]) + list(fc["Lower_CI"][::-1]),
        fill="toself",
        fillcolor="rgba(13,148,136,0.15)",  # teal 15% opacity
        line=dict(color="rgba(13,148,136,0)"),
        name="95% CI",
        showlegend=True,
    )

    fig = go.Figure([hist_trace, fc_trace, ci_trace])
    fig.update_layout(
        **plotly_layout,
        xaxis_title="Year",
        yaxis_title="Life Expectancy",
    )

    if forecast.last_history_year:
        fig.add_vline(
            x=forecast.last_history_year,
            line_width=2,
            line_dash="dot",
            line_color="#E5E7EB",
        )
        fig.add_annotation(
            x=forecast.last_history_year + 0.5,
            y=max(
                history_df["Life_Expectancy"].max(),
                fc["Forecast"].max(),
            ),
            text="Forecast Period →",
            showarrow=False,
            font=dict(color="#E5E7EB"),
        )
    return fig


def forecast_page(
    master_df: pd.DataFrame,
    region_colors: dict[str, str],
    plotly_layout: dict,
) -> None:
    st.header("Where Are Outcomes Headed?")
    st.caption("5-year forecast using historical WHO trend data.")

    if master_df.empty:
        st.error("No data available for forecasting.")
        return

    countries = sorted(master_df["Country_Name"].dropna().unique())
    selected_country = st.selectbox("Select Country for Forecast", countries)
    country_df = master_df[master_df["Country_Name"] == selected_country]
    country_df = country_df.dropna(subset=["Life_Expectancy"])

    if country_df.empty:
        st.warning("No life expectancy data available for the selected country.")
        return

    # Section A — Life Expectancy Forecast
    st.subheader("Life Expectancy Forecast")
    forecast_res = forecast_life_expectancy(country_df, steps=5)
    hist_df = country_df[["Year", "Life_Expectancy"]].dropna()

    fig = _forecast_figure(hist_df, forecast_res, region_colors, plotly_layout)
    st.plotly_chart(fig, use_container_width=True)

    # 5 forecast value cards
    cols = st.columns(len(forecast_res.df) if not forecast_res.df.empty else 1)
    for col, (_, row) in zip(cols, forecast_res.df.iterrows()):
        with col:
            st.metric(
                label=str(int(row["Year"])),
                value=f"{row['Forecast']:.1f} years",
            )

    # Regional forecast comparison chart
    st.subheader("Regional Forecast Comparison")
    region_results: list[pd.DataFrame] = []
    for region in sorted(master_df["Region"].dropna().unique()):
        reg_res = forecast_region_average(master_df, region, steps=5)
        if reg_res is None or reg_res.df.empty:
            continue
        tmp = reg_res.df.copy()
        tmp["Region"] = region
        region_results.append(tmp)
    if region_results:
        reg_all = pd.concat(region_results, ignore_index=True)
        fig_reg = px.line(
            reg_all,
            x="Year",
            y="Forecast",
            color="Region",
            color_discrete_map=region_colors,
        )
        fig_reg.update_layout(
            **plotly_layout,
            xaxis_title="Year",
            yaxis_title="Forecast Life Expectancy",
        )
        st.plotly_chart(fig_reg, use_container_width=True)
    else:
        st.info("Not enough data to compute regional forecasts.")

    # Section B — Interactive Risk Score Predictor
    st.subheader("Interactive Risk Score Predictor")
    artifacts: RiskModelArtifacts = train_risk_model(master_df)

    with st.sidebar:
        st.markdown("### Model Performance")
        st.metric("R²", f"{artifacts.r2:.3f}")
        st.metric("MAE", f"{artifacts.mae:.2f}")

    current_year = int(master_df["Year"].max())
    # Slider defaults with sensible fallbacks
    le_default = float(hist_df["Life_Expectancy"].iloc[-1]) if not hist_df.empty else 70.0
    ncd_default = float(country_df["NCD_Mortality"].dropna().mean()) if country_df["NCD_Mortality"].notna().any() else 15.0
    if "Admission_Rate" in country_df.columns and country_df["Admission_Rate"].notna().any():
        adm_default = float(country_df["Admission_Rate"].dropna().mean())
    else:
        adm_default = 12.0

    le_input = st.slider("Life Expectancy", 50.0, 90.0, le_default)
    ncd_input = st.slider("NCD Mortality Rate", 5.0, 35.0, ncd_default)
    # Only show Admission slider if we have that feature in the model (4 features)
    show_admission = artifacts.feature_means.shape[0] == 4
    if show_admission:
        adm_input = st.slider("Readmission Rate (Admission Proxy)", 5.0, 30.0, adm_default)
    else:
        adm_input = None

    # Replace any NaNs from widgets with specified defaults
    if np.isnan(le_input):
        le_input = 70.0
    if np.isnan(ncd_input):
        ncd_input = 15.0
    if adm_input is not None and np.isnan(adm_input):
        adm_input = 12.0

    pred_score = predict_risk_score(
        artifacts,
        life_expectancy=le_input,
        ncd_mortality=ncd_input,
        admission_rate=adm_input,
        year=current_year,
    )

    if pred_score < 30:
        risk_label = "Low Risk"
        color = "#22C55E"
    elif pred_score <= 60:
        risk_label = "Medium Risk"
        color = "#F59E0B"
    else:
        risk_label = "High Risk"
        color = "#DC2626"

    st.markdown(
        f"<div style='background-color:{color}33;padding:1rem;border-radius:0.5rem;'>"
        f"<h3 style='margin:0;color:white;'>Predicted Risk Score: {pred_score:.1f} ({risk_label})</h3>"
        "</div>",
        unsafe_allow_html=True,
    )

    # Section C — Scatter Plot: Actual vs Predicted Risk Scores
    st.subheader("Actual vs Predicted Risk Scores")
    scatter_df = actual_vs_predicted_df(master_df, artifacts.model)
    if scatter_df.empty:
        st.info("Not enough data to compute actual vs predicted scatter.")
    else:
        fig_scatter = px.scatter(
            scatter_df,
            x="Actual",
            y="Predicted",
            color="Region",
            color_discrete_map=region_colors,
            hover_data=["Country_Code"],
        )
        max_val = float(max(scatter_df["Actual"].max(), scatter_df["Predicted"].max()))
        fig_scatter.add_trace(
            go.Scatter(
                x=[0, max_val],
                y=[0, max_val],
                mode="lines",
                name="Perfect Prediction",
                line=dict(color="#9CA3AF", dash="dash"),
            )
        )
        fig_scatter.update_layout(
            **plotly_layout,
            xaxis_title="Actual Risk Score",
            yaxis_title="Predicted Risk Score",
            annotations=[
                dict(
                    x=0.05 * max_val,
                    y=0.95 * max_val,
                    text=f"R² = {artifacts.r2:.3f}",
                    showarrow=False,
                    font=dict(color="#E5E7EB"),
                )
            ],
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    # Section D — 2028 Forecast Summary Table
    st.subheader("2028 Forecast Summary")
    last_year = int(master_df["Year"].max())
    target_year = last_year + 5

    rows = []
    for code, group in master_df.groupby("Country_Code"):
        name = group["Country_Name"].iloc[0]
        region = group["Region"].iloc[0]
        hist = group[["Year", "Life_Expectancy"]].dropna()
        if hist.empty:
            continue
        fc = forecast_life_expectancy(hist, steps=5)
        if fc.df.empty or target_year not in fc.df["Year"].values:
            continue
        le_2023 = float(hist[hist["Year"] == last_year]["Life_Expectancy"].iloc[0])
        le_2028 = float(fc.df[fc.df["Year"] == target_year]["Forecast"].iloc[0])
        change = le_2028 - le_2023

        risk_2023 = compute_risk_tier(le_2023)
        risk_2028 = compute_risk_tier(le_2028)
        score_2023 = compute_risk_score(
            life_expectancy=le_2023,
            ncd_mortality=float(group[group["Year"] == last_year]["NCD_Mortality"].fillna(0).iloc[0]),
            admission_rate=float(group[group["Year"] == last_year]["Admission_Rate"].fillna(0).iloc[0]) if "Admission_Rate" in group.columns else 0.0,
        )
        score_2028 = compute_risk_score(
            life_expectancy=le_2028,
            ncd_mortality=float(group[group["Year"] == last_year]["NCD_Mortality"].fillna(0).iloc[0]),
            admission_rate=float(group[group["Year"] == last_year]["Admission_Rate"].fillna(0).iloc[0]) if "Admission_Rate" in group.columns else 0.0,
        )

        if score_2028 is None or score_2023 is None:
            status = "Stable"
        elif score_2028 < score_2023 - 2:
            status = "Improving"
        elif score_2028 > score_2023 + 2:
            status = "Concern"
        else:
            status = "Stable"

        rows.append(
            {
                "Country": name,
                "Region": region,
                f"LE {last_year}": le_2023,
                f"LE {target_year}": le_2028,
                "Change": change,
                f"Risk Tier {last_year}": risk_2023,
                f"Risk Tier {target_year}": risk_2028,
                "Status": status,
            }
        )

    if rows:
        summary_df = pd.DataFrame(rows)
        # Apply styling
        def highlight_row(row):
            if row["Status"] == "Improving":
                return ["background-color: rgba(34,197,94,0.15)"] * len(row)
            if row["Status"] == "Concern":
                return ["background-color: rgba(239,68,68,0.15)"] * len(row)
            return [""] * len(row)

        st.dataframe(
            summary_df.style.apply(highlight_row, axis=1).format(
                {f"LE {last_year}": "{:.1f}", f"LE {target_year}": "{:.1f}", "Change": "{:+.1f}"}
            ),
            use_container_width=True,
        )
        csv_bytes = summary_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download 2028 Forecast Summary CSV",
            data=csv_bytes,
            file_name="forecast_2028_summary.csv",
            mime="text/csv",
        )
    else:
        st.info("Not enough data to build a 2028 forecast summary.")

