from __future__ import annotations

import base64
from pathlib import Path

import streamlit as st

from components.country_spotlight import country_spotlight_page
from components.disease_burden import disease_burden_page
from components.forecast_page import forecast_page
from components.hero import render_hero_kpis
from components.insights import insights_page
from components.regional_chart import regional_overview_charts
from data.transform import build_master_dataframe

REGION_COLORS = {
    "Africa": "#F59E0B",
    "Europe": "#0D9488",
    "Asia": "#6366F1",
    "North America": "#EC4899",
}

PLOTLY_LAYOUT = {
    "template": "plotly_dark",
    "paper_bgcolor": "#1E293B",
    "plot_bgcolor": "#1E293B",
    "font": {"color": "white"},
}


def inject_css() -> None:
    st.markdown(
        """
        <style>
        header[data-testid="stHeader"] { display: none; }
        #MainMenu { display: none; }
        .stApp { background-color: #0F172A; }
        .block-container { padding-top: 1.5rem; padding-bottom: 1.5rem; }
        section[data-testid="stSidebar"] { display: none; }
        div[data-testid="stHorizontalBlock"] { gap: 0.75rem; }

        /* Nav bar styling */
        .nav-container {
            display: flex;
            gap: 8px;
            background: #0A1628;
            padding: 12px 20px;
            border-radius: 12px;
            border: 1px solid #1E293B;
            margin-bottom: 20px;
            align-items: center;
        }
        .nav-title {
            font-size: 15px;
            font-weight: 700;
            color: #F1F5F9;
            margin-right: 16px;
            white-space: nowrap;
        }

        /* Filter bar styling */
        .filter-bar {
            background: #1E293B;
            border-radius: 12px;
            padding: 16px 20px;
            border: 1px solid #334155;
            margin-bottom: 20px;
        }

        /* Streamlit selectbox and multiselect dark styling */
        .stSelectbox > div > div {
            background-color: #0F172A !important;
            color: #F1F5F9 !important;
            border: 1px solid #334155 !important;
            border-radius: 8px !important;
        }
        .stMultiSelect > div > div {
            background-color: #0F172A !important;
            border: 1px solid #334155 !important;
            border-radius: 8px !important;
        }
        .stSlider > div { color: #F1F5F9; }
        label { color: #94A3B8 !important; font-size: 11px !important;
                letter-spacing: 0.5px; text-transform: uppercase; }
        .stMultiSelect span[data-baseweb="tag"] {
            background-color: #0D9488 !important;
            color: white !important;
            border-radius: 6px !important;
        }
        div[data-testid="stVerticalBlockBorderWrapper"] {
            background: #0A1628 !important;
            border: 1px solid #1E293B !important;
            border-radius: 12px !important;
            padding: 8px 16px !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

def get_logo_base64() -> str:
    logo_path = Path(__file__).parent / "assets" / "logo.png"
    if logo_path.exists():
        with open(logo_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return ""

def render_nav(current_page: str) -> str:


    pages = [
        "Regional Overview",
        "Disease Burden Story",
        "Country Spotlight",
        "Predictive Forecast",
        "Stakeholder Insights",
    ]
    icons = ["🌍", "🦠", "🔍", "🔮", "💡"]

    cols = st.columns([1.2] + [1] * len(pages))
    with cols[0]:
        logo_b64 = get_logo_base64()
        if logo_b64:
            st.markdown(
                f'<img src="data:image/png;base64,{logo_b64}" '
                f'style="height:52px;filter:invert(1) sepia(1) '
                f'saturate(5) hue-rotate(140deg);padding-top:4px;">',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div style="font-size:14px;font-weight:700;color:#F1F5F9;'
                'padding-top:8px;">🏥 WHO Outcomes</div>',
                unsafe_allow_html=True,
            )

    selected = current_page
    for i, (page, icon) in enumerate(zip(pages, icons)):
        with cols[i + 1]:
            is_active = page == current_page
            btn_style = (
                "background:#0D9488;color:white;border:none;border-radius:8px;"
                "padding:8px 12px;cursor:pointer;font-size:12px;font-weight:600;"
                "width:100%;white-space:nowrap;"
                if is_active else
                "background:#0F172A;color:#94A3B8;border:1px solid #334155;"
                "border-radius:8px;padding:8px 12px;cursor:pointer;font-size:12px;"
                "width:100%;white-space:nowrap;"
            )
            if st.button(f"{icon} {page}", key=f"nav_{page}", use_container_width=True):
                selected = page

    st.markdown("</div>", unsafe_allow_html=True)
    return selected


def render_filters(master_df) -> tuple:
    with st.container(border=True):
        col1, col2, col3 = st.columns([1, 1, 2])

        with col1:
            region_options = ["All"] + sorted(master_df["Region"].dropna().unique())
            selected_region = st.selectbox("Region", region_options, key="region_filter")

        with col2:
            all_years = sorted(master_df["Year"].dropna().astype(int).unique())
            min_year, max_year = all_years[0], all_years[-1]
            default_start = max(min_year, max_year - 5)
            year_range = st.slider(
                "Year Range",
                min_value=int(min_year),
                max_value=int(max_year),
                value=(int(default_start), int(max_year)),
                key="year_filter",
            )

        with col3:
            st.markdown(
                '<div style="padding-top:24px;font-size:12px;color:#64748B;">'
                'Filtering across 20 countries · 4 regions · WHO Global Health Observatory</div>',
                unsafe_allow_html=True,
            )

    if selected_region == "All":
        codes = master_df["Country_Code"].dropna().unique().tolist()
    else:
        codes = (
            master_df[master_df["Region"] == selected_region]["Country_Code"]
            .dropna().unique().tolist()
        )

    return selected_region, codes, year_range

def get_profile_base64() -> str:
    profile_path = Path(__file__).parent / "assets" / "profile.jpg"
    if profile_path.exists():
        with open(profile_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return ""


def main() -> None:
    st.set_page_config(
        layout="wide",
        page_title="WHO Outcomes Dashboard",
        page_icon="assets/favicon.png",
    )
    inject_css()

    # Session state for page navigation
    if "current_page" not in st.session_state:
        st.session_state.current_page = "Regional Overview"

    selected = render_nav(st.session_state.current_page)
    if selected != st.session_state.current_page:
        st.session_state.current_page = selected
        st.rerun()

    page = st.session_state.current_page

    with st.spinner("Fetching WHO data..."):
        master_df = build_master_dataframe()

    if master_df.empty:
        st.error("No WHO data could be loaded. Please try again later.")
        return

    # Top filter bar (shown on Regional Overview only)
    if page == "Regional Overview":
        title_col, profile_col = st.columns([3, 1])
        with title_col:
            st.header("Regional Overview")
        with profile_col:
            img_b64 = get_profile_base64()
            if img_b64:
                st.markdown(
                    f'<div style="display:flex;align-items:center;gap:8px;justify-content:flex-end;padding-top:8px;">'
                    f'<img src="data:image/jpeg;base64,{img_b64}" style="width:45px;height:45px;border-radius:50%;object-fit:cover;object-position:center 80%;border:2px solid #0D9488;flex-shrink:0;">'
                    f'<div><div style="font-size:18px;font-weight:700;color:#F1F5F9;">Osondu Samuel Elijah</div>'
                    f'<div style="font-size:12px;color:#0D9488;">Data Analyst</div></div></div>',
                    unsafe_allow_html=True,
                )
        if "region_filter" in st.session_state:
            cap_region = st.session_state["region_filter"]
        else:
            cap_region = "All"
        if cap_region == "All":
            st.caption("Macro-level view of life expectancy, risk, and mortality across 4 regions.")
        else:
            country_count = len(master_df[master_df["Region"] == cap_region]["Country_Code"].unique())
            st.caption(f"Macro-level view of life expectancy, risk, and mortality across {country_count} countries in {cap_region}.")
        selected_region, codes, year_range = render_filters(master_df)
        filtered_df = master_df[master_df["Country_Code"].isin(codes)]
        render_hero_kpis(filtered_df, year=year_range[1], prev_year=year_range[1] - 1)
        regional_overview_charts(
            master_df,
            region_filter=selected_region,
            country_filter=codes,
            year_range=year_range,
            region_colors=REGION_COLORS,
            plotly_layout=PLOTLY_LAYOUT,
        )

    elif page == "Disease Burden Story":
        disease_burden_page(
            master_df,
            region_colors=REGION_COLORS,
            plotly_layout=PLOTLY_LAYOUT,
        )

    elif page == "Country Spotlight":
        country_spotlight_page(
            master_df,
            region_colors=REGION_COLORS,
            plotly_layout=PLOTLY_LAYOUT,
        )

    elif page == "Predictive Forecast":
        forecast_page(
            master_df,
            region_colors=REGION_COLORS,
            plotly_layout=PLOTLY_LAYOUT,
        )

    elif page == "Stakeholder Insights":
        insights_page(master_df)


if __name__ == "__main__":
    main()