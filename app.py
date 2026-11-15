"""
ELYSIUM — Mars Terrain Analysis & Mission Planning Dashboard
Run from the project root with: streamlit run app.py

This app orchestrates the existing project modules. It does not replace or
modify the processing, analysis, or simulation implementations.
"""

from __future__ import annotations

import traceback
import json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import (
    PROJECT_NAME,
    PROJECT_VERSION,
    PLANET_NAME,
    DETAILED_AOI,
    INSIGHT_SITE,
    OUTPUT_DIR,
    validate_configuration,
)
from processing.mola import process_mola
from data_engine.discovery import discover_observations
from processing.ctx import build_ctx_catalogue, OUTPUT_FILE as CTX_CATALOGUE_FILE
from processing.hirise import run_hirise_pipeline, CATALOGUE_FILE as HIRISE_CATALOGUE_FILE
from processing.themis import process_themis, THEMIS_INVENTORY_PATH
from processing.footprint_analysis import analyse_footprints
from processing.footprint_map import create_footprint_map
from analysis.elevation import analyse_elevation
from analysis.roughness import analyse_roughness
from analysis.slope import analyse_slope
from analysis.terrain_cost import analyse_terrain_cost
from simulation.mission import run_mission, generate_mission_report
from simulation.route_planner import DEFAULT_START
from simulation.terrain_3d import create_terrain_3d

# ---------------------------------------------------------------------
# PAGE CONFIGURATION AND THEME
# ---------------------------------------------------------------------

st.set_page_config(
    page_title="ELYSIUM | Mars Terrain Intelligence",
    page_icon="🔴",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Manrope', sans-serif;
    }
    .stApp {
        background:
            radial-gradient(circle at 85% 0%, rgba(63, 105, 166, 0.16), transparent 28%),
            linear-gradient(180deg, #0d1420 0%, #080d16 100%);
        color: #edf0f2;
    }
    [data-testid="stSidebar"] {
        background: #0d1724;
        border-right: 1px solid rgba(88, 155, 206, 0.20);
    }
    [data-testid="stHeader"] {
        background: rgba(11, 14, 17, 0.75);
    }
    h1, h2, h3 {
        letter-spacing: -0.035em;
    }
    .eyebrow {
        color: #79c7e8;
        font-family: 'DM Mono', monospace;
        font-size: 0.72rem;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        margin-bottom: 0.45rem;
    }
    .hero-title {
        font-size: clamp(2.2rem, 5vw, 4.2rem);
        line-height: 1.0;
        font-weight: 800;
        margin: 0 0 0.6rem 0;
    }
    .hero-subtitle {
        color: #aeb8bd;
        max-width: 780px;
        font-size: 1.02rem;
        line-height: 1.7;
    }
    .section-note {
        color: #9ba7ad;
        font-size: 0.88rem;
        line-height: 1.6;
    }
    .metric-card {
        background: linear-gradient(145deg, rgba(25, 39, 58, 0.94), rgba(13, 22, 35, 0.96));
        border: 1px solid rgba(93, 161, 205, 0.22);
        border-radius: 14px;
        padding: 1rem 1.1rem;
        min-height: 112px;
    }
    .metric-label {
        color: #aeb8bd;
        font-family: 'DM Mono', monospace;
        font-size: 0.69rem;
        letter-spacing: 0.09em;
        text-transform: uppercase;
        margin-bottom: 0.55rem;
    }
    .metric-value {
        color: #eaf5ff;
        font-size: 1.7rem;
        font-weight: 800;
        line-height: 1.15;
    }
    .metric-help {
        color: #87949a;
        font-size: 0.75rem;
        margin-top: 0.35rem;
    }
    .status-pill {
        display: inline-block;
        border: 1px solid rgba(92, 177, 213, 0.48);
        color: #8bd7f2;
        background: rgba(92, 177, 213, 0.10);
        border-radius: 99px;
        padding: 0.25rem 0.65rem;
        font-family: 'DM Mono', monospace;
        font-size: 0.68rem;
        letter-spacing: 0.08em;
    }
    div[data-testid="stMetric"] {
        background: rgba(22, 37, 54, 0.78);
        border: 1px solid rgba(93, 161, 205, 0.18);
        padding: 0.8rem 0.95rem;
        border-radius: 12px;
    }
    div[data-testid="stMetricLabel"] {
        color: #aeb8bd;
    }
    div[data-testid="stMetricValue"] {
        color: #eaf5ff;
    }
    .stButton > button, .stDownloadButton > button {
        border-radius: 9px;
        border: 1px solid rgba(100, 195, 225, 0.48);
        background: #176b91;
        color: white;
        font-weight: 700;
    }
    .stButton > button:hover, .stDownloadButton > button:hover {
        border-color: #8bdcf5;
        background: #2185ad;
        color: white;
    }
    hr { border-color: rgba(93, 161, 205, 0.18); }
    [data-testid="stVerticalBlock"] > [data-testid="stHorizontalBlock"] { gap: 1rem; }
    .block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 1500px; }
    div[data-testid="stDataFrame"] { border: 1px solid rgba(93,161,205,.18); border-radius: 12px; overflow: hidden; }
    .stTabs [data-baseweb="tab-list"] { gap: .45rem; }
    .stTabs [data-baseweb="tab"] { background: rgba(27,45,65,.55); border-radius: 8px 8px 0 0; padding: .55rem .9rem; }
    .eyebrow { color: #79c7e8; }
    .hero-title span { color: #79c7e8 !important; }
    
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------
# DATA PIPELINE
# ---------------------------------------------------------------------

@st.cache_data(show_spinner=False, max_entries=2)
def load_and_analyse_terrain():
    """
    Run the existing MOLA -> elevation -> roughness/slope -> terrain-cost
    pipeline. Cached results prevent repeated downloads and calculations
    during normal Streamlit reruns.
    """
    validate_configuration()

    region_data = process_mola()
    elevation_results = analyse_elevation(region_data)
    roughness_results = analyse_roughness(elevation_results)
    slope_results = analyse_slope(elevation_results)
    terrain_results = analyse_terrain_cost(
        elevation_results,
        roughness_results,
        slope_results,
    )

    return {
        "region": region_data,
        "elevation": elevation_results,
        "roughness": roughness_results,
        "slope": slope_results,
        "terrain": terrain_results,
    }

# ---------------------------------------------------------------------
# SMALL PRESENTATION HELPERS
# ---------------------------------------------------------------------

def metric_card(label: str, value: str, help_text: str = ""):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-help">{help_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def mars_heatmap(
    z,
    longitude,
    latitude,
    title,
    colorbar_title,
    colorscale="Inferno",
    zmin=None,
    zmax=None,
    zsmooth=None,
):
    fig = go.Figure(
        go.Heatmap(
            z=np.asarray(z),
            x=np.asarray(longitude),
            y=np.asarray(latitude),
            colorscale=colorscale,
            zmin=zmin,
            zmax=zmax,
            zsmooth=zsmooth,
            colorbar={"title": {"text": colorbar_title, "font": {"color": "#e8edef"}}},
            hovertemplate=(
                "Longitude: %{x:.4f}°E<br>"
                "Latitude: %{y:.4f}°N<br>"
                f"{colorbar_title}: %{{z:.3f}}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title={"text": title, "x": 0.02, "xanchor": "left"},
        height=520,
        margin={"l": 15, "r": 15, "t": 55, "b": 15},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#111619",
        font={"color": "#e8edef"},
        xaxis={
            "title": "Longitude (°E)",
            "gridcolor": "rgba(255,255,255,0.08)",
            "zeroline": False,
        },
        yaxis= {
            "title": "Latitude (°N)",
            "gridcolor": "rgba(255,255,255,0.08)",
            "zeroline": False,
            "scaleanchor": "x",
            "scaleratio": 1,
        },
    )
    return fig


def terrain_3d_figure(elevation_results):
    """Build an interactive 3D surface from the detailed-area MOLA grid."""
    lon = elevation_results["longitude_1d"]
    lat = elevation_results["latitude_1d"]
    elev = elevation_results["elevation_m"]

    # Use the actual MOLA elevation values; exaggeration is only a display
    # setting and is clearly identified in the plot controls.
    vertical_exaggeration = 2.5
    z0 = float(np.nanmedian(elev))
    z_display = (np.asarray(elev, dtype=float) - z0) * vertical_exaggeration

    fig = go.Figure(
        data=[
            go.Surface(
                x=lon,
                y=lat,
                z=z_display,
                surfacecolor=elev,
                customdata=np.asarray(elev),
                colorscale=[
                    [0.0, "#241b1a"],
                    [0.22, "#553126"],
                    [0.48, "#8c4932"],
                    [0.72, "#c07850"],
                    [1.0, "#e3b18a"],
                ],
                colorbar={
                    "title": "MOLA elevation<br>(m)",
                    "tickfont": {"color": "#e8edef"},
                    "title": {"text": "MOLA elevation<br>(m)", "font": {"color": "#e8edef"}},
                },
                contours={
                    "z": {
                        "show": True,
                        "usecolormap": True,
                        "project_z": True,
                        "width": 1,
                    }
                },
                hovertemplate=(
                    "Longitude: %{x:.4f}°E<br>"
                    "Latitude: %{y:.4f}°N<br>"
                    "MOLA elevation: %{customdata:.1f} m<extra></extra>"
                ),
            )
        ]
    )
    fig.update_layout(
        title="Western Elysium Planitia — MOLA topography",
        height=680,
        margin={"l": 0, "r": 0, "t": 55, "b": 0},
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "#e8edef"},
        scene={
            "xaxis_title": "Longitude (°E)",
            "yaxis_title": "Latitude (°N)",
            "zaxis_title": "Relative relief (m, ×2.5)",
            "bgcolor": "#0d1113",
            "aspectmode": "manual",
            "aspectratio": {"x": 1.35, "y": 1.0, "z": 0.20},
            "camera": {"eye": {"x": 1.55, "y": 1.45, "z": 0.85}},
        },
    )
    return fig


def make_terrain_dataframe(data):
    elev = data["elevation"]
    terrain = data["terrain"]
    slope = data["slope"]
    roughness = data["roughness"]

    return pd.DataFrame(
        {
            "latitude_deg": elev["latitude"].ravel(),
            "longitude_deg": elev["longitude"].ravel(),
            "elevation_m": elev["elevation_m"].ravel(),
            "slope_degrees": slope["slope_degrees"].ravel(),
            "roughness_TRI_m": roughness["tri_m"].ravel(),
            "local_std_m": roughness["local_std_m"].ravel(),
            "terrain_cost": terrain["terrain_cost"].ravel(),
            "terrain_class": terrain["terrain_classes"].ravel(),
            "valid": terrain["valid_mask"].ravel(),
        }
    )


def show_load_error(error):
    st.error("The terrain pipeline could not be completed.")
    st.code(f"{type(error).__name__}: {error}")
    with st.expander("Technical traceback"):
        st.code(traceback.format_exc())
    st.info(
        "Check that the project folders and Python packages are present. "
        "On the first run, MOLA data must be downloaded from the PDS server."
    )


# ---------------------------------------------------------------------
# HEADER AND SIDEBAR
# ---------------------------------------------------------------------

st.markdown(
    '<div class="eyebrow">PLANETARY SURFACE INTELLIGENCE / MARS</div>'
    '<div style="display:flex;align-items:flex-end;justify-content:space-between;gap:1rem;flex-wrap:wrap;">'
    '<div><div class="hero-title">ELYSIUM<span>.</span></div>'
    '<div class="hero-subtitle">Orbital observations · Terrain intelligence · Mission simulation</div></div>'
    '<div class="status-pill">WESTERN ELYSIUM PLANITIA</div></div>',
    unsafe_allow_html=True,
)
st.write("")

with st.sidebar:
    st.markdown("## 🔴 ELYSIUM")
    st.caption(f"{PROJECT_NAME} · v{PROJECT_VERSION}")
    st.markdown("---")
    page = st.radio(
        "WORKSPACE",
        [
            "Mission Control",
            "Terrain Intelligence",
            "3D Terrain Explorer",
            "Surface Mapping",
            "Mission Planner",
            "Orbital Data Explorer",
            "Data & Methodology",
        ],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown("**STUDY AREA**")
    st.caption(DETAILED_AOI["name"])
    st.caption(
        f'{DETAILED_AOI["west"]:.1f}–{DETAILED_AOI["east"]:.1f}°E'
    )
    st.caption(
        f'{DETAILED_AOI["south"]:.1f}–{DETAILED_AOI["north"]:.1f}°N'
    )
    st.markdown("---")
    if st.button("Reload MOLA terrain", use_container_width=True):
        load_and_analyse_terrain.clear()
        st.session_state.pop("elysium_data", None)
        st.session_state.pop("last_mission", None)
        st.session_state.pop("terrain_load_error", None)
        st.session_state.pop("terrain_load_traceback", None)
        st.rerun()


# ---------------------------------------------------------------------
# LOAD DATA — ONLY FOR TERRAIN-DEPENDENT PAGES
# ---------------------------------------------------------------------

TERRAIN_PAGES = {
    "Mission Control",
    "Terrain Intelligence",
    "3D Terrain Explorer",
    "Mission Planner",
}

if page in TERRAIN_PAGES and "elysium_data" not in st.session_state:
    with st.spinner("Loading MOLA and calculating terrain intelligence..."):
        try:
            st.session_state["elysium_data"] = load_and_analyse_terrain()
            st.session_state["terrain_load_error"] = None
        except Exception as exc:
            st.session_state["terrain_load_error"] = f"{type(exc).__name__}: {exc}"
            st.session_state["terrain_load_traceback"] = traceback.format_exc()

data = st.session_state.get("elysium_data")
load_error = st.session_state.get("terrain_load_error")

if page in TERRAIN_PAGES and data is None:
    st.error("MOLA terrain data is not available, so this workspace cannot be displayed yet.")
    if load_error:
        st.code(load_error)
    with st.expander("Technical traceback"):
        st.code(st.session_state.get("terrain_load_traceback", "No traceback available."))
    st.info("The Orbital Data Explorer and Surface Mapping workspaces do not require the MOLA analysis pipeline and remain available from the sidebar.")
    st.stop()

if data is not None:
    elevation = data["elevation"]
    roughness = data["roughness"]
    slope = data["slope"]
    terrain = data["terrain"]
    elev_stats = elevation["statistics"]
    slope_stats = slope["statistics"]
    rough_stats = roughness["tri_statistics"]
    cost_stats = terrain["statistics"]


# ---------------------------------------------------------------------
# PAGE 1 — MISSION OVERVIEW
# ---------------------------------------------------------------------

if page == "Mission Control":
    regional_elevation = elevation
    regional_terrain = terrain
    st.markdown('<div class="eyebrow">01 / MISSION CONTROL</div>', unsafe_allow_html=True)
    st.header("Terrain intelligence at a glance")
    st.markdown(
        '<p class="section-note">A compact view of the current MOLA-derived '
        'study-area model. Values below describe the processed grid, not '
        'real-time surface conditions.</p>',
        unsafe_allow_html=True,
    )
    status_cols = st.columns(3)
    for col, label, path in [
        (status_cols[0], "CTX catalogue", CTX_CATALOGUE_FILE),
        (status_cols[1], "HiRISE catalogue", HIRISE_CATALOGUE_FILE),
        (status_cols[2], "THEMIS inventory", THEMIS_INVENTORY_PATH),
    ]:
        col.caption(f"{label}: {'AVAILABLE' if Path(path).exists() else 'NOT BUILT'}")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Valid terrain cells", f'{cost_stats["valid_pixels"]:,}')
    with c2:
        st.metric("Mean elevation", f'{elev_stats["mean_m"]:,.1f} m')
    with c3:
        st.metric("Mean slope", f'{slope_stats["mean"]:.3f}°')
    with c4:
        st.metric("Mean terrain cost", f'{cost_stats["mean"]:.4f}')

    st.write("")
    elevation_display_min = np.nanpercentile(
    regional_elevation["elevation_m"], 2
    )
    elevation_display_max = np.nanpercentile(
        regional_elevation["elevation_m"], 98
    )
    left, right = st.columns([1, 1])
    with left:
        st.plotly_chart(
            mars_heatmap(
                regional_elevation["elevation_m"],
                regional_elevation["longitude_1d"],
                regional_elevation["latitude_1d"],
                "MOLA elevation model",
                "Elevation (m)",
                colorscale=[
                    [0.0, "#211713"],
                    [0.15, "#49271e"],
                    [0.30, "#713a29"],
                    [0.45, "#a45132"],
                    [0.60, "#c87543"],
                    [0.75, "#df9860"],
                    [0.90, "#edbd88"],
                    [1.0, "#fff0c2"],
                ],
            zmin=elevation_display_min,
            zmax=elevation_display_max,
            ),
            use_container_width=True,
        )
    with right:
        st.plotly_chart(
            mars_heatmap(
                regional_terrain["terrain_cost"],
                regional_terrain["longitude"][0, :],
                regional_terrain["latitude"][:, 0],
                "Relative terrain traversal cost",
                "Cost",
                colorscale="Inferno",
                zmin=0,
                zmax=1,
            ),
            use_container_width=True,
        )

    st.markdown("### Study-area profile")
    p1, p2, p3 = st.columns(3)
    with p1:
        st.metric("Elevation range", f'{elev_stats["minimum_m"]:.0f} to {elev_stats["maximum_m"]:.0f} m')
    with p2:
        st.metric("MOLA grid", f'{elevation["elevation_m"].shape[1]} × {elevation["elevation_m"].shape[0]}')
    with p3:
        st.metric("Grid spacing", f'{elevation["grid_spacing_km"]["east_west_km"]:.2f} km EW')

    st.info(
        "Interpretation: the terrain-cost surface is a relative index derived "
        "from MOLA-scale slope and roughness. It is not a direct measurement "
        "of vehicle performance or astronaut safety."
    )


# ---------------------------------------------------------------------
# PAGE 2 — TERRAIN ANALYSIS
# ---------------------------------------------------------------------

elif page == "Terrain Intelligence":
    st.markdown('<div class="eyebrow">02 / TERRAIN INTELLIGENCE</div>', unsafe_allow_html=True)
    st.header("Terrain analysis")
    st.markdown(
        '<p class="section-note">Inspect the component layers used to derive '
        'the terrain traversal-cost surface.</p>',
        unsafe_allow_html=True,
    )

    layer = st.selectbox(
        "Select terrain layer",
        [
            "Elevation",
            "Slope",
            "Roughness (TRI)",
            "Local elevation variability",
            "Terrain traversal cost",
            "Terrain cost classes",
        ],
    )

    if layer == "Elevation":
        z = elevation["elevation_m"]
        title, unit, scale = "MOLA elevation", "m", [
            [0.0, "#211713"],
            [0.15, "#49271e"],
            [0.30, "#713a29"],
            [0.45, "#a45132"],
            [0.60, "#c87543"],
            [0.75, "#df9860"],
            [0.90, "#edbd88"],
            [1.0, "#fff0c2"],
        ]
        st_stats = elev_stats
        metric_values = [
            ("Minimum", f'{st_stats["minimum_m"]:.2f} m'),
            ("Maximum", f'{st_stats["maximum_m"]:.2f} m'),
            ("Mean", f'{st_stats["mean_m"]:.2f} m'),
            ("Median", f'{st_stats["median_m"]:.2f} m'),
        ]
    elif layer == "Slope":
        z = slope["slope_degrees"]
        title, unit, scale = "Slope gradient", "degrees", "YlOrBr"
        st_stats = slope_stats
        metric_values = [
            ("Minimum", f'{st_stats["minimum"]:.3f}°'),
            ("Maximum", f'{st_stats["maximum"]:.3f}°'),
            ("Mean", f'{st_stats["mean"]:.3f}°'),
            ("95th percentile", f'{st_stats["percentile_95"]:.3f}°'),
        ]
    elif layer == "Roughness (TRI)":
        z = roughness["tri_m"]
        title, unit, scale = "Terrain Ruggedness Index (8-neighbour)", "m", "Inferno"
        st_stats = rough_stats
        metric_values = [
            ("Minimum", f'{st_stats["minimum_m"]:.3f} m'),
            ("Maximum", f'{st_stats["maximum_m"]:.3f} m'),
            ("Mean", f'{st_stats["mean_m"]:.3f} m'),
            ("Median", f'{st_stats["median_m"]:.3f} m'),
        ]
    elif layer == "Local elevation variability":
        z = roughness["local_std_m"]
        title, unit, scale = "Local 3 × 3 elevation standard deviation", "m", "Magma"
        st_stats = roughness["local_std_statistics"]
        metric_values = [
            ("Minimum", f'{st_stats["minimum_m"]:.3f} m'),
            ("Maximum", f'{st_stats["maximum_m"]:.3f} m'),
            ("Mean", f'{st_stats["mean_m"]:.3f} m'),
            ("Median", f'{st_stats["median_m"]:.3f} m'),
        ]
    elif layer == "Terrain traversal cost":
        z = terrain["terrain_cost"]
        title, unit, scale = "Combined terrain traversal cost", "relative index", "Inferno"
        st_stats = cost_stats
        metric_values = [
            ("Minimum", f'{st_stats["minimum"]:.4f}'),
            ("Maximum", f'{st_stats["maximum"]:.4f}'),
            ("Mean", f'{st_stats["mean"]:.4f}'),
            ("Median", f'{st_stats["median"]:.4f}'),
        ]
    else:
        z = terrain["terrain_classes"]
        title, unit, scale = "Terrain traversal-cost classes", "class", "Turbo"
        metric_values = [
            ("Class 0", f'{terrain["class_distribution"].get(0, {}).get("pixels", 0):,} cells'),
            ("Class 1", f'{terrain["class_distribution"].get(1, {}).get("pixels", 0):,} cells'),
            ("Class 2", f'{terrain["class_distribution"].get(2, {}).get("pixels", 0):,} cells'),
            ("Class 3+", f'{sum(v.get("pixels", 0) for k, v in terrain["class_distribution"].items() if int(k) >= 3):,} cells'),
        ]

    cols = st.columns(4)
    for col, (label, value) in zip(cols, metric_values):
        with col:
            st.metric(label, value)

    # Display-only contrast enhancement.
    # Original scientific arrays remain unchanged.
    if layer in (
        "Elevation",
        "Slope",
        "Roughness (TRI)",
        "Local elevation variability",
    ):
        valid_values = np.asarray(z)[np.isfinite(z)]

        display_min, display_max = np.percentile(
            valid_values, [2, 98]
        )

    elif layer == "Terrain traversal cost":
        display_min, display_max = 0, 1

    else:
        display_min, display_max = None, None

    st.plotly_chart(
        mars_heatmap(
            z,
            elevation["longitude_1d"],
            elevation["latitude_1d"],
            title,
            unit,
            colorscale=scale,
            zmin=display_min,
            zmax=display_max,
            zsmooth="best",
        ),
        use_container_width=True,
    )

    st.markdown("### Terrain-cost model")
    cfg = terrain["configuration"]
    a, b, c, d = st.columns(4)
    a.metric("Slope weight", f'{cfg["slope_weight"]:.0%}')
    b.metric("Roughness weight", f'{cfg["roughness_weight"]:.0%}')
    c.metric("Lower percentile", f'{cfg["lower_percentile"]:.0f}%')
    d.metric("Upper percentile", f'{cfg["upper_percentile"]:.0f}%')

    st.markdown(
        "The combined cost uses normalised slope and TRI roughness. "
        "Local 3 × 3 standard deviation is retained as a diagnostic layer "
        "rather than added as a third independent cost term."
    )


    st.markdown("### Interactive model exploration")

    st.caption(
        "Experiment with the relative contribution of slope and "
        "roughness. These settings affect this exploration view only; "
        "the original mission-planning model remains unchanged."
    )

    slope_weight_percent = st.slider(
        "Slope contribution",
        min_value=0,
        max_value=100,
        value=70,
        step=5,
        format="%d%%",
    )

    roughness_weight_percent = 100 - slope_weight_percent

    weight_col1, weight_col2 = st.columns(2)

    weight_col1.metric(
        "Slope weight",
        f"{slope_weight_percent}%"
    )

    weight_col2.metric(
        "Roughness weight",
        f"{roughness_weight_percent}%"
    )

    lower_percentile = st.slider(
        "Lower normalisation percentile",
        min_value=0,
        max_value=20,
        value=2,
        step=1,
        format="%d%%",
    )

    upper_percentile = st.slider(
        "Upper normalisation percentile",
        min_value=80,
        max_value=100,
        value=98,
        step=1,
        format="%d%%",
    )

    if lower_percentile >= upper_percentile:
        st.error("The upper percentile must be greater than the lower percentile.")
    else:
        exploratory_terrain = analyse_terrain_cost(
            elevation,
            roughness,
            slope,
            slope_weight=slope_weight_percent / 100,
            roughness_weight=roughness_weight_percent / 100,
            lower_percentile=lower_percentile,
            upper_percentile=upper_percentile,
        )
        exploratory_stats = exploratory_terrain["statistics"]
        st.markdown("#### Updated terrain-cost results")
        result_col1, result_col2, result_col3 = st.columns(3)
        result_col1.metric("Mean terrain cost", f'{exploratory_stats["mean"]:.4f}')
        result_col2.metric("Median terrain cost", f'{exploratory_stats["median"]:.4f}')
        result_col3.metric("Valid cells", f'{exploratory_stats["valid_pixels"]:,}')
        st.plotly_chart(
            mars_heatmap(
                exploratory_terrain["terrain_cost"],
                elevation["longitude_1d"],
                elevation["latitude_1d"],
                "Interactive terrain traversal cost",
                "Relative cost",
                colorscale="Inferno",
                zmin=0,
                zmax=1,
                zsmooth="best",
            ),
            use_container_width=True,
        )
        st.caption(
            "This is an experimental relative-cost surface. Changing weights "
            "does not establish actual astronaut mobility or surface safety."
        )


    with st.expander("Weight sensitivity analysis"):
        sensitivity = terrain["sensitivity_analysis"]
        rows = []
        for name, result in sensitivity.items():
            stats = result.get("statistics", {})
            rows.append(
                {
                    "Scenario": name,
                    "Mean terrain cost": stats.get("mean", np.nan),
                    "Median terrain cost": stats.get("median", np.nan),
                }
            )
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.caption(
            "This sensitivity comparison reports changes in the overall "
            "surface statistics; it does not establish that route rankings "
            "are stable under every weighting choice."
        )


# ---------------------------------------------------------------------
# PAGE 3 — 3D TERRAIN EXPLORER
# ---------------------------------------------------------------------

elif page == "3D Terrain Explorer":
    st.markdown('<div class="eyebrow">03 / 3D TERRAIN EXPLORER</div>', unsafe_allow_html=True)
    st.header("3D terrain explorer")
    st.markdown(
        '<p class="section-note">Interactive MOLA surface for the detailed '
        'Western Elysium Planitia AOI. Rotate, zoom, and hover to inspect '
        'the gridded topography.</p>',
        unsafe_allow_html=True,
    )

    try:
        with st.spinner("Building your original regional MOLA terrain explorer..."):
            terrain_3d_result = create_terrain_3d(show=False)

        st.plotly_chart(
            terrain_3d_result["figure"],
            use_container_width=True,
            config={"scrollZoom": True, "displaylogo": False},
        )

    except Exception as exc:
        st.error(f"Could not build the regional 3D terrain: {type(exc).__name__}: {exc}")
        with st.expander("Technical traceback"):
            st.code(traceback.format_exc())
        st.caption(
            "The regional surface uses 6× vertical exaggeration for visualisation. "
            "Elevation hover values remain in metres; horizontal axes are local Martian kilometres."
        )
    st.info(
        "CTX footprints are included as a toggleable legend layer when validated footprint "
        "files are available. MOLA is a regional-scale dataset and does not resolve "
        "metre-scale rocks or small surface hazards."
    )


# ---------------------------------------------------------------------
# PAGE 4 — MISSION PLANNER
# ---------------------------------------------------------------------

elif page == "Mission Planner":
    st.markdown('<div class="eyebrow">05 / ROUTE SIMULATION</div>', unsafe_allow_html=True)
    st.header("Mission planner")
    st.markdown(
        '<p class="section-note">Choose a destination within the detailed '
        'study area and run the existing A* route planner over the terrain-cost '
        'surface.</p>',
        unsafe_allow_html=True,
    )

    with st.form("mission_form"):
        mission_name = st.text_input(
            "Mission name",
            value="ELYSIUM Terrain Exploration",
            max_chars=100,
        )
        start_col, dest_col = st.columns(2)
        with start_col:
            st.markdown("**Start — InSight reference site**")
            start_lat = st.number_input(
                "Start latitude (°N)",
                min_value=-90.0,
                max_value=90.0,
                value=float(DEFAULT_START[0]),
                step=0.01,
                format="%.6f",
            )
            start_lon = st.number_input(
                "Start longitude (°E)",
                min_value=-360.0,
                max_value=360.0,
                value=float(DEFAULT_START[1]),
                step=0.01,
                format="%.6f",
            )
        with dest_col:
            st.markdown("**Destination**")
            destination_lat = st.number_input(
                "Destination latitude (°N)",
                min_value=float(DETAILED_AOI["south"]),
                max_value=float(DETAILED_AOI["north"]),
                value=4.70,
                step=0.01,
                format="%.6f",
            )
            destination_lon = st.number_input(
                "Destination longitude (°E)",
                min_value=float(DETAILED_AOI["west"]),
                max_value=float(DETAILED_AOI["east"]),
                value=136.00,
                step=0.01,
                format="%.6f",
            )

        submitted = st.form_submit_button(
            "Simulate mission route",
            use_container_width=True,
        )

    if submitted:
        try:
            with st.spinner("Running A* route simulation..."):
                mission = run_mission(
                    terrain_results=terrain,
                    destination=(destination_lat, destination_lon),
                    start=(start_lat, start_lon),
                    mission_name=mission_name,
                )
            st.session_state["last_mission"] = mission
            st.success("Mission simulation completed.")
        except Exception as exc:
            st.error(f"Mission simulation failed: {type(exc).__name__}: {exc}")

    mission = st.session_state.get("last_mission")
    if mission:
        st.markdown("---")
        st.markdown(f"### {mission['mission_name']}")
        st.markdown(
            f'<span class="status-pill">{mission["status"]}</span>',
            unsafe_allow_html=True,
        )
        summary = mission["summary"]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Route distance", f'{summary["route_distance_km"]:.3f} km')
        m2.metric(
            "Traversal cost (km-equivalent)",
            f'{summary["cumulative_traversal_cost_km_equivalent"]:.3f}'
        )
        m3.metric("Route cells", f'{summary["route_cells"]:,}')
        m4.metric("Expanded cells", f'{summary["expanded_cells"]:,}')

        route_points = mission["route"]["route_coordinates"]
        route_lats = [p["latitude"] for p in route_points]
        route_lons = [p["longitude"] for p in route_points]

        route_fig = mars_heatmap(
            terrain["terrain_cost"],
            elevation["longitude_1d"],
            elevation["latitude_1d"],
            "A* route over terrain traversal cost",
            "Relative cost",
            colorscale="Inferno",
            zmin=0,
            zmax=1,
            zsmooth="best",
        )
        route_fig.add_trace(
            go.Scatter(
                x=route_lons,
                y=route_lats,
                mode="lines",
                name="Simulated route",
                line={"color": "#55e0c1", "width": 3},
                hovertemplate=(
                    "Longitude: %{x:.5f}°E<br>"
                    "Latitude: %{y:.5f}°N<extra>Route</extra>"
                ),
            )
        )
        actual_start = mission["mission_parameters"]["actual_start"]
        actual_end = mission["mission_parameters"]["actual_destination"]
        route_fig.add_trace(
            go.Scatter(
                x=[actual_start[1]],
                y=[actual_start[0]],
                mode="markers",
                name="Snapped start",
                marker={"size": 11, "color": "#71c7ff", "symbol": "circle"},
            )
        )
        route_fig.add_trace(
            go.Scatter(
                x=[actual_end[1]],
                y=[actual_end[0]],
                mode="markers",
                name="Snapped destination",
                marker={"size": 12, "color": "#fff0a6", "symbol": "star"},
            )
        )
        route_fig.update_layout(
            margin={"l": 15, "r": 85, "t": 150, "b": 15},
            title={
                "text": "A* route over terrain traversal cost",
                "y": 0.98,
                "x": 0.02,
                "xanchor": "left",
                "yanchor": "top",
            },
            legend={
                "orientation": "h",
                "yanchor": "bottom",
                "y": 1.02,
                "xanchor": "left",
                "x": 0,
                "bgcolor": "rgba(8, 13, 22, 0.85)",
                "font": {"color": "#eaf5ff", "size": 11},
            },
        )

        st.plotly_chart(route_fig, use_container_width=True)

        st.markdown("### Route details")
        d1, d2, d3 = st.columns(3)
        d1.metric("Mean route terrain cost", f'{summary["mean_terrain_cost"]:.5f}')
        d2.metric("Route elevation range", f'{summary["minimum_elevation_m"]:.1f} to {summary["maximum_elevation_m"]:.1f} m')
        d3.metric("Mean route elevation", f'{summary["mean_elevation_m"]:.1f} m')

        route_df = pd.DataFrame(route_points)
        route_df["route_step"] = np.arange(1, len(route_df) + 1)
        profile_tab1, profile_tab2 = st.tabs(["Elevation profile", "Terrain-cost profile"])
        with profile_tab1:
            profile = go.Figure(go.Scatter(
                x=route_df["route_step"],
                y=route_df["elevation_m"],
                mode="lines",
                line={"color": "#79c7e8", "width": 2.5},
                hovertemplate="Route step: %{x}<br>Elevation: %{y:.1f} m<extra></extra>"
            ))
            profile.update_layout(title="MOLA elevation along simulated route", height=300,
                margin={"l": 10, "r": 10, "t": 45, "b": 10},
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#101a28",
                font={"color": "#eaf5ff"}, xaxis_title="Route step", yaxis_title="Elevation (m)")
            profile.update_yaxes(
                range=[
                    float(route_df["elevation_m"].min()) - 10,
                    float(route_df["elevation_m"].max()) + 10,
                ]
            )
            st.plotly_chart(profile, use_container_width=True)
        with profile_tab2:
            profile = go.Figure(go.Scatter(
                x=route_df["route_step"], y=route_df["terrain_cost"],
                mode="lines", line={"color": "#e6a06d", "width": 2.5},
                hovertemplate="Route step: %{x}<br>Relative terrain cost: %{y:.4f}<extra></extra>"
            ))
            profile.update_layout(title="Relative terrain cost along simulated route", height=300,
                margin={"l": 10, "r": 10, "t": 45, "b": 10},
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#101a28",
                font={"color": "#eaf5ff"}, xaxis_title="Route step", yaxis_title="Relative cost")
            st.plotly_chart(profile, use_container_width=True)

        p = mission["mission_parameters"]
        st.write(
            f'**Requested start:** {p["requested_start"][0]:.5f}°N, '
            f'{p["requested_start"][1]:.5f}°E'
        )
        st.write(
            f'**Snapped start:** {p["actual_start"][0]:.5f}°N, '
            f'{p["actual_start"][1]:.5f}°E '
            f'({summary["start_snap_distance_km"]:.3f} km from requested point)'
        )
        st.write(
            f'**Requested destination:** {p["requested_destination"][0]:.5f}°N, '
            f'{p["requested_destination"][1]:.5f}°E'
        )
        st.write(
            f'**Snapped destination:** {p["actual_destination"][0]:.5f}°N, '
            f'{p["actual_destination"][1]:.5f}°E '
            f'({summary["destination_snap_distance_km"]:.3f} km from requested point)'
        )

        st.warning(mission["limitations"])
        report = generate_mission_report(mission)
        st.download_button(
            "Download mission report (.txt)",
            data=report.encode("utf-8"),
            file_name="elysium_mission_report.txt",
            mime="text/plain",
            use_container_width=True,
        )
    else:
        st.info(
            "No mission has been simulated in this session yet. Set a "
            "destination and select **Simulate mission route**."
        )


# ---------------------------------------------------------------------
# PAGE 5 — ORBITAL DATA EXPLORER
# ---------------------------------------------------------------------

elif page == "Orbital Data Explorer":
    st.markdown('<div class="eyebrow">06 / ORBITAL OBSERVATIONS</div>', unsafe_allow_html=True)
    st.header("Orbital Data Explorer")
    st.markdown(
        '<p class="section-note">Manage observation inventories from CTX, HiRISE and THEMIS. '
        'Discovery and download operations run only when you request them.</p>',
        unsafe_allow_html=True,
    )

    def read_json_records(path):
        path = Path(path)
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def show_records_table(records, key):
        if not records:
            st.info("No records are available yet. Run the inventory action above.")
            return
        if isinstance(records, dict):
            records = records.get("records", records.get("products", records.get("inventory", [records])))
        if not isinstance(records, list):
            records = [records]
        if not records:
            st.info("The catalogue exists but contains no records.")
            return
        frame = pd.json_normalize(records, sep=".")
        st.dataframe(frame, use_container_width=True, hide_index=True, height=380)
        st.download_button(
            "Export visible catalogue as CSV",
            data=frame.to_csv(index=False).encode("utf-8"),
            file_name=f"elysium_{key}_catalogue.csv",
            mime="text/csv",
            key=f"download_{key}_catalogue",
        )

    ctx_tab, hirise_tab, themis_tab = st.tabs(["CTX · Context Camera", "HiRISE · High Resolution", "THEMIS · Thermal / Visible"])

    with ctx_tab:
        st.subheader("CTX observation inventory")
        st.write("First discover CTX observations from NASA ODE, then build an inventory of local footprints, downloaded images and validated PDS metadata.")
        ctx_records = None
        try:
            ctx_records = read_json_records(CTX_CATALOGUE_FILE)
        except Exception as exc:
            st.warning(f"Could not read the existing CTX catalogue: {exc}")
        if ctx_records is not None:
            ctx_frame = pd.json_normalize(ctx_records, sep=".") if isinstance(ctx_records, list) else pd.DataFrame()
            st.metric("Catalogued CTX observations", f"{len(ctx_records):,}" if isinstance(ctx_records, list) else "—")
        discover_col, inventory_col = st.columns(2)
        with discover_col:
            if st.button("Discover CTX observations", type="primary", key="discover_ctx"):
                try:
                    with st.spinner("Searching NASA Mars ODE for CTX observations..."):
                        discovered = discover_observations(save=True, page_size=500)
                    st.success(f"Discovery complete: {len(discovered):,} selected observations saved.")
                except Exception as exc:
                    st.error(f"CTX discovery failed: {type(exc).__name__}: {exc}")
                    with st.expander("Technical traceback"):
                        st.code(traceback.format_exc())
        with inventory_col:
            if st.button("Build / refresh CTX inventory", key="build_ctx"):
                try:
                    with st.spinner("Scanning CTX catalogue, footprints and local image metadata..."):
                        ctx_records = build_ctx_catalogue()
                    st.session_state["ctx_records"] = ctx_records
                    st.success(f"CTX inventory built: {len(ctx_records):,} observations.")
                except Exception as exc:
                    st.error(f"CTX inventory failed: {type(exc).__name__}: {exc}")
                    with st.expander("Technical traceback"):
                        st.code(traceback.format_exc())
        ctx_records = st.session_state.get("ctx_records", ctx_records)
        show_records_table(ctx_records, "ctx")
        st.caption("CTX EDR pixel values are raw digital numbers unless a separately calibrated product is used.")

    with hirise_tab:
        st.subheader("HiRISE acquisition")
        st.write("Searches MRO HiRISE processed RDRV11 and EDR products, prioritises processed products, then downloads a deliberately limited selection.")
        try:
            existing_hirise = read_json_records(HIRISE_CATALOGUE_FILE)
        except Exception as exc:
            existing_hirise = None
            st.warning(f"Could not read the existing HiRISE catalogue: {exc}")
        if existing_hirise is not None:
            st.metric("Existing HiRISE catalogue records", f"{len(existing_hirise):,}" if isinstance(existing_hirise, list) else "Available")
        max_hirise = st.select_slider("Maximum HiRISE products to download", options=[1, 2, 3], value=2, key="hirise_limit")
        if st.button("Discover and acquire HiRISE products", type="primary", key="run_hirise"):
            try:
                with st.spinner("Searching NASA ODE and processing the selected HiRISE products..."):
                    records = run_hirise_pipeline(max_products=max_hirise)
                st.session_state["hirise_records"] = records
                st.success(f"HiRISE pipeline completed. Catalogue records: {len(records):,}.")
            except Exception as exc:
                st.error(f"HiRISE pipeline failed: {type(exc).__name__}: {exc}")
                with st.expander("Technical traceback"):
                    st.code(traceback.format_exc())
        records = st.session_state.get("hirise_records", existing_hirise)
        show_records_table(records, "hirise")
        st.caption("The existing HiRISE pipeline caps downloads at three products by default. Download sizes and network time can vary.")

    with themis_tab:
        st.subheader("THEMIS discovery and inventory")
        st.write("Discovers AOI-intersecting THEMIS products, downloads a limited selection, and inspects available rasters.")
        try:
            existing_themis = read_json_records(THEMIS_INVENTORY_PATH)
        except Exception as exc:
            existing_themis = None
            st.warning(f"Could not read the existing THEMIS inventory: {exc}")
        if existing_themis is not None:
            existing_count = len(existing_themis) if isinstance(existing_themis, (list, dict)) else 0
            st.metric("Existing THEMIS inventory entries", f"{existing_count:,}")
        if st.button("Discover, download and inspect THEMIS", type="primary", key="run_themis"):
            try:
                with st.spinner("Running the THEMIS discovery and processing pipeline..."):
                    result = process_themis()
                st.session_state["themis_result"] = result
                st.success(
                    f"THEMIS pipeline completed: {len(result.get('products', []))} selected products, "
                    f"{len(result.get('inventory', []))} inventory entries."
                )
            except Exception as exc:
                st.error(f"THEMIS pipeline failed: {type(exc).__name__}: {exc}")
                with st.expander("Technical traceback"):
                    st.code(traceback.format_exc())
        result = st.session_state.get("themis_result")
        inventory_records = result.get("inventory") if isinstance(result, dict) else existing_themis
        show_records_table(inventory_records, "themis")
        st.caption("THEMIS raw values are not automatically calibrated temperature, thermal inertia or reflectance. Geographic overlays require validated CRS and longitude conventions.")


# ---------------------------------------------------------------------
# PAGE 6 — SURFACE MAPPING
# ---------------------------------------------------------------------

elif page == "Surface Mapping":
    st.markdown('<div class="eyebrow">04 / OBSERVATION GEOMETRY</div>', unsafe_allow_html=True)
    st.header("Surface mapping & footprint coverage")
    st.markdown(
        '<p class="section-note">Inspect where downloaded orbital observation footprints intersect the detailed study area.</p>',
        unsafe_allow_html=True,
    )
    st.info("Coverage is calculated from unique geometric union within the AOI. Overlapping observations are not double-counted.")
    if st.button("Analyse footprint coverage", type="primary", key="analyse_footprints"):
        try:
            with st.spinner("Validating Mars footprint CRS and calculating AOI intersections..."):
                coverage = analyse_footprints()
                create_footprint_map()
            st.session_state["footprint_coverage"] = coverage
            st.success("Footprint analysis and map generation completed.")
        except Exception as exc:
            st.error(f"Footprint analysis failed: {type(exc).__name__}: {exc}")
            with st.expander("Technical traceback"):
                st.code(traceback.format_exc())

    coverage = st.session_state.get("footprint_coverage")
    if coverage:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Observation folders", f'{len(coverage["records"]):,}')
        c2.metric("Unique AOI coverage", f'{coverage["coverage_percent"]:.2f}%')
        c3.metric("Covered area", f'{coverage["covered_area_km2"]:,.1f} km²')
        c4.metric("Intersecting observations", f'{coverage["status_counts"].get("Intersecting AOI", 0):,}')
        coverage_frame = pd.DataFrame(coverage["records"])
        st.dataframe(coverage_frame, use_container_width=True, hide_index=True, height=350)
        st.download_button(
            "Download footprint analysis CSV",
            data=coverage_frame.to_csv(index=False).encode("utf-8"),
            file_name="elysium_footprint_coverage.csv",
            mime="text/csv",
        )
    else:
        st.caption("Run footprint analysis to populate the coverage statistics and table.")

    footprint_map_path = Path(OUTPUT_DIR) / "footprint_coverage_map.png"
    if footprint_map_path.exists():
        st.markdown("### Observation footprint map")
        st.image(str(footprint_map_path), use_container_width=True)
    else:
        st.info("No generated footprint map is available yet. Run the analysis above.")


# ---------------------------------------------------------------------
# PAGE 7 — DATA AND METHODOLOGY
# ---------------------------------------------------------------------

elif page == "Data & Methodology":
    st.markdown('<div class="eyebrow">07 / DATA PROVENANCE</div>', unsafe_allow_html=True)
    st.header("Data & methodology")

    st.markdown("### Study region")
    st.write(
        f'**{DETAILED_AOI["name"]}** — '
        f'{DETAILED_AOI["west"]:.1f}–{DETAILED_AOI["east"]:.1f}°E, '
        f'{DETAILED_AOI["south"]:.1f}–{DETAILED_AOI["north"]:.1f}°N.'
    )
    st.write(
        f'InSight reference coordinate: {INSIGHT_SITE["latitude"]:.3f}°N, '
        f'{INSIGHT_SITE["longitude"]:.3f}°E.'
    )

    st.markdown("### Primary elevation source")
    st.write("**MOLA MEGDR — MEGT90N000GB.IMG**")
    st.write(
        "Mars Orbiter Laser Altimeter gridded elevation product from the "
        "NASA Planetary Data System. The project configuration uses a "
        "64-pixel-per-degree grid and extracts the detailed AOI."
    )
    st.markdown(
        "[NASA PDS Geosciences Node — MOLA MEGDR dataset](https://pds-geosciences.wustl.edu/mgs/mgs-m-mola-5-megdr-l3-v1/)"
    )

    st.markdown("### Orbital observation sources")
    st.write("**CTX (MRO):** Context imagery and observation footprints support regional surface context. Raw EDR digital numbers are not calibrated reflectance.")
    st.write("**HiRISE (MRO):** High-resolution RDRV11 products are preferred when available; EDR products may be raw observations.")
    st.write("**THEMIS (Mars Odyssey):** Visible and infrared products provide complementary observations. Raw values are not automatically calibrated temperature or thermal inertia, and raster overlays require validated georeferencing.")

    st.markdown("### Processing and analysis sequence")
    st.code(
        """process_mola()
    ↓
analyse_elevation()
    ├── analyse_roughness()  → TRI + local 3×3 standard deviation
    └── analyse_slope()      → slope gradient
             ↓
analyse_terrain_cost()
    ↓
run_mission()  → A* route planning
    ↓
generate_mission_report()""",
        language="text",
    )

    st.markdown("### Terrain-cost model")
    if data is not None:
        cfg = terrain["configuration"]
        st.write(
            f'The current model combines normalised slope ({cfg["slope_weight"]:.0%}) '
            f'and normalised TRI roughness ({cfg["roughness_weight"]:.0%}). '
            f'Layer normalisation uses the {cfg["lower_percentile"]:.0f}th to '
            f'{cfg["upper_percentile"]:.0f}th percentile range.'
        )
    else:
        st.write("The model combines normalised slope and TRI roughness using the project configuration.")
    st.write(
        "Local elevation standard deviation is available as a diagnostic "
        "layer and is not independently added to the combined cost, reducing "
        "the risk of counting closely related roughness measures twice."
    )

    st.markdown("### Resolution and interpretation limits")
    st.warning(
        "MOLA is suitable for regional topographic context, not for resolving "
        "small rocks, metre-scale obstacles, or detailed surface hazards. "
        "The terrain-cost weights and percentile scaling are model choices, "
        "not empirically calibrated astronaut-mobility parameters."
    )
    st.warning(
        "The A* route is a relative route-planning simulation over the "
        "available grid. It must not be interpreted as a verified safe route "
        "for astronauts, rovers, or landing systems."
    )

    st.markdown("### Export processed terrain grid")
    if data is not None:
        df = make_terrain_dataframe(data)
        st.download_button(
            "Download terrain analysis as CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="elysium_terrain_analysis.csv",
            mime="text/csv",
            use_container_width=True,
        )
        st.caption(
            f"CSV contains {len(df):,} grid cells and the elevation, slope, "
            "roughness, terrain-cost, class, and validity fields."
        )
    else:
        st.info("Load MOLA from a terrain workspace to export the processed terrain grid.")


# ---------------------------------------------------------------------
# FOOTER
# ---------------------------------------------------------------------

st.markdown("---")
st.markdown(
    f'<div style="display:flex;justify-content:space-between;gap:1rem;'
    f'flex-wrap:wrap;color:#77858b;font-family:DM Mono,monospace;'
    f'font-size:0.68rem;letter-spacing:0.06em;">'
    f'<span>ELYSIUM · {PLANET_NAME.upper()} TERRAIN INTELLIGENCE</span>'
    f'<span>BUILD {PROJECT_VERSION} · '
    f'{datetime.now(timezone.utc).strftime("%Y-%m-%d UTC")}</span>'
    f'</div>',
    unsafe_allow_html=True,
)
