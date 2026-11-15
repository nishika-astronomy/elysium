"""
ELYSIUM — 3D Terrain Explorer
-----------------------------
Creates an interactive 3D regional terrain visualization using MOLA
elevation data, with CTX observation footprint overlays and the
InSight landing-site marker.

The visualization is regional-scale and is not a high-resolution
surface hazard map or a certified astronaut-safety assessment.
"""

from pathlib import Path

import geopandas as gpd
import numpy as np
import plotly.graph_objects as go

from shapely.geometry import (
    box,
    Polygon,
    MultiPolygon,
    GeometryCollection,
)

from processing.mola import MOLA_DIR
from config import RAW_DIR, OUTPUT_DIR


# ============================================================
# 1. CONFIGURATION
# ============================================================

WEST, EAST = 130.0, 140.0
SOUTH, NORTH = 1.0, 9.0

PIXELS_PER_DEGREE = 64
VERTICAL_EXAGGERATION = 6.0

MARS_RADIUS_KM = 3396.19

MAP_CENTRE_LON = (WEST + EAST) / 2.0
MAP_CENTRE_LAT = (SOUTH + NORTH) / 2.0

INSIGHT_LON = 135.623
INSIGHT_LAT = 4.502

CTX_COLOR = "#e7a27f"
CTX_LINE_WIDTH = 1.2

MOLA_ROWS = 5760
MOLA_COLS = 11520
MOLA_DTYPE = ">i2"

MOLA_NODATA_VALUE = None

MOLA_FILE = Path(MOLA_DIR) / "megt90n000gb.img"
FOOTPRINT_DIR = Path(RAW_DIR) / "footprints"

TERRAIN_OUTPUT_FILE = (
    Path(OUTPUT_DIR) / "elysium_terrain_3d.html"
)


# ============================================================
# 2. LOAD MOLA TERRAIN
# ============================================================

def load_mola_terrain():
    """Load MOLA elevation data and extract the regional study area."""

    print("Loading MOLA elevation data...")

    if not MOLA_FILE.exists():
        raise FileNotFoundError(
            f"MOLA elevation file not found: {MOLA_FILE}"
        )

    expected_values = MOLA_ROWS * MOLA_COLS

    image = np.fromfile(
        MOLA_FILE,
        dtype=MOLA_DTYPE
    )

    if image.size != expected_values:
        raise ValueError(
            "Unexpected MOLA image dimensions. "
            f"Expected {expected_values:,} values, "
            f"found {image.size:,}."
        )

    image = image.reshape((MOLA_ROWS, MOLA_COLS))

    row_start = int(
        (90.0 - NORTH) * PIXELS_PER_DEGREE
    )
    row_end = int(
        (90.0 - SOUTH) * PIXELS_PER_DEGREE
    )

    col_start = int(WEST * PIXELS_PER_DEGREE)
    col_end = int(EAST * PIXELS_PER_DEGREE)

    if not (
        0 <= row_start < row_end <= MOLA_ROWS
        and 0 <= col_start < col_end <= MOLA_COLS
    ):
        raise ValueError(
            "Requested region falls outside the MOLA grid."
        )

    elevation = image[
        row_start:row_end,
        col_start:col_end
    ].astype(np.float32)

    rows, cols = elevation.shape

    longitude = WEST + (
        np.arange(cols, dtype=np.float64) + 0.5
    ) / PIXELS_PER_DEGREE

    latitude = NORTH - (
        np.arange(rows, dtype=np.float64) + 0.5
    ) / PIXELS_PER_DEGREE

    lon_grid, lat_grid = np.meshgrid(
        longitude,
        latitude
    )

    valid_mask = np.isfinite(elevation)

    if MOLA_NODATA_VALUE is not None:
        valid_mask &= (
            elevation != MOLA_NODATA_VALUE
        )

    if not np.any(valid_mask):
        raise ValueError(
            "No valid MOLA elevation pixels found."
        )

    mean_elevation = float(
        np.mean(elevation[valid_mask])
    )

    visual_elevation = np.full(
        elevation.shape,
        np.nan,
        dtype=np.float32
    )

    visual_elevation[valid_mask] = (
        (elevation[valid_mask] - mean_elevation) / 1000.0
    ) * VERTICAL_EXAGGERATION

    print("MOLA regional terrain loaded.")
    print("Grid dimensions:", elevation.shape)

    print(
        "Elevation range:",
        float(np.min(elevation[valid_mask])),
        "to",
        float(np.max(elevation[valid_mask])),
        "m"
    )

    print("Mean elevation:", mean_elevation, "m")
    print("Valid pixels:", int(np.count_nonzero(valid_mask)))

    return (
        elevation,
        visual_elevation,
        lon_grid,
        lat_grid,
        mean_elevation,
        valid_mask
    )


# ============================================================
# 3. GEOGRAPHIC COORDINATES TO LOCAL MARTIAN KILOMETRES
# ============================================================

def geographic_to_local_km(longitude, latitude):
    """
    Convert Mars longitude/latitude into local east/north distances.

    Uses a local equirectangular approximation centred on the
    regional study area. All horizontal scene coordinates are km.
    """

    lon = np.asarray(longitude, dtype=np.float64)
    lat = np.asarray(latitude, dtype=np.float64)

    lat0 = np.deg2rad(MAP_CENTRE_LAT)

    x_km = (
        MARS_RADIUS_KM
        * np.cos(lat0)
        * np.deg2rad(lon - MAP_CENTRE_LON)
    )

    y_km = (
        MARS_RADIUS_KM
        * np.deg2rad(lat - MAP_CENTRE_LAT)
    )

    return x_km, y_km


# ============================================================
# 4. SAMPLE TERRAIN ELEVATION
# ============================================================

def get_visual_elevation(
    lon,
    lat,
    elevation,
    mean_elevation,
    valid_mask
):
    """Sample the MOLA grid and return display elevation in km."""

    if not (
        np.isfinite(lon)
        and np.isfinite(lat)
        and WEST <= lon <= EAST
        and SOUTH <= lat <= NORTH
    ):
        return float("nan")

    row = int(
        (NORTH - lat) * PIXELS_PER_DEGREE
    )

    col = int(
        (lon - WEST) * PIXELS_PER_DEGREE
    )

    row = int(
        np.clip(row, 0, elevation.shape[0] - 1)
    )

    col = int(
        np.clip(col, 0, elevation.shape[1] - 1)
    )

    if not valid_mask[row, col]:
        return float("nan")

    value = float(elevation[row, col])

    return (
        (value - mean_elevation) / 1000.0
    ) * VERTICAL_EXAGGERATION


# ============================================================
# 5. EXTRACT POLYGON BOUNDARIES
# ============================================================

def extract_polygon_lines(geometry):
    """Extract exterior boundaries from supported geometries."""

    if geometry is None or geometry.is_empty:
        return []

    if isinstance(geometry, Polygon):
        return [geometry.exterior]

    if isinstance(geometry, MultiPolygon):
        return [
            polygon.exterior
            for polygon in geometry.geoms
            if not polygon.is_empty
        ]

    if isinstance(geometry, GeometryCollection):
        lines = []

        for part in geometry.geoms:
            lines.extend(
                extract_polygon_lines(part)
            )

        return lines

    return []


# ============================================================
# 6. LOAD CTX OBSERVATION FOOTPRINTS
# ============================================================

def load_ctx_coverage(
    elevation,
    mean_elevation,
    valid_mask
):
    """Load and clip CTX observation footprints to the study region."""

    print("Loading CTX observation footprints...")

    if not FOOTPRINT_DIR.exists():
        print(
            "Footprint directory not found:",
            FOOTPRINT_DIR
        )
        return [], [], [], []

    folders = sorted(
        FOOTPRINT_DIR.glob("*_footprint")
    )

    regional_area = box(
        WEST,
        SOUTH,
        EAST,
        NORTH
    )

    x_values = []
    y_values = []
    z_values = []
    observation_ids = []

    processed = 0
    skipped = 0

    for index, folder in enumerate(folders, start=1):

        shapefiles = sorted(folder.glob("*.shp"))

        if not shapefiles:
            skipped += 1
            continue

        try:
            data = gpd.read_file(shapefiles[0])

            if data.empty or data.crs is None:
                skipped += 1
                continue

            try:
                radius = data.crs.ellipsoid.semi_major_metre
            except Exception:
                skipped += 1
                continue

            if (
                radius is None
                or abs(radius - 3396190) > 1
            ):
                skipped += 1
                continue

            clipped_geometries = (
                data.geometry.intersection(regional_area)
            )

            observation_id = (
                folder.name[:-len("_footprint")]
            )

            has_geometry = False

            for geometry in clipped_geometries:

                if geometry is None or geometry.is_empty:
                    continue

                simplified = geometry.simplify(
                    0.003,
                    preserve_topology=True
                )

                lines = extract_polygon_lines(simplified)

                for line in lines:

                    coordinates = list(line.coords)

                    if len(coordinates) < 2:
                        continue

                    for lon, lat in coordinates:

                        x_values.append(float(lon))
                        y_values.append(float(lat))

                        z_values.append(
                            get_visual_elevation(
                                float(lon),
                                float(lat),
                                elevation,
                                mean_elevation,
                                valid_mask
                            )
                        )

                        observation_ids.append(observation_id)

                    x_values.append(None)
                    y_values.append(None)
                    z_values.append(None)
                    observation_ids.append(None)

                    has_geometry = True

            if has_geometry:
                processed += 1
            else:
                skipped += 1

        except Exception as error:
            print(f"Skipped {folder.name}: {error}")
            skipped += 1

        if index % 50 == 0:
            print(
                f"CTX folders checked: "
                f"{index}/{len(folders)}"
            )

    print("CTX observations plotted:", processed)
    print("CTX observations skipped:", skipped)

    return (
        x_values,
        y_values,
        z_values,
        observation_ids
    )

# ============================================================
# 7. CREATE INTERACTIVE 3D TERRAIN
# ============================================================


def create_terrain_3d(show=True):
    """
    Create an interactive 3D MOLA terrain visualization.

    Horizontal coordinates: local Martian kilometres.
    Vertical coordinates: mean-centred, vertically exaggerated.
    """

    print("\nCreating ELYSIUM 3D terrain visualization...")

    (
        elevation,
        visual_elevation,
        lon_grid,
        lat_grid,
        mean_elevation,
        valid_mask
    ) = load_mola_terrain()

    # Convert geographic coordinates into local Martian kilometres.
    x_grid, y_grid = geographic_to_local_km(
        lon_grid,
        lat_grid
    )

    fig = go.Figure()

    # Preserve the original MOLA elevation values for hover data.
    terrain_hover = np.where(
        valid_mask,
        elevation,
        np.nan
    )

    # Mars-inspired scientific colour palette.
    mars_colors = [
        [0.00, "#211b35"],  # Deep volcanic shadow
        [0.15, "#44304b"],  # Dark purple
        [0.30, "#71394a"],  # Martian burgundy
        [0.45, "#a64a3e"],  # Deep rust
        [0.60, "#cf6542"],  # Terracotta
        [0.75, "#e58b50"],  # Copper orange
        [0.90, "#f0b66f"],  # Sandstone
        [1.00, "#f8dfad"]   # Pale Martian dust
    ]

    # Full-resolution MOLA surface: no artificial upscaling.
    fig.add_trace(
        go.Surface(
            x=x_grid,
            y=y_grid,
            z=visual_elevation,

            surfacecolor=terrain_hover,

            colorscale=mars_colors,
            cmin=-4200,
            cmax=-500,

            colorbar=dict(
                title=dict(
                    text="Elevation (m)"
                ),
                thickness=16,
                len=0.65,
                x=1.02,
                tickfont=dict(size=11)
            ),

            customdata=terrain_hover,

            hovertemplate=(
                "<b>MOLA Terrain</b><br>"
                "Elevation: %{customdata:.1f} m"
                "<extra></extra>"
            ),

            connectgaps=False,

            lighting=dict(
                ambient=0.35,
                diffuse=0.90,
                specular=0.18,
                roughness=0.82,
                fresnel=0.08
            ),

            lightposition=dict(
                x=1200,
                y=-800,
                z=2500
            ),

            name="MOLA Terrain",
            showscale=True
        )
    )

    # --------------------------------------------------------
    # InSight landing-site marker
    # --------------------------------------------------------

    insight_x, insight_y = geographic_to_local_km(
        INSIGHT_LON,
        INSIGHT_LAT
    )

    insight_z = get_visual_elevation(
        INSIGHT_LON,
        INSIGHT_LAT,
        elevation,
        mean_elevation,
        valid_mask
    )

    if not np.isfinite(insight_z):
        insight_z = 0.0

    fig.add_trace(
        go.Scatter3d(
            x=[float(insight_x)],
            y=[float(insight_y)],
            z=[insight_z + 0.2],

            mode="markers+text",

            marker=dict(
                size=7,
                color="#fff4d6",
                symbol="diamond",
                line=dict(
                    color="#382b32",
                    width=2
                )
            ),

            text=["InSight"],
            textposition="top center",

            textfont=dict(
                color="#fff4d6",
                size=12
            ),

            name="InSight Landing Site",

            hovertemplate=(
                "<b>InSight Landing Site</b><br>"
                f"Longitude: {INSIGHT_LON:.3f}°E<br>"
                f"Latitude: {INSIGHT_LAT:.3f}°N"
                "<extra></extra>"
            )
        )
    )

    # --------------------------------------------------------
    # Layout and camera
    # --------------------------------------------------------

    x_min = float(np.nanmin(x_grid))
    x_max = float(np.nanmax(x_grid))

    y_min = float(np.nanmin(y_grid))
    y_max = float(np.nanmax(y_grid))

    fig.update_layout(

        title=dict(
            text=(
                "ELYSIUM | WESTERN ELYSIUM PLANITIA"
                "<br><sup>MOLA Digital Terrain Model "
                "| InSight Landing Site</sup>"
            ),
            x=0.5,
            xanchor="center",
            font=dict(
                size=19,
                color="#f5dfb5"
            )
        ),

        paper_bgcolor="#090b14",
        plot_bgcolor="#090b14",

        font=dict(
            color="#dedce8",
            family="Arial"
        ),

        margin=dict(
            l=15,
            r=65,
            t=90,
            b=25
        ),

        height=900,

        showlegend=True,

        legend=dict(
            x=0.02,
            y=0.97,
            bgcolor="rgba(15,15,28,0.75)",
            bordercolor="rgba(220,190,160,0.2)",
            borderwidth=1,
            font=dict(size=11)
        ),

        scene=dict(

            bgcolor="#090b14",

            xaxis=dict(
                title=dict(
                    text="East–West (km)",
                    font=dict(size=12)
                ),
                backgroundcolor="#090b14",
                gridcolor="rgba(150,140,170,0.12)",
                zeroline=False,
                showbackground=False,
                showgrid=False,
                showline=False,
                tickfont=dict(size=10)
            ),

            yaxis=dict(
                title=dict(
                    text="North–South (km)",
                    font=dict(size=12)
                ),
                backgroundcolor="#090b14",
                gridcolor="rgba(150,140,170,0.12)",
                zeroline=False,
                showbackground=False,
                showgrid=False,
                showline=False,
                tickfont=dict(size=10)
            ),

            zaxis=dict(
                title=dict(
                    text="Relative elevation (km)",
                    font=dict(size=12)
                ),
                backgroundcolor="#090b14",
                gridcolor="rgba(150,140,170,0.12)",
                zeroline=False,
                showbackground=False,
                showgrid=False,
                showline=False,
                tickfont=dict(size=10)
            ),

            aspectmode="manual",

            aspectratio=dict(
                x=1.0,
                y=1.35,
                z=0.34
            ),

            camera=dict(
                eye=dict(
                    x=1.2,
                    y=-1.4,
                    z=1.35
                ),
                center=dict(
                    x=0,
                    y=0,
                    z=0
                ),
                up=dict(
                    x=0,
                    y=0,
                    z=1
                ),
                projection=dict(
                    type="perspective"
                )
            ),

            xaxis_range=[x_min, x_max],
            yaxis_range=[y_min, y_max]
        )
    )

    # --------------------------------------------------------
    # Save interactive visualization
    # --------------------------------------------------------

    TERRAIN_OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    fig.write_html(
        str(TERRAIN_OUTPUT_FILE),
        include_plotlyjs=True,
        full_html=True,
        config={
            "responsive": True,
            "displaylogo": False,
            "scrollZoom": True
        }
    )

    print("\n3D terrain visualization saved to:")
    print(TERRAIN_OUTPUT_FILE)

    if show:
        fig.show()

    return {
        "figure": fig,
        "output_file": TERRAIN_OUTPUT_FILE,
        "mean_elevation": mean_elevation,
        "valid_pixels": int(
            np.count_nonzero(valid_mask)
        ),
        "ctx_observations": 0
    }


# ============================================================
# 8. MAIN
# ============================================================

if __name__ == "__main__":

    result = create_terrain_3d(show=True)

    print("\nELYSIUM 3D terrain generation complete.")