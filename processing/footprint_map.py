
"""
ELYSIUM — Footprint Coverage Map

Creates a geographic visualization of orbital observation
footprints over the Western Elysium Planitia study area.

Features:
    - Selects one preferred footprint per observation.
    - Validates Mars CRS and coordinate bounds.
    - Clips footprints to the detailed AOI.
    - Marks the InSight landing site.
    - Reports processed, skipped and outside-AOI observations.
    - Saves a high-resolution map.

The original NASA footprint files are never modified.
"""

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt

from shapely.geometry import box, Point

from config import (
    RAW_DIR,
    OUTPUT_DIR,
    DETAILED_AOI,
    INSIGHT_SITE,
    MARS_EQUATORIAL_RADIUS_KM,
)


# ==================================================
# CONFIGURATION
# ==================================================

FOOTPRINT_DIR = RAW_DIR / "footprints"

MARS_RADIUS_METRES = MARS_EQUATORIAL_RADIUS_KM * 1000
CRS_RADIUS_TOLERANCE = 1.0

PREFERRED_VARIANTS = (
    "_ga.shp",
    "_c0a.shp",
    "_c180a.shp",
)


# ==================================================
# FOOTPRINT VALIDATION
# ==================================================

def is_valid_mars_footprint(data):
    """Check Mars CRS and longitude/latitude coordinate ranges."""

    if data.empty or data.crs is None:
        return False

    if not data.crs.is_geographic:
        return False

    try:
        radius = data.crs.ellipsoid.semi_major_metre
    except Exception:
        return False

    if radius is None:
        return False

    if abs(radius - MARS_RADIUS_METRES) > CRS_RADIUS_TOLERANCE:
        return False

    min_x, min_y, max_x, max_y = data.total_bounds

    if not (
        0 <= min_x <= 360
        and 0 <= max_x <= 360
        and -90 <= min_y <= 90
        and -90 <= max_y <= 90
    ):
        return False

    return True


# ==================================================
# FOOTPRINT SELECTION
# ==================================================

def select_preferred_shapefile(folder):
    """Select one preferred footprint representation."""

    shapefiles = sorted(
        path
        for path in folder.rglob("*")
        if path.is_file()
        and path.suffix.lower() == ".shp"
    )

    if not shapefiles:
        return None

    for variant in PREFERRED_VARIANTS:
        for path in shapefiles:
            if path.name.lower().endswith(variant):
                return path

    # Deterministic fallback if preferred variants are absent.
    return shapefiles[0]


# ==================================================
# MAP GENERATION
# ==================================================

def create_footprint_map():
    """Create the ELYSIUM orbital footprint coverage map."""

    if not FOOTPRINT_DIR.exists():
        raise FileNotFoundError(
            f"Footprint directory not found: {FOOTPRINT_DIR}"
        )

    folders = sorted(
        folder
        for folder in FOOTPRINT_DIR.glob("*_footprint")
        if folder.is_dir()
    )

    area = DETAILED_AOI

    aoi = box(
        area["west"],
        area["south"],
        area["east"],
        area["north"],
    )

    footprints = []

    processed = 0
    skipped = 0
    outside_aoi = 0

    print("\nELYSIUM | FOOTPRINT MAP")
    print("=" * 50)

    print("Loading footprint geometries...")
    print("Total folders:", len(folders))
    print("Preferred variants:", PREFERRED_VARIANTS)

    # ----------------------------------------------
    # LOAD AND PROCESS FOOTPRINTS
    # ----------------------------------------------

    for index, folder in enumerate(folders, start=1):

        shapefile = select_preferred_shapefile(folder)

        if shapefile is None:
            skipped += 1
            continue

        try:
            data = gpd.read_file(shapefile)

            if not is_valid_mars_footprint(data):
                print(
                    "Invalid or unsupported footprint:",
                    folder.name,
                )
                skipped += 1
                continue

            # Both verified CRS groups use longitude/latitude
            # coordinates and the same Mars spherical radius.
            clipped = data.geometry.intersection(aoi)

            clipped = clipped[
                clipped.notna()
                & ~clipped.is_empty
            ]

            if clipped.empty:
                outside_aoi += 1
                continue

            footprints.extend(clipped.tolist())
            processed += 1

        except Exception as error:
            print(f"Skipped {folder.name}: {error}")
            skipped += 1

        if index % 50 == 0:
            print(
                f"Checked {index}/{len(folders)} folders"
            )

    # ----------------------------------------------
    # CHECK RESULTS
    # ----------------------------------------------

    if not footprints:
        print("\nNo footprints available for mapping.")
        print("Skipped or invalid:", skipped)
        print("Outside AOI:", outside_aoi)
        return

    # Use one consistent Mars CRS for the combined display.
    # This does not transform or modify the coordinates.
    mars_crs = (
        "GEOGCS["
        '"GCS_Mars_2000_Sphere",'
        'DATUM["Mars_2000_Sphere",'
        'SPHEROID["Mars_2000_Sphere_IAU_IAG",3396190,0]],'
        'PRIMEM["Reference_Meridian",0],'
        'UNIT["Degree",0.0174532925199433]]'
    )

    footprint_gdf = gpd.GeoDataFrame(
        geometry=footprints,
        crs=mars_crs,
    )

    aoi_gdf = gpd.GeoDataFrame(
        geometry=[aoi],
        crs=mars_crs,
    )

    insight = gpd.GeoDataFrame(
        geometry=[
            Point(
                INSIGHT_SITE["longitude"],
                INSIGHT_SITE["latitude"],
            )
        ],
        crs=mars_crs,
    )

    # ----------------------------------------------
    # CREATE MAP
    # ----------------------------------------------

    fig, ax = plt.subplots(figsize=(14, 9))

    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    footprint_gdf.plot(
        ax=ax,
        facecolor="none",
        edgecolor="#ff7043",
        linewidth=0.45,
        alpha=0.45,
    )

    aoi_gdf.boundary.plot(
        ax=ax,
        color="#00d9ff",
        linewidth=2,
    )

    insight.plot(
        ax=ax,
        color="white",
        marker="*",
        markersize=180,
        label="InSight landing site",
    )

    ax.set_xlim(area["west"], area["east"])
    ax.set_ylim(area["south"], area["north"])

    ax.set_title(
        "ELYSIUM | CTX Observation Footprints",
        color="white",
        fontsize=17,
        pad=16,
    )

    ax.set_xlabel("Longitude (°E)", color="white")
    ax.set_ylabel("Latitude (°N)", color="white")

    ax.tick_params(colors="white")

    ax.grid(
        color="gray",
        linestyle=":",
        alpha=0.3,
    )

    legend = ax.legend(
        facecolor="#151515",
        edgecolor="white",
    )

    for text in legend.get_texts():
        text.set_color("white")

    for spine in ax.spines.values():
        spine.set_color("gray")

    # ----------------------------------------------
    # SAVE MAP
    # ----------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = OUTPUT_DIR / "footprint_coverage_map.png"

    plt.tight_layout()

    plt.savefig(
        output_file,
        dpi=300,
        facecolor="black",
        bbox_inches="tight",
    )

    plt.close(fig)

    # ----------------------------------------------
    # FINAL REPORT
    # ----------------------------------------------

    print("\nFOOTPRINT MAP CREATED")
    print("=" * 50)

    print("Total folders:", len(folders))
    print("Successfully processed:", processed)
    print("Outside AOI:", outside_aoi)
    print("Skipped or invalid:", skipped)
    print("Footprint geometries:", len(footprints))

    print("\nSaved:", output_file)


# ==================================================
# MAIN
# ==================================================

if __name__ == "__main__":
    create_footprint_map()
