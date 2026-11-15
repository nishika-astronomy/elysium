
"""
ELYSIUM — Footprint Coverage Analysis
Version: 0.2.0

Analyses orbital observation footprints over the
Western Elysium Planitia detailed study area.

Features:
    - Selects one preferred footprint per observation.
    - Validates Mars CRS and coordinate bounds.
    - Calculates footprint intersections with the AOI.
    - Uses a Mars spherical equal-area projection.
    - Calculates unique combined AOI coverage.
    - Records processing outcomes and skipped observations.

The original NASA footprint files are never modified.
"""

from pathlib import Path
from collections import Counter

import geopandas as gpd
import pandas as pd

from pyproj import CRS, Transformer
import numpy as np
from shapely.geometry import box
from shapely.ops import unary_union

try:
    from shapely import transform as shapely_transform
except ImportError:
    from shapely.ops import transform as shapely_transform

from config import (
    RAW_DIR,
    PROCESSED_DIR,
    DETAILED_AOI,
    MARS_EQUATORIAL_RADIUS_KM,
)


# ==================================================
# CONFIGURATION
# ==================================================

FOOTPRINT_DIR = RAW_DIR / "footprints"

COVERAGE_OUTPUT = PROCESSED_DIR / "footprint_coverage.csv"
SUMMARY_OUTPUT = PROCESSED_DIR / "footprint_analysis_summary.csv"

MARS_RADIUS_METRES = MARS_EQUATORIAL_RADIUS_KM * 1000

CRS_RADIUS_TOLERANCE_METRES = 1.0

# Preferred footprint representation.
PREFERRED_VARIANT = "_ga.shp"

FALLBACK_VARIANTS = (
    "_c0a.shp",
    "_c180a.shp",
)


# ==================================================
# AOI AND PROJECTION
# ==================================================

def build_projection():
    """
    Build a Mars spherical Lambert azimuthal equal-area
    projection centred on the detailed AOI.

    The projection is equal-area, making it suitable
    for calculating footprint coverage areas.
    """

    lon0 = (
        DETAILED_AOI["west"] +
        DETAILED_AOI["east"]
    ) / 2

    lat0 = (
        DETAILED_AOI["south"] +
        DETAILED_AOI["north"]
    ) / 2

    mars_geographic = CRS.from_proj4(
        f"+proj=longlat "
        f"+R={MARS_RADIUS_METRES} "
        f"+no_defs +type=crs"
    )

    mars_equal_area = CRS.from_proj4(
        f"+proj=laea "
        f"+lat_0={lat0} "
        f"+lon_0={lon0} "
        f"+R={MARS_RADIUS_METRES} "
        f"+units=m +no_defs +type=crs"
    )

    transformer = Transformer.from_crs(
        mars_geographic,
        mars_equal_area,
        always_xy=True,
    )

    return transformer


def project_geometry(geometry, transformer):
    """Transform a Mars longitude/latitude geometry to metres."""

    def project_coordinates(coords):
        x, y = transformer.transform(
            coords[:, 0],
            coords[:, 1],
        )
        return np.column_stack((x, y))

    try:
        return shapely_transform(
            geometry,
            project_coordinates,
        )
    except TypeError:
        # Compatibility with older Shapely versions.
        from shapely.ops import transform

        return transform(
            transformer.transform,
            geometry,
        )


def build_aoi():
    """Create the detailed study-area geometry."""

    return box(
        DETAILED_AOI["west"],
        DETAILED_AOI["south"],
        DETAILED_AOI["east"],
        DETAILED_AOI["north"],
    )


# ==================================================
# FOOTPRINT DISCOVERY
# ==================================================

def discover_footprint_folders():
    """Find all extracted observation footprint folders."""

    if not FOOTPRINT_DIR.exists():
        raise FileNotFoundError(
            f"Footprint directory not found: {FOOTPRINT_DIR}"
        )

    return sorted(
        folder
        for folder in FOOTPRINT_DIR.glob("*_footprint")
        if folder.is_dir()
    )


def select_preferred_shapefile(folder):
    """
    Select one footprint representation per observation.

    Priority:
        1. GA
        2. C0A
        3. C180A

    If none of these variants exists, return None.
    """

    shapefiles = sorted(
        path
        for path in folder.rglob("*")
        if path.is_file()
        and path.suffix.lower() == ".shp"
    )

    if not shapefiles:
        return None

    by_name = {
        path.name.lower(): path
        for path in shapefiles
    }

    preferred_name = (
        folder.name[:-len("_footprint")].lower()
    )

    # Search for the preferred GA variant.
    for path in shapefiles:
        if path.name.lower().endswith(PREFERRED_VARIANT):
            return path

    # Use deterministic fallbacks.
    for variant in FALLBACK_VARIANTS:

        for path in shapefiles:
            if path.name.lower().endswith(variant):
                return path

    # Last-resort fallback: first sorted shapefile.
    return shapefiles[0]


# ==================================================
# CRS VALIDATION
# ==================================================

def validate_mars_crs(crs):
    """
    Check that the footprint CRS represents the
    expected Mars spherical reference system.

    Both identified ESRI CRS groups are accepted
    when their sphere radius matches the project radius.
    """

    if crs is None:
        raise ValueError("Missing CRS information.")

    parsed_crs = CRS.from_user_input(crs)

    if not parsed_crs.is_geographic:
        raise ValueError(
            "Expected a geographic longitude/latitude CRS."
        )

    radius = parsed_crs.ellipsoid.semi_major_metre

    if radius is None:
        raise ValueError(
            "Could not identify the CRS sphere radius."
        )

    if abs(radius - MARS_RADIUS_METRES) > CRS_RADIUS_TOLERANCE_METRES:
        raise ValueError(
            f"Unexpected planetary radius: {radius} m."
        )

    return parsed_crs


def validate_coordinate_bounds(geometry):
    """
    Confirm that stored coordinates are plausible
    longitude/latitude values in x/y order.

    Mars east-positive longitudes may use 0–360 degrees.
    """

    if geometry is None or geometry.is_empty:
        raise ValueError("Empty geometry.")

    min_x, min_y, max_x, max_y = geometry.bounds

    if not (
        -180 <= min_x <= 360
        and -180 <= max_x <= 360
    ):
        raise ValueError(
            "Longitude bounds are outside the accepted range."
        )

    if not (
        -90 <= min_y <= 90
        and -90 <= max_y <= 90
    ):
        raise ValueError(
            "Latitude bounds are outside the accepted range. "
            "Possible axis-order mismatch."
        )

    return True


# ==================================================
# GEOMETRY PROCESSING
# ==================================================

def read_footprint(shapefile):
    """
    Read and validate a selected footprint shapefile.

    Returns:
        footprint geometry
        CRS description
    """

    data = gpd.read_file(shapefile)

    if data.empty:
        raise ValueError("Shapefile contains no records.")

    parsed_crs = validate_mars_crs(data.crs)

    geometries = [
        geometry
        for geometry in data.geometry
        if geometry is not None and not geometry.is_empty
    ]

    if not geometries:
        raise ValueError("No usable geometries found.")

    footprint = unary_union(geometries)

    if footprint.is_empty:
        raise ValueError("Combined footprint is empty.")

    if not footprint.is_valid:
        raise ValueError("Footprint geometry is invalid.")

    validate_coordinate_bounds(footprint)

    return footprint, parsed_crs.to_string()


# ==================================================
# MAIN ANALYSIS
# ==================================================

def analyse_footprints():
    """
    Calculate individual and combined footprint coverage
    within the detailed Western Elysium Planitia AOI.
    """

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    folders = discover_footprint_folders()

    transformer = build_projection()

    aoi = build_aoi()

    projected_aoi = project_geometry(
        aoi,
        transformer,
    )

    total_area_km2 = projected_aoi.area / 1_000_000

    records = []
    intersections = []

    status_counts = Counter()

    print("\nELYSIUM | FOOTPRINT COVERAGE ANALYSIS")
    print("=" * 55)

    print("Footprint folders:", len(folders))
    print("Preferred variant:", PREFERRED_VARIANT)
    print("AOI:", DETAILED_AOI["name"])
    print("AOI area:", round(total_area_km2, 2), "km²")

    for index, folder in enumerate(folders, start=1):

        observation_id = folder.name[:-len("_footprint")]

        shapefile = select_preferred_shapefile(folder)

        record = {
            "observation_id": observation_id,
            "selected_shapefile": (
                shapefile.name if shapefile else None
            ),
            "crs": None,
            "intersection_area_km2": 0.0,
            "coverage_percent_of_aoi": 0.0,
            "status": None,
            "error": None,
        }

        try:

            if shapefile is None:
                raise FileNotFoundError(
                    "No shapefiles found in observation folder."
                )

            footprint, crs_name = read_footprint(shapefile)

            record["crs"] = crs_name

            projected_footprint = project_geometry(
                footprint,
                transformer,
            )

            intersection = projected_footprint.intersection(
                projected_aoi
            )

            if intersection.is_empty:
                record["status"] = "Outside AOI"

            elif intersection.area <= 0:
                record["status"] = "Zero-area intersection"

            else:

                area_km2 = (
                    intersection.area / 1_000_000
                )

                record["intersection_area_km2"] = round(
                    area_km2,
                    6,
                )

                record["coverage_percent_of_aoi"] = round(
                    (area_km2 / total_area_km2) * 100,
                    6,
                )

                record["status"] = "Intersecting AOI"

                intersections.append(intersection)

        except Exception as error:

            record["status"] = "Processing failed"
            record["error"] = str(error)

        status_counts[record["status"]] += 1

        records.append(record)

        if index % 25 == 0:
            print(
                f"Processed {index}/{len(folders)} folders..."
            )

    # ==================================================
    # COMBINED AOI COVERAGE
    # ==================================================

    if intersections:

        covered_geometry = unary_union(intersections)

        # Clip once more to ensure the result cannot
        # extend outside the defined AOI.
        covered_geometry = covered_geometry.intersection(
            projected_aoi
        )

        covered_area_km2 = (
            covered_geometry.area / 1_000_000
        )

    else:
        covered_area_km2 = 0.0

    coverage_percent = (
        covered_area_km2 / total_area_km2
    ) * 100

    # ==================================================
    # SAVE RESULTS
    # ==================================================

    results = pd.DataFrame(records)

    results.to_csv(
        COVERAGE_OUTPUT,
        index=False,
    )

    summary = pd.DataFrame([
        {
            "metric": "total_observation_folders",
            "value": len(folders),
        },
        {
            "metric": "total_aoi_area_km2",
            "value": round(total_area_km2, 6),
        },
        {
            "metric": "unique_covered_area_km2",
            "value": round(covered_area_km2, 6),
        },
        {
            "metric": "unique_aoi_coverage_percent",
            "value": round(coverage_percent, 6),
        },
        {
            "metric": "intersecting_observations",
            "value": status_counts["Intersecting AOI"],
        },
        {
            "metric": "outside_aoi_observations",
            "value": status_counts["Outside AOI"],
        },
        {
            "metric": "failed_observations",
            "value": status_counts["Processing failed"],
        },
        {
            "metric": "zero_area_intersections",
            "value": status_counts["Zero-area intersection"],
        },
    ])

    summary.to_csv(
        SUMMARY_OUTPUT,
        index=False,
    )

    # ==================================================
    # FINAL REPORT
    # ==================================================

    print("\nFOOTPRINT ANALYSIS COMPLETE")
    print("=" * 55)

    print("Total observation folders:", len(folders))
    print(
        "Intersecting observations:",
        status_counts["Intersecting AOI"],
    )
    print(
        "Outside AOI:",
        status_counts["Outside AOI"],
    )
    print(
        "Processing failures:",
        status_counts["Processing failed"],
    )

    print("\nAOI area:", round(total_area_km2, 2), "km²")
    print(
        "Unique covered area:",
        round(covered_area_km2, 2),
        "km²",
    )
    print(
        "Unique AOI coverage:",
        round(coverage_percent, 4),
        "%",
    )

    print("\nSaved files:")
    print(COVERAGE_OUTPUT)
    print(SUMMARY_OUTPUT)

    return {
        "records": records,
        "total_area_km2": total_area_km2,
        "covered_area_km2": covered_area_km2,
        "coverage_percent": coverage_percent,
        "status_counts": dict(status_counts),
    }


# ==================================================
# MAIN
# ==================================================

if __name__ == "__main__":
    analyse_footprints()
