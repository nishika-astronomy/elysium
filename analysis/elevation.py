
"""
ELYSIUM — Elevation Analysis

Analyses MOLA elevation measurements for
Western Elysium Planitia.

Elevation units: metres.
Horizontal distance units: kilometres.

Coordinates are supplied directly by the MOLA
processing pipeline using PDS pixel registration.

Slope calculations are handled separately
in analysis/slope.py.
"""

import numpy as np

from config import MARS_EQUATORIAL_RADIUS_KM


# --------------------------------------------------
# ELEVATION VALIDATION
# --------------------------------------------------

def validate_elevation_grid(elevation):
    """Validate a two-dimensional elevation array."""

    elevation = np.asarray(elevation, dtype=np.float64)

    if elevation.ndim != 2:
        raise ValueError(
            "Elevation data must be a 2D array."
        )

    if min(elevation.shape) < 2:
        raise ValueError(
            "Elevation grid is too small."
        )

    if not np.any(np.isfinite(elevation)):
        raise ValueError(
            "Elevation grid contains no valid values."
        )

    return elevation


# --------------------------------------------------
# MOLA REGION VALIDATION
# --------------------------------------------------

def validate_region_data(region_data):
    """
    Validate the dictionary returned by process_mola().

    Expected keys:
        elevation
        latitudes
        longitudes
        bounds
    """

    if not isinstance(region_data, dict):
        raise TypeError(
            "MOLA region data must be a dictionary."
        )

    required_keys = {
        "elevation",
        "latitudes",
        "longitudes",
        "bounds"
    }

    missing = required_keys - region_data.keys()

    if missing:
        raise ValueError(
            f"MOLA region data is missing keys: {sorted(missing)}"
        )

    elevation = validate_elevation_grid(
        region_data["elevation"]
    )

    latitudes = np.asarray(
        region_data["latitudes"],
        dtype=np.float64
    )

    longitudes = np.asarray(
        region_data["longitudes"],
        dtype=np.float64
    )

    bounds = region_data["bounds"]

    if latitudes.ndim != 1 or longitudes.ndim != 1:
        raise ValueError(
            "MOLA latitude and longitude arrays must be 1D."
        )

    if len(latitudes) != elevation.shape[0]:
        raise ValueError(
            "Latitude count does not match elevation rows."
        )

    if len(longitudes) != elevation.shape[1]:
        raise ValueError(
            "Longitude count does not match elevation columns."
        )

    if not np.all(np.isfinite(latitudes)):
        raise ValueError(
            "Latitude coordinates contain non-finite values."
        )

    if not np.all(np.isfinite(longitudes)):
        raise ValueError(
            "Longitude coordinates contain non-finite values."
        )

    if (
        np.any(np.diff(latitudes) == 0)
        or np.any(np.diff(longitudes) == 0)
    ):
        raise ValueError(
            "MOLA coordinates contain repeated pixel centres."
        )

    if not (
        np.all(np.diff(latitudes) > 0)
        or np.all(np.diff(latitudes) < 0)
    ):
        raise ValueError(
            "Latitude coordinates must be strictly monotonic."
        )

    if not (
        np.all(np.diff(longitudes) > 0)
        or np.all(np.diff(longitudes) < 0)
    ):
        raise ValueError(
            "Longitude coordinates must be strictly monotonic."
        )

    if not isinstance(bounds, dict):
        raise TypeError(
            "MOLA bounds must be a dictionary."
        )

    required_bounds = {"west", "east", "south", "north"}

    if not required_bounds.issubset(bounds):
        raise ValueError(
            "MOLA bounds are incomplete."
        )

    west = float(bounds["west"])
    east = float(bounds["east"])
    south = float(bounds["south"])
    north = float(bounds["north"])

    if not (
        0 <= west < east <= 180
        and 0 <= south < north <= 90
    ):
        raise ValueError(
            "Invalid MOLA geographic bounds."
        )

    if (
        np.min(longitudes) < west
        or np.max(longitudes) > east
        or np.min(latitudes) < south
        or np.max(latitudes) > north
    ):
        raise ValueError(
            "MOLA pixel centres fall outside the supplied AOI bounds."
        )

    return {
        "elevation": elevation,
        "latitudes": latitudes,
        "longitudes": longitudes,
        "bounds": {
            "west": west,
            "east": east,
            "south": south,
            "north": north
        }
    }


# --------------------------------------------------
# GEOGRAPHIC COORDINATE GRID
# --------------------------------------------------

def generate_coordinate_grid(latitudes, longitudes):
    """
    Generate 2D coordinate grids from MOLA's actual
    one-dimensional pixel-centre coordinate arrays.

    Does not reconstruct coordinates from AOI boundaries.
    """

    latitudes = np.asarray(latitudes, dtype=np.float64)
    longitudes = np.asarray(longitudes, dtype=np.float64)

    if latitudes.ndim != 1 or longitudes.ndim != 1:
        raise ValueError(
            "Latitude and longitude inputs must be 1D."
        )

    if latitudes.size < 2 or longitudes.size < 2:
        raise ValueError(
            "Coordinate arrays must contain at least two values."
        )

    longitude_grid, latitude_grid = np.meshgrid(
        longitudes,
        latitudes
    )

    return longitude_grid, latitude_grid


# --------------------------------------------------
# ELEVATION STATISTICS
# --------------------------------------------------

def calculate_elevation_statistics(elevation):
    """Calculate descriptive elevation statistics."""

    elevation = validate_elevation_grid(elevation)

    valid = elevation[np.isfinite(elevation)]

    percentiles = np.percentile(
        valid,
        [1, 5, 25, 50, 75, 95, 99]
    )

    return {
        "valid_pixels": int(valid.size),
        "minimum_m": float(np.min(valid)),
        "maximum_m": float(np.max(valid)),
        "mean_m": float(np.mean(valid)),
        "median_m": float(np.median(valid)),
        "standard_deviation_m": float(np.std(valid)),
        "elevation_range_m": float(
            np.max(valid) - np.min(valid)
        ),
        "percentiles": {
            "p01": float(percentiles[0]),
            "p05": float(percentiles[1]),
            "p25": float(percentiles[2]),
            "p50": float(percentiles[3]),
            "p75": float(percentiles[4]),
            "p95": float(percentiles[5]),
            "p99": float(percentiles[6])
        }
    }


# --------------------------------------------------
# APPROXIMATE PIXEL SPACING
# --------------------------------------------------

def calculate_grid_spacing(latitudes, longitudes):
    """
    Calculate approximate horizontal pixel spacing.

    Uses actual MOLA pixel-centre coordinates and
    a spherical Mars approximation.

    East-west spacing is evaluated at mean latitude.

    This is a regional spacing estimate, not a
    substitute for a full geodesic calculation.
    """

    latitudes = np.asarray(latitudes, dtype=np.float64)
    longitudes = np.asarray(longitudes, dtype=np.float64)

    if latitudes.ndim != 1 or longitudes.ndim != 1:
        raise ValueError(
            "Coordinate arrays must be one-dimensional."
        )

    if latitudes.size < 2 or longitudes.size < 2:
        raise ValueError(
            "At least two coordinates are required per axis."
        )

    if not (
        np.all(np.isfinite(latitudes))
        and np.all(np.isfinite(longitudes))
    ):
        raise ValueError(
            "Coordinate arrays contain non-finite values."
        )

    mean_latitude = np.radians(
        np.mean(latitudes)
    )

    longitude_spacing = np.radians(
        np.median(np.abs(np.diff(longitudes)))
    )

    latitude_spacing = np.radians(
        np.median(np.abs(np.diff(latitudes)))
    )

    dx_km = (
        MARS_EQUATORIAL_RADIUS_KM
        * np.cos(mean_latitude)
        * longitude_spacing
    )

    dy_km = (
        MARS_EQUATORIAL_RADIUS_KM
        * latitude_spacing
    )

    return {
        "east_west_km": float(dx_km),
        "north_south_km": float(dy_km)
    }


# --------------------------------------------------
# ELEVATION CLASSIFICATION
# --------------------------------------------------

def classify_elevation(elevation, thresholds=None):
    """
    Divide elevation into relative visualisation classes.

    Classes represent relative elevation bands.

    They are NOT geological units or safety categories.

    Class values:
        0-4: relative elevation bands
        -1: invalid pixels
    """

    elevation = validate_elevation_grid(elevation)

    if thresholds is None:
        thresholds = [0.20, 0.40, 0.60, 0.80]

    thresholds = np.asarray(
        thresholds,
        dtype=np.float64
    )

    if (
        thresholds.ndim != 1
        or np.any(~np.isfinite(thresholds))
        or np.any(thresholds <= 0)
        or np.any(thresholds >= 1)
        or np.any(np.diff(thresholds) <= 0)
    ):
        raise ValueError(
            "Thresholds must be finite, strictly increasing "
            "values between 0 and 1."
        )

    valid = np.isfinite(elevation)

    minimum = np.min(elevation[valid])
    maximum = np.max(elevation[valid])

    normalized = np.full(
        elevation.shape,
        np.nan,
        dtype=np.float32
    )

    if maximum == minimum:
        normalized[valid] = 0.5
    else:
        normalized[valid] = (
            (elevation[valid] - minimum)
            / (maximum - minimum)
        )

    classes = np.full(
        elevation.shape,
        -1,
        dtype=np.int8
    )

    classes[valid] = np.digitize(
        normalized[valid],
        thresholds
    )

    return classes


# --------------------------------------------------
# COMPLETE ELEVATION ANALYSIS
# --------------------------------------------------

def analyse_elevation(region_data):
    """
    Run the complete MOLA elevation analysis.

    Input:
        Dictionary returned by process_mola().

    Output:
        Statistics, coordinate grids, pixel spacing,
        elevation classes and elevation measurements.
    """

    region = validate_region_data(region_data)

    elevation = region["elevation"]
    latitudes = region["latitudes"]
    longitudes = region["longitudes"]
    bounds = region["bounds"]

    longitude_grid, latitude_grid = generate_coordinate_grid(
        latitudes,
        longitudes
    )

    statistics = calculate_elevation_statistics(
        elevation
    )

    spacing = calculate_grid_spacing(
        latitudes,
        longitudes
    )

    elevation_classes = classify_elevation(
        elevation
    )

    return {
        "statistics": statistics,
        "longitude": longitude_grid,
        "latitude": latitude_grid,
        "longitude_1d": longitudes.copy(),
        "latitude_1d": latitudes.copy(),
        "bounds": bounds,
        "grid_spacing_km": spacing,
        "elevation_classes": elevation_classes,
        "elevation_m": elevation.copy()
    }


# --------------------------------------------------
# MAIN
# --------------------------------------------------

if __name__ == "__main__":

    from processing.mola import process_mola

    print("\nELYSIUM ELEVATION ANALYSIS")
    print("=" * 45)

    region_data = process_mola()

    results = analyse_elevation(region_data)

    stats = results["statistics"]
    spacing = results["grid_spacing_km"]

    print("\nElevation statistics:")
    print("Valid pixels:", stats["valid_pixels"])
    print("Minimum:", round(stats["minimum_m"], 2), "m")
    print("Maximum:", round(stats["maximum_m"], 2), "m")
    print("Mean:", round(stats["mean_m"], 2), "m")
    print("Median:", round(stats["median_m"], 2), "m")
    print(
        "Standard deviation:",
        round(stats["standard_deviation_m"], 2),
        "m"
    )

    print("\nGrid spacing:")
    print(
        "East-west:",
        round(spacing["east_west_km"], 4),
        "km/pixel"
    )
    print(
        "North-south:",
        round(spacing["north_south_km"], 4),
        "km/pixel"
    )

    print("\nCoordinate grid:", results["longitude"].shape)

    print(
        "Elevation classes:",
        np.unique(results["elevation_classes"])
    )

    print("\nElevation analysis completed successfully.")
