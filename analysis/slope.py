
"""
ELYSIUM — Slope Analysis

Calculates terrain slope from the MOLA elevation grid
for Western Elysium Planitia.

Slope is calculated using numerical elevation gradients.

Elevation units: metres
Horizontal distance units: kilometres
Slope units: degrees

Grid coordinates and pixel spacing are obtained from
the MOLA processing and elevation analysis pipelines.

Slope classes are provisional visualisation categories,
not certified astronaut mobility limits.
"""

import numpy as np


# --------------------------------------------------
# INPUT VALIDATION
# --------------------------------------------------

def validate_inputs(elevation_results):
    """
    Validate the output dictionary from analyse_elevation().

    Required:
        elevation_m
        grid_spacing_km
    """

    if not isinstance(elevation_results, dict):
        raise TypeError(
            "Elevation analysis output must be a dictionary."
        )

    required_keys = {
        "elevation_m",
        "grid_spacing_km"
    }

    missing = required_keys - elevation_results.keys()

    if missing:
        raise ValueError(
            f"Elevation results are missing keys: {sorted(missing)}"
        )

    elevation = np.asarray(
        elevation_results["elevation_m"],
        dtype=np.float64
    )

    if elevation.ndim != 2:
        raise ValueError(
            "Elevation data must be a 2D array."
        )

    if min(elevation.shape) < 2:
        raise ValueError(
            "Elevation grid must contain at least 2 rows and columns."
        )

    if not np.any(np.isfinite(elevation)):
        raise ValueError(
            "No valid elevation values found."
        )

    spacing = elevation_results["grid_spacing_km"]

    if not isinstance(spacing, dict):
        raise TypeError(
            "Grid spacing must be a dictionary."
        )

    required_spacing = {
        "east_west_km",
        "north_south_km"
    }

    missing_spacing = required_spacing - spacing.keys()

    if missing_spacing:
        raise ValueError(
            f"Grid spacing is missing: {sorted(missing_spacing)}"
        )

    east_west_km = float(spacing["east_west_km"])
    north_south_km = float(spacing["north_south_km"])

    if not (
        np.isfinite(east_west_km)
        and east_west_km > 0
        and np.isfinite(north_south_km)
        and north_south_km > 0
    ):
        raise ValueError(
            "Grid spacing must contain finite positive values."
        )

    return elevation, {
        "east_west_km": east_west_km,
        "north_south_km": north_south_km
    }


# --------------------------------------------------
# SLOPE CALCULATION
# --------------------------------------------------

def calculate_slope(
    elevation,
    east_west_km,
    north_south_km
):
    """
    Calculate terrain slope using numerical elevation gradients.

    Elevation is converted from metres to kilometres
    so that the vertical and horizontal units match.

    Invalid elevation neighbourhoods produce NaN slopes.
    """

    elevation = np.asarray(
        elevation,
        dtype=np.float64
    )

    if elevation.ndim != 2 or min(elevation.shape) < 2:
        raise ValueError(
            "Slope calculation requires a 2D grid "
            "with at least 2 rows and columns."
        )

    if not (
        np.isfinite(east_west_km)
        and east_west_km > 0
        and np.isfinite(north_south_km)
        and north_south_km > 0
    ):
        raise ValueError(
            "Pixel spacing must be finite and positive."
        )

    valid_elevation = np.isfinite(elevation)

    if not np.any(valid_elevation):
        raise ValueError(
            "No valid elevation values available."
        )

    # Convert elevation from metres to kilometres.
    elevation_km = elevation / 1000.0

    # Calculate north-south and east-west gradients.
    gradient_north_south, gradient_east_west = np.gradient(
        elevation_km,
        north_south_km,
        east_west_km
    )

    # Calculate gradient magnitude.
    gradient_magnitude = np.sqrt(
        gradient_east_west ** 2
        + gradient_north_south ** 2
    )

    # Convert gradient magnitude into degrees.
    slope = np.degrees(
        np.arctan(gradient_magnitude)
    )

    # Exclude locations where the gradient is invalid.
    valid_gradient = (
        valid_elevation
        & np.isfinite(gradient_north_south)
        & np.isfinite(gradient_east_west)
    )

    slope[~valid_gradient] = np.nan

    return slope


# --------------------------------------------------
# SLOPE STATISTICS
# --------------------------------------------------

def calculate_slope_statistics(slope):
    """
    Calculate descriptive statistics for slope measurements.

    All slope values are expressed in degrees.
    """

    slope = np.asarray(
        slope,
        dtype=np.float64
    )

    if slope.ndim != 2:
        raise ValueError(
            "Slope data must be a 2D array."
        )

    valid = slope[np.isfinite(slope)]

    if valid.size == 0:
        raise ValueError(
            "No valid slope values available."
        )

    return {
        "minimum": float(np.min(valid)),
        "maximum": float(np.max(valid)),
        "mean": float(np.mean(valid)),
        "median": float(np.median(valid)),
        "std_dev": float(np.std(valid)),
        "percentile_90": float(np.percentile(valid, 90)),
        "percentile_95": float(np.percentile(valid, 95)),
        "valid_pixels": int(valid.size)
    }


# --------------------------------------------------
# SLOPE CLASSIFICATION
# --------------------------------------------------

def classify_slope(slope):
    """
    Classify slope for visualisation.

    Classification ranges:

        Class 0: Below 5 degrees
        Class 1: 5 to below 10 degrees
        Class 2: 10 to below 20 degrees
        Class 3: 20 to below 30 degrees
        Class 4: 30 degrees and above

    Invalid pixels are assigned -1.

    These classes are provisional analytical categories,
    not certified astronaut mobility limits.
    """

    slope = np.asarray(
        slope,
        dtype=np.float64
    )

    if slope.ndim != 2:
        raise ValueError(
            "Slope data must be a 2D array."
        )

    valid = np.isfinite(slope)

    if np.any(slope[valid] < 0) or np.any(slope[valid] > 90):
        raise ValueError(
            "Slope values must be between 0 and 90 degrees."
        )

    classes = np.full(
        slope.shape,
        -1,
        dtype=np.int8
    )

    classes[valid & (slope < 5)] = 0

    classes[
        valid & (slope >= 5) & (slope < 10)
    ] = 1

    classes[
        valid & (slope >= 10) & (slope < 20)
    ] = 2

    classes[
        valid & (slope >= 20) & (slope < 30)
    ] = 3

    classes[valid & (slope >= 30)] = 4

    return classes


# --------------------------------------------------
# SLOPE CLASS DISTRIBUTION
# --------------------------------------------------

def calculate_class_distribution(slope_classes, valid_pixels):
    """
    Calculate the number and percentage of pixels
    belonging to each slope class.
    """

    if valid_pixels <= 0:
        raise ValueError(
            "Valid pixel count must be positive."
        )

    distribution = {}

    for class_id in range(5):

        count = int(
            np.sum(slope_classes == class_id)
        )

        percentage = (
            count / valid_pixels
        ) * 100.0

        distribution[class_id] = {
            "pixels": count,
            "percentage": float(percentage)
        }

    return distribution


# --------------------------------------------------
# COMPLETE SLOPE ANALYSIS
# --------------------------------------------------

def analyse_slope(elevation_results):
    """
    Run the complete slope-analysis pipeline.

    Input:
        Dictionary returned by analyse_elevation().

    Output:
        Slope array, statistics, classification,
        pixel spacing and class distribution.
    """

    print("\nELYSIUM SLOPE ANALYSIS")
    print("=" * 45)

    elevation, spacing = validate_inputs(
        elevation_results
    )

    print("\nGrid spacing:")

    print(
        f"East-west: {spacing['east_west_km']:.4f} km/pixel"
    )

    print(
        f"North-south: {spacing['north_south_km']:.4f} km/pixel"
    )

    # Calculate slope.
    slope = calculate_slope(
        elevation,
        spacing["east_west_km"],
        spacing["north_south_km"]
    )

    # Calculate statistics.
    statistics = calculate_slope_statistics(
        slope
    )

    # Classify slope.
    slope_classes = classify_slope(
        slope
    )

    # Calculate class distribution.
    class_distribution = calculate_class_distribution(
        slope_classes,
        statistics["valid_pixels"]
    )

    print("\nSlope statistics:")

    print(
        f"Minimum: {statistics['minimum']:.4f} degrees"
    )

    print(
        f"Maximum: {statistics['maximum']:.4f} degrees"
    )

    print(
        f"Mean: {statistics['mean']:.4f} degrees"
    )

    print(
        f"Median: {statistics['median']:.4f} degrees"
    )

    print(
        f"Standard deviation: {statistics['std_dev']:.4f} degrees"
    )

    print(
        f"90th percentile: {statistics['percentile_90']:.4f} degrees"
    )

    print(
        f"95th percentile: {statistics['percentile_95']:.4f} degrees"
    )

    print(
        f"Valid pixels: {statistics['valid_pixels']}"
    )

    print("\nSlope class distribution:")

    for class_id, data in class_distribution.items():

        print(
            f"Class {class_id}: "
            f"{data['pixels']} pixels "
            f"({data['percentage']:.2f}%)"
        )

    print("\nSlope analysis completed successfully.")

    return {
        "slope_degrees": slope,
        "statistics": statistics,
        "slope_classes": slope_classes,
        "class_distribution": class_distribution,
        "grid_spacing_km": spacing
    }


# --------------------------------------------------
# MAIN
# --------------------------------------------------

if __name__ == "__main__":

    from processing.mola import process_mola
    from analysis.elevation import analyse_elevation

    print("\nLoading MOLA elevation data...")

    region_data = process_mola()

    print("\nRunning elevation analysis...")

    elevation_results = analyse_elevation(
        region_data
    )

    print("\nRunning slope analysis...")

    results = analyse_slope(
        elevation_results
    )
