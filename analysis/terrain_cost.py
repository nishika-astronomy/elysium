
"""
ELYSIUM — Terrain Traversal Cost Analysis

Combines MOLA-derived slope and terrain ruggedness
into a relative regional-scale traversal cost surface.

Inputs:
    Elevation analysis:
        elevation_m
        latitude
        longitude
        grid_spacing_km

    Roughness analysis:
        tri_m
        local_std_m

    Slope analysis:
        slope_degrees

Model:
    Weighted additive cost model.

    Terrain cost =
        slope_weight * normalised_slope
        + roughness_weight * normalised_TRI

Normalisation:
    Robust percentile scaling with clipping.

Important:
    The output is a relative terrain traversal cost.
    It is not a certified astronaut safety assessment.

    MOLA resolution limits this analysis to regional-scale
    terrain variation. Small rocks, loose soil and
    astronaut-scale obstacles cannot be resolved.
"""

import numpy as np


# --------------------------------------------------
# DEFAULT MODEL CONFIGURATION
# --------------------------------------------------

DEFAULT_SLOPE_WEIGHT = 0.70
DEFAULT_ROUGHNESS_WEIGHT = 0.30

DEFAULT_LOWER_PERCENTILE = 2.0
DEFAULT_UPPER_PERCENTILE = 98.0

COST_CLASS_THRESHOLDS = [0.20, 0.40, 0.60, 0.80]


# --------------------------------------------------
# INPUT VALIDATION
# --------------------------------------------------

def validate_cost_inputs(
    elevation_results,
    roughness_results,
    slope_results
):
    """
    Validate outputs from the three analysis modules.

    All arrays must have identical dimensions.
    """

    if not all(
        isinstance(item, dict)
        for item in (
            elevation_results,
            roughness_results,
            slope_results
        )
    ):
        raise TypeError(
            "All analysis inputs must be dictionaries."
        )

    required = {
        "elevation": {
            "elevation_m",
            "latitude",
            "longitude",
            "grid_spacing_km"
        },
        "roughness": {
            "tri_m",
            "local_std_m"
        },
        "slope": {
            "slope_degrees"
        }
    }

    inputs = {
        "elevation": elevation_results,
        "roughness": roughness_results,
        "slope": slope_results
    }

    for name, keys in required.items():

        missing = keys - inputs[name].keys()

        if missing:
            raise ValueError(
                f"{name.title()} results are missing keys: "
                f"{sorted(missing)}"
            )

    elevation = np.asarray(
        elevation_results["elevation_m"],
        dtype=np.float64
    )

    latitude = np.asarray(
        elevation_results["latitude"],
        dtype=np.float64
    )

    longitude = np.asarray(
        elevation_results["longitude"],
        dtype=np.float64
    )

    tri = np.asarray(
        roughness_results["tri_m"],
        dtype=np.float64
    )

    local_std = np.asarray(
        roughness_results["local_std_m"],
        dtype=np.float64
    )

    slope = np.asarray(
        slope_results["slope_degrees"],
        dtype=np.float64
    )

    arrays = {
        "elevation": elevation,
        "latitude": latitude,
        "longitude": longitude,
        "TRI": tri,
        "local standard deviation": local_std,
        "slope": slope
    }

    for name, array in arrays.items():

        if array.ndim != 2:
            raise ValueError(
                f"{name} must be a two-dimensional array."
            )

        if array.shape != elevation.shape:
            raise ValueError(
                f"{name} shape does not match elevation grid."
            )

    if min(elevation.shape) < 2:
        raise ValueError(
            "Terrain grid is too small for analysis."
        )

    spacing = elevation_results["grid_spacing_km"]

    if not isinstance(spacing, dict):
        raise TypeError(
            "Grid spacing must be a dictionary."
        )

    for key in ("east_west_km", "north_south_km"):

        if key not in spacing:
            raise ValueError(
                f"Grid spacing is missing '{key}'."
            )

        value = float(spacing[key])

        if not np.isfinite(value) or value <= 0:
            raise ValueError(
                f"Invalid grid spacing: {key}."
            )

    if np.any(np.isfinite(slope) & ((slope < 0) | (slope > 90))):
        raise ValueError(
            "Slope values must be between 0 and 90 degrees."
        )

    if np.any(np.isfinite(tri) & (tri < 0)):
        raise ValueError(
            "TRI values cannot be negative."
        )

    if np.any(np.isfinite(local_std) & (local_std < 0)):
        raise ValueError(
            "Local standard deviation cannot be negative."
        )

    return {
        "elevation_m": elevation,
        "latitude": latitude,
        "longitude": longitude,
        "tri_m": tri,
        "local_std_m": local_std,
        "slope_degrees": slope,
        "grid_spacing_km": dict(spacing)
    }


# --------------------------------------------------
# WEIGHT VALIDATION
# --------------------------------------------------

def validate_weights(slope_weight, roughness_weight):
    """
    Validate and normalise model weights.

    Both weights must be non-negative and their sum
    must be greater than zero.
    """

    slope_weight = float(slope_weight)
    roughness_weight = float(roughness_weight)

    if not (
        np.isfinite(slope_weight)
        and np.isfinite(roughness_weight)
    ):
        raise ValueError(
            "Model weights must be finite."
        )

    if slope_weight < 0 or roughness_weight < 0:
        raise ValueError(
            "Model weights cannot be negative."
        )

    total = slope_weight + roughness_weight

    if total <= 0:
        raise ValueError(
            "At least one model weight must be positive."
        )

    return (
        slope_weight / total,
        roughness_weight / total
    )


# --------------------------------------------------
# ROBUST NORMALISATION
# --------------------------------------------------

def normalise_cost_layer(
    values,
    valid_mask,
    lower_percentile=DEFAULT_LOWER_PERCENTILE,
    upper_percentile=DEFAULT_UPPER_PERCENTILE
):
    """
    Convert a terrain measurement into a dimensionless
    relative cost layer between 0 and 1.

    Percentile clipping limits the influence of extreme
    outliers.

    Values below the lower percentile become 0.
    Values above the upper percentile become 1.

    Invalid pixels remain NaN.
    """

    values = np.asarray(
        values,
        dtype=np.float64
    )

    valid_mask = np.asarray(
        valid_mask,
        dtype=bool
    )

    if values.ndim != 2 or valid_mask.shape != values.shape:
        raise ValueError(
            "Values and validity mask must be matching 2D arrays."
        )

    if not (
        np.isfinite(lower_percentile)
        and np.isfinite(upper_percentile)
        and 0 <= lower_percentile < upper_percentile <= 100
    ):
        raise ValueError(
            "Percentiles must satisfy 0 <= lower < upper <= 100."
        )

    valid = valid_mask & np.isfinite(values)

    if not np.any(valid):
        raise ValueError(
            "No valid values available for normalisation."
        )

    lower, upper = np.percentile(
        values[valid],
        [lower_percentile, upper_percentile]
    )

    normalised = np.full(
        values.shape,
        np.nan,
        dtype=np.float64
    )

    if upper == lower:
        normalised[valid] = 0.0
    else:
        clipped = np.clip(
            values[valid],
            lower,
            upper
        )

        normalised[valid] = (
            (clipped - lower) / (upper - lower)
        )

    metadata = {
        "lower_percentile": float(lower_percentile),
        "upper_percentile": float(upper_percentile),
        "lower_value": float(lower),
        "upper_value": float(upper)
    }

    return normalised, metadata


# --------------------------------------------------
# COMBINED TERRAIN COST
# --------------------------------------------------

def calculate_cost_surface(
    normalised_slope,
    normalised_roughness,
    valid_mask,
    slope_weight=DEFAULT_SLOPE_WEIGHT,
    roughness_weight=DEFAULT_ROUGHNESS_WEIGHT
):
    """
    Calculate the weighted additive terrain cost surface.

    Both input layers must be normalised to [0, 1].

    The output is also between 0 and 1.
    """

    slope_weight, roughness_weight = validate_weights(
        slope_weight,
        roughness_weight
    )

    normalised_slope = np.asarray(
        normalised_slope,
        dtype=np.float64
    )

    normalised_roughness = np.asarray(
        normalised_roughness,
        dtype=np.float64
    )

    valid_mask = np.asarray(
        valid_mask,
        dtype=bool
    )

    if not (
        normalised_slope.ndim == 2
        and normalised_roughness.ndim == 2
        and valid_mask.ndim == 2
    ):
        raise ValueError(
            "Cost layers and validity mask must be 2D."
        )

    if not (
        normalised_slope.shape
        == normalised_roughness.shape
        == valid_mask.shape
    ):
        raise ValueError(
            "Cost layers and validity mask must have matching shapes."
        )

    valid = (
        valid_mask
        & np.isfinite(normalised_slope)
        & np.isfinite(normalised_roughness)
    )

    if not np.any(valid):
        raise ValueError(
            "No common valid pixels available for cost calculation."
        )

    if np.any(
        (normalised_slope[valid] < 0)
        | (normalised_slope[valid] > 1)
        | (normalised_roughness[valid] < 0)
        | (normalised_roughness[valid] > 1)
    ):
        raise ValueError(
            "Normalised cost layers must be between 0 and 1."
        )

    cost = np.full(
        valid_mask.shape,
        np.nan,
        dtype=np.float64
    )

    cost[valid] = (
        slope_weight * normalised_slope[valid]
        + roughness_weight * normalised_roughness[valid]
    )

    return cost


# --------------------------------------------------
# TERRAIN COST CLASSIFICATION
# --------------------------------------------------

def classify_terrain_cost(
    cost_surface,
    thresholds=COST_CLASS_THRESHOLDS
):
    """
    Classify relative terrain cost.

    Class 0: Lowest relative cost
    Class 1: Low-to-moderate relative cost
    Class 2: Moderate relative cost
    Class 3: High relative cost
    Class 4: Highest relative cost

    Invalid pixels: -1

    These are relative visualisation categories,
    not physical safety thresholds.
    """

    cost_surface = np.asarray(
        cost_surface,
        dtype=np.float64
    )

    if cost_surface.ndim != 2:
        raise ValueError(
            "Terrain cost surface must be 2D."
        )

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
            "Classification thresholds must be finite, "
            "strictly increasing values between 0 and 1."
        )

    valid = np.isfinite(cost_surface)

    if np.any(
        (cost_surface[valid] < 0)
        | (cost_surface[valid] > 1)
    ):
        raise ValueError(
            "Terrain cost values must be between 0 and 1."
        )

    classes = np.full(
        cost_surface.shape,
        -1,
        dtype=np.int8
    )

    classes[valid] = np.digitize(
        cost_surface[valid],
        thresholds
    )

    return classes


# --------------------------------------------------
# TERRAIN COST STATISTICS
# --------------------------------------------------

def calculate_cost_statistics(cost_surface):
    """
    Calculate descriptive statistics for the cost surface.
    """

    cost_surface = np.asarray(
        cost_surface,
        dtype=np.float64
    )

    if cost_surface.ndim != 2:
        raise ValueError(
            "Cost surface must be a 2D array."
        )

    valid = cost_surface[np.isfinite(cost_surface)]

    if valid.size == 0:
        raise ValueError(
            "No valid terrain cost measurements available."
        )

    percentiles = np.percentile(
        valid,
        [5, 25, 50, 75, 90, 95]
    )

    return {
        "valid_pixels": int(valid.size),
        "minimum": float(np.min(valid)),
        "maximum": float(np.max(valid)),
        "mean": float(np.mean(valid)),
        "median": float(np.median(valid)),
        "standard_deviation": float(np.std(valid)),
        "percentile_05": float(percentiles[0]),
        "percentile_25": float(percentiles[1]),
        "percentile_50": float(percentiles[2]),
        "percentile_75": float(percentiles[3]),
        "percentile_90": float(percentiles[4]),
        "percentile_95": float(percentiles[5])
    }


# --------------------------------------------------
# CLASS DISTRIBUTION
# --------------------------------------------------

def calculate_class_distribution(classes):
    """
    Calculate pixel count and percentage for each class.
    """

    classes = np.asarray(classes)

    if classes.ndim != 2:
        raise ValueError(
            "Terrain classes must be a 2D array."
        )

    valid = classes >= 0

    valid_count = int(np.sum(valid))

    if valid_count == 0:
        raise ValueError(
            "No valid terrain classes available."
        )

    distribution = {}

    for class_id in range(5):

        count = int(
            np.sum(classes == class_id)
        )

        distribution[class_id] = {
            "pixels": count,
            "percentage": float(
                count / valid_count * 100
            )
        }

    return distribution


# --------------------------------------------------
# WEIGHT SENSITIVITY ANALYSIS
# --------------------------------------------------

def calculate_weight_sensitivity(
    normalised_slope,
    normalised_roughness,
    valid_mask
):
    """
    Calculate alternative cost surfaces using different
    slope and roughness weights.

    These scenarios are sensitivity tests, not claims
    that any particular weights are scientifically optimal.
    """

    scenarios = {
        "slope_dominant": (0.80, 0.20),
        "balanced": (0.50, 0.50),
        "roughness_dominant": (0.30, 0.70)
    }

    results = {}

    for name, weights in scenarios.items():

        surface = calculate_cost_surface(
            normalised_slope,
            normalised_roughness,
            valid_mask,
            slope_weight=weights[0],
            roughness_weight=weights[1]
        )

        results[name] = {
            "slope_weight": weights[0],
            "roughness_weight": weights[1],
            "statistics": calculate_cost_statistics(surface)
        }

    return results


# --------------------------------------------------
# COMPLETE TERRAIN COST ANALYSIS
# --------------------------------------------------

def analyse_terrain_cost(
    elevation_results,
    roughness_results,
    slope_results,
    slope_weight=DEFAULT_SLOPE_WEIGHT,
    roughness_weight=DEFAULT_ROUGHNESS_WEIGHT,
    lower_percentile=DEFAULT_LOWER_PERCENTILE,
    upper_percentile=DEFAULT_UPPER_PERCENTILE
):
    """
    Run the complete terrain traversal cost analysis.

    Inputs:
        Outputs from analyse_elevation(),
        analyse_roughness() and analyse_slope().

    Returns:
        Normalised slope and roughness layers,
        local standard deviation diagnostic,
        combined cost surface, classes, statistics,
        validity mask, configuration and sensitivity results.
    """

    print("\nELYSIUM TERRAIN COST ANALYSIS")
    print("=" * 50)

    slope_weight, roughness_weight = validate_weights(
        slope_weight,
        roughness_weight
    )

    data = validate_cost_inputs(
        elevation_results,
        roughness_results,
        slope_results
    )

    elevation = data["elevation_m"]
    slope = data["slope_degrees"]
    tri = data["tri_m"]
    local_std = data["local_std_m"]

    # Common validity mask for the combined cost model.
    valid_mask = (
        np.isfinite(elevation)
        & np.isfinite(slope)
        & np.isfinite(tri)
    )

    if not np.any(valid_mask):
        raise ValueError(
            "No common valid terrain pixels available."
        )

    # Normalise slope.
    normalised_slope, slope_normalisation = (
        normalise_cost_layer(
            slope,
            valid_mask,
            lower_percentile,
            upper_percentile
        )
    )

    # Normalise TRI.
    normalised_roughness, roughness_normalisation = (
        normalise_cost_layer(
            tri,
            valid_mask,
            lower_percentile,
            upper_percentile
        )
    )

    # Keep local standard deviation as a diagnostic layer.
    std_mask = valid_mask & np.isfinite(local_std)

    normalised_local_std, std_normalisation = (
        normalise_cost_layer(
            local_std,
            std_mask,
            lower_percentile,
            upper_percentile
        )
    )

    # Calculate combined terrain cost.
    cost_surface = calculate_cost_surface(
        normalised_slope,
        normalised_roughness,
        valid_mask,
        slope_weight,
        roughness_weight
    )

    # Classify terrain cost.
    cost_classes = classify_terrain_cost(
        cost_surface
    )

    # Calculate statistics.
    statistics = calculate_cost_statistics(
        cost_surface
    )

    class_distribution = calculate_class_distribution(
        cost_classes
    )

    # Sensitivity analysis.
    sensitivity = calculate_weight_sensitivity(
        normalised_slope,
        normalised_roughness,
        valid_mask
    )

    print("\nModel configuration:")

    print(f"Slope weight: {slope_weight:.2f}")
    print(f"Roughness weight: {roughness_weight:.2f}")

    print(
        f"Normalisation percentiles: "
        f"{lower_percentile:.1f}–{upper_percentile:.1f}"
    )

    print("\nTerrain cost statistics:")

    print(f"Valid pixels: {statistics['valid_pixels']}")
    print(f"Minimum: {statistics['minimum']:.4f}")
    print(f"Maximum: {statistics['maximum']:.4f}")
    print(f"Mean: {statistics['mean']:.4f}")
    print(f"Median: {statistics['median']:.4f}")
    print(
        f"Standard deviation: "
        f"{statistics['standard_deviation']:.4f}"
    )

    print("\nTerrain cost class distribution:")

    for class_id, result in class_distribution.items():

        print(
            f"Class {class_id}: "
            f"{result['pixels']} pixels "
            f"({result['percentage']:.2f}%)"
        )

    print("\nWeight sensitivity analysis:")

    for name, result in sensitivity.items():

        print(
            f"{name}: "
            f"mean cost = "
            f"{result['statistics']['mean']:.4f}"
        )

    print("\nTerrain cost analysis completed successfully.")

    return {
        "terrain_cost": cost_surface,
        "terrain_classes": cost_classes,
        "normalised_slope": normalised_slope,
        "normalised_roughness": normalised_roughness,
        "normalised_local_std": normalised_local_std,
        "local_std_m": local_std.copy(),
        "elevation_m": elevation.copy(),
        "latitude": data["latitude"].copy(),
        "longitude": data["longitude"].copy(),
        "valid_mask": valid_mask,
        "statistics": statistics,
        "class_distribution": class_distribution,
        "grid_spacing_km": data["grid_spacing_km"],
        "configuration": {
            "slope_weight": slope_weight,
            "roughness_weight": roughness_weight,
            "lower_percentile": lower_percentile,
            "upper_percentile": upper_percentile,
            "classification_thresholds": (
                COST_CLASS_THRESHOLDS.copy()
            )
        },
        "normalisation": {
            "slope": slope_normalisation,
            "roughness": roughness_normalisation,
            "local_std": std_normalisation
        },
        "sensitivity_analysis": sensitivity
    }


# --------------------------------------------------
# MAIN
# --------------------------------------------------

if __name__ == "__main__":

    from processing.mola import process_mola
    from analysis.elevation import analyse_elevation
    from analysis.roughness import analyse_roughness
    from analysis.slope import analyse_slope

    print("\nLoading MOLA data...")

    region_data = process_mola()

    print("\nRunning elevation analysis...")

    elevation_results = analyse_elevation(
        region_data
    )

    print("\nRunning roughness analysis...")

    roughness_results = analyse_roughness(
        elevation_results
    )

    print("\nRunning slope analysis...")

    slope_results = analyse_slope(
        elevation_results
    )

    print("\nRunning terrain cost analysis...")

    results = analyse_terrain_cost(
        elevation_results,
        roughness_results,
        slope_results
    )
