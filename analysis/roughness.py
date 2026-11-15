
"""
ELYSIUM — Terrain Roughness Analysis

Calculates regional-scale terrain roughness from MOLA
elevation measurements for Western Elysium Planitia.

Metrics:
    1. Terrain Ruggedness Index (TRI)
    2. Local elevation standard deviation (3x3)

Elevation and roughness units: metres.

MOLA resolution limits interpretation to regional-scale
terrain variation, not individual surface hazards.
"""

import numpy as np


# --------------------------------------------------
# INPUT VALIDATION
# --------------------------------------------------

def validate_roughness_input(elevation_results):
    """
    Validate the output dictionary from analyse_elevation().
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
            "Elevation grid must be two-dimensional."
        )

    if min(elevation.shape) < 3:
        raise ValueError(
            "Elevation grid must have at least 3 rows and columns."
        )

    if not np.any(np.isfinite(elevation)):
        raise ValueError(
            "Elevation grid contains no valid measurements."
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
                f"Invalid pixel spacing: {key}."
            )

    return elevation


# --------------------------------------------------
# NEIGHBOURHOOD EXTRACTION
# --------------------------------------------------

def get_neighbourhoods(elevation):
    """
    Extract the eight neighbouring pixel arrays.

    NaN values are preserved.
    """

    rows, columns = elevation.shape

    padded = np.pad(
        elevation,
        pad_width=1,
        mode="constant",
        constant_values=np.nan
    )

    neighbours = []

    for row_offset in (-1, 0, 1):

        for column_offset in (-1, 0, 1):

            if row_offset == 0 and column_offset == 0:
                continue

            neighbour = padded[
                1 + row_offset:1 + row_offset + rows,
                1 + column_offset:1 + column_offset + columns
            ]

            neighbours.append(neighbour)

    return neighbours


# --------------------------------------------------
# TERRAIN RUGGEDNESS INDEX
# --------------------------------------------------

def calculate_tri(elevation):
    """
    Calculate Riley Terrain Ruggedness Index.

    TRI = sqrt(sum((neighbour - centre)^2))

    Uses all eight surrounding neighbours.

    Pixels without a complete valid neighbourhood
    are assigned NaN.
    """

    elevation = np.asarray(
        elevation,
        dtype=np.float64
    )

    if elevation.ndim != 2 or min(elevation.shape) < 3:
        raise ValueError(
            "TRI requires a 2D grid of at least 3x3."
        )

    neighbours = get_neighbourhoods(elevation)

    centre_valid = np.isfinite(elevation)

    squared_difference_sum = np.zeros(
        elevation.shape,
        dtype=np.float64
    )

    complete_neighbourhood = centre_valid.copy()

    for neighbour in neighbours:

        valid = np.isfinite(neighbour)

        complete_neighbourhood &= valid

        difference = np.zeros(
            elevation.shape,
            dtype=np.float64
        )

        mask = centre_valid & valid

        difference[mask] = (
            neighbour[mask] - elevation[mask]
        )

        squared_difference_sum += difference ** 2

    tri = np.full(
        elevation.shape,
        np.nan,
        dtype=np.float64
    )

    tri[complete_neighbourhood] = np.sqrt(
        squared_difference_sum[complete_neighbourhood]
    )

    return tri


# --------------------------------------------------
# LOCAL ELEVATION STANDARD DEVIATION
# --------------------------------------------------

def calculate_local_std(elevation):
    """
    Calculate local elevation standard deviation
    within a 3x3 neighbourhood.

    All nine pixels must contain valid measurements.

    Pixels without a complete valid neighbourhood
    are assigned NaN.
    """

    elevation = np.asarray(
        elevation,
        dtype=np.float64
    )

    if elevation.ndim != 2 or min(elevation.shape) < 3:
        raise ValueError(
            "Local standard deviation requires a 2D grid of at least 3x3."
        )

    rows, columns = elevation.shape

    padded = np.pad(
        elevation,
        pad_width=1,
        mode="constant",
        constant_values=np.nan
    )

    centre = elevation

    valid_neighbourhood = np.isfinite(centre)

    sum_difference = np.zeros(
        elevation.shape,
        dtype=np.float64
    )

    sum_squared_difference = np.zeros(
        elevation.shape,
        dtype=np.float64
    )

    for row_offset in (-1, 0, 1):

        for column_offset in (-1, 0, 1):

            neighbour = padded[
                1 + row_offset:1 + row_offset + rows,
                1 + column_offset:1 + column_offset + columns
            ]

            valid = np.isfinite(neighbour)

            valid_neighbourhood &= valid

            difference = np.zeros(
                elevation.shape,
                dtype=np.float64
            )

            mask = np.isfinite(centre) & valid

            difference[mask] = (
                neighbour[mask] - centre[mask]
            )

            sum_difference += difference
            sum_squared_difference += difference ** 2

    local_std = np.full(
        elevation.shape,
        np.nan,
        dtype=np.float64
    )

    mask = valid_neighbourhood

    mean_difference = sum_difference[mask] / 9

    variance = (
        sum_squared_difference[mask] / 9
        - mean_difference ** 2
    )

    # Protect against tiny floating-point rounding errors.
    variance = np.maximum(variance, 0.0)

    local_std[mask] = np.sqrt(variance)

    return local_std


# --------------------------------------------------
# ROUGHNESS STATISTICS
# --------------------------------------------------

def calculate_roughness_statistics(roughness):
    """
    Calculate descriptive statistics for a roughness array.
    """

    roughness = np.asarray(
        roughness,
        dtype=np.float64
    )

    if roughness.ndim != 2:
        raise ValueError(
            "Roughness data must be a 2D array."
        )

    valid = roughness[np.isfinite(roughness)]

    if valid.size == 0:
        raise ValueError(
            "Roughness array contains no valid measurements."
        )

    return {
        "valid_pixels": int(valid.size),
        "minimum_m": float(np.min(valid)),
        "maximum_m": float(np.max(valid)),
        "mean_m": float(np.mean(valid)),
        "median_m": float(np.median(valid)),
        "standard_deviation_m": float(np.std(valid))
    }


# --------------------------------------------------
# COMPLETE ROUGHNESS ANALYSIS
# --------------------------------------------------

def analyse_roughness(elevation_results):
    """
    Run the complete terrain roughness analysis.

    Input:
        Dictionary returned by analyse_elevation().

    Output:
        TRI array, local standard deviation array,
        and descriptive statistics.
    """

    elevation = validate_roughness_input(
        elevation_results
    )

    tri = calculate_tri(elevation)

    local_std = calculate_local_std(elevation)

    tri_statistics = calculate_roughness_statistics(
        tri
    )

    local_std_statistics = calculate_roughness_statistics(
        local_std
    )

    return {
        "tri_m": tri,
        "local_std_m": local_std,
        "tri_statistics": tri_statistics,
        "local_std_statistics": local_std_statistics,
        "grid_spacing_km": dict(
            elevation_results["grid_spacing_km"]
        )
    }


# --------------------------------------------------
# MAIN
# --------------------------------------------------

if __name__ == "__main__":

    from processing.mola import process_mola
    from analysis.elevation import analyse_elevation

    print("\nELYSIUM TERRAIN ROUGHNESS ANALYSIS")
    print("=" * 45)

    region_data = process_mola()

    elevation_results = analyse_elevation(
        region_data
    )

    results = analyse_roughness(
        elevation_results
    )

    print("\nTerrain Ruggedness Index (TRI):")

    for key, value in results["tri_statistics"].items():
        print(f"{key}: {value}")

    print("\nLocal Elevation Standard Deviation:")

    for key, value in results["local_std_statistics"].items():
        print(f"{key}: {value}")

    print("\nOutput array shape:", results["tri_m"].shape)

    print("\nTerrain roughness analysis completed successfully.")
