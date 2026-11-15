
"""
ELYSIUM — Data Validation

Checks downloaded Mars observation files,
CTX raster metadata, and geographic footprints.

CTX raw EDR images may be non-georeferenced.
Their pixel coordinates must never be interpreted
as Martian geographic coordinates.
"""

from pathlib import Path

import numpy as np
import rasterio
import geopandas as gpd

from rasterio.windows import Window


# --------------------------------------------------
# FILE EXISTENCE
# --------------------------------------------------

def check_file_exists(file_path: Path) -> bool:
    """Check whether a file exists and is not empty."""

    file_path = Path(file_path)

    if not file_path.exists():
        print(f"Missing file: {file_path}")
        return False

    if not file_path.is_file():
        print(f"Not a file: {file_path}")
        return False

    if file_path.stat().st_size == 0:
        print(f"Empty file: {file_path}")
        return False

    return True


# --------------------------------------------------
# CTX IMAGE VALIDATION
# --------------------------------------------------

def inspect_ctx_file(file_path: Path) -> dict:
    """
    Inspect a CTX raster image.

    Handles:
    - Raster dimensions and band metadata
    - NoData values and valid pixel counts
    - CRS and affine georeferencing
    - GCP and RPC georeferencing

    Raw CTX EDR images may have no georeferencing.
    Their bounds are then pixel coordinates only.
    """

    file_path = Path(file_path)

    if not check_file_exists(file_path):
        raise FileNotFoundError(file_path)

    try:
        with rasterio.open(file_path) as dataset:

            if dataset.width <= 0 or dataset.height <= 0:
                raise ValueError("Invalid raster dimensions.")

            if dataset.count < 1:
                raise ValueError("Raster contains no image bands.")

            sample_width = min(64, dataset.width)
            sample_height = min(64, dataset.height)

            window = Window(
                0,
                0,
                sample_width,
                sample_height
            )

            sample = dataset.read(1, window=window)

            if sample.size == 0:
                raise ValueError("Unable to read image pixels.")

            # Exclude NaN and infinite values.
            valid_sample = np.isfinite(sample)

            # Exclude the declared NoData value.
            nodata = dataset.nodata

            if nodata is not None:
                if np.isnan(nodata):
                    valid_sample &= ~np.isnan(sample)
                else:
                    valid_sample &= sample != nodata

            # Respect Rasterio's internal mask as well.
            sample_mask = dataset.read_masks(1, window=window)
            valid_sample &= sample_mask > 0

            valid_values = sample[valid_sample]

            gcps, gcp_crs = dataset.gcps
            has_gcps = len(gcps) > 0
            has_rpcs = dataset.rpcs is not None

            has_affine_georeferencing = (
                dataset.crs is not None
                and not dataset.transform.is_identity
            )

            is_georeferenced = (
                has_affine_georeferencing
                or has_gcps
                or has_rpcs
            )

            if is_georeferenced:
                coordinate_interpretation = "Georeferenced coordinates"
            else:
                coordinate_interpretation = (
                    "Image pixel coordinates only"
                )

            metadata = {
                "filename": file_path.name,
                "file_size_bytes": file_path.stat().st_size,
                "width": dataset.width,
                "height": dataset.height,
                "bands": dataset.count,
                "data_type": list(dataset.dtypes),
                "nodata": nodata,
                "crs": (
                    str(dataset.crs)
                    if dataset.crs is not None
                    else None
                ),
                "bounds": tuple(dataset.bounds),
                "transform": tuple(dataset.transform),
                "has_affine_georeferencing": (
                    has_affine_georeferencing
                ),
                "has_gcps": has_gcps,
                "gcp_crs": str(gcp_crs) if gcp_crs else None,
                "has_rpcs": has_rpcs,
                "is_georeferenced": is_georeferenced,
                "coordinate_interpretation": (
                    coordinate_interpretation
                ),
                "sample_pixels_checked": int(sample.size),
                "valid_sample_pixels": int(valid_values.size),
                "invalid_sample_pixels": int(
                    sample.size - valid_values.size
                ),
                "sample_minimum": (
                    float(valid_values.min())
                    if valid_values.size
                    else None
                ),
                "sample_maximum": (
                    float(valid_values.max())
                    if valid_values.size
                    else None
                ),
                "validation_status": "Passed"
            }

    except Exception as error:
        raise RuntimeError(
            f"CTX image validation failed for "
            f"{file_path.name}: {error}"
        ) from error

    return metadata


# --------------------------------------------------
# FOOTPRINT ATTRIBUTE VALIDATION
# --------------------------------------------------

def find_attribute(footprint, attribute_name):
    """
    Find a numeric attribute without depending
    on field-name case.

    Returns the first usable numeric value.
    """

    if footprint.empty:
        return None

    field_lookup = {
        str(column).lower(): column
        for column in footprint.columns
    }

    actual_name = field_lookup.get(attribute_name.lower())

    if actual_name is None:
        return None

    for value in footprint[actual_name]:

        if value is None:
            continue

        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            continue

        if np.isfinite(numeric_value):
            return numeric_value

    return None


# --------------------------------------------------
# FOOTPRINT VALIDATION
# --------------------------------------------------

def inspect_footprint(shapefile_path: Path) -> dict:
    """
    Extract geographic metadata from a Mars footprint.

    Requires a Mars geographic CRS.
    Does not automatically repair invalid geometries.

    Bounds and centroid coordinates use the geometry's
    stored x/y order (longitude/latitude for this dataset).
    """

    shapefile_path = Path(shapefile_path)

    if not check_file_exists(shapefile_path):
        raise FileNotFoundError(shapefile_path)

    footprint = gpd.read_file(shapefile_path)

    if footprint.empty:
        raise ValueError(
            "Footprint contains no geographic features."
        )

    if footprint.crs is None:
        raise ValueError(
            "Footprint has no CRS; geographic coordinates "
            "cannot be verified."
        )

    crs = footprint.crs

    if not crs.is_geographic:
        raise ValueError(
            "Footprint CRS is not geographic. "
            "Cannot interpret its bounds as longitude/latitude."
        )

    # Verify that the geographic CRS uses the Mars sphere.
    mars_radius = crs.ellipsoid.semi_major_metre

    if not np.isclose(mars_radius, 3396190.0, atol=1.0):
        raise ValueError(
            f"Unexpected reference radius: {mars_radius}. "
            "Expected the Mars 2000 spherical radius "
            "of approximately 3,396,190 metres."
        )

    geometries = footprint.geometry

    empty_mask = geometries.is_empty | geometries.isna()
    usable_geometries = geometries[~empty_mask]

    if usable_geometries.empty:
        raise ValueError(
            "Footprint contains no usable geometries."
        )

    invalid_geometry_count = int(
        (~usable_geometries.is_valid).sum()
    )

    bounds = usable_geometries.total_bounds

    if not np.all(np.isfinite(bounds)):
        raise ValueError(
            "Footprint contains non-finite geographic bounds."
        )

    west, south, east, north = map(float, bounds)

    # Mars geographic coordinate range checks.
    if not (-360 <= west <= 360 and -360 <= east <= 360):
        raise ValueError(
            "Footprint longitude values are out of range."
        )

    if not (-90 <= south <= 90 and -90 <= north <= 90):
        raise ValueError(
            "Footprint latitude values are out of range."
        )

    if west > east:
        raise ValueError(
            "Footprint west longitude exceeds east longitude."
        )

    # Use supplied centre attributes when available.
    center_latitude = find_attribute(
        footprint, "CenterLat"
    )

    center_longitude = find_attribute(
        footprint, "CenterLon"
    )

    # Otherwise calculate an approximate planar centroid.
    if center_latitude is None or center_longitude is None:

        combined_geometry = usable_geometries.union_all()
        centroid = combined_geometry.centroid

        if center_longitude is None:
            center_longitude = float(centroid.x)

        if center_latitude is None:
            center_latitude = float(centroid.y)

    metadata = {
        "filename": shapefile_path.name,
        "crs": str(footprint.crs),
        "feature_count": len(footprint),
        "empty_geometry_count": int(empty_mask.sum()),
        "invalid_geometry_count": invalid_geometry_count,
        "west_longitude": west,
        "south_latitude": south,
        "east_longitude": east,
        "north_latitude": north,
        "center_latitude": center_latitude,
        "center_longitude": center_longitude,
        "validation_status": (
            "Passed"
            if invalid_geometry_count == 0
            and empty_mask.sum() == 0
            else "Warning: geometry issues found"
        )
    }

    return metadata


# --------------------------------------------------
# BASIC VALIDATION TEST
# --------------------------------------------------

if __name__ == "__main__":

    print("ELYSIUM DATA VALIDATION")
    print("=" * 40)

    print("\nValidation module loaded successfully.")

    print("Available checks:")
    print("1. File existence and size")
    print("2. CTX raster readability and metadata")
    print("3. Mars geographic CRS and footprint geometry")
