
"""
ELYSIUM — MOLA MEGDR Data Processing
=====================================

Downloads and processes the MOLA MEGDR topography product
for the detailed Western Elysium Planitia study area.

Source:
NASA Planetary Data System (PDS)
Dataset: MGS-M-MOLA-5-MEGDR-L3-V1.0
Product: MEGT90N000GB.IMG

Product characteristics (verified from PDS3 label):
- Projection: Simple cylindrical
- Coordinate system: Planetocentric
- Longitude direction: East
- Coverage: 0–90 degrees North, 0–180 degrees East
- Resolution: 64 pixels per degree
- Dimensions: 5760 rows x 11520 columns
- Sample type: MSB_INTEGER
- Sample bits: 16
- Elevation unit: metres
- Record bytes: 23040
- Line projection offset: 5760.5
- Sample projection offset: 11520.5

Important:
MOLA elevation represents planetary radius minus the areoid
radius. It is not elevation above a local surface.
"""

from pathlib import Path

import numpy as np
import requests

from config import RAW_DIR, DETAILED_AOI


# ============================================================
# PRODUCT CONFIGURATION
# ============================================================

MOLA_URL = (
    "https://pds-geosciences.wustl.edu/"
    "mgs/mgs-m-mola-5-megdr-l3-v1/"
    "mgsl_300x/meg064/megt90n000gb.img"
)

MOLA_LABEL_URL = MOLA_URL.replace(".img", ".lbl")

MOLA_DIR = Path(RAW_DIR) / "mola"

MOLA_IMAGE_PATH = MOLA_DIR / "megt90n000gb.img"
MOLA_LABEL_PATH = MOLA_DIR / "megt90n000gb.lbl"

MOLA_ROWS = 5760
MOLA_COLUMNS = 11520

MOLA_PIXELS_PER_DEGREE = 64

MOLA_SAMPLE_BYTES = 2
MOLA_EXPECTED_SIZE = (
    MOLA_ROWS * MOLA_COLUMNS * MOLA_SAMPLE_BYTES
)

MOLA_DTYPE = ">i2"

# PDS map projection metadata
LINE_PROJECTION_OFFSET = 5760.5
SAMPLE_PROJECTION_OFFSET = 11520.5

CENTER_LONGITUDE = 180.0

# No missing-value constant is specified in the supplied label.
# Do not automatically convert -32768 or other values to NaN.


# ============================================================
# DIRECTORY SETUP
# ============================================================

def create_mola_directory():
    """Create the MOLA data directory if it does not exist."""

    MOLA_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# DOWNLOAD UTILITIES
# ============================================================

def download_file(url, destination, expected_size=None):
    """
    Download a file using streaming requests.

    The download is first written to a temporary .part file.
    The final file is created only after validation succeeds.
    """

    destination = Path(destination)
    temporary_path = destination.with_suffix(
        destination.suffix + ".part"
    )

    destination.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    if destination.exists():
        if (
            expected_size is None
            or destination.stat().st_size == expected_size
        ):
            print(f"Already downloaded: {destination.name}")
            return destination

        print("Existing file has an unexpected size. Re-downloading.")

    try:
        with requests.get(
            url,
            stream=True,
            timeout=(30, 120)
        ) as response:

            response.raise_for_status()

            content_type = response.headers.get(
                "Content-Type", ""
            ).lower()

            if "text/html" in content_type:
                raise ValueError(
                    "The server returned HTML instead of the data file."
                )

            downloaded_size = 0

            with open(temporary_path, "wb") as file:

                for chunk in response.iter_content(
                    chunk_size=1024 * 1024
                ):

                    if not chunk:
                        continue

                    file.write(chunk)
                    downloaded_size += len(chunk)

        if downloaded_size == 0:
            raise ValueError("Downloaded file is empty.")

        if (
            expected_size is not None
            and downloaded_size != expected_size
        ):
            raise ValueError(
                f"Unexpected file size: {downloaded_size} bytes. "
                f"Expected: {expected_size} bytes."
            )

        # Atomic replacement after successful validation
        temporary_path.replace(destination)

        print(
            f"Downloaded successfully: {destination.name} "
            f"({downloaded_size:,} bytes)"
        )

        return destination

    except Exception:
        if temporary_path.exists():
            temporary_path.unlink()

        raise


# ============================================================
# MOLA IMAGE DOWNLOAD
# ============================================================

def download_mola_image():
    """Download and validate the MOLA MEGDR image."""

    create_mola_directory()

    return download_file(
        url=MOLA_URL,
        destination=MOLA_IMAGE_PATH,
        expected_size=MOLA_EXPECTED_SIZE
    )


# ============================================================
# MOLA LABEL DOWNLOAD
# ============================================================

def download_mola_label():
    """Download the PDS3 label associated with the MOLA image."""

    create_mola_directory()

    return download_file(
        url=MOLA_LABEL_URL,
        destination=MOLA_LABEL_PATH
    )


# ============================================================
# BINARY GRID LOADING
# ============================================================

def load_mola_grid(image_path=MOLA_IMAGE_PATH):
    """
    Load the MOLA image as a two-dimensional elevation grid.

    The PDS label specifies:
    SAMPLE_TYPE = MSB_INTEGER
    SAMPLE_BITS = 16

    Therefore, the correct NumPy dtype is big-endian
    signed 16-bit integer: >i2.
    """

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"MOLA image not found: {image_path}"
        )

    actual_size = image_path.stat().st_size

    if actual_size != MOLA_EXPECTED_SIZE:
        raise ValueError(
            f"Invalid MOLA image size: {actual_size:,} bytes. "
            f"Expected: {MOLA_EXPECTED_SIZE:,} bytes."
        )

    grid = np.fromfile(
        image_path,
        dtype=MOLA_DTYPE
    )

    expected_samples = MOLA_ROWS * MOLA_COLUMNS

    if grid.size != expected_samples:
        raise ValueError(
            f"Unexpected number of samples: {grid.size:,}. "
            f"Expected: {expected_samples:,}."
        )

    grid = grid.reshape(
        (MOLA_ROWS, MOLA_COLUMNS)
    )

    print("MOLA grid loaded successfully.")
    print(f"Grid shape: {grid.shape}")
    print(f"Data type: {grid.dtype}")

    return grid


# ============================================================
# AOI EXTRACTION
# ============================================================

def load_mola_region(grid):
    """
    Extract the detailed AOI from the MOLA grid.

    Uses the PDS map projection offsets to account for
    the half-pixel registration.

    Returns:
        dict:
            elevation  - 2D elevation array in metres
            latitudes  - latitude of each pixel centre
            longitudes - longitude of each pixel centre
            bounds     - requested AOI boundaries
    """

    if grid.shape != (MOLA_ROWS, MOLA_COLUMNS):
        raise ValueError(
            f"Unexpected grid shape: {grid.shape}"
        )

    west = float(DETAILED_AOI["west"])
    east = float(DETAILED_AOI["east"])
    south = float(DETAILED_AOI["south"])
    north = float(DETAILED_AOI["north"])

    if not (
        0 <= west < east <= 180
        and 0 <= south < north <= 90
    ):
        raise ValueError(
            "The requested AOI is outside the coverage "
            "of this MOLA tile."
        )

    resolution = MOLA_PIXELS_PER_DEGREE

    # Convert geographic boundaries into zero-based
    # array indices using the PDS projection offsets.

    row_start = int(np.ceil(
        LINE_PROJECTION_OFFSET - north * resolution
    )) - 1

    row_end = int(np.ceil(
        LINE_PROJECTION_OFFSET - south * resolution
    )) - 1

    col_start = int(np.ceil(
        SAMPLE_PROJECTION_OFFSET
        + (west - CENTER_LONGITUDE) * resolution
    )) - 1

    col_end = int(np.ceil(
        SAMPLE_PROJECTION_OFFSET
        + (east - CENTER_LONGITUDE) * resolution
    )) - 1

    # Check array boundaries
    if not (
        0 <= row_start < row_end <= MOLA_ROWS
        and 0 <= col_start < col_end <= MOLA_COLUMNS
    ):
        raise ValueError(
            "Calculated AOI indices fall outside the MOLA grid."
        )

    region = grid[
        row_start:row_end,
        col_start:col_end
    ].astype(np.float32)

    # Expected dimensions for 4 x 2 degrees at 64 px/degree
    expected_rows = int((north - south) * resolution)
    expected_columns = int((east - west) * resolution)

    if region.shape != (
        expected_rows,
        expected_columns
    ):
        raise ValueError(
            f"Unexpected AOI shape: {region.shape}. "
            f"Expected: {(expected_rows, expected_columns)}."
        )

    # Calculate pixel-centre coordinates from PDS offsets.
    latitudes = (
        LINE_PROJECTION_OFFSET
        - (np.arange(row_start, row_end) + 1)
    ) / resolution

    longitudes = (
        (
            np.arange(col_start, col_end) + 1
            - SAMPLE_PROJECTION_OFFSET
        ) / resolution
    ) + CENTER_LONGITUDE

    # The supplied PDS label does not specify a missing-value
    # constant. Preserve the original samples without masking.

    if not np.all(np.isfinite(region)):
        raise ValueError(
            "The extracted elevation grid contains "
            "non-finite values."
        )

    print("\nMOLA AOI extraction successful.")
    print(
        f"AOI: {west}°E–{east}°E, "
        f"{south}°N–{north}°N"
    )
    print(f"Region shape: {region.shape}")
    print(
        f"Latitude centres: "
        f"{latitudes.min():.6f}°N to "
        f"{latitudes.max():.6f}°N"
    )
    print(
        f"Longitude centres: "
        f"{longitudes.min():.6f}°E to "
        f"{longitudes.max():.6f}°E"
    )
    print(
        f"Minimum elevation: {region.min():.2f} m"
    )
    print(
        f"Maximum elevation: {region.max():.2f} m"
    )
    print(
        f"Mean elevation: {region.mean():.2f} m"
    )

    return {
        "elevation": region,
        "latitudes": latitudes,
        "longitudes": longitudes,
        "bounds": {
            "west": west,
            "east": east,
            "south": south,
            "north": north,
        },
    }


# ============================================================
# COMPLETE MOLA PROCESSING PIPELINE
# ============================================================

def process_mola():
    """
    Execute the complete MOLA workflow:

    1. Download the MOLA image.
    2. Download the PDS label.
    3. Load the binary elevation grid.
    4. Extract the detailed AOI.
    5. Return the processed region.
    """

    print("\n" + "=" * 55)
    print("ELYSIUM — MOLA PROCESSING")
    print("=" * 55)

    download_mola_image()
    download_mola_label()

    grid = load_mola_grid()

    region_data = load_mola_region(grid)

    print("\nMOLA processing completed successfully.")

    return region_data


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    mola_data = process_mola()

    elevation = mola_data["elevation"]

    print("\nFinal output:")
    print(f"Elevation array: {elevation.shape}")
    print(f"Data type: {elevation.dtype}")
