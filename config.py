
"""
ELYSIUM — Central Project Configuration

Purpose:
    Defines project paths, study areas, Mars reference
    parameters and shared data-selection settings.

All project modules should import shared configuration
from this file instead of duplicating constants.
"""

from pathlib import Path


# ==================================================
# 1. PROJECT IDENTITY
# ==================================================

PROJECT_NAME = "ELYSIUM"
PROJECT_VERSION = "0.1.0"
PLANET_NAME = "Mars"


# ==================================================
# 2. PROJECT DIRECTORIES
# ==================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"

RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CATALOGUE_DIR = DATA_DIR / "catalogue"

OUTPUT_DIR = BASE_DIR / "outputs"


# Create required directories automatically.
REQUIRED_DIRECTORIES = (
    DATA_DIR,
    RAW_DIR,
    PROCESSED_DIR,
    CATALOGUE_DIR,
    OUTPUT_DIR,
)

for directory in REQUIRED_DIRECTORIES:
    directory.mkdir(parents=True, exist_ok=True)


# ==================================================
# 3. STUDY AREAS
# ==================================================

# Detailed scientific analysis region.
DETAILED_AOI = {
    "name": "Western Elysium Planitia",
    "west": 134.0,
    "east": 138.0,
    "south": 3.5,
    "north": 5.5,
}

# Broader regional context.
REGIONAL_AOI = {
    "name": "Regional Elysium Planitia",
    "west": 130.0,
    "east": 180.0,
    "south": -5.0,
    "north": 10.0,
}


# ==================================================
# 4. REFERENCE LOCATION
# ==================================================

# InSight landing-site reference coordinates.
# These are approximate coordinates used by this project.
INSIGHT_SITE = {
    "latitude": 4.502,
    "longitude": 135.623,
}


# ==================================================
# 5. DATA SELECTION PARAMETERS
# ==================================================

MIN_COVERAGE_PERCENT = 50.0

MAX_RESOLUTION_METRES = {
    "MOLA": 1000.0,
    "CTX": 12.0,
    "HIRISE": 1.0,
    "THEMIS_VIS": 100.0,
    "THEMIS_IR": 100.0,
}

MAX_RESULTS_PER_SEARCH = 500


# ==================================================
# 6. MARS REFERENCE PARAMETERS
# ==================================================

# Mars equatorial radius.
MARS_EQUATORIAL_RADIUS_KM = 3396.19

# MOLA MEGDR dataset resolution.
# 64 samples per degree, not 64 samples per kilometre.
MOLA_PIXELS_PER_DEGREE = 64


# ==================================================
# 7. CONFIGURATION VALIDATION
# ==================================================

def validate_aoi(area):
    """Validate an area-of-interest dictionary."""

    required_keys = {
        "name",
        "west",
        "east",
        "south",
        "north",
    }

    missing = required_keys - set(area)

    if missing:
        raise ValueError(
            f"AOI is missing required fields: {sorted(missing)}"
        )

    west = float(area["west"])
    east = float(area["east"])
    south = float(area["south"])
    north = float(area["north"])

    if not (0.0 <= west < east <= 360.0):
        raise ValueError(
            f"Invalid longitude boundaries for {area['name']}."
        )

    if not (-90.0 <= south < north <= 90.0):
        raise ValueError(
            f"Invalid latitude boundaries for {area['name']}."
        )

    return True


def validate_configuration():
    """Validate the project's shared configuration."""

    validate_aoi(DETAILED_AOI)
    validate_aoi(REGIONAL_AOI)

    if not (0.0 <= MIN_COVERAGE_PERCENT <= 100.0):
        raise ValueError("Coverage threshold must be between 0 and 100.")

    if MAX_RESULTS_PER_SEARCH < 1:
        raise ValueError("Maximum search results must be positive.")

    if MARS_EQUATORIAL_RADIUS_KM <= 0:
        raise ValueError("Mars radius must be positive.")

    if MOLA_PIXELS_PER_DEGREE <= 0:
        raise ValueError("MOLA pixel density must be positive.")

    if not (
        -90.0 <= INSIGHT_SITE["latitude"] <= 90.0
    ):
        raise ValueError("Invalid InSight latitude.")

    if not (
        0.0 <= INSIGHT_SITE["longitude"] <= 360.0
    ):
        raise ValueError("Invalid InSight longitude.")

    for dataset, resolution in MAX_RESOLUTION_METRES.items():
        if resolution <= 0:
            raise ValueError(
                f"Invalid resolution threshold for {dataset}."
            )

    return True


# ==================================================
# 8. CONFIGURATION TEST
# ==================================================

if __name__ == "__main__":

    print("=" * 50)
    print("ELYSIUM CONFIGURATION VALIDATION")
    print("=" * 50)

    validate_configuration()

    print(f"Project: {PROJECT_NAME}")
    print(f"Version: {PROJECT_VERSION}")
    print(f"Planet: {PLANET_NAME}")

    print("\nDetailed study area:")
    print(DETAILED_AOI)

    print("\nRegional study area:")
    print(REGIONAL_AOI)

    print("\nInSight reference:")
    print(INSIGHT_SITE)

    print("\nProject directories:")

    for directory in REQUIRED_DIRECTORIES:
        print(f"{directory.name}: {directory}")

    print("\nMars equatorial radius:")
    print(f"{MARS_EQUATORIAL_RADIUS_KM} km")

    print("\nMOLA resolution:")
    print(f"{MOLA_PIXELS_PER_DEGREE} pixels/degree")

    print("\nConfiguration validation passed.")
