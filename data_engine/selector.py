
"""
ELYSIUM — Observation Selector
Version: 0.2.0

Filters Mars orbital observations using available metadata.

Important:
    Observation discovery is not the same as scientific
    suitability assessment.

    Coverage, resolution and data quality are only used
    when verified measurements are supplied.
"""

from copy import deepcopy
from dataclasses import dataclass
from math import isfinite
from typing import Optional


# ==================================================
# SELECTION CRITERIA
# ==================================================

@dataclass(frozen=True)
class SelectionCriteria:
    """Rules used to filter orbital observations."""

    preferred_instruments: tuple = (
        "CTX",
        "HIRISE",
        "THEMIS",
    )

    require_data_product: bool = True

    # Optional because discovery metadata may not contain
    # measured coverage or resolution.
    min_coverage_percent: Optional[float] = None

    max_resolution_metres: Optional[float] = None

    def __post_init__(self):

        if not self.preferred_instruments:
            raise ValueError(
                "At least one preferred instrument is required."
            )

        if self.min_coverage_percent is not None:

            if not _valid_number(self.min_coverage_percent):
                raise ValueError(
                    "Minimum coverage must be a finite number."
                )

            if not 0 <= self.min_coverage_percent <= 100:
                raise ValueError(
                    "Minimum coverage must be between 0 and 100."
                )

        if self.max_resolution_metres is not None:

            if not _valid_number(self.max_resolution_metres):
                raise ValueError(
                    "Maximum resolution must be a finite number."
                )

            if self.max_resolution_metres <= 0:
                raise ValueError(
                    "Maximum resolution must be greater than zero."
                )


# ==================================================
# NUMERIC VALIDATION
# ==================================================

def _valid_number(value) -> bool:
    """Return True only for finite numeric values."""

    if isinstance(value, bool):
        return False

    try:
        return isfinite(float(value))

    except (TypeError, ValueError, OverflowError):
        return False


def _get_numeric_value(product: dict, key: str):
    """Read a numeric measurement without guessing its value."""

    value = product.get(key)

    if not _valid_number(value):
        return None

    return float(value)


# ==================================================
# INDIVIDUAL SCORING FUNCTIONS
# ==================================================

def calculate_resolution_score(
    resolution_metres: float,
    maximum_resolution: float
) -> float:
    """
    Score spatial resolution.

    Smaller pixel size represents finer spatial resolution.

    A resolution equal to or better than the maximum
    threshold receives 100 points.
    """

    if (
        not _valid_number(resolution_metres)
        or not _valid_number(maximum_resolution)
    ):
        raise ValueError(
            "Resolution values must be finite numbers."
        )

    resolution_metres = float(resolution_metres)
    maximum_resolution = float(maximum_resolution)

    if resolution_metres <= 0 or maximum_resolution <= 0:
        raise ValueError(
            "Resolution values must be greater than zero."
        )

    score = 100 * maximum_resolution / resolution_metres

    return round(min(100.0, score), 2)


def calculate_coverage_score(
    coverage_percent: float
) -> float:
    """Convert measured coverage percentage into a score."""

    if not _valid_number(coverage_percent):
        raise ValueError(
            "Coverage must be a finite number."
        )

    coverage_percent = float(coverage_percent)

    if not 0 <= coverage_percent <= 100:
        raise ValueError(
            "Coverage must be between 0 and 100."
        )

    return round(coverage_percent, 2)


def calculate_quality_score(
    valid_data_percent: float
) -> float:
    """Convert measured valid-data percentage into a score."""

    if not _valid_number(valid_data_percent):
        raise ValueError(
            "Valid-data percentage must be a finite number."
        )

    valid_data_percent = float(valid_data_percent)

    if not 0 <= valid_data_percent <= 100:
        raise ValueError(
            "Valid-data percentage must be between 0 and 100."
        )

    return round(valid_data_percent, 2)


# ==================================================
# COMPOSITE OBSERVATION SCORE
# ==================================================

def calculate_observation_score(
    resolution_score: float,
    coverage_score: float,
    quality_score: float
) -> float:
    """
    Calculate a weighted observation score.

    Weights:
        Resolution: 40%
        Coverage:   35%
        Quality:    25%

    This is a project-defined ranking metric,
    not an official NASA suitability rating.
    """

    scores = (
        resolution_score,
        coverage_score,
        quality_score,
    )

    if not all(_valid_number(value) for value in scores):
        raise ValueError(
            "All component scores must be finite numbers."
        )

    if not all(0 <= float(value) <= 100 for value in scores):
        raise ValueError(
            "All component scores must be between 0 and 100."
        )

    result = (
        float(resolution_score) * 0.40
        + float(coverage_score) * 0.35
        + float(quality_score) * 0.25
    )

    return round(result, 2)


# ==================================================
# PRODUCT FILE VALIDATION
# ==================================================

def _has_data_product(product: dict) -> bool:
    """Check whether an original product data file is available."""

    files_data = product.get("Product_files", [])

    if isinstance(files_data, dict):
        files = files_data.get("Product_file", [])

    elif isinstance(files_data, list):
        files = files_data

    else:
        return False

    if isinstance(files, dict):
        files = [files]

    if not isinstance(files, list):
        return False

    for file_info in files:

        if not isinstance(file_info, dict):
            continue

        file_type = str(
            file_info.get("Type", "")
        ).strip().lower()

        url = file_info.get("URL")

        if file_type == "product" and isinstance(url, str) and url.strip():
            return True

    return False


# ==================================================
# OBSERVATION ELIGIBILITY
# ==================================================

def is_eligible_observation(
    product: dict,
    criteria: SelectionCriteria
) -> bool:
    """Determine whether an observation meets selection criteria."""

    if not isinstance(product, dict):
        return False

    instrument = str(
        product.get("iid", "")
    ).strip().upper()

    product_type = str(
        product.get("pt", "")
    ).strip().upper()

    preferred = {
        str(item).strip().upper()
        for item in criteria.preferred_instruments
    }

    # Instrument filter.
    if instrument not in preferred:
        return False

    # Accept standard CTX/HiRISE/THEMIS EDR or RDR products.
    if product_type not in {"EDR", "RDR"}:
        return False

    # A usable product identifier is required.
    product_id = product.get("pdsid")

    if not isinstance(product_id, str) or not product_id.strip():
        return False

    # Original data product availability.
    if criteria.require_data_product:
        if not _has_data_product(product):
            return False

    # Apply coverage threshold only when explicitly requested.
    if criteria.min_coverage_percent is not None:

        coverage = _get_numeric_value(
            product,
            "coverage_percent"
        )

        if coverage is None:
            return False

        if coverage < criteria.min_coverage_percent:
            return False

    # Apply resolution threshold only when explicitly requested.
    if criteria.max_resolution_metres is not None:

        resolution = _get_numeric_value(
            product,
            "resolution_metres"
        )

        if resolution is None:
            return False

        if resolution > criteria.max_resolution_metres:
            return False

    return True


# ==================================================
# OBSERVATION SELECTION
# ==================================================

def select_observations(
    products: list,
    criteria: Optional[SelectionCriteria] = None
) -> list:
    """
    Filter and organise orbital observations.

    Original catalogue records are not modified.

    Scientific suitability remains pending until
    footprint, resolution and image-quality analysis.
    """

    if criteria is None:
        criteria = SelectionCriteria()

    if not isinstance(criteria, SelectionCriteria):
        raise TypeError(
            "criteria must be a SelectionCriteria instance."
        )

    if not isinstance(products, list):
        raise TypeError(
            "products must be a list of observation dictionaries."
        )

    selected = []
    seen_ids = set()

    for product in products:

        if not is_eligible_observation(product, criteria):
            continue

        product_id = str(
            product["pdsid"]
        ).strip()

        # Prevent duplicate observations.
        duplicate_key = product_id.upper()

        if duplicate_key in seen_ids:
            continue

        seen_ids.add(duplicate_key)

        # Work on a copy instead of modifying source data.
        observation = deepcopy(product)

        # Never invent scientific measurements.
        observation["scientific_score"] = None

        observation["assessment_status"] = (
            "Pending footprint and quality analysis"
        )

        selected.append(observation)

    # Sort by instrument preference, then product ID.
    priority = {
        str(instrument).strip().upper(): index
        for index, instrument in enumerate(
            criteria.preferred_instruments
        )
    }

    selected.sort(
        key=lambda item: (
            priority.get(
                str(item.get("iid", "")).strip().upper(),
                999
            ),
            str(item.get("pdsid", "")).upper()
        )
    )

    return selected


# ==================================================
# SUMMARY
# ==================================================

def get_selection_summary(
    original_products: list,
    selected_products: list
) -> dict:
    """Generate a simple observation-selection summary."""

    return {
        "total_received": len(original_products),
        "total_selected": len(selected_products),
        "total_rejected": (
            len(original_products) - len(selected_products)
        ),
        "instruments": sorted({
            str(item.get("iid", "")).upper()
            for item in selected_products
        }),
        "product_types": sorted({
            str(item.get("pt", "")).upper()
            for item in selected_products
        }),
    }


# ==================================================
# TEST
# ==================================================

if __name__ == "__main__":

    from data_engine.catalogue import load_observations

    observations = load_observations()

    criteria = SelectionCriteria(
        preferred_instruments=("CTX",),
        require_data_product=True
    )

    selected = select_observations(
        observations,
        criteria
    )

    print("\nELYSIUM OBSERVATION SELECTOR TEST")
    print("=" * 45)

    print(
        get_selection_summary(
            observations,
            selected
        )
    )

    print(
        "\nResolution score test:",
        calculate_resolution_score(6, 12)
    )

    print(
        "Coverage score test:",
        calculate_coverage_score(85)
    )

    print(
        "Quality score test:",
        calculate_quality_score(95)
    )

    print(
        "Composite score test:",
        calculate_observation_score(100, 85, 95)
    )

    print("\nSelector test completed.")
