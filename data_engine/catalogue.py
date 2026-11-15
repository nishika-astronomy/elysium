
"""
ELYSIUM — Observation Catalogue

Stores, loads and manages discovered Mars observations.

Catalogue format:
    JSON list of observation dictionaries.
"""

import json
import os

from pathlib import Path
from typing import Any

from config import CATALOGUE_DIR


# ==================================================
# 1. CATALOGUE CONFIGURATION
# ==================================================

CATALOGUE_FILE = CATALOGUE_DIR / "observations.json"
TEMP_FILE = CATALOGUE_DIR / "observations.tmp"


# ==================================================
# 2. SAVE OBSERVATIONS
# ==================================================

def save_observations(observations: list[dict[str, Any]]) -> None:
    """
    Save observations to the JSON catalogue.

    Uses a temporary file and atomic replacement to
    reduce the risk of corrupting an existing catalogue.
    """

    if not isinstance(observations, list):
        raise TypeError("Observations must be provided as a list.")

    if not all(isinstance(item, dict) for item in observations):
        raise TypeError(
            "Every observation must be a dictionary."
        )

    CATALOGUE_DIR.mkdir(parents=True, exist_ok=True)

    try:
        with open(TEMP_FILE, "w", encoding="utf-8") as file:
            json.dump(
                observations,
                file,
                indent=4,
                ensure_ascii=False,
                allow_nan=False
            )

            file.flush()
            os.fsync(file.fileno())

        os.replace(TEMP_FILE, CATALOGUE_FILE)

    finally:
        if TEMP_FILE.exists():
            TEMP_FILE.unlink()

    print(f"Saved {len(observations)} observations.")
    print(f"Catalogue location: {CATALOGUE_FILE}")


# ==================================================
# 3. LOAD OBSERVATIONS
# ==================================================

def load_observations() -> list[dict[str, Any]]:
    """
    Load observations from the JSON catalogue.

    Returns an empty list if no catalogue exists.
    Raises an error if the catalogue is corrupted.
    """

    if not CATALOGUE_FILE.exists():
        print("No observation catalogue found.")
        return []

    try:
        with open(CATALOGUE_FILE, "r", encoding="utf-8") as file:
            observations = json.load(file)

    except json.JSONDecodeError as error:
        raise ValueError(
            f"Catalogue contains invalid JSON: {error}"
        ) from error

    if not isinstance(observations, list):
        raise ValueError(
            "Catalogue format is invalid: expected a JSON list."
        )

    if not all(isinstance(item, dict) for item in observations):
        raise ValueError(
            "Catalogue contains invalid observation records."
        )

    print(f"Loaded {len(observations)} observations.")

    return observations


# ==================================================
# 4. CATALOGUE SUMMARY
# ==================================================

def get_catalogue_summary() -> dict[str, Any]:
    """Generate a basic summary of the catalogue."""

    observations = load_observations()

    instruments = {}

    for observation in observations:
        instrument = str(
            observation.get("iid", "Unknown")
        ).upper()

        instruments[instrument] = (
            instruments.get(instrument, 0) + 1
        )

    return {
        "total_observations": len(observations),
        "instruments": instruments,
        "catalogue_path": str(CATALOGUE_FILE)
    }


# ==================================================
# 5. MODULE TEST
# ==================================================

if __name__ == "__main__":

    print("=" * 50)
    print("ELYSIUM OBSERVATION CATALOGUE")
    print("=" * 50)

    summary = get_catalogue_summary()

    print("\nCatalogue summary:")
    print(f"Total observations: {summary['total_observations']}")

    print("\nInstrument distribution:")

    for instrument, count in summary["instruments"].items():
        print(f"{instrument}: {count}")

    print("\nCatalogue inspection completed.")
