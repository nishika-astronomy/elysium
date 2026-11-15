
"""
ELYSIUM — Mars ODE Discovery Engine

Discovers orbital observations intersecting the
Western Elysium Planitia study region.

Data source:
    NASA/PDS Geosciences Node — Mars Orbital Data Explorer

Workflow:
    1. Query Mars ODE.
    2. Validate the API response.
    3. Select eligible observations.
    4. Save the selected observations to the catalogue.

Important:
    A search reaching its requested result limit may be
    incomplete. This module does not assume that one
    API response contains every matching observation.
"""

import requests

from config import DETAILED_AOI, MAX_RESULTS_PER_SEARCH

from data_engine.selector import (
    select_observations,
    SelectionCriteria,
)

from data_engine.catalogue import save_observations


# ==================================================
# 1. API CONFIGURATION
# ==================================================

ODE_URL = "https://oderest.rsl.wustl.edu/live2/"
ODE_WEBSITE_URL = "https://ode.rsl.wustl.edu/mars/"

DEFAULT_INSTRUMENT = "CTX"
DEFAULT_MISSION = "MRO"
DEFAULT_PRODUCT_TYPE = "EDR"

DEFAULT_PAGE_SIZE = MAX_RESULTS_PER_SEARCH
DEFAULT_TIMEOUT = 120

USER_AGENT = "ELYSIUM-Mars-Exploration/0.1.0"


# ==================================================
# 2. API RESPONSE VALIDATION
# ==================================================

def parse_ode_response(data):
    """
    Extract product records from a Mars ODE JSON response.

    Supports:
        - Multiple product records
        - A single product dictionary
        - Empty product results

    Raises an error for malformed responses.
    """

    if not isinstance(data, dict):
        raise ValueError(
            "Mars ODE returned an invalid JSON structure."
        )

    ode_results = data.get("ODEResults")

    if not isinstance(ode_results, dict):
        raise ValueError(
            "Mars ODE response is missing ODEResults."
        )

    api_error = ode_results.get("Error")

    if api_error:
        raise RuntimeError(
            f"Mars ODE reported an error: {api_error}"
        )

    products = ode_results.get("Products")

    if not isinstance(products, dict):
        raise ValueError(
            "Mars ODE response is missing the Products section."
        )

    product_list = products.get("Product", [])

    if product_list is None:
        return []

    if isinstance(product_list, dict):
        product_list = [product_list]

    if not isinstance(product_list, list):
        raise ValueError(
            "Mars ODE returned products in an unexpected format."
        )

    if not all(isinstance(product, dict) for product in product_list):
        raise ValueError(
            "Mars ODE returned an invalid observation record."
        )

    return product_list


# ==================================================
# 3. SEARCH MARS ODE
# ==================================================

def search_mars_products(
    instrument=DEFAULT_INSTRUMENT,
    mission=DEFAULT_MISSION,
    product_type=DEFAULT_PRODUCT_TYPE,
    page_size=DEFAULT_PAGE_SIZE,
    timeout=DEFAULT_TIMEOUT,
):
    """
    Search Mars ODE for orbital products intersecting
    the configured detailed study area.

    Returns:
        list: Raw observation dictionaries.

    Note:
        page_size is the requested result limit, not a
        guarantee that the API supports pagination here.
    """

    if not isinstance(page_size, int) or page_size < 1:
        raise ValueError(
            "page_size must be a positive integer."
        )

    if page_size > MAX_RESULTS_PER_SEARCH:
        raise ValueError(
            f"page_size cannot exceed {MAX_RESULTS_PER_SEARCH}."
        )

    if timeout <= 0:
        raise ValueError("Timeout must be positive.")

    area = DETAILED_AOI

    params = {
        "query": "product",
        "results": "fp",
        "output": "JSON",
        "limit": page_size,
        "target": "mars",
        "ihid": mission,
        "iid": instrument,
        "westlon": area["west"],
        "eastlon": area["east"],
        "minlat": area["south"],
        "maxlat": area["north"],
        "pt": product_type,
    }

    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/json",
    }

    print("\nConnecting to Mars ODE...")
    print(f"Instrument: {instrument}")
    print(f"Mission: {mission}")
    print(f"Product type: {product_type}")
    print(f"Study area: {area['name']}")

    try:
        response = requests.get(
            ODE_URL,
            params=params,
            headers=headers,
            timeout=timeout,
        )

        response.raise_for_status()

    except requests.exceptions.Timeout as error:
        raise RuntimeError(
            "Mars ODE request timed out."
        ) from error

    except requests.exceptions.RequestException as error:
        raise RuntimeError(
            f"Could not communicate with Mars ODE: {error}"
        ) from error

    try:
        data = response.json()

    except ValueError as error:
        raise RuntimeError(
            "Mars ODE did not return valid JSON."
        ) from error

    products = parse_ode_response(data)

    print(f"Raw observations returned: {len(products)}")

    if len(products) >= page_size:
        print(
            "WARNING: The result count reached the requested "
            "limit. The search may be incomplete."
        )

    return products


# ==================================================
# 4. DISCOVER AND SELECT OBSERVATIONS
# ==================================================

def discover_observations(
    save=True,
    page_size=DEFAULT_PAGE_SIZE,
):
    """
    Discover, filter and optionally catalogue observations.

    Returns:
        list: Selected observation dictionaries.

    Existing catalogue data is preserved if the API
    returns zero observations.
    """

    print("\nELYSIUM DATA DISCOVERY")
    print("=" * 50)

    observations = search_mars_products(
        instrument=DEFAULT_INSTRUMENT,
        mission=DEFAULT_MISSION,
        product_type=DEFAULT_PRODUCT_TYPE,
        page_size=page_size,
    )

    if not observations:
        print("\nNo observations were returned.")
        print("Existing catalogue has not been modified.")
        return []

    criteria = SelectionCriteria(
        preferred_instruments=(DEFAULT_INSTRUMENT,),
        require_data_product=True,
    )

    selected = select_observations(
        observations,
        criteria,
    )

    print("\nDiscovery summary:")
    print(f"Raw observations: {len(observations)}")
    print(f"Eligible observations: {len(selected)}")
    print(f"Rejected observations: {len(observations) - len(selected)}")

    if not selected:
        print(
            "\nNo eligible observations found. "
            "Existing catalogue has not been modified."
        )
        return []

    if save:
        save_observations(selected)
        print("Observation catalogue updated successfully.")

    print("\nDiscovery completed.")

    return selected


# ==================================================
# 5. CONNECTION TEST
# ==================================================

def test_connection(timeout=30):
    """
    Test whether the Mars ODE website is reachable.
    """

    try:
        response = requests.get(
            ODE_WEBSITE_URL,
            headers={"User-Agent": USER_AGENT},
            timeout=timeout,
        )

        response.raise_for_status()

        return True

    except requests.exceptions.RequestException as error:
        raise RuntimeError(
            f"Mars ODE connection test failed: {error}"
        ) from error


# ==================================================
# 6. MODULE EXECUTION
# ==================================================

if __name__ == "__main__":

    try:
        print("=" * 50)
        print("ELYSIUM NASA DATA DISCOVERY")
        print("=" * 50)

        if test_connection():
            print("\nMars ODE website is reachable.")

        discovered = discover_observations()

        print(
            f"\nFinal selected observation count: {len(discovered)}"
        )

    except Exception as error:
        print(f"\nDiscovery failed: {error}")
