
"""
ELYSIUM — HiRISE Data Acquisition and Processing

Automatically discovers, selects, downloads and catalogues
HiRISE observations from NASA Mars Orbital Data Explorer.

Workflow:
    1. Search NASA ODE for MRO HiRISE observations.
    2. Search processed RDRV11 products and raw EDR products.
    3. Filter and deduplicate observations.
    4. Prefer processed RDRV11 products.
    5. Download selected products and available labels.
    6. Store products in data/raw/hirise/.
    7. Generate a HiRISE master catalogue.

Notes:
    - The detailed AOI is inherited from config.py.
    - Existing CTX catalogue files are never overwritten.
    - Original NASA products are preserved.
    - Download count is deliberately limited.
"""

import json
import shutil

from pathlib import Path

from config import RAW_DIR, PROCESSED_DIR

from data_engine.discovery import search_mars_products

from data_engine.downloader import (
    get_product_files,
    get_product_download_url,
    download_product,
)


# ==================================================
# CONFIGURATION
# ==================================================

HIRISE_DIR = RAW_DIR / "hirise"

CATALOGUE_FILE = (
    PROCESSED_DIR / "hirise_master_catalogue.json"
)

# Prefer processed, map-projected imagery.
PRODUCT_TYPES = (
    "RDRV11",
    "EDR",
)

# Avoid downloading hundreds of large images.
MAX_PRODUCTS_TO_DOWNLOAD = 3

PAGE_SIZE = 500

INSTRUMENT = "HIRISE"
MISSION = "MRO"


# ==================================================
# PRODUCT NORMALISATION
# ==================================================

def get_product_id(product):
    """Get the NASA PDS product identifier."""

    return str(
        product.get("pdsid")
        or product.get("ProductId")
        or product.get("productId")
        or ""
    ).strip()


def get_product_type(product):
    """Get the ODE product type."""

    return str(
        product.get("pt")
        or product.get("ProductType")
        or product.get("productType")
        or ""
    ).upper().strip()


def get_product_instrument(product):
    """Get the instrument identifier."""

    return str(
        product.get("iid")
        or product.get("Instrument")
        or product.get("instrument")
        or ""
    ).upper().strip()


# ==================================================
# NASA ODE DISCOVERY
# ==================================================

def discover_hirise_observations():
    """
    Search NASA ODE for HiRISE observations
    intersecting the configured detailed AOI.
    """

    print("\nELYSIUM | HIRISE DISCOVERY")
    print("=" * 50)

    discovered = []
    search_errors = []

    for product_type in PRODUCT_TYPES:

        print(
            f"\nSearching NASA ODE: "
            f"{MISSION} / {INSTRUMENT} / {product_type}"
        )

        try:
            products = search_mars_products(
                instrument=INSTRUMENT,
                mission=MISSION,
                product_type=product_type,
                page_size=PAGE_SIZE,
            )

            print("Products returned:", len(products))

            for product in products:

                if not isinstance(product, dict):
                    continue

                if get_product_instrument(product) != INSTRUMENT:
                    continue

                product_id = get_product_id(product)

                if not product_id:
                    continue

                product["pdsid"] = product_id
                product["iid"] = INSTRUMENT
                product["pt"] = product_type

                discovered.append(product)

            if len(products) >= PAGE_SIZE:
                print(
                    "Warning: Search reached the page limit. "
                    "The results may be incomplete."
                )

        except Exception as error:

            search_errors.append(
                f"{product_type}: {error}"
            )

            print(
                f"Search failed for {product_type}: {error}"
            )

    if not discovered and search_errors:
        raise RuntimeError(
            "All HiRISE discovery searches failed:\n"
            + "\n".join(search_errors)
        )

    # Deduplicate by product ID.
    unique_products = {}

    for product in discovered:

        product_id = get_product_id(product)

        if product_id not in unique_products:
            unique_products[product_id] = product

    products = list(unique_products.values())

    # Prefer processed RDRV11 products over raw EDR products.
    priority = {
        "RDRV11": 0,
        "RDR": 1,
        "EDR": 2,
    }

    products.sort(
        key=lambda product: (
            priority.get(get_product_type(product), 99),
            get_product_id(product),
        )
    )

    print("\nDISCOVERY SUMMARY")
    print("Unique HiRISE products:", len(products))
    print("Search errors:", len(search_errors))

    return products


# ==================================================
# PRODUCT SELECTION
# ==================================================

def select_hirise_products(products, max_products=None):
    """
    Select products with available original data files.

    RDRV11 products receive priority because they are
    map-projected reduced data products.
    """

    selected = []

    for product in products:

        product_type = get_product_type(product)

        if product_type not in PRODUCT_TYPES:
            continue

        product_url = get_product_download_url(product)

        if not product_url:
            continue

        selected.append(product)

    if max_products is not None:
        selected = selected[:max_products]

    print("\nHIRISE PRODUCT SELECTION")
    print("Products with download URLs:", len(selected))

    for index, product in enumerate(selected, start=1):

        print(
            f"{index}. {get_product_id(product)} "
            f"({get_product_type(product)})"
        )

    return selected


# ==================================================
# DOWNLOAD MANAGEMENT
# ==================================================

def move_download_to_hirise(downloaded_path):
    """Move a verified downloader result into the HiRISE directory."""

    downloaded_path = Path(downloaded_path)

    HIRISE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    destination = HIRISE_DIR / downloaded_path.name

    if destination.exists() and destination.stat().st_size > 0:
        print("Already present in HiRISE directory:", destination.name)

        if downloaded_path.exists() and downloaded_path != destination:
            downloaded_path.unlink()

        return destination

    if not downloaded_path.exists():
        raise FileNotFoundError(downloaded_path)

    shutil.move(
        str(downloaded_path),
        str(destination),
    )

    return destination


def get_label_download_url(product):
    """
    Find the PDS label matching the selected HiRISE product.

    Never associate an unrelated label with an image.
    """

    product_id = get_product_id(product).casefold()

    for file_info in get_product_files(product):

        filename = str(
            file_info.get("FileName", "")
        ).replace("\\", "/").split("/")[-1]

        file_type = str(
            file_info.get("Type", "")
        ).lower()

        if not filename.lower().endswith(".lbl"):
            continue

        label_stem = Path(filename).stem.casefold()

        if label_stem == product_id:
            return file_info.get("URL")

    return None


def download_hirise_product(product):
    """Download the original HiRISE product and optional label."""

    product_id = get_product_id(product)

    product_url = get_product_download_url(product)

    if not product_url:
        raise ValueError(
            f"No original product URL found for {product_id}"
        )

    HIRISE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Reuse an existing local product.
    filename = Path(
        product_url.split("?")[0]
    ).name

    destination = HIRISE_DIR / filename

    if destination.exists() and destination.stat().st_size > 0:

        print("Product already downloaded:", filename)

        product_path = destination

    else:

        # The shared downloader performs streaming and
        # integrity checks. It initially saves to RAW_DIR.
        downloaded_path = download_product(
            product_url,
            product_id,
        )

        product_path = move_download_to_hirise(
            downloaded_path
        )

    # Download a separate label when ODE provides one.
    label_url = get_label_download_url(product)

    label_path = None

    if label_url:

        label_filename = Path(
            label_url.split("?")[0]
        ).name

        expected_label = HIRISE_DIR / label_filename

        if (
            expected_label.exists()
            and expected_label.stat().st_size > 0
        ):

            label_path = expected_label

        else:

            downloaded_label = download_product(
                label_url,
                f"{product_id}_label",
            )

            label_path = move_download_to_hirise(
                downloaded_label
            )

    return {
        "product_path": str(product_path),
        "label_path": (
            str(label_path)
            if label_path
            else None
        ),
        "product_size_bytes": product_path.stat().st_size,
    }


# ==================================================
# MASTER CATALOGUE
# ==================================================

def save_hirise_catalogue(records):
    """Save the HiRISE catalogue atomically."""

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_file = CATALOGUE_FILE.with_suffix(".tmp")

    try:

        with open(
            temporary_file,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                records,
                file,
                indent=4,
                ensure_ascii=False,
            )

            file.flush()

            import os
            os.fsync(file.fileno())

        temporary_file.replace(CATALOGUE_FILE)

    finally:

        if temporary_file.exists():
            temporary_file.unlink()


# ==================================================
# COMPLETE HIRISE PIPELINE
# ==================================================

def run_hirise_pipeline(
    max_products=MAX_PRODUCTS_TO_DOWNLOAD,
):
    """
    Run discovery, selection, downloading and cataloguing.

    Set max_products=None to remove the download-count limit.
    """

    print("\n")
    print("=" * 55)
    print("ELYSIUM | HIRISE DATA ACQUISITION")
    print("=" * 55)

    HIRISE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # 1. NASA ODE discovery.
    products = discover_hirise_observations()

    # 2. Select useful products.
    selected = select_hirise_products(
        products,
        max_products=max_products,
    )

    records = []

    successful = 0
    failed = 0

    # 3. Download selected products.
    for index, product in enumerate(selected, start=1):

        product_id = get_product_id(product)

        print(
            f"\n[{index}/{len(selected)}] "
            f"{product_id}"
        )

        record = {
            "product_id": product_id,
            "mission": MISSION,
            "instrument": INSTRUMENT,
            "product_type": get_product_type(product),
            "observation_start": product.get(
                "starttime",
                product.get("Observation_start_time"),
            ),
            "observation_end": product.get(
                "stoptime",
                product.get("Observation_end_time"),
            ),
            "detail_url": product.get("DetailURL"),
            "download_status": None,
            "product_path": None,
            "label_path": None,
            "product_size_bytes": None,
            "error": None,
        }

        try:

            result = download_hirise_product(product)

            record.update(result)
            record["download_status"] = "Downloaded"

            successful += 1

        except Exception as error:

            record["download_status"] = "Failed"
            record["error"] = str(error)

            failed += 1

            print("Download failed:", error)

        records.append(record)

    # 4. Save the independent HiRISE catalogue.
    save_hirise_catalogue(records)

    # 5. Final report.
    print("\n")
    print("=" * 55)
    print("HIRISE ACQUISITION COMPLETE")
    print("=" * 55)

    print("Discovered:", len(products))
    print("Selected:", len(selected))
    print("Downloaded or already available:", successful)
    print("Failed:", failed)

    print("\nHiRISE data directory:")
    print(HIRISE_DIR)

    print("\nMaster catalogue:")
    print(CATALOGUE_FILE)

    return records


# ==================================================
# MAIN
# ==================================================

if __name__ == "__main__":

    try:

        run_hirise_pipeline()

    except Exception as error:

        print("\nHIRISE PIPELINE FAILED")
        print(error)
