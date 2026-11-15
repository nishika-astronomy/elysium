"""
ELYSIUM — THEMIS Discovery, Download and Processing

Discovers Mars Odyssey THEMIS observations through NASA PDS ODE.

Workflow:
1. Query available THEMIS product types.
2. Discover observations intersecting the ELYSIUM AOI.
3. Select a balanced set of products.
4. Download image files and matching product labels.
5. Validate and inventory downloaded rasters.
6. Extract geographic regions where a valid Mars CRS exists.

Scientific precautions:
- Visible brightness is not temperature.
- Infrared brightness temperature is not thermal inertia.
- Raw, unprojected images are not geographic maps.
- Original pixel values are preserved.
- Calibration is not automatically applied.
- ODE search coordinates use positive-east longitude (0–360)
  and planetocentric latitude.
"""

from pathlib import Path
import json
import math
import time
import xml.etree.ElementTree as ET

import numpy as np
import requests
import rasterio

from rasterio.windows import Window, from_bounds
from rasterio.warp import transform_bounds
from rasterio.crs import CRS as RasterioCRS
from pyproj import CRS as PyprojCRS

from config import RAW_DIR, PROCESSED_DIR, DETAILED_AOI


# ============================================================
# CONFIGURATION
# ============================================================

ODE_BASE_URL = "https://oderest.rsl.wustl.edu/live2"

THEMIS_RAW_DIR = Path(RAW_DIR) / "themis"
THEMIS_PROCESSED_DIR = Path(PROCESSED_DIR) / "themis"

THEMIS_CATALOGUE_PATH = (
    THEMIS_PROCESSED_DIR / "themis_catalogue.json"
)
THEMIS_INVENTORY_PATH = (
    THEMIS_PROCESSED_DIR / "themis_inventory.json"
)

PREFERRED_PRODUCT_TYPES = [
    "IRGEO4",
    "IRPBT4",
    "VGEO4",
    "VISALB4",
    "IRGEO2",
    "IRPBT2",
    "VGEO2",
    "VISALB",
    "IRGEO1",
    "IRPBT1",
    "VGEO1",
]

MAX_DOWNLOAD_PRODUCTS = 6
MAX_RESULTS_PER_TYPE = 100

SUPPORTED_EXTENSIONS = {
    ".cub",
    ".img",
    ".tif",
    ".tiff",
    ".dat",
}

LABEL_EXTENSIONS = {".lbl", ".xml"}

MARS_RADIUS_MIN_METRES = 3_000_000
MARS_RADIUS_MAX_METRES = 4_000_000

REQUEST_TIMEOUT = (30, 180)
DOWNLOAD_CHUNK_SIZE = 1024 * 1024

THEMIS_RAW_DIR.mkdir(parents=True, exist_ok=True)
THEMIS_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# AOI UTILITIES
# ============================================================

def get_themis_aoi():
    """Return and validate the detailed ELYSIUM AOI."""

    west = float(DETAILED_AOI["west"])
    east = float(DETAILED_AOI["east"])
    south = float(DETAILED_AOI["south"])
    north = float(DETAILED_AOI["north"])

    if not (
        0 <= west < east <= 360
        and -90 <= south < north <= 90
    ):
        raise ValueError("Invalid ELYSIUM AOI.")

    return west, south, east, north


# ============================================================
# XML UTILITIES
# ============================================================

def local_tag(element):
    """Return an XML tag without its namespace."""
    return element.tag.split("}")[-1].lower()


def get_child_text(element, *names):
    """Find a child element by one of its possible names."""

    wanted = {name.lower() for name in names}

    for child in element.iter():
        if local_tag(child) in wanted:
            if child.text:
                return child.text.strip()

    return None


def parse_ode_xml(response):
    """Parse an ODE XML response and detect API errors."""

    try:
        root = ET.fromstring(response.content)
    except ET.ParseError as error:
        raise ValueError(
            "ODE returned an invalid XML response."
        ) from error

    status = get_child_text(root, "Status")

    if status and status.upper() == "ERROR":
        message = get_child_text(root, "Error")
        raise RuntimeError(
            f"NASA ODE error: {message or 'Unknown API error'}"
        )

    return root


# ============================================================
# ODE REQUEST
# ============================================================

def ode_request(params):
    """Send a GET request to the NASA ODE REST interface."""

    try:
        response = requests.get(
            ODE_BASE_URL,
            params=params,
            timeout=REQUEST_TIMEOUT,
            headers={
                "User-Agent": "ELYSIUM-Mars-Research/1.0"
            },
        )
        response.raise_for_status()

    except requests.RequestException as error:
        raise RuntimeError(
            f"Unable to communicate with NASA ODE: {error}"
        ) from error

    return response


# ============================================================
# DISCOVER AVAILABLE THEMIS PRODUCT TYPES
# ============================================================

def get_available_themis_product_types():
    """Query ODE's IIPT endpoint for available THEMIS types."""

    print("\nQuerying NASA ODE for THEMIS product types...")

    response = ode_request({
        "query": "iipt",
        "target": "mars",
        "output": "XML",
    })

    root = parse_ode_xml(response)
    available = {}

    for element in root.iter():
        if local_tag(element) != "iiptset":
            continue

        host = get_child_text(element, "IHID")
        instrument = get_child_text(element, "IID")
        product_type = get_child_text(element, "PT")
        product_name = get_child_text(element, "PTName")

        if not all([host, instrument, product_type]):
            continue

        if (
            host.strip().lower() == "ody"
            and instrument.strip().lower() == "themis"
        ):
            available[product_type.strip().upper()] = {
                "product_type": product_type.strip(),
                "product_name": product_name,
            }

    if not available:
        raise RuntimeError(
            "ODE did not return any THEMIS product types. "
            "Check the ODE service and its current holdings."
        )

    print(
        "Available THEMIS product types:",
        ", ".join(sorted(available)),
    )

    return available


# ============================================================
# PRODUCT FILE EXTRACTION
# ============================================================

def extract_product_files(product):
    """Extract downloadable file metadata from an ODE product."""

    files = []

    for element in product.iter():
        if local_tag(element) != "product_file":
            continue

        filename = get_child_text(element, "FileName")
        url = get_child_text(element, "URL")
        file_type = get_child_text(element, "Type")
        size_kb = get_child_text(element, "KBytes")

        if not filename or not url:
            continue

        try:
            size_bytes = (
                int(float(size_kb) * 1024)
                if size_kb
                else None
            )
        except (ValueError, TypeError):
            size_bytes = None

        files.append({
            "filename": filename.replace("\\", "/").split("/")[-1],
            "url": url,
            "type": file_type,
            "size_bytes": size_bytes,
        })

    return files


def choose_image_file(files):
    """Choose the primary image/cube file from a product."""

    candidates = []

    for file_info in files:
        filename = file_info["filename"]
        extension = Path(filename).suffix.lower()

        if extension in SUPPORTED_EXTENSIONS:
            candidates.append(file_info)

    if not candidates:
        return None

    priority = {
        ".cub": 0,
        ".tif": 1,
        ".tiff": 2,
        ".img": 3,
        ".dat": 4,
    }

    candidates.sort(
        key=lambda item: priority.get(
            Path(item["filename"]).suffix.lower(),
            99,
        )
    )

    return candidates[0]


def choose_label_files(files, image_file):
    """
    Find labels belonging to the exact selected image.

    Only ODE files classified as Product are accepted.
    Browse, Referenced and Derived files are excluded.
    """

    image_stem = Path(
        image_file["filename"]
    ).stem.casefold()

    labels = []

    for file_info in files:
        filename = file_info["filename"]
        extension = Path(filename).suffix.lower()

        if extension not in LABEL_EXTENSIONS:
            continue

        if str(file_info.get("type", "")).casefold() != "product":
            continue

        label_stem = Path(filename).stem.casefold()

        if label_stem == image_stem:
            labels.append(file_info)

    labels.sort(
        key=lambda item: (
            0
            if Path(item["filename"]).suffix.lower() == ".lbl"
            else 1
        )
    )

    return labels


# ============================================================
# DISCOVER THEMIS OBSERVATIONS
# ============================================================

def discover_themis_products():
    """
    Search NASA ODE for THEMIS products intersecting the AOI.
    """

    west, south, east, north = get_themis_aoi()
    available_types = get_available_themis_product_types()

    selected_types = [
        product_type
        for product_type in PREFERRED_PRODUCT_TYPES
        if product_type in available_types
    ]

    if not selected_types:
        raise RuntimeError(
            "None of the preferred THEMIS product types "
            "are currently available in ODE. "
            f"Available types: {list(available_types)}"
        )

    print("\nELYSIUM THEMIS DISCOVERY")
    print("=" * 55)
    print(
        f"AOI: {west}°E–{east}°E, "
        f"{south}°N–{north}°N"
    )

    discovered = []

    for product_type in selected_types:
        print(f"\nSearching product type: {product_type}")

        response = ode_request({
            "query": "product",
            "target": "mars",
            "output": "XML",
            "results": "opmf",
            "ihid": "ODY",
            "iid": "THEMIS",
            "pt": product_type,
            "westernlon": west,
            "easternlon": east,
            "minlat": south,
            "maxlat": north,
            "loc": "f",
            "limit": MAX_RESULTS_PER_TYPE,
            "offset": 0,
        })

        root = parse_ode_xml(response)
        products_found = 0

        for product in root.iter():
            if local_tag(product) != "product":
                continue

            product_id = get_child_text(product, "pdsid")

            if not product_id:
                continue

            files = extract_product_files(product)
            image_file = choose_image_file(files)

            if image_file is None:
                continue

            # IMPORTANT: use the plural function and pass the image.
            label_files = choose_label_files(files, image_file)

            item = {
                "product_id": product_id,
                "product_type": product_type,
                "instrument_host": get_child_text(product, "ihid"),
                "instrument": get_child_text(product, "iid"),
                "dataset_id": get_child_text(product, "Data_Set_Id"),
                "observation_time": get_child_text(
                    product, "Observation_time"
                ),
                "start_time": get_child_text(
                    product, "UTC_start_time"
                ),
                "stop_time": get_child_text(
                    product, "UTC_stop_time"
                ),
                "center_latitude": get_child_text(
                    product, "Center_latitude"
                ),
                "center_longitude": get_child_text(
                    product, "Center_longitude"
                ),
                "minimum_latitude": get_child_text(
                    product, "Minimum_latitude"
                ),
                "maximum_latitude": get_child_text(
                    product, "Maximum_latitude"
                ),
                "westernmost_longitude": get_child_text(
                    product, "Westernmost_longitude"
                ),
                "easternmost_longitude": get_child_text(
                    product, "Easternmost_longitude"
                ),
                "map_resolution": get_child_text(
                    product, "Map_resolution"
                ),
                "description": get_child_text(product, "Description"),
                "image_file": image_file,
                "label_files": label_files,
                "all_files": files,
            }

            discovered.append(item)
            products_found += 1

        print(
            "Products with downloadable image files:",
            products_found,
        )

        time.sleep(0.25)

    # Remove duplicate product IDs while preserving discovery order.
    unique = {}

    for product in discovered:
        unique[product["product_id"]] = product

    discovered = list(unique.values())

    print("\n" + "=" * 55)
    print("THEMIS DISCOVERY COMPLETE")
    print("Unique products:", len(discovered))

    return discovered


# ============================================================
# PRODUCT SELECTION
# ============================================================

def select_themis_products(products):
    """Select a balanced set of THEMIS product types."""

    priority = {
        "IRGEO4": 0,
        "IRPBT4": 1,
        "VGEO4": 2,
        "VISALB4": 3,
        "IRGEO2": 4,
        "IRPBT2": 5,
        "VGEO2": 6,
        "VISALB": 7,
        "IRGEO1": 8,
        "IRPBT1": 9,
        "VGEO1": 10,
    }

    grouped = {}

    for product in products:
        product_type = product["product_type"]
        grouped.setdefault(product_type, []).append(product)

    for product_type in grouped:
        grouped[product_type].sort(
            key=lambda item: item["product_id"]
        )

    ordered_types = sorted(
        grouped,
        key=lambda item: priority.get(item, 99),
    )

    selected = []

    # Take at most two products from each type per round.
    for round_index in range(2):
        for product_type in ordered_types:
            candidates = grouped[product_type]

            if round_index < len(candidates):
                selected.append(candidates[round_index])

                if len(selected) >= MAX_DOWNLOAD_PRODUCTS:
                    break

        if len(selected) >= MAX_DOWNLOAD_PRODUCTS:
            break

    print("\nTHEMIS PRODUCT SELECTION")
    print("=" * 45)
    print("Discovered:", len(products))
    print("Selected:", len(selected))

    for item in selected:
        print(
            item["product_id"],
            "|",
            item["product_type"],
        )

    return selected


# ============================================================
# DOWNLOAD
# ============================================================

def download_themis_file(file_info, destination):
    """Download one THEMIS product file safely."""

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists() and destination.stat().st_size > 0:
        print("Already downloaded:", destination.name)
        return destination

    temporary_path = destination.with_suffix(
        destination.suffix + ".part"
    )

    try:
        with requests.get(
            file_info["url"],
            stream=True,
            timeout=REQUEST_TIMEOUT,
            headers={
                "User-Agent": "ELYSIUM-Mars-Research/1.0"
            },
        ) as response:

            response.raise_for_status()

            content_type = response.headers.get(
                "Content-Type", ""
            ).lower()

            if "text/html" in content_type:
                raise ValueError(
                    "Server returned HTML instead of a product file."
                )

            downloaded_bytes = 0

            with open(temporary_path, "wb") as output:
                for chunk in response.iter_content(
                    chunk_size=DOWNLOAD_CHUNK_SIZE
                ):
                    if not chunk:
                        continue

                    output.write(chunk)
                    downloaded_bytes += len(chunk)

        if downloaded_bytes == 0:
            raise ValueError("Downloaded product file is empty.")

        temporary_path.replace(destination)

        print(
            f"Downloaded: {destination.name} "
            f"({downloaded_bytes:,} bytes)"
        )

        return destination

    except Exception:
        if temporary_path.exists():
            temporary_path.unlink()
        raise


def download_themis_products(products):
    """Download selected THEMIS images and their matching labels."""

    download_results = []

    print("\nTHEMIS DOWNLOAD")
    print("=" * 55)

    for index, product in enumerate(products, start=1):
        product_id = product["product_id"]
        product_dir = THEMIS_RAW_DIR / product_id
        product_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[{index}/{len(products)}] {product_id}")

        result = {
            "product_id": product_id,
            "product_type": product["product_type"],
            "status": "Failed",
            "image_path": None,
            "label_paths": [],
            "error": None,
        }

        try:
            image_info = product["image_file"]
            image_path = product_dir / image_info["filename"]

            download_themis_file(image_info, image_path)
            result["image_path"] = str(image_path)

            # Recalculate matching labels from the complete ODE file list.
            label_files = choose_label_files(
                product["all_files"],
                image_info,
            )

            if label_files:
                for label_info in label_files:
                    label_path = (
                        product_dir / label_info["filename"]
                    )

                    download_themis_file(label_info, label_path)
                    result["label_paths"].append(str(label_path))

            else:
                print(
                    "Warning: No matching Product label "
                    "was listed by ODE."
                )

            result["status"] = "Downloaded"

        except Exception as error:
            result["error"] = str(error)
            print("Download failed:", error)

        download_results.append(result)

    return download_results


# ============================================================
# CATALOGUE SAVING
# ============================================================

def save_themis_catalogue(products, download_results):
    """Save discovery metadata and download results."""

    catalogue = {
        "project": "ELYSIUM",
        "source": "NASA PDS Geosciences Node ODE",
        "instrument_host": "2001 Mars Odyssey",
        "instrument": "THEMIS",
        "aoi": {
            "west": DETAILED_AOI["west"],
            "east": DETAILED_AOI["east"],
            "south": DETAILED_AOI["south"],
            "north": DETAILED_AOI["north"],
        },
        "discovered_product_count": len(products),
        "download_results": download_results,
        "products": products,
    }

    with open(
        THEMIS_CATALOGUE_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            catalogue,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print("\nTHEMIS catalogue saved:", THEMIS_CATALOGUE_PATH)

    return catalogue


# ============================================================
# LOCAL FILE DISCOVERY
# ============================================================

def find_themis_files():
    """Find locally downloaded THEMIS raster files."""

    files = [
        path
        for path in THEMIS_RAW_DIR.rglob("*")
        if (
            path.is_file()
            and path.suffix.lower() in SUPPORTED_EXTENSIONS
            and not path.name.endswith(".part")
        )
    ]

    return sorted(files)


# ============================================================
# MARS CRS VALIDATION
# ============================================================

def validate_mars_crs(dataset_crs):
    """
    Validate the raster CRS using its geodetic ellipsoid.

    A Mars-sized ellipsoid is a sanity check, not proof that
    longitude convention and map orientation are correct.
    """

    if dataset_crs is None:
        raise ValueError("THEMIS raster has no CRS.")

    try:
        pyproj_crs = PyprojCRS.from_wkt(dataset_crs.to_wkt())
        geographic_crs = pyproj_crs.geodetic_crs

        if geographic_crs is None:
            raise ValueError("Raster CRS has no geodetic CRS.")

        radius = geographic_crs.ellipsoid.semi_major_metre

        if not (
            MARS_RADIUS_MIN_METRES
            <= radius
            <= MARS_RADIUS_MAX_METRES
        ):
            raise ValueError(
                "CRS does not have a Mars-sized ellipsoid. "
                f"Semi-major axis: {radius} metres."
            )

        return geographic_crs

    except ValueError:
        raise

    except Exception as error:
        raise ValueError(
            f"Unable to validate Mars CRS: {error}"
        ) from error


# ============================================================
# RASTER INSPECTION
# ============================================================

def inspect_themis_file(file_path: Path) -> dict:
    """Inspect a downloaded THEMIS raster."""

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(file_path)

    if file_path.stat().st_size == 0:
        raise ValueError("THEMIS file is empty.")

    with rasterio.open(file_path) as dataset:

        if dataset.count < 1:
            raise ValueError("THEMIS raster contains no bands.")

        sample = dataset.read(
            1,
            out_shape=(
                min(dataset.height, 128),
                min(dataset.width, 128),
            ),
        )

        valid = np.isfinite(sample)

        if dataset.nodata is not None:
            valid &= sample != dataset.nodata

        mars_crs_valid = False
        geographic_crs_name = None
        crs_error = None

        if dataset.crs is not None:
            try:
                geographic_crs = validate_mars_crs(dataset.crs)
                mars_crs_valid = True
                geographic_crs_name = geographic_crs.name

            except ValueError as error:
                crs_error = str(error)

        metadata = {
            "filename": file_path.name,
            "file_size_bytes": file_path.stat().st_size,
            "width": dataset.width,
            "height": dataset.height,
            "band_count": dataset.count,
            "data_types": list(dataset.dtypes),
            "band_descriptions": list(dataset.descriptions),
            "band_units": list(dataset.units),
            "band_scales": list(dataset.scales),
            "band_offsets": list(dataset.offsets),
            "crs": (
                str(dataset.crs)
                if dataset.crs is not None
                else None
            ),
            "geographic_crs": geographic_crs_name,
            "is_georeferenced": dataset.crs is not None,
            "is_valid_mars_crs": mars_crs_valid,
            "crs_validation_error": crs_error,
            "bounds": tuple(dataset.bounds),
            "transform": tuple(dataset.transform),
            "nodata": dataset.nodata,
            "valid_sample_pixels": int(np.count_nonzero(valid)),
            "sample_pixel_count": int(sample.size),
            "sample_minimum": (
                float(np.min(sample[valid]))
                if np.any(valid)
                else None
            ),
            "sample_maximum": (
                float(np.max(sample[valid]))
                if np.any(valid)
                else None
            ),
            "band_1_tags": dataset.tags(1),
        }

    return metadata


# ============================================================
# GEOGRAPHIC REGION LOADING
# ============================================================

def load_themis_region(file_path: Path, bounds=None, band=1):
    """
    Extract a geographic AOI from a georeferenced THEMIS raster.

    Bounds are (west, south, east, north).

    Original pixel values are retained. No calibration or
    temperature conversion is applied.
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(file_path)

    if bounds is None:
        bounds = get_themis_aoi()

    west, south, east, north = map(float, bounds)

    if not (
        -180 <= west < east <= 360
        and -90 <= south < north <= 90
    ):
        raise ValueError("Invalid geographic bounds.")

    with rasterio.open(file_path) as dataset:

        if dataset.crs is None:
            raise ValueError(
                "THEMIS raster has no CRS. "
                "Geographic extraction is not possible."
            )

        if not isinstance(band, int) or isinstance(band, bool):
            raise ValueError("Band number must be an integer.")

        if band < 1 or band > dataset.count:
            raise ValueError(
                f"Invalid band {band}. "
                f"Available bands: 1 to {dataset.count}"
            )

        geographic_crs = validate_mars_crs(dataset.crs)

        # Convert positive-east 0–360 longitudes to the
        # equivalent -180–180 convention when appropriate.
        if (
            west >= 180
            and geographic_crs.prime_meridian.longitude == 0
        ):
            west_for_transform = west - 360
            east_for_transform = east - 360
        else:
            west_for_transform = west
            east_for_transform = east

        source_crs = RasterioCRS.from_wkt(
            geographic_crs.to_wkt()
        )

        if dataset.crs.is_geographic:
            raster_bounds = (
                west_for_transform,
                south,
                east_for_transform,
                north,
            )
        else:
            raster_bounds = transform_bounds(
                source_crs,
                dataset.crs,
                west_for_transform,
                south,
                east_for_transform,
                north,
                densify_pts=21,
            )

        raw_window = from_bounds(
            *raster_bounds,
            transform=dataset.transform,
        )

        col_start = max(
            0,
            math.floor(raw_window.col_off),
        )
        row_start = max(
            0,
            math.floor(raw_window.row_off),
        )
        col_end = min(
            dataset.width,
            math.ceil(raw_window.col_off + raw_window.width),
        )
        row_end = min(
            dataset.height,
            math.ceil(raw_window.row_off + raw_window.height),
        )

        if col_end <= col_start or row_end <= row_start:
            raise ValueError(
                "Requested AOI does not overlap the raster."
            )

        window = Window(
            col_start,
            row_start,
            col_end - col_start,
            row_end - row_start,
        )

        data = dataset.read(
            band,
            window=window,
            masked=True,
        )

        values = data.astype(np.float32).filled(np.nan)
        valid = np.isfinite(values)

        if not np.any(valid):
            raise ValueError(
                "Extracted region contains no valid pixels."
            )

        metadata = {
            "filename": file_path.name,
            "band": band,
            "crs": str(dataset.crs),
            "geographic_crs": geographic_crs.name,
            "bounds_requested": (
                west,
                south,
                east,
                north,
            ),
            "bounds_in_raster_crs": tuple(raster_bounds),
            "window": (
                int(row_start),
                int(col_start),
                int(row_end - row_start),
                int(col_end - col_start),
            ),
            "shape": values.shape,
            "raster_resolution": tuple(dataset.res),
            "nodata": dataset.nodata,
            "data_type": dataset.dtypes[band - 1],
            "band_description": dataset.descriptions[band - 1],
            "band_unit": dataset.units[band - 1],
            "band_scale": dataset.scales[band - 1],
            "band_offset": dataset.offsets[band - 1],
            "valid_pixels": int(np.count_nonzero(valid)),
            "total_pixels": int(values.size),
            "minimum": float(np.min(values[valid])),
            "maximum": float(np.max(values[valid])),
            "mean": float(np.mean(values[valid])),
            "calibration_applied": False,
        }

    return values, metadata


# ============================================================
# DATASET INVENTORY
# ============================================================

def build_themis_inventory():
    """Inspect all locally downloaded THEMIS raster files."""

    files = find_themis_files()
    inventory = []

    print("\nELYSIUM THEMIS INVENTORY")
    print("=" * 55)
    print("Files found:", len(files))

    for index, file_path in enumerate(files, start=1):
        print(f"\n[{index}/{len(files)}] {file_path.name}")

        try:
            metadata = inspect_themis_file(file_path)

            inventory.append({
                "status": "Passed",
                "metadata": metadata,
            })

            print(
                "Dimensions:",
                metadata["width"],
                "x",
                metadata["height"],
            )
            print("Bands:", metadata["band_count"])
            print("CRS:", metadata["crs"])
            print(
                "Valid Mars CRS:",
                metadata["is_valid_mars_crs"],
            )

        except Exception as error:
            inventory.append({
                "status": "Failed",
                "filename": file_path.name,
                "error": str(error),
            })

            print("Validation failed:", error)

    passed = sum(
        item["status"] == "Passed"
        for item in inventory
    )
    failed = sum(
        item["status"] == "Failed"
        for item in inventory
    )

    print("\n" + "=" * 55)
    print("THEMIS INVENTORY COMPLETE")
    print("Files inspected:", len(files))
    print("Passed:", passed)
    print("Failed:", failed)

    with open(
        THEMIS_INVENTORY_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            inventory,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print("Inventory saved:", THEMIS_INVENTORY_PATH)

    return inventory


# ============================================================
# COMPLETE PIPELINE
# ============================================================

def process_themis():
    """Execute the complete THEMIS discovery and processing workflow."""

    print("\n" + "=" * 55)
    print("ELYSIUM — THEMIS PIPELINE")
    print("=" * 55)

    products = discover_themis_products()

    if not products:
        raise RuntimeError(
            "No downloadable THEMIS products were found "
            "for the detailed ELYSIUM AOI."
        )

    selected = select_themis_products(products)

    download_results = download_themis_products(selected)

    save_themis_catalogue(
        selected,
        download_results,
    )

    inventory = build_themis_inventory()

    successful_downloads = sum(
        item["status"] == "Downloaded"
        for item in download_results
    )

    print("\n" + "=" * 55)
    print("THEMIS PIPELINE COMPLETE")
    print("Selected products:", len(selected))
    print("Successful downloads:", successful_downloads)
    print("Raster files inspected:", len(inventory))

    return {
        "products": selected,
        "downloads": download_results,
        "inventory": inventory,
    }


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    results = process_themis()