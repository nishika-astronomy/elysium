
"""
ELYSIUM — CTX Image Processing and Inventory
Version: 0.2.0

Manages Context Camera (CTX) observation metadata,
local image inventory, footprint inventory and PDS3
image validation.

Important:
    CTX EDR images contain raw pixel digital numbers (DN).
    They are not calibrated reflectance products and must
    not be interpreted as map-projected terrain data.

This module does not modify original NASA products.
"""

import json
import os
import re
from pathlib import Path
from collections import Counter

import numpy as np

from config import RAW_DIR, PROCESSED_DIR, CATALOGUE_DIR


# ==================================================
# PATHS AND SETTINGS
# ==================================================

OBSERVATION_CATALOGUE = CATALOGUE_DIR / "observations.json"
FOOTPRINT_DIR = RAW_DIR / "footprints"
CTX_IMAGE_DIR = RAW_DIR

OUTPUT_FILE = PROCESSED_DIR / "ctx_master_catalogue.json"
TEMP_OUTPUT_FILE = PROCESSED_DIR / "ctx_master_catalogue.tmp"

SAMPLE_STEP = 100
LABEL_READ_LIMIT = 4 * 1024 * 1024


# ==================================================
# CATALOGUE LOADING
# ==================================================

def load_observations():
    """
    Load the original NASA ODE observation catalogue.

    Supports a list directly or a dictionary containing
    observations, products or data.
    """

    if not OBSERVATION_CATALOGUE.exists():
        raise FileNotFoundError(
            f"Observation catalogue not found: "
            f"{OBSERVATION_CATALOGUE}"
        )

    with OBSERVATION_CATALOGUE.open(
        "r", encoding="utf-8"
    ) as file:
        data = json.load(file)

    if isinstance(data, list):
        observations = data

    elif isinstance(data, dict):
        observations = None

        for key in ("observations", "products", "data"):
            if isinstance(data.get(key), list):
                observations = data[key]
                break

        if observations is None:
            raise ValueError(
                "Could not identify observation list "
                "in the catalogue."
            )

    else:
        raise ValueError(
            "Unexpected observation catalogue format."
        )

    if not all(isinstance(item, dict) for item in observations):
        raise ValueError(
            "Catalogue contains records that are not dictionaries."
        )

    return observations


# ==================================================
# IDENTIFIERS
# ==================================================

def normalize_id(value):
    """Normalize product identifiers for case-insensitive matching."""

    if value is None:
        return None

    return str(value).strip().upper()


# ==================================================
# FOOTPRINT INVENTORY
# ==================================================

def build_footprint_inventory():
    """
    Discover extracted footprint folders and all nested
    shapefiles.

    Multiple shapefiles may belong to one observation.
    They are counted individually but do not represent
    separate observations.
    """

    inventory = {}

    if not FOOTPRINT_DIR.exists():
        return inventory

    folders = sorted(
        path for path in FOOTPRINT_DIR.rglob("*_footprint")
        if path.is_dir()
    )

    for folder in folders:

        observation_id = normalize_id(
            folder.name[:-len("_footprint")]
        )

        shapefiles = sorted(
            path for path in folder.rglob("*")
            if path.is_file() and path.suffix.lower() == ".shp"
        )

        inventory[observation_id] = {
            "available": bool(shapefiles),
            "shapefile_count": len(shapefiles),
            "shapefiles": [
                str(path) for path in shapefiles
            ],
            "folder": str(folder),
        }

    return inventory


# ==================================================
# CTX IMAGE INVENTORY
# ==================================================


def build_image_inventory():
    """
    Discover local CTX images that match product IDs
    in the NASA ODE observation catalogue.

    Excludes unrelated IMG files such as MOLA rasters.
    """

    inventory = {}

    if not CTX_IMAGE_DIR.exists():
        return inventory

    observations = load_observations()

    catalogue_ids = {
        normalize_id(item.get("pdsid"))
        for item in observations
        if item.get("pdsid")
    }

    for image_path in sorted(CTX_IMAGE_DIR.rglob("*")):

        if not image_path.is_file():
            continue

        if image_path.suffix.lower() != ".img":
            continue

        observation_id = normalize_id(image_path.stem)

        if observation_id not in catalogue_ids:
            continue

        inventory[observation_id] = {
            "available": True,
            "filename": image_path.name,
            "size_bytes": image_path.stat().st_size,
            "path": str(image_path),
        }

    return inventory

# ==================================================
# PDS3 LABEL READER
# ==================================================

def read_pds_label(image_path):
    """
    Read the PDS3 label preceding the image data.

    Returns a dictionary of label key-value pairs.
    """

    image_path = Path(image_path)

    if not image_path.is_file():
        raise FileNotFoundError(
            f"Image file not found: {image_path}"
        )

    with image_path.open("rb") as file:
        raw = file.read(LABEL_READ_LIMIT)

    text = raw.decode("ascii", errors="replace")

    match = re.search(
        r"(?m)^\s*END\s*(?:\r?\n|$)",
        text
    )

    if not match:
        raise ValueError(
            "PDS label END marker not found within "
            f"{LABEL_READ_LIMIT} bytes."
        )

    label_text = text[:match.start()]

    metadata = {}

    for line in label_text.splitlines():

        line = line.strip()

        if not line or line.startswith("/*"):
            continue

        if "=" not in line:
            continue

        key, value = line.split("=", 1)

        # Remove comments without cutting quoted values.
        value = value.strip()

        if "/*" in value:
            value = value.split("/*", 1)[0].strip()

        value = value.strip().strip('"').strip("'")

        metadata[key.strip().upper()] = value

    return metadata


# ==================================================
# PDS IMAGE POINTER
# ==================================================

def calculate_image_offset(metadata):
    """
    Calculate the image-data byte offset from ^IMAGE.

    Supports:
        ^IMAGE = 2
        ^IMAGE = (2 <RECORDS>)
        ^IMAGE = (1024 <BYTES>)
    """

    pointer = metadata.get("^IMAGE")

    if not pointer:
        raise ValueError("Missing ^IMAGE pointer.")

    match = re.search(r"\(\s*(\d+)\s*(?:<\s*(RECORDS|BYTES)\s*>)?", pointer, re.I)

    if match:
        pointer_value = int(match.group(1))
        pointer_unit = (match.group(2) or "RECORDS").upper()

    else:
        match = re.fullmatch(r"\s*(\d+)\s*", pointer)

        if not match:
            raise ValueError(
                f"Invalid ^IMAGE pointer: {pointer}"
            )

        pointer_value = int(match.group(1))
        pointer_unit = "RECORDS"

    if pointer_value < 1:
        raise ValueError(
            "^IMAGE pointer must be one-based and positive."
        )

    if pointer_unit == "BYTES":
        return pointer_value - 1

    record_bytes = int(metadata["RECORD_BYTES"])

    if record_bytes <= 0:
        raise ValueError("Invalid RECORD_BYTES value.")

    return (pointer_value - 1) * record_bytes


# ==================================================
# IMAGE FORMAT VALIDATION
# ==================================================

DTYPE_MAP = {
    ("MSB_INTEGER", 16): ">i2",
    ("LSB_INTEGER", 16): "<i2",
    ("MSB_UNSIGNED_INTEGER", 16): ">u2",
    ("LSB_UNSIGNED_INTEGER", 16): "<u2",
    ("UNSIGNED_INTEGER", 8): "u1",
    ("UNSIGNED_INTEGER", 16): ">u2",
    ("MSB_INTEGER", 32): ">i4",
    ("LSB_INTEGER", 32): "<i4",
    ("IEEE_REAL", 32): ">f4",
    ("PC_REAL", 32): "<f4",
}


def inspect_image(image_path):
    """
    Validate a CTX image's PDS metadata and image layout.

    Samples pixel values only when image rows have no
    prefix or suffix bytes.

    Sample statistics describe raw digital numbers (DN),
    not calibrated reflectance.
    """

    image_path = Path(image_path)
    metadata = read_pds_label(image_path)

    required_fields = (
        "LINES",
        "LINE_SAMPLES",
        "SAMPLE_BITS",
        "SAMPLE_TYPE",
        "RECORD_BYTES",
    )

    missing = [
        field for field in required_fields
        if field not in metadata
    ]

    if missing:
        raise ValueError(
            f"Missing required PDS fields: {missing}"
        )

    lines = int(metadata["LINES"])
    samples = int(metadata["LINE_SAMPLES"])
    sample_bits = int(metadata["SAMPLE_BITS"])
    record_bytes = int(metadata["RECORD_BYTES"])

    sample_type = metadata["SAMPLE_TYPE"].upper()

    line_prefix = int(metadata.get("LINE_PREFIX_BYTES", 0))
    line_suffix = int(metadata.get("LINE_SUFFIX_BYTES", 0))

    if lines <= 0 or samples <= 0:
        raise ValueError(
            "Image dimensions must be positive."
        )

    if record_bytes <= 0:
        raise ValueError(
            "RECORD_BYTES must be positive."
        )

    dtype_string = DTYPE_MAP.get(
        (sample_type, sample_bits)
    )

    if dtype_string is None:
        raise ValueError(
            f"Unsupported image format: "
            f"{sample_type}, {sample_bits} bits."
        )

    dtype = np.dtype(dtype_string)

    image_offset = calculate_image_offset(metadata)

    row_data_bytes = samples * dtype.itemsize

    row_bytes = (
        line_prefix + row_data_bytes + line_suffix
    )

    if row_bytes > record_bytes:
        raise ValueError(
            "Image row exceeds RECORD_BYTES. "
            "Check PDS layout metadata."
        )

    required_file_size = (
        image_offset + (lines - 1) * record_bytes + row_bytes
    )

    actual_file_size = image_path.stat().st_size

    if actual_file_size < required_file_size:
        raise ValueError(
            "Image data is incomplete. "
            f"Expected at least {required_file_size:,} bytes, "
            f"found {actual_file_size:,} bytes."
        )

    sample_min = None
    sample_max = None
    sample_mean = None
    sampling_status = "Skipped: row prefix/suffix present"

    # Direct memory mapping is safe when each image row
    # contains only image pixels and the record stride
    # matches the PDS record length.
    
    if line_prefix == 0 and line_suffix == 0:

        # Map the image as raw bytes first. This supports
        # PDS records with padding between image rows.
        raw_image = np.memmap(
            image_path,
            dtype=np.uint8,
            mode="r",
            offset=image_offset,
            shape=(lines * record_bytes,),
        )

        # Create a typed image view with the correct row stride.
        image = np.ndarray(
            shape=(lines, samples),
            dtype=dtype,
            buffer=raw_image,
            offset=line_prefix,
            strides=(record_bytes, dtype.itemsize),
        )

        sampled = np.asarray(
            image[::SAMPLE_STEP, ::SAMPLE_STEP]
        )

        if sampled.size == 0:
            raise ValueError(
                "No pixel values available for sampling."
            )

        if not np.all(np.isfinite(sampled)):
            raise ValueError(
                "Sampled image contains non-finite values."
            )

        sample_min = float(np.min(sampled))
        sample_max = float(np.max(sampled))
        sample_mean = float(np.mean(sampled))

        del image
        del raw_image

        sampling_status = "Sampled successfully"

    result = {
        "lines": lines,
        "samples": samples,
        "sample_bits": sample_bits,
        "sample_type": sample_type,
        "record_bytes": record_bytes,
        "image_offset_bytes": image_offset,
        "line_prefix_bytes": line_prefix,
        "line_suffix_bytes": line_suffix,
        "start_time": metadata.get("START_TIME"),
        "stop_time": metadata.get("STOP_TIME"),
        "data_quality": metadata.get("DATA_QUALITY_DESC"),
        "instrument_id": metadata.get("INSTRUMENT_ID"),
        "product_id": metadata.get("PRODUCT_ID"),
        "data_set_id": metadata.get("DATA_SET_ID"),
        "sample_min": sample_min,
        "sample_max": sample_max,
        "sample_mean": sample_mean,
        "sample_units": "Raw digital numbers (DN)",
        "sampling_status": sampling_status,
        "validation": "Passed",
    }

    return result


# ==================================================
# SAFE JSON OUTPUT
# ==================================================

def save_master_catalogue(records):
    """
    Write the master catalogue to a temporary file,
    then atomically replace the previous output.
    """

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    try:
        with TEMP_OUTPUT_FILE.open(
            "w", encoding="utf-8"
        ) as file:

            json.dump(
                records,
                file,
                indent=4,
                ensure_ascii=False,
                default=str,
            )

            file.flush()
            os.fsync(file.fileno())

        os.replace(
            TEMP_OUTPUT_FILE,
            OUTPUT_FILE
        )

    finally:
        if TEMP_OUTPUT_FILE.exists():
            TEMP_OUTPUT_FILE.unlink()


# ==================================================
# MASTER CTX CATALOGUE
# ==================================================

def build_ctx_catalogue():
    """
    Build a master catalogue combining:
        - Original NASA ODE observation records
        - Local footprint availability
        - Local CTX image availability
        - Validated CTX PDS metadata
    """

    print("\nELYSIUM | CTX MASTER DATA INVENTORY")
    print("=" * 55)

    observations = load_observations()

    footprint_inventory = build_footprint_inventory()
    image_inventory = build_image_inventory()

    print("Discovered observations:", len(observations))
    print("Local footprint folders:", len(footprint_inventory))
    print("Local CTX image files:", len(image_inventory))

    master_records = []

    for index, observation in enumerate(
        observations, start=1
    ):

        observation_id = normalize_id(
            observation.get("pdsid")
        )

        if not observation_id:
            observation_id = f"UNKNOWN_{index:04d}"
            print(
                f"Observation {index} has no pdsid."
            )

        footprint = footprint_inventory.get(
            observation_id,
            {
                "available": False,
                "shapefile_count": 0,
                "shapefiles": [],
            },
        )

        image = image_inventory.get(
            observation_id,
            {"available": False},
        )

        image_metadata = None
        processing_status = "Not downloaded"

        if image["available"]:

            image_path = Path(image["path"])

            try:
                image_metadata = inspect_image(image_path)

                processing_status = "Metadata validated"

            except Exception as error:
                processing_status = (
                    f"Validation failed: {error}"
                )

        record = {
            "observation_id": observation_id,

            # Preserve the complete original NASA ODE record.
            "source_metadata": observation,

            "footprint": footprint,

            "image": image,

            "image_metadata": image_metadata,

            "download_status": (
                "Downloaded"
                if image["available"]
                else "Not downloaded"
            ),

            "processing_status": processing_status,
        }

        master_records.append(record)

        if index % 50 == 0:
            print(
                f"Processed {index}/{len(observations)} observations"
            )

    save_master_catalogue(master_records)

    # --------------------------------------------------
    # SUMMARY
    # --------------------------------------------------

    total = len(master_records)

    with_footprints = sum(
        item["footprint"]["available"]
        for item in master_records
    )

    downloaded = sum(
        item["image"]["available"]
        for item in master_records
    )

    validated = sum(
        item["processing_status"] == "Metadata validated"
        for item in master_records
    )

    failed = sum(
        item["processing_status"].startswith("Validation failed")
        for item in master_records
    )

    quality_counts = Counter(
        (
            item["image_metadata"].get("data_quality")
            if item["image_metadata"]
            else "Unavailable"
        )
        for item in master_records
    )

    print("\nCTX INVENTORY COMPLETE")
    print("=" * 55)
    print("Total observations:", total)
    print("Observations with footprints:", with_footprints)
    print("Downloaded CTX images:", downloaded)
    print("Images with validated metadata:", validated)
    print("Image validation failures:", failed)
    print("Images not downloaded:", total - downloaded)

    print("\nImage data-quality labels:")
    for quality, count in sorted(quality_counts.items()):
        print(f"  {quality}: {count}")

    print("\nMaster catalogue:", OUTPUT_FILE)

    return master_records


# ==================================================
# MAIN
# ==================================================

if __name__ == "__main__":
    build_ctx_catalogue()
