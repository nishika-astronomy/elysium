
"""
ELYSIUM — Scientific Data Downloader
Version: 0.2.0

Downloads original Mars orbital observation products and
geographic footprint archives from NASA Mars ODE.

Supported footprint formats:
    - ZIP
    - TAR.GZ
    - TGZ

Safety:
    - HTTPS-only downloads
    - Product ID validation
    - Streaming downloads
    - Temporary files before final replacement
    - Archive integrity validation
    - ZIP and TAR path traversal protection
    - Rejects archive links and special files
    - Validates extracted shapefile components

Original .IMG products are downloaded separately from
geographic footprint archives.
"""

import json
import os
import re
import shutil
import tarfile
import zipfile

from pathlib import Path
from urllib.parse import urlparse, unquote

import requests

from config import RAW_DIR
from data_engine.catalogue import CATALOGUE_FILE


# ==================================================
# PATHS AND SETTINGS
# ==================================================

FOOTPRINT_DIR = RAW_DIR / "footprints"

REQUEST_TIMEOUT = 120
CHUNK_SIZE = 1024 * 1024

SUPPORTED_ARCHIVES = (".tar.gz", ".tgz", ".zip")


# ==================================================
# VALIDATION HELPERS
# ==================================================

def validate_product_id(product_id: str) -> str:
    """Validate a NASA product identifier before using it in a path."""

    if not isinstance(product_id, str):
        raise ValueError("Product ID must be a string.")

    product_id = product_id.strip()

    if not product_id or not re.fullmatch(
        r"[A-Za-z0-9_-]+", product_id
    ):
        raise ValueError(f"Invalid product ID: {product_id!r}")

    return product_id


def validate_url(url: str) -> str:
    """Accept only valid HTTPS download URLs."""

    if not isinstance(url, str) or not url.strip():
        raise ValueError("Download URL is missing.")

    parsed = urlparse(url.strip())

    if parsed.scheme.lower() != "https":
        raise ValueError("Only HTTPS download URLs are permitted.")

    if not parsed.hostname:
        raise ValueError("Download URL has no hostname.")

    if parsed.username or parsed.password:
        raise ValueError("URLs containing credentials are not permitted.")

    return url.strip()


def get_archive_extension(url: str) -> str:
    """Determine the supported archive extension from its URL path."""

    path = unquote(urlparse(url).path).lower()

    for extension in SUPPORTED_ARCHIVES:
        if path.endswith(extension):
            return extension

    raise ValueError(
        f"Unsupported footprint archive format: {path}"
    )


# ==================================================
# CATALOGUE
# ==================================================

def load_catalogue() -> list:
    """Load selected observations from the JSON catalogue."""

    if not CATALOGUE_FILE.exists():
        raise FileNotFoundError(
            "Observation catalogue not found. Run discovery first."
        )

    with open(CATALOGUE_FILE, "r", encoding="utf-8") as file:
        observations = json.load(file)

    if not isinstance(observations, list):
        raise ValueError(
            "Observation catalogue must contain a JSON list."
        )

    return observations


# ==================================================
# NASA ODE PRODUCT METADATA
# ==================================================

def get_product_files(observation: dict) -> list:
    """Normalize NASA ODE Product_files into a list."""

    if not isinstance(observation, dict):
        return []

    files_data = observation.get("Product_files", [])

    if isinstance(files_data, dict):
        files_list = files_data.get("Product_file", [])

    elif isinstance(files_data, list):
        files_list = files_data

    else:
        return []

    if isinstance(files_list, dict):
        files_list = [files_list]

    if not isinstance(files_list, list):
        return []

    return [
        item for item in files_list
        if isinstance(item, dict)
    ]


def get_product_download_url(observation: dict):
    """Find the original data-product URL."""

    for file_info in get_product_files(observation):

        file_type = str(
            file_info.get("Type", "")
        ).strip().lower()

        url = file_info.get("URL")

        if file_type == "product" and url:
            return validate_url(url)

    return None


def get_footprint_archive_url(observation: dict):
    """
    Find the geographic footprint archive URL.

    Prefer the explicitly labelled footprint shapefile.
    Supports ZIP, TAR.GZ and TGZ.
    """

    files = get_product_files(observation)

    # First pass: explicitly labelled footprint archives.
    for file_info in files:

        description = str(
            file_info.get("Description", "")
        ).upper()

        filename = str(
            file_info.get("FileName", "")
        ).lower()

        url = file_info.get("URL")

        if not url:
            continue

        is_footprint = (
            "FOOTPRINT" in description
            or "SHAPEFILE" in description
            or "FOOTPRINT" in filename
        )

        if is_footprint:
            try:
                get_archive_extension(url)
                return validate_url(url)
            except ValueError:
                continue

    # Second pass: archive filename fallback.
    for file_info in files:

        url = file_info.get("URL")

        if not url:
            continue

        try:
            get_archive_extension(url)
            return validate_url(url)
        except ValueError:
            continue

    return None


def get_footprint_zip_url(observation: dict):
    """
    Backward-compatible function name.

    Returns ZIP, TAR.GZ or TGZ URLs.
    """

    return get_footprint_archive_url(observation)


# ==================================================
# STREAMING DOWNLOAD
# ==================================================

def stream_download(url: str, destination: Path) -> Path:
    """Download a file into a temporary path, then replace destination."""

    url = validate_url(url)

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    temporary = destination.with_name(
        destination.name + ".part"
    )

    try:
        with requests.get(
            url,
            stream=True,
            timeout=REQUEST_TIMEOUT
        ) as response:

            response.raise_for_status()

            content_type = response.headers.get(
                "Content-Type", ""
            ).lower()

            if "text/html" in content_type:
                raise ValueError(
                    "Server returned HTML instead of a data file."
                )

            expected_header = response.headers.get(
                "Content-Length"
            )

            try:
                expected_size = (
                    int(expected_header)
                    if expected_header
                    else None
                )
            except ValueError:
                expected_size = None

            downloaded_size = 0

            with open(temporary, "wb") as output:

                for chunk in response.iter_content(
                    chunk_size=CHUNK_SIZE
                ):
                    if chunk:
                        output.write(chunk)
                        downloaded_size += len(chunk)

            if downloaded_size == 0:
                raise ValueError("Downloaded file is empty.")

            if (
                expected_size is not None
                and downloaded_size != expected_size
            ):
                raise ValueError(
                    "Downloaded file size does not match "
                    "the server's Content-Length."
                )

        os.replace(temporary, destination)

        return destination

    except Exception:

        if temporary.exists():
            temporary.unlink()

        raise


# ==================================================
# ARCHIVE VALIDATION
# ==================================================

def is_valid_archive(archive_path: Path) -> bool:
    """Check archive format and integrity."""

    archive_path = Path(archive_path)

    if not archive_path.is_file():
        return False

    filename = archive_path.name.lower()

    try:

        if filename.endswith(".zip"):

            if not zipfile.is_zipfile(archive_path):
                return False

            with zipfile.ZipFile(archive_path, "r") as archive:
                return archive.testzip() is None

        if filename.endswith((".tar.gz", ".tgz")):

            if not tarfile.is_tarfile(archive_path):
                return False

            with tarfile.open(archive_path, "r:gz") as archive:

                for member in archive.getmembers():

                    if member.isfile():

                        source = archive.extractfile(member)

                        if source is None:
                            return False

                        with source:
                            while source.read(CHUNK_SIZE):
                                pass

            return True

    except (
        OSError,
        EOFError,
        tarfile.TarError,
        zipfile.BadZipFile,
        RuntimeError,
    ):
        return False

    return False


# ==================================================
# FOOTPRINT DOWNLOAD
# ==================================================

def download_footprint(url: str, product_id: str) -> Path:
    """Download and validate a ZIP or TAR.GZ footprint archive."""

    product_id = validate_product_id(product_id)
    url = validate_url(url)

    extension = get_archive_extension(url)

    FOOTPRINT_DIR.mkdir(parents=True, exist_ok=True)

    destination = FOOTPRINT_DIR / (
        f"{product_id}_footprint{extension}"
    )

    if destination.exists():

        if is_valid_archive(destination):
            print(f"Footprint archive already downloaded: {destination}")
            return destination

        print("Existing archive is invalid. Re-downloading.")
        destination.unlink()

    print(f"Downloading footprint: {product_id}")

    stream_download(url, destination)

    if not is_valid_archive(destination):
        destination.unlink(missing_ok=True)

        raise ValueError(
            "Downloaded footprint archive failed integrity validation."
        )

    print(f"Footprint saved: {destination}")

    return destination


# ==================================================
# SHAPEFILE VALIDATION
# ==================================================

def is_footprint_extracted(extract_dir: Path) -> bool:
    """
    Check for a complete shapefile component set.

    A shapefile requires at least:
        .shp — geometry
        .shx — geometry index
        .dbf — attribute table
    """

    extract_dir = Path(extract_dir)

    if not extract_dir.exists():
        return False

    for shapefile in extract_dir.rglob("*"):

        if not shapefile.is_file():
            continue

        if shapefile.suffix.lower() != ".shp":
            continue

        required = {".shp", ".shx", ".dbf"}

        available = {
            item.suffix.lower()
            for item in shapefile.parent.iterdir()
            if item.is_file()
            and item.stem.lower() == shapefile.stem.lower()
        }

        if required.issubset(available):
            return True

    return False


# ==================================================
# SAFE ARCHIVE EXTRACTION
# ==================================================

def _safe_archive_target(root: Path, member_name: str) -> Path:
    """Resolve an archive member path and prevent path traversal."""

    root = Path(root).resolve()

    # Archive paths use POSIX separators, even on Windows.
    normalized = member_name.replace("\\", "/")

    if normalized.startswith("/") or re.match(
        r"^[A-Za-z]:", normalized
    ):
        raise ValueError(
            f"Absolute archive path is not permitted: {member_name}"
        )

    parts = Path(normalized).parts

    if ".." in parts:
        raise ValueError(
            f"Path traversal detected: {member_name}"
        )

    target = (root / normalized).resolve()

    if target != root and root not in target.parents:
        raise ValueError(
            f"Unsafe archive path detected: {member_name}"
        )

    return target


def _extract_zip_safely(
    archive_path: Path,
    destination: Path
):
    """Extract ZIP members without allowing links or unsafe paths."""

    with zipfile.ZipFile(archive_path, "r") as archive:

        for member in archive.infolist():

            target = _safe_archive_target(
                destination,
                member.filename
            )

            unix_mode = member.external_attr >> 16
            file_type = unix_mode & 0o170000

            if file_type == 0o120000:
                raise ValueError(
                    f"ZIP symbolic link not permitted: {member.filename}"
                )

            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue

            if file_type not in (0, 0o100000):
                raise ValueError(
                    f"Unsupported ZIP member type: {member.filename}"
                )

            target.parent.mkdir(parents=True, exist_ok=True)

            with archive.open(member, "r") as source:
                with open(target, "wb") as output:
                    shutil.copyfileobj(source, output)


def _extract_tar_safely(
    archive_path: Path,
    destination: Path
):
    """Extract TAR.GZ/TGZ files without links or special files."""

    with tarfile.open(archive_path, "r:gz") as archive:

        for member in archive.getmembers():

            target = _safe_archive_target(
                destination,
                member.name
            )

            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue

            if not member.isfile():
                raise ValueError(
                    f"Unsupported TAR member type: {member.name}"
                )

            target.parent.mkdir(parents=True, exist_ok=True)

            source = archive.extractfile(member)

            if source is None:
                raise ValueError(
                    f"Could not read TAR member: {member.name}"
                )

            with source:
                with open(target, "wb") as output:
                    shutil.copyfileobj(source, output)


def extract_footprint(archive_path: Path) -> Path:
    """
    Safely extract a verified footprint archive.

    The final folder is replaced only after extraction
    and shapefile validation succeed.
    """

    archive_path = Path(archive_path)

    if not archive_path.is_file():
        raise FileNotFoundError(archive_path)

    if not is_valid_archive(archive_path):
        raise ValueError("Footprint archive is invalid.")

    filename = archive_path.name.lower()

    if filename.endswith(".tar.gz"):
        product_id = archive_path.name[:-7]

    elif filename.endswith(".tgz"):
        product_id = archive_path.name[:-4]

    elif filename.endswith(".zip"):
        product_id = archive_path.stem

    else:
        raise ValueError("Unsupported archive extension.")

    if product_id.endswith("_footprint"):
        product_id = product_id[:-10]

    product_id = validate_product_id(product_id)

    extract_dir = archive_path.parent / (
        f"{product_id}_footprint"
    )

    temporary_dir = archive_path.parent / (
        f"{product_id}_footprint.extracting"
    )

    backup_dir = archive_path.parent / (
        f"{product_id}_footprint.backup"
    )

    if is_footprint_extracted(extract_dir):
        print(f"Footprint already extracted: {extract_dir}")
        return extract_dir

    if temporary_dir.exists():
        shutil.rmtree(temporary_dir)

    temporary_dir.mkdir(parents=True)

    try:

        if filename.endswith(".zip"):
            _extract_zip_safely(
                archive_path,
                temporary_dir
            )

        else:
            _extract_tar_safely(
                archive_path,
                temporary_dir
            )

        if not is_footprint_extracted(temporary_dir):
            raise ValueError(
                "Extraction finished, but a complete shapefile "
                "component set (.shp, .shx, .dbf) was not found."
            )

        if backup_dir.exists():
            shutil.rmtree(backup_dir)

        if extract_dir.exists():
            os.replace(extract_dir, backup_dir)

        try:
            os.replace(temporary_dir, extract_dir)

        except Exception:

            if backup_dir.exists():
                os.replace(backup_dir, extract_dir)

            raise

        if backup_dir.exists():
            shutil.rmtree(backup_dir)

        print(f"Footprint extracted to: {extract_dir}")

        return extract_dir

    except Exception:

        if temporary_dir.exists():
            shutil.rmtree(temporary_dir)

        raise


# ==================================================
# ORIGINAL PRODUCT DOWNLOAD
# ==================================================

def download_product(url: str, product_id: str) -> Path:
    """Download an original observation product such as .IMG."""

    product_id = validate_product_id(product_id)
    url = validate_url(url)

    filename = Path(
        unquote(urlparse(url).path)
    ).name

    if not filename:
        raise ValueError(
            f"Invalid download URL for {product_id}"
        )

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    destination = RAW_DIR / filename

    if destination.exists() and destination.stat().st_size > 0:
        print(f"Already downloaded: {filename}")
        return destination

    print(f"Downloading original product: {product_id}")
    print(f"File: {filename}")

    stream_download(url, destination)

    print(f"Saved: {destination}")
    print(
        f"Size: {destination.stat().st_size / (1024 ** 2):.2f} MB"
    )

    return destination


# ==================================================
# ORIGINAL PRODUCT DOWNLOAD MANAGER
# ==================================================

def download_all_observations(max_observations=None):
    """
    Download original observation products.

    This function is intentionally not run automatically.
    """

    observations = load_catalogue()

    if max_observations is not None:

        if not isinstance(max_observations, int) or max_observations < 1:
            raise ValueError(
                "max_observations must be a positive integer."
            )

        observations = observations[:max_observations]

    print("\nELYSIUM ORIGINAL PRODUCT DOWNLOAD")
    print("=" * 45)
    print("Observations to process:", len(observations))

    successful = 0
    skipped = 0
    failed = 0

    for index, observation in enumerate(observations, start=1):

        product_id = observation.get("pdsid", "unknown")

        print(f"\n[{index}/{len(observations)}] {product_id}")

        try:
            url = get_product_download_url(observation)

            if not url:
                print("No original product URL found.")
                skipped += 1
                continue

            download_product(url, product_id)
            successful += 1

        except Exception as error:
            print(f"Download failed: {error}")
            failed += 1

    summary = {
        "successful": successful,
        "skipped": skipped,
        "failed": failed,
    }

    print("\nORIGINAL PRODUCT DOWNLOAD COMPLETE")
    print(summary)

    return summary


# ==================================================
# FOOTPRINT DOWNLOAD MANAGER
# ==================================================

def download_all_footprints(max_observations=None):
    """
    Download and extract geographic footprints.

    Supports ZIP, TAR.GZ and TGZ archives.

    Does not download original image products.
    """

    observations = load_catalogue()

    if max_observations is not None:

        if not isinstance(max_observations, int) or max_observations < 1:
            raise ValueError(
                "max_observations must be a positive integer."
            )

        observations = observations[:max_observations]

    FOOTPRINT_DIR.mkdir(parents=True, exist_ok=True)

    successful = 0
    already_available = 0
    unavailable = 0
    failed = 0

    print("\nELYSIUM FOOTPRINT ACQUISITION")
    print("=" * 45)
    print("Observations to process:", len(observations))

    for index, observation in enumerate(observations, start=1):

        product_id = observation.get("pdsid", "unknown")

        print(f"\n[{index}/{len(observations)}] {product_id}")

        try:
            product_id = validate_product_id(product_id)

            extract_dir = FOOTPRINT_DIR / (
                f"{product_id}_footprint"
            )

            # Reuse complete extracted footprints.
            if is_footprint_extracted(extract_dir):
                print("Footprint already available.")
                already_available += 1
                continue

            url = get_footprint_archive_url(observation)

            if not url:
                print("No footprint archive available.")
                unavailable += 1
                continue

            extension = get_archive_extension(url)

            archive_path = FOOTPRINT_DIR / (
                f"{product_id}_footprint{extension}"
            )

            # Reuse a valid archive.
            if archive_path.exists():

                if is_valid_archive(archive_path):
                    print("Using existing footprint archive.")

                else:
                    print("Existing archive is invalid.")
                    archive_path.unlink()

            if not archive_path.exists():
                archive_path = download_footprint(
                    url,
                    product_id
                )

            extract_footprint(archive_path)

            successful += 1

        except Exception as error:
            print(f"Footprint processing failed: {error}")
            failed += 1

    summary = {
        "successfully_processed": successful,
        "already_available": already_available,
        "unavailable": unavailable,
        "failed": failed,
    }

    print("\nFOOTPRINT ACQUISITION COMPLETE")
    print(summary)

    return summary


# ==================================================
# MAIN
# ==================================================

if __name__ == "__main__":
    download_all_footprints()
