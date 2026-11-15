
"""
ELYSIUM — Footprint CRS Diagnostic
Version: 0.2.0

Inspects the coordinate reference systems (CRS) of
extracted Mars orbital observation footprint shapefiles.

This script is read-only:
    - Does not modify shapefiles.
    - Does not reproject geometries.
    - Does not modify the observation catalogue.

Checks:
    - CRS identification
    - Missing CRS information
    - Shapefile reading errors
    - CRS grouping across observations
    - Geographic/projected coordinate bounds
"""

from collections import defaultdict
from pathlib import Path

import geopandas as gpd

from config import RAW_DIR


# ==================================================
# CONFIGURATION
# ==================================================

FOOTPRINT_DIR = RAW_DIR / "footprints"

# Display a short WKT preview for each CRS group.
WKT_PREVIEW_LENGTH = 500


# ==================================================
# FOOTPRINT DISCOVERY
# ==================================================

def discover_shapefiles():
    """Find all shapefiles inside extracted footprint folders."""

    if not FOOTPRINT_DIR.exists():
        raise FileNotFoundError(
            f"Footprint directory not found: {FOOTPRINT_DIR}"
        )

    folders = sorted(
        path for path in FOOTPRINT_DIR.glob("*_footprint")
        if path.is_dir()
    )

    shapefiles = []

    for folder in folders:
        shapefiles.extend(
            sorted(folder.rglob("*.shp"))
        )

    return folders, shapefiles


# ==================================================
# CRS ANALYSIS
# ==================================================

def inspect_footprints(shapefiles):
    """Read shapefiles and group them by their CRS."""

    crs_groups = defaultdict(list)
    missing_crs = []
    failed_files = []

    total_empty = 0

    for index, shapefile in enumerate(shapefiles, start=1):

        observation = shapefile.parent.name

        try:
            data = gpd.read_file(shapefile)

            if data.empty:
                total_empty += 1

            if data.crs is None:
                missing_crs.append(shapefile)
                continue

            crs_name = data.crs.to_string()

            bounds = None

            if not data.empty:
                bounds = data.total_bounds.tolist()

            crs_groups[crs_name].append({
                "observation": observation,
                "file": shapefile,
                "bounds": bounds,
                "wkt": data.crs.to_wkt(),
            })

        except Exception as error:

            failed_files.append({
                "file": shapefile,
                "error": str(error),
            })

            print(
                f"[ERROR] {shapefile.name}: {error}"
            )

        if index % 50 == 0:
            print(
                f"Inspected {index}/{len(shapefiles)} shapefiles..."
            )

    return {
        "crs_groups": crs_groups,
        "missing_crs": missing_crs,
        "failed_files": failed_files,
        "empty_files": total_empty,
    }


# ==================================================
# REPORTING
# ==================================================

def print_report(folders, shapefiles, results):
    """Display the footprint CRS diagnostic report."""

    crs_groups = results["crs_groups"]
    missing_crs = results["missing_crs"]
    failed_files = results["failed_files"]

    print("\n" + "=" * 60)
    print("ELYSIUM FOOTPRINT CRS DIAGNOSTIC")
    print("=" * 60)

    print(f"Footprint folders found: {len(folders)}")
    print(f"Shapefiles found: {len(shapefiles)}")
    print(f"CRS groups found: {len(crs_groups)}")
    print(f"Shapefiles without CRS: {len(missing_crs)}")
    print(f"Empty shapefiles: {results['empty_files']}")
    print(f"Failed shapefiles: {len(failed_files)}")

    # --------------------------------------------------
    # CRS GROUPS
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("CRS GROUPS FOUND")
    print("=" * 60)

    if not crs_groups:
        print("No valid CRS information was found.")

    for index, (crs_name, records) in enumerate(
        sorted(crs_groups.items()),
        start=1
    ):

        observations = sorted({
            record["observation"]
            for record in records
        })

        print(f"\nGROUP {index}")
        print("-" * 45)

        print("CRS:", crs_name)
        print("Shapefiles:", len(records))
        print("Unique observations:", len(observations))

        print("Example observation:", observations[0])
        print("Example shapefile:", records[0]["file"].name)

        print("\nWKT:")
        print(
            records[0]["wkt"][:WKT_PREVIEW_LENGTH]
        )

        print("\nExample coordinate bounds:")

        bounds_record = next(
            (
                record for record in records
                if record["bounds"] is not None
            ),
            None
        )

        if bounds_record:
            print(bounds_record["bounds"])
        else:
            print("No non-empty geometry available.")

    # --------------------------------------------------
    # MISSING CRS
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("SHAPEFILES WITHOUT CRS")
    print("=" * 60)

    if missing_crs:

        for shapefile in missing_crs:
            print(shapefile)

    else:
        print("All readable shapefiles contain CRS information.")

    # --------------------------------------------------
    # FAILED FILES
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("SHAPEFILES THAT COULD NOT BE READ")
    print("=" * 60)

    if failed_files:

        for item in failed_files:
            print("\nFile:", item["file"])
            print("Error:", item["error"])

    else:
        print("No shapefile reading errors.")

    # --------------------------------------------------
    # FINAL STATUS
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 60)

    if not failed_files and not missing_crs and crs_groups:
        print("All readable shapefiles have CRS information.")

    elif missing_crs or failed_files:
        print("Review the missing-CRS and error reports.")

    else:
        print("No CRS groups could be established.")


# ==================================================
# MAIN
# ==================================================

def main():
    """Run the footprint CRS diagnostic."""

    print("Discovering extracted footprint shapefiles...")

    folders, shapefiles = discover_shapefiles()

    print(f"Found {len(folders)} footprint folders.")
    print(f"Found {len(shapefiles)} shapefiles.")

    if not shapefiles:
        print("No shapefiles found. Nothing to inspect.")
        return

    print("\nInspecting coordinate reference systems...")

    results = inspect_footprints(shapefiles)

    print_report(
        folders,
        shapefiles,
        results
    )


if __name__ == "__main__":
    main()
