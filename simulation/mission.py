
"""
ELYSIUM — Mission Orchestration
--------------------------------
Coordinates mission parameters, terrain route planning,
and mission-level reporting.

This module does not perform terrain analysis itself.
It uses the existing terrain_cost and route_planner outputs.
"""

from datetime import datetime, timezone

from simulation.route_planner import DEFAULT_START, plan_route


MISSION_STATUS = "SIMULATED"

MISSION_LIMITATIONS = (
    "This is a regional-scale mission simulation based on available "
    "orbital terrain data. It is not a certified astronaut mobility "
    "or safety assessment."
)


def validate_coordinates(coordinate, name):
    """Validate a geographic coordinate supplied as (latitude, longitude)."""

    if not isinstance(coordinate, (tuple, list)) or len(coordinate) != 2:
        raise ValueError(
            f"{name} must be a (latitude, longitude) pair."
        )

    latitude, longitude = coordinate

    try:
        latitude = float(latitude)
        longitude = float(longitude)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must contain numeric coordinates.")

    if not (-90 <= latitude <= 90):
        raise ValueError(f"{name} latitude must be between -90 and 90.")

    if not (-360 <= longitude <= 360):
        raise ValueError(f"{name} longitude is outside the supported range.")

    return latitude, longitude


def validate_mission_name(mission_name):
    """Validate the mission name."""

    if not isinstance(mission_name, str):
        raise TypeError("Mission name must be a string.")

    mission_name = mission_name.strip()

    if not mission_name:
        raise ValueError("Mission name cannot be empty.")

    return mission_name


def create_mission_summary(route_result):
    """Extract mission-level information from a route-planning result."""

    statistics = route_result["statistics"]

    return {
        "route_distance_km": statistics["distance_km"],
        "cumulative_traversal_cost_km_equivalent": (
            statistics["cumulative_traversal_cost_km_equivalent"]
        ),
        "route_cells": statistics["path_cells"],
        "mean_terrain_cost": statistics["mean_terrain_cost"],
        "maximum_terrain_cost": statistics["maximum_terrain_cost"],
        "minimum_elevation_m": statistics["minimum_elevation_m"],
        "maximum_elevation_m": statistics["maximum_elevation_m"],
        "mean_elevation_m": statistics["mean_elevation_m"],
        "terrain_class_distribution": (
            statistics["terrain_class_distribution"]
        ),
        "expanded_cells": route_result["search"]["expanded_cells"],
        "start_snap_distance_km": (
            route_result["start_snap_distance_km"]
        ),
        "destination_snap_distance_km": (
            route_result["destination_snap_distance_km"]
        ),
    }


def run_mission(
    terrain_results,
    destination,
    start=DEFAULT_START,
    mission_name="ELYSIUM Terrain Exploration",
):
    """
    Run a mission simulation using the existing A* route planner.

    Parameters
    ----------
    terrain_results : dict
        Output from analyse_terrain_cost().

    destination : tuple
        Destination coordinate as (latitude, longitude).

    start : tuple
        Starting coordinate as (latitude, longitude).
        Defaults to the InSight lander location.

    mission_name : str
        Name assigned to the mission.

    Returns
    -------
    dict
        Mission metadata, route results and mission summary.
    """

    mission_name = validate_mission_name(mission_name)
    start = validate_coordinates(start, "Start")
    destination = validate_coordinates(destination, "Destination")

    # Generate the route using the existing A* implementation.
    route_result = plan_route(
        terrain_results=terrain_results,
        start=start,
        destination=destination,
    )

    # Build the mission-level summary.
    summary = create_mission_summary(route_result)

    mission = {
        "mission_name": mission_name,
        "status": MISSION_STATUS,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),

        "mission_parameters": {
            "requested_start": start,
            "requested_destination": destination,
            "actual_start": route_result["start_grid_coordinate"],
            "actual_destination": (
                route_result["destination_grid_coordinate"]
            ),
        },

        "route": route_result,

        "summary": summary,

        "planning": {
            "algorithm": route_result["search"]["algorithm"],
            "movement": route_result["search"]["movement"],
            "expanded_cells": route_result["search"]["expanded_cells"],
        },

        "limitations": MISSION_LIMITATIONS,
    }

    return mission


def generate_mission_report(mission):
    """Generate a readable text report from a mission result."""

    if not isinstance(mission, dict):
        raise TypeError("Mission must be a dictionary.")

    required_keys = (
        "mission_name",
        "status",
        "mission_parameters",
        "summary",
        "planning",
        "limitations",
    )

    missing = [
        key for key in required_keys
        if key not in mission
    ]

    if missing:
        raise KeyError(f"Mission result is missing keys: {missing}")

    parameters = mission["mission_parameters"]
    summary = mission["summary"]
    planning = mission["planning"]

    lines = [
        "=" * 55,
        "ELYSIUM — MISSION REPORT",
        "=" * 55,
        "",
        f"Mission: {mission['mission_name']}",
        f"Status: {mission['status']}",
        "",
        "MISSION COORDINATES",
        f"Requested start: {parameters['requested_start']}",
        f"Actual start: {parameters['actual_start']}",
        f"Requested destination: {parameters['requested_destination']}",
        f"Actual destination: {parameters['actual_destination']}",
        "",
        "ROUTE SUMMARY",
        f"Route distance: {summary['route_distance_km']:.3f} km",
        (
            "Cumulative traversal cost: "
            f"{summary['cumulative_traversal_cost_km_equivalent']:.3f}"
        ),
        f"Route cells: {summary['route_cells']}",
        f"Mean terrain cost: {summary['mean_terrain_cost']:.5f}",
        f"Maximum terrain cost: {summary['maximum_terrain_cost']:.5f}",
        "",
        "ELEVATION",
        f"Minimum: {summary['minimum_elevation_m']:.2f} m",
        f"Maximum: {summary['maximum_elevation_m']:.2f} m",
        f"Mean: {summary['mean_elevation_m']:.2f} m",
        "",
        "PLANNING",
        f"Algorithm: {planning['algorithm']}",
        f"Movement: {planning['movement']}",
        f"Expanded cells: {planning['expanded_cells']}",
        "",
        "COORDINATE SNAPPING",
        (
            "Start: "
            f"{summary['start_snap_distance_km']:.3f} km"
        ),
        (
            "Destination: "
            f"{summary['destination_snap_distance_km']:.3f} km"
        ),
        "",
        "TERRAIN CLASS DISTRIBUTION",
    ]

    for class_id, count in sorted(
        summary["terrain_class_distribution"].items()
    ):
        lines.append(f"Class {class_id}: {count} cells")

    lines.extend([
        "",
        "SCIENTIFIC LIMITATION",
        mission["limitations"],
        "",
        "=" * 55,
    ])

    return "\n".join(lines)


if __name__ == "__main__":
    print("ELYSIUM Mission module loaded successfully.")
    print("Waiting for terrain results and mission coordinates.")
