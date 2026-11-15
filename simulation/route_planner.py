
"""
ELYSIUM — Terrain Route Planner
--------------------------------
A* route planning over the MOLA-derived terrain cost surface.

Routes are regional-scale simulations, not certified astronaut
mobility or safety assessments.
"""

import heapq
import math

import numpy as np


MARS_RADIUS_KM = 3396.19

DEFAULT_START = (4.502, 135.623)  # latitude, longitude

MOVEMENT_DIRECTIONS = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
)


def validate_route_inputs(terrain_results):
    """Validate the output from analyse_terrain_cost()."""

    if not isinstance(terrain_results, dict):
        raise TypeError("Terrain results must be a dictionary.")

    required_keys = (
        "terrain_cost",
        "terrain_classes",
        "latitude",
        "longitude",
        "elevation_m",
        "valid_mask",
        "grid_spacing_km",
    )

    missing = [
        key for key in required_keys
        if key not in terrain_results
    ]

    if missing:
        raise KeyError(f"Missing terrain result keys: {missing}")

    arrays = {
        key: np.asarray(terrain_results[key])
        for key in (
            "terrain_cost",
            "terrain_classes",
            "latitude",
            "longitude",
            "elevation_m",
            "valid_mask",
        )
    }

    shape = arrays["terrain_cost"].shape

    if len(shape) != 2 or min(shape) < 2:
        raise ValueError("Terrain grid must be a 2D array of at least 2x2.")

    for name, array in arrays.items():
        if array.shape != shape:
            raise ValueError(
                f"{name} has shape {array.shape}; expected {shape}."
            )

    spacing = terrain_results["grid_spacing_km"]

    if not isinstance(spacing, dict):
        raise TypeError("grid_spacing_km must be a dictionary.")

    for direction in ("east_west_km", "north_south_km"):
        value = spacing.get(direction)

        if value is None or not np.isfinite(value) or value <= 0:
            raise ValueError(f"Invalid grid spacing: {direction}.")

    cost = arrays["terrain_cost"].astype(float)
    latitude = arrays["latitude"].astype(float)
    longitude = arrays["longitude"].astype(float)
    elevation = arrays["elevation_m"].astype(float)
    classes = arrays["terrain_classes"]

    if not np.issubdtype(classes.dtype, np.number):
        raise TypeError("Terrain classes must be numeric.")

    if np.any(
        np.isfinite(cost) &
        ((cost < 0) | (cost > 1))
    ):
        raise ValueError("Terrain costs must be between 0 and 1.")

    if np.any(
        np.isfinite(latitude) &
        ((latitude < -90) | (latitude > 90))
    ):
        raise ValueError("Latitude values must be between -90 and 90.")

    if np.any(
        np.isfinite(longitude) &
        ((longitude < -360) | (longitude > 360))
    ):
        raise ValueError("Longitude values are outside the supported range.")

    valid_mask = (
        arrays["valid_mask"].astype(bool)
        & np.isfinite(cost)
        & np.isfinite(latitude)
        & np.isfinite(longitude)
        & np.isfinite(elevation)
        & np.isfinite(classes)
        & (classes >= 0)
    )

    if not np.any(valid_mask):
        raise ValueError("No valid terrain cells are available for routing.")

    return {
        "terrain_cost": cost,
        "terrain_classes": classes,
        "latitude": latitude,
        "longitude": longitude,
        "elevation_m": elevation,
        "valid_mask": valid_mask,
        "grid_spacing_km": spacing,
    }


def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate great-circle distance on Mars, in kilometres."""

    lat1, lon1, lat2, lon2 = map(
        math.radians, (lat1, lon1, lat2, lon2)
    )

    delta_lat = lat2 - lat1
    delta_lon = lon2 - lon1

    a = (
        math.sin(delta_lat / 2) ** 2
        + math.cos(lat1)
        * math.cos(lat2)
        * math.sin(delta_lon / 2) ** 2
    )

    a = min(1.0, max(0.0, a))

    return (
        2 * MARS_RADIUS_KM
        * math.asin(math.sqrt(a))
    )


def find_nearest_valid_cell(latitude, longitude, valid_mask, coordinate):
    """Find the nearest valid grid cell to a geographic coordinate."""

    target_lat, target_lon = coordinate

    if not (
        np.isfinite(target_lat)
        and np.isfinite(target_lon)
        and -90 <= target_lat <= 90
        and -360 <= target_lon <= 360
    ):
        raise ValueError("Invalid geographic coordinate.")

    valid_lats = latitude[valid_mask]
    valid_lons = longitude[valid_mask]

    if not (
        valid_lats.min() <= target_lat <= valid_lats.max()
        and valid_lons.min() <= target_lon <= valid_lons.max()
    ):
        raise ValueError(
            "Requested coordinate is outside the valid terrain grid."
        )

    distances = np.full(latitude.shape, np.inf)

    distances[valid_mask] = np.vectorize(
        haversine_distance
    )(
        target_lat,
        target_lon,
        valid_lats,
        valid_lons,
    )

    index = np.unravel_index(
        np.argmin(distances),
        distances.shape,
    )

    return index, float(distances[index])


def calculate_edge_cost(
    current,
    neighbour,
    terrain_cost,
    latitude,
    longitude,
):
    """
    Calculate movement cost between two adjacent cells.

    Edge cost = physical distance x (1 + mean terrain difficulty).

    This preserves a distance penalty while accounting for
    relative terrain difficulty.
    """

    r1, c1 = current
    r2, c2 = neighbour

    distance = haversine_distance(
        latitude[r1, c1],
        longitude[r1, c1],
        latitude[r2, c2],
        longitude[r2, c2],
    )

    mean_terrain_cost = (
        terrain_cost[r1, c1]
        + terrain_cost[r2, c2]
    ) / 2

    return distance * (1 + mean_terrain_cost)


def reconstruct_path(came_from, current):
    """Reconstruct the route from the A* predecessor map."""

    path = [current]

    while current in came_from:
        current = came_from[current]
        path.append(current)

    path.reverse()
    return path


def astar_search(
    terrain_cost,
    latitude,
    longitude,
    valid_mask,
    start,
    goal,
    max_expansions=1000000,
):
    """Find a minimum-cost route using A*."""

    def heuristic(cell):
        r, c = cell
        gr, gc = goal

        return haversine_distance(
            latitude[r, c],
            longitude[r, c],
            latitude[gr, gc],
            longitude[gr, gc],
        )

    open_heap = [(heuristic(start), 0.0, start)]

    came_from = {}
    g_score = {start: 0.0}
    closed = set()

    expansions = 0

    while open_heap:

        _, current_cost, current = heapq.heappop(open_heap)

        if current in closed:
            continue

        if current_cost > g_score.get(current, math.inf):
            continue

        if current == goal:
            return reconstruct_path(came_from, current), expansions

        closed.add(current)
        expansions += 1

        if expansions > max_expansions:
            raise RuntimeError(
                "A* exceeded the maximum number of cell expansions."
            )

        r, c = current

        for dr, dc in MOVEMENT_DIRECTIONS:
            nr, nc = r + dr, c + dc
            neighbour = (nr, nc)

            if not (
                0 <= nr < valid_mask.shape[0]
                and 0 <= nc < valid_mask.shape[1]
            ):
                continue

            if not valid_mask[nr, nc]:
                continue

            # Prevent diagonal movement through blocked corners.
            if dr != 0 and dc != 0:
                if not (
                    valid_mask[r + dr, c]
                    and valid_mask[r, c + dc]
                ):
                    continue

            if neighbour in closed:
                continue

            edge_cost = calculate_edge_cost(
                current,
                neighbour,
                terrain_cost,
                latitude,
                longitude,
            )

            tentative_cost = current_cost + edge_cost

            if tentative_cost < g_score.get(neighbour, math.inf):
                came_from[neighbour] = current
                g_score[neighbour] = tentative_cost

                priority = tentative_cost + heuristic(neighbour)

                heapq.heappush(
                    open_heap,
                    (priority, tentative_cost, neighbour),
                )

    raise RuntimeError("No valid route could be found.")


def calculate_route_statistics(path, results):
    """Calculate route distance, terrain and elevation statistics."""

    costs = np.array([
        results["terrain_cost"][r, c]
        for r, c in path
    ])

    elevations = np.array([
        results["elevation_m"][r, c]
        for r, c in path
    ])

    classes = np.array([
        int(results["terrain_classes"][r, c])
        for r, c in path
    ])

    route_distance = sum(
        haversine_distance(
            results["latitude"][r1, c1],
            results["longitude"][r1, c1],
            results["latitude"][r2, c2],
            results["longitude"][r2, c2],
        )
        for (r1, c1), (r2, c2) in zip(path[:-1], path[1:])
    )

    class_distribution = {
        int(class_id): int(np.sum(classes == class_id))
        for class_id in np.unique(classes)
    }

    return {
        "distance_km": float(route_distance),
        "cumulative_traversal_cost_km_equivalent": float(
            sum(
                calculate_edge_cost(
                    first,
                    second,
                    results["terrain_cost"],
                    results["latitude"],
                    results["longitude"],
                )
                for first, second in zip(path[:-1], path[1:])
            )
        ),
        "mean_terrain_cost": float(np.mean(costs)),
        "maximum_terrain_cost": float(np.max(costs)),
        "minimum_terrain_cost": float(np.min(costs)),
        "minimum_elevation_m": float(np.min(elevations)),
        "maximum_elevation_m": float(np.max(elevations)),
        "mean_elevation_m": float(np.mean(elevations)),
        "terrain_class_distribution": class_distribution,
        "path_cells": len(path),
    }


def plan_route(
    terrain_results,
    destination,
    start=DEFAULT_START,
    max_expansions=1000000,
):
    """
    Plan a route from the InSight site (by default) to a destination.

    Coordinates must be supplied as (latitude, longitude).

    Returns route coordinates, distance, traversal cost,
    terrain statistics and start/end snapping information.
    """

    results = validate_route_inputs(terrain_results)

    if (
        not isinstance(start, (tuple, list))
        or len(start) != 2
        or not isinstance(destination, (tuple, list))
        or len(destination) != 2
    ):
        raise ValueError(
            "Start and destination must be (latitude, longitude) pairs."
        )

    start_cell, start_snap = find_nearest_valid_cell(
        results["latitude"],
        results["longitude"],
        results["valid_mask"],
        start,
    )

    goal_cell, goal_snap = find_nearest_valid_cell(
        results["latitude"],
        results["longitude"],
        results["valid_mask"],
        destination,
    )

    path, expansions = astar_search(
        results["terrain_cost"],
        results["latitude"],
        results["longitude"],
        results["valid_mask"],
        start_cell,
        goal_cell,
        max_expansions,
    )

    route_coordinates = [
        {
            "latitude": float(results["latitude"][r, c]),
            "longitude": float(results["longitude"][r, c]),
            "elevation_m": float(results["elevation_m"][r, c]),
            "terrain_cost": float(results["terrain_cost"][r, c]),
            "terrain_class": int(results["terrain_classes"][r, c]),
            "grid_index": (int(r), int(c)),
        }
        for r, c in path
    ]

    statistics = calculate_route_statistics(path, results)

    return {
        "route_coordinates": route_coordinates,
        "route_indices": path,
        "start_requested": tuple(start),
        "destination_requested": tuple(destination),
        "start_grid_coordinate": (
            float(results["latitude"][start_cell]),
            float(results["longitude"][start_cell]),
        ),
        "destination_grid_coordinate": (
            float(results["latitude"][goal_cell]),
            float(results["longitude"][goal_cell]),
        ),
        "start_snap_distance_km": start_snap,
        "destination_snap_distance_km": goal_snap,
        "statistics": statistics,
        "search": {
            "algorithm": "A*",
            "movement": "8-directional",
            "expanded_cells": expansions,
        },
        "model_note": (
            "Relative regional-scale route simulation. "
            "Not a certified astronaut safety assessment."
        ),
    }
