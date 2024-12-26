import numpy as np
from math import sqrt
from collections import defaultdict
import matplotlib.pyplot as plt
import itertools
from heapq import heappop, heappush


class Coordinate:
    def __init__(self, x_val, y_val):
        self.x = x_val
        self.y = y_val
        self.position = (x_val, y_val)

    def __getitem__(self, idx):
        return self.position[idx]

    def __repr__(self):
        return f"{self[0]},{self[1]}"


class NeighborRegion:
    def __init__(self, coord_list, radius_limit, dist_func):
        self.coord_list = coord_list
        self.radius_limit = radius_limit
        self.adjacency = defaultdict(list)
        self.closest_dist = {}
        self._initialize(dist_func)

    def _initialize(self, dist_func):
        for coord_i in self.coord_list:
            min_distance = float("inf")
            for coord_j in self.coord_list:
                if coord_i == coord_j:
                    continue
                distance = dist_func(coord_i, coord_j)
                if distance <= self.radius_limit:
                    self.adjacency[coord_i].append(coord_j)
                if distance < min_distance:
                    min_distance = distance
            self.closest_dist[coord_i] = min_distance

    def get_neighbors(self, coord):
        return self.adjacency[coord]


def compute_distance(point1, point2):
    return np.sqrt(np.sum((np.array([point1.x - point2.x, point1.y - point2.y]) ** 2)))


def distance_2d(p1, p2):
    return sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def cluster_extraction(ordered_points, reach_threshold, min_points):
    cluster_list = []
    boundary_indices = []
    for idx, point in enumerate(ordered_points):
        rd = (
            point.reachability_distance if point.reachability_distance else float("inf")
        )
        if rd > reach_threshold:
            boundary_indices.append(idx)

    boundary_indices.append(len(ordered_points))

    for i in range(len(boundary_indices) - 1):
        start, end = boundary_indices[i], boundary_indices[i + 1]
        if end - start >= min_points:
            cluster_list.append(ordered_points[start:end])

    return cluster_list


def perform_optics(coord_points, radius, min_points, dist_func=compute_distance):
    ordered_sequence = []
    region = NeighborRegion(coord_points, radius, dist_func)
    unvisited = coord_points[:]

    # Initialize points
    for coord in coord_points:
        coord.processed = False
        coord.core_distance = None
        coord.reachability_distance = None

    # Priority queue related variables
    counter = itertools.count()
    REMOVED_MARKER = "<removed>"

    def insert_seed(pq, coord, finder, reach_dist):
        if coord in finder:
            delete_seed(coord, finder)
        count = next(counter)
        entry = [reach_dist, count, coord]
        finder[coord] = entry
        heappush(pq, entry)

    def delete_seed(coord, finder):
        entry = finder.pop(coord)
        entry[-1] = REMOVED_MARKER

    def extract_min_seed(pq, finder):
        while pq:
            rd, _, coord = heappop(pq)
            if coord is not REMOVED_MARKER:
                del finder[coord]
                return coord
        return None

    def update_reach_dist(neighbors, current_coord, core_dist, seeds_pq, seed_finder):
        for neighbor in neighbors:
            if neighbor.processed:
                continue
            new_reach = max(core_dist, dist_func(current_coord, neighbor))
            if neighbor.reachability_distance is None:
                neighbor.reachability_distance = new_reach
                insert_seed(seeds_pq, neighbor, seed_finder, new_reach)
            elif new_reach < neighbor.reachability_distance:
                neighbor.reachability_distance = new_reach
                insert_seed(seeds_pq, neighbor, seed_finder, new_reach)

    def determine_core_distance(coord, neighbors, min_pts_required):
        if coord.core_distance is not None:
            return coord.core_distance
        if len(neighbors) >= min_pts_required - 1:
            sorted_dists = sorted([dist_func(n, coord) for n in neighbors])
            coord.core_distance = sorted_dists[min_pts_required - 2]
            return coord.core_distance
        return None

    while unvisited:
        current = unvisited.pop()
        if current.processed:
            continue

        current.processed = True
        ordered_sequence.append(current)

        neighbors = region.get_neighbors(current)
        core_distance = determine_core_distance(current, neighbors, min_points)

        if core_distance is not None:
            seed_queue = []
            seed_finder = {}
            update_reach_dist(
                neighbors, current, core_distance, seed_queue, seed_finder
            )
            while seed_queue:
                next_seed = extract_min_seed(seed_queue, seed_finder)
                if next_seed is None:
                    continue
                next_seed.processed = True
                ordered_sequence.append(next_seed)
                next_neighbors = region.get_neighbors(next_seed)
                next_core_dist = determine_core_distance(
                    next_seed, next_neighbors, min_points
                )
                if next_core_dist is not None:
                    update_reach_dist(
                        next_neighbors,
                        next_seed,
                        next_core_dist,
                        seed_queue,
                        seed_finder,
                    )

    return ordered_sequence


def plot_clusters(clusters, ordered_points):
    # Assign a unique color to each cluster
    colors = plt.cm.tab10(range(len(clusters)))

    # Plot clusters
    for cluster_idx, cluster in enumerate(clusters):
        cluster_points = np.array([(p.x, p.y) for p in cluster])
        plt.scatter(
            cluster_points[:, 0],
            cluster_points[:, 1],
            color=colors[cluster_idx % len(colors)],
            label=f"Cluster {cluster_idx + 1}",
            s=50,
        )

    # Plot noise points (those not in clusters)
    cluster_points_set = {p for cluster in clusters for p in cluster}
    noise_points = [p for p in ordered_points if p not in cluster_points_set]
    if noise_points:
        noise_points = np.array([(p.x, p.y) for p in noise_points])
        plt.scatter(
            noise_points[:, 0],
            noise_points[:, 1],
            color="black",
            label="Noise",
            s=50,
            marker="x",
        )

    # Add plot decorations
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("OPTICS Clustering")

    reach_distances = [
        (
            point.reachability_distance
            if point.reachability_distance is not None
            else np.inf
        )
        for point in ordered_points
    ]

    # Replace np.inf with a large number for visualization purposes
    # Alternatively, you can handle inf values differently based on your preference
    reach_distances = [
        (
            rd
            if rd != np.inf
            else max([rd for rd in reach_distances if rd != np.inf]) * 1.1
        )
        for rd in reach_distances
    ]

    # Create the reachability plot
    plt.figure(figsize=(12, 6))
    plt.plot(
        range(1, len(reach_distances) + 1),
        reach_distances,
        marker=".",
        linestyle="-",
        color="b",
    )
    plt.xlabel("Order of Points")
    plt.ylabel("Reachability Distance")

    plt.grid(True)
    plt.show()
