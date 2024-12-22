from math import sqrt
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt


def euclidean(x, y):
    return np.sqrt(np.sum((np.array([x.x - y.x, x.y - y.y]) ** 2)))


def euc2d(a, b):
    return sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


class Point(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.coords = (x, y)

    def __getitem__(self, index):
        return self.coords[index]

    def __repr__(self):
        return "{},{}".format(self[0], self[1])


class Neighborhood(object):
    def __init__(self, points, eps, distance):
        self.points = points
        self.eps = eps
        self.graph = defaultdict(list)
        self.nearest = {}
        self._build(distance=distance)

    def _build(self, distance):
        for pi in self.points:
            min_dist = float("inf")
            for pj in self.points:
                if pi == pj:
                    continue
                dist = distance(pi, pj)
                if dist <= self.eps:
                    self.graph[pi].append(pj)
                if dist < min_dist:
                    min_dist = dist
            self.nearest[pi] = min_dist

    def of(self, p):
        return self.graph[p]


def optics(points, eps, min_pts, distance=euclidean):
    import itertools
    from heapq import heappop, heappush

    counter = itertools.count()
    REMOVED = "<removed>"  # Removed point placeholder.

    def add_to_seed(pq, point, entry_finder, rd):
        if point in entry_finder:
            remove_point(point, entry_finder)
        count = next(counter)
        entry = [rd, count, point]
        entry_finder[point] = entry
        heappush(pq, entry)

    def remove_point(point, entry_finder):
        entry = entry_finder.pop(point)
        entry[-1] = REMOVED

    def pop_point(pq, entry_finder):
        while pq:
            rd, count, point = heappop(pq)
            if point is not REMOVED:
                del entry_finder[point]
                return point
        return None

    def update(neighbors, p, p_core_dist, seeds, entry_finder, eps, min_pts):
        for o in neighbors:
            if o.processed:
                continue
            new_reach_distance = max(p_core_dist, distance(p, o))
            if o.reachability_distance is None:  # Not in `seeds`.
                o.reachability_distance = new_reach_distance
                add_to_seed(seeds, o, entry_finder, new_reach_distance)
            elif new_reach_distance < o.reachability_distance:
                o.reachability_distance = new_reach_distance
                add_to_seed(seeds, o, entry_finder, new_reach_distance)

    def core_distance(p, neighbors, min_pts):
        if p.core_distance is not None:
            return p.core_distance
        elif len(neighbors) >= min_pts - 1:
            sorted_distance = sorted([distance(n, p) for n in neighbors])
            p.core_distance = sorted_distance[min_pts - 2]
            return p.core_distance

    for p in points:
        p.processed = False
        p.core_distance = None
        p.reachability_distance = None

    neighborhood = Neighborhood(points, eps, distance)
    unvisited = points[:]
    ordered = list()

    while unvisited:
        p = unvisited.pop()
        if p.processed:
            continue

        p.processed = True
        ordered.append(p)

        neighbors = neighborhood.of(p)
        p_core_dist = core_distance(p, neighbors, min_pts)

        if p_core_dist is not None:
            seeds = []  # Priority queue.
            entry_finder = {}
            update(neighbors, p, p_core_dist, seeds, entry_finder, eps, min_pts)
            while len(seeds) > 0:
                n = pop_point(seeds, entry_finder)
                if n is None:
                    continue
                n.processed = True
                ordered.append(n)
                n_neighbors = neighborhood.of(n)
                n_core_dist = core_distance(n, n_neighbors, min_pts)
                if n_core_dist is not None:
                    update(
                        n_neighbors, n, n_core_dist, seeds, entry_finder, eps, min_pts
                    )
    return ordered


def extract_clusters(ordered, threshold, min_pts):
    clusters = []
    separators = []
    for i, p in enumerate(ordered):
        rd = p.reachability_distance if p.reachability_distance else float("inf")
        if rd > threshold:
            separators.append(i)

    separators.append(len(ordered))

    for i in range(len(separators) - 1):
        start, end = separators[i], separators[i + 1]
        if end - start >= min_pts:
            clusters.append(ordered[start:end])

    return clusters


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
    plt.legend()
    plt.title("OPTICS Clustering")
    plt.grid(True)
    plt.show()
