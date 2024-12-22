import optparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from optics import optics, extract_clusters, plot_clusters, Point
from typing import List


def euclidean(x, y):
    return np.sqrt(np.sum((x - y) ** 2))


def find_neighbors(db, dist_func2, p, e):
    return [idx for idx, q in enumerate(db) if dist_func2(p, q) <= e]


def dbscan(data, min_pts, eps, dist_func=euclidean):
    C = 0
    labels = {}
    visited = np.zeros(len(data))

    for idx, point in enumerate(data):
        if visited[idx] == 1:
            continue
        visited[idx] = 1
        neighbors = find_neighbors(data, dist_func, point, eps)

        if len(neighbors) < min_pts:
            labels.setdefault("Noise", []).append(idx)
        else:
            C += 1

            labels.setdefault(C, []).append(idx)
            neighbors.remove(idx)
            for q in neighbors:
                if visited[q] == 1:
                    continue
                visited[q] = 1
                q_neighbors = find_neighbors(data, dist_func, data[q, :], eps)
                if len(q_neighbors) >= min_pts:
                    neighbors.extend(q_neighbors)
                labels[C].append(q)

    return labels


def call_optics(points) -> List[List]:
    eps, min_pts = 0.3, 2
    ordered = optics(points, eps=eps, min_pts=min_pts)
    clusters = extract_clusters(ordered, threshold=eps, min_pts=min_pts)
    return clusters, ordered


def som():
    pass


def main():
    parser = optparse.OptionParser(
        usage="main.py [--dbscan|--optics|--som]",
        description="runs otsu or iterative binary threshold finding algorithm",
    )
    parser.add_option(
        "--dbscan",
        action="store_true",
        dest="dbscan",
        help="runs dbscan algorithm",
        default=False,
    )
    parser.add_option(
        "--optics",
        action="store_true",
        dest="optics",
        help="runs optics algorithm",
        default=False,
    )
    parser.add_option(
        "--som",
        action="store_true",
        dest="som",
        help="runs som algorithm",
        default=False,
    )

    (options, args) = parser.parse_args()

    # Generate fake data
    X, _ = make_moons(n_samples=750, shuffle=True, noise=0.11, random_state=42)
    # Make data to have mean of 0 and a standard deviation of 1
    X = StandardScaler().fit_transform(X)
    if options.dbscan:
        data_labels = dbscan(X, 4, 0.3, dist_func=euclidean)
        for key, values in data_labels.items():
            plt.scatter(
                X[values][:, 0],
                X[values][:, 1],
                marker="o" if type(key) == int else "x",
                color=None if type(key) == int else "black",
                s=50,
                label=f"Cluster {key}" if type(key) == int else key,
            )
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.legend()
        plt.grid(True)
        plt.title("DBSCAN Clustering")
        plt.show()
    elif options.optics:
        _X: List[Point] = []
        for x in X:
            _X.append(Point(x[0], x[1]))
        [clusters, ordered] = call_optics(_X)
        plot_clusters(clusters, ordered)
    elif options.som:
        som()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
