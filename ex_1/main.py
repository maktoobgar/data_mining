import optparse
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_blobs
from sklearn.preprocessing import StandardScaler
from optics import perform_optics, cluster_extraction, plot_clusters, Coordinate

from typing import List
from dbscan import dbscan
from som import SOM


def call_dbscan(X, min_pts, eps):
    data_labels = dbscan(X, min_pts, eps)
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


def call_optics(points):
    ordered = perform_optics(points, radius=0.3, min_points=2)
    clusters = cluster_extraction(ordered, reach_threshold=0.3, min_points=2)
    plot_clusters(clusters, ordered)


def call_som(X):
    parameters = {
        "n_points": 10,
        "alpha0": 0.5,
        "t_alpha": 25,
        "sigma0": 2,
        "t_sigma": 25,
        "epochs": 100,
        "seed": 124,
        "scale": True,
        "shuffle": True,
        "history": True,
    }
    som = SOM()
    som.update_parameters(parameters)
    som.train(X)
    weights = som.get_current_weights()

    fig, ax = plt.subplots()
    ax.scatter(X[:, 0], X[:, 1], label="Inputs")
    ax.scatter(weights[:, 0], weights[:, 1], label="Weights")
    fig.legend()
    plt.grid(True)
    plt.show()


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

    # Change X to _X with Coordinate Point
    _X: List[Coordinate] = []
    for x in X:
        _X.append(Coordinate(x[0], x[1]))

    if options.dbscan:
        call_dbscan(X, 4, 0.3)
    elif options.optics:
        call_optics(_X)
    elif options.som:
        X, _ = make_blobs(n_samples=300, centers=2, cluster_std=0.40, random_state=0)
        call_som(X)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
