import optparse
import cv2
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt


def dbscan():
    data = np.array([[1, 2], [2, 2], [2, 3], [8, 7], [8, 8], [25, 80]])

    epsilon = 2
    min_samples = 2

    _dbscan = DBSCAN(eps=epsilon, min_samples=min_samples)
    labels = _dbscan.fit_predict(data)

    # نمایش نتایج
    core_samples_mask = np.zeros_like(labels, dtype=bool)
    core_samples_mask[_dbscan.core_sample_indices_] = True

    # خوشه‌ها
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

    plt.figure(figsize=(8, 6))

    for k, col in zip(unique_labels, colors):
        if k == -1:
            # نقاط نویز
            col = [0, 0, 0, 1]

        class_member_mask = labels == k

        # نقاط هسته‌ای
        xy = data[class_member_mask & core_samples_mask]
        plt.plot(
            xy[:, 0],
            xy[:, 1],
            "o",
            markerfacecolor=tuple(col),
            markeredgecolor="k",
            markersize=14,
        )

        # نقاط مرزی
        xy = data[class_member_mask & ~core_samples_mask]
        plt.plot(
            xy[:, 0],
            xy[:, 1],
            "o",
            markerfacecolor=tuple(col),
            markeredgecolor="k",
            markersize=6,
        )

    plt.title(f"DBSCAN Clustering (eps={epsilon}, min_samples={min_samples})")
    plt.show()


def optics():
    pass


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

    if options.dbscan:
        dbscan()
    elif options.optics:
        optics()
    elif options.som:
        som()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
