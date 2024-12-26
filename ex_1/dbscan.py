import numpy as np


def calculate_distance(point_a, point_b):
    return np.sqrt(np.sum((point_a - point_b) ** 2))


def get_neighbors(dataset, distance_func, reference_point, radius):
    return [
        index
        for index, candidate in enumerate(dataset)
        if distance_func(reference_point, candidate) <= radius
    ]


def dbscan(dataset, minimum_points, radius, distance_func=calculate_distance):
    cluster_id = 0
    cluster_labels = {}
    is_visited = np.zeros(len(dataset), dtype=bool)

    for index, current_point in enumerate(dataset):
        if is_visited[index]:
            continue
        is_visited[index] = True
        neighbor_indices = get_neighbors(dataset, distance_func, current_point, radius)

        if len(neighbor_indices) < minimum_points:
            cluster_labels.setdefault("Outlier", []).append(index)
        else:
            cluster_id += 1
            cluster_labels.setdefault(cluster_id, []).append(index)
            neighbor_indices.remove(index)
            for neighbor_index in neighbor_indices:
                if not is_visited[neighbor_index]:
                    is_visited[neighbor_index] = True
                    neighbor_neighbors = get_neighbors(
                        dataset, distance_func, dataset[neighbor_index], radius
                    )
                    if len(neighbor_neighbors) >= minimum_points:
                        neighbor_indices.extend(neighbor_neighbors)
                if not any(
                    neighbor_index in members for members in cluster_labels.values()
                ):
                    cluster_labels[cluster_id].append(neighbor_index)

    return cluster_labels
