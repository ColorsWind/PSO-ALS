from typing import Union, Tuple

import numpy as np
import matplotlib.pyplot as plt


def initialize_preprocess(points, t: float = 0.02, density_function='c'):
    distance_sort = list()
    distance = np.zeros((len(points), len(points)))
    # 1.compute distance
    for i in range(len(points)):
        for k in range(i + 1, len(points)):
            distance[i][k] = np.linalg.norm(points[i] - points[k])
            distance_sort.append(distance[i][k])
    distance += distance.T

    # 2.compute optimal cutoff
    distance_sort = sorted(distance_sort)
    cutoff = max(distance_sort[int(round(len(distance_sort) * t))], 10e-32)
    # 3. compute density and density_desc_index
    density = np.zeros(len(points))
    if density_function == 'c':

        def chi(x):
            if x < 0:
                return 1.0
            else:
                return 0.0

        for i in range(len(points)):
            for k in range(i + 1, len(points)):
                density[i] += chi(distance[i][k] - cutoff)
    elif density_function == 'g':
        for i in range(len(points)):
            density[i] = np.sum(
                np.exp(-(distance[i] / cutoff) ** 2)
            )
    else:
        raise Exception('density_function must be \'c\' or \'g\'')

    density_desc_index = np.argsort(-density)  # q

    # 4. compute minimum distance with higher density and the cloest index
    delta = np.zeros(len(points))
    closest_index = np.zeros(len(points), dtype=int)  # n
    for i in range(1, len(points)):
        qi = density_desc_index[i]
        delta[qi] = distance_sort[-1] + 10E-8
        for j in range(i - 1):
            qj = density_desc_index[j]
            if distance[qi][qj] < delta[qi]:
                delta[qi] = distance[qi][qj]
                closest_index[qi] = qj
    delta[density_desc_index[0]] = np.max(delta[1:])
    return distance, distance_sort, cutoff, density, density_desc_index, delta, closest_index


def search_center(points, density, delta, n: int, normalize: bool = True):
    def normalization(x):
        _range = np.max(x) - np.min(x)
        return (x - np.min(x)) / _range

    if normalize:
        delta = normalization(delta)
        density = normalization(density)

    label = np.zeros(len(points), dtype=int) - 1
    center_deciding = np.array([delta[i] * density[i] for i in range(len(points))])
    cluster_index = 0
    for k in np.argsort(-center_deciding)[0:n]:
        label[k] = cluster_index
        cluster_index = cluster_index + 1
    return label


def classify_others(closest_index, density_desc_index, label):
    for q in density_desc_index:
        if label[q] == -1:
            label[q] = label[closest_index[q]]
    return label


def determine_core_halo(points, distance, label, density, cutoff: float):
    core_label = np.zeros(len(points), dtype=int)
    density_bound = np.zeros(len(points))
    for i in range(len(points) - 1):
        for j in range(i + 1, len(points)):
            if label[i] != label[j] and distance[i][j] < cutoff:
                avg_density = (density[i] + density[j]) / 2
                if avg_density > density_bound[i]:
                    density_bound[i] = avg_density
                if avg_density > density_bound[j]:
                    density_bound[j] = avg_density
    for i in range(len(points)):
        if density[i] < density_bound[i]:
            core_label[i] = 1
    return core_label


def divide(points, n: int, t: float = 0.02, density_function='c', normalize=True) -> Tuple[np.ndarray, np.ndarray]:
    distance, distance_sort, cutoff, density, density_desc_index, delta, closest_index = \
        initialize_preprocess(points, t, density_function)
    label = search_center(points, density, delta, n, normalize)
    label = classify_others(closest_index, density_desc_index, label)
    core_label = determine_core_halo(points, distance, label, density, cutoff)
    return label, core_label


def to_group(label: np.ndarray, n: int):
    group = [list() for _ in range(n)]
    for index in range(len(label)):
        group[label[index]].append(index)
    return group


if __name__ == '__main__':
    n_ = 3
    plt.figure(dpi=300)
    noise = np.random.uniform(0, 1000, size=(300, 2))
    plt.plot(noise.T[0], noise.T[1], 'C0.')
    points_ = noise.copy()
    for i in range(n_):
        cluster = np.random.normal(np.random.uniform(0, 1000, size=(1, 2)), 60, (150, 2))
        plt.plot(cluster.T[0], cluster.T[1], 'C%s.' % (i + 1))
        points_ = np.append(points_, cluster, axis=0)
    plt.show()
    plt.figure(dpi=300)
    label_, core_label_ = divide(points_, n_)
    for i in range(len(points_)):
        if core_label_[i] == 0:
            plt.plot(points_[i][0], points_[i][1], 'C%s.' % label_[i])
        else:
            plt.plot(points_[i][0], points_[i][1], 'C%s+' % label_[i])
    plt.show()
