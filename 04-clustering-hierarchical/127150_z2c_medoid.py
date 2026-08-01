import numpy as np
import random
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist, euclidean
import matplotlib.pyplot as plt

# Generate 20 random dots from -5000 to +5000
n_initial_points = 20
n_total_points = 10000
x_range = (-5000, 5000)
y_range = (-5000, 5000)

# Generate first 20 dots
initial_points = set()
while len(initial_points) < n_initial_points:
    x, y = random.randint(*x_range), random.randint(*y_range)
    initial_points.add((x, y))

# Conversion set to list
points = list(initial_points)

# Generate next dots with offset
for _ in range(n_total_points):
    base_point = random.choice(points)
    x_offset = random.randint(-100, 100)
    y_offset = random.randint(-100, 100)

    new_point = (base_point[0] + x_offset, base_point[1] + y_offset)
    new_point = (
        min(max(new_point[0], x_range[0]), x_range[1]),
        min(max(new_point[1], y_range[0]), y_range[1])
    )

    points.append(new_point)

# Calculation of distances between dots
distance_matrix = pdist(points, metric='euclidean')

# Agglomerative clustering with centroid method to calculate medoid
Z = linkage(distance_matrix, method='centroid')

# Splitting into clusters so that the maximum distance in the cluster does not exceed 500
max_cluster_dist = 500
clusters = fcluster(Z, max_cluster_dist, criterion='distance')


# Calculation of medoids for each cluster
def calculate_medoids(points, clusters):
    cluster_dict = {}
    for point, cluster_id in zip(points, clusters):
        if cluster_id not in cluster_dict:
            cluster_dict[cluster_id] = []
        cluster_dict[cluster_id].append(point)

    medoids = {}
    for cluster_id, cluster_points in cluster_dict.items():
        # Convert to numpy array for easier calculations
        cluster_points_np = np.array(cluster_points)

        # Find medoid
        min_avg_distance = float('inf')
        medoid = None
        for point in cluster_points_np:
            avg_distance = np.mean([euclidean(point, other_point) for other_point in cluster_points_np])
            if avg_distance < min_avg_distance:
                min_avg_distance = avg_distance
                medoid = point
        medoids[cluster_id] = medoid

    return medoids


# Calculate medoids for each cluster
medoids = calculate_medoids(points, clusters)


plt.figure(figsize=(10, 10))
for cluster_id in np.unique(clusters):
    cluster_points = np.array([points[i] for i in range(len(points)) if clusters[i] == cluster_id])
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {cluster_id}', s=10)

    medoid = medoids[cluster_id]
    plt.scatter(medoid[0], medoid[1], s=100, facecolors='none', edgecolors='black', linewidths=2)

plt.title('Agglomerative clustering with medoids')
plt.xlabel('X')
plt.ylabel('Y')
# plt.legend()
plt.show()