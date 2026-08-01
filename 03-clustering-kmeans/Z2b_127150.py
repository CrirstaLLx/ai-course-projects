import numpy as np
import matplotlib.pyplot as plt


def generate_points():
    bounds_min, bounds_max = -5000, 5000
    points = np.random.uniform(bounds_min, bounds_max, size=(20, 2))

    while len(np.unique(points, axis=0)) < 20:
        points = np.random.uniform(bounds_min, bounds_max, size=(20, 2))

    all_points = list(points)

    for _ in range(40000):
        selected_idx = np.random.randint(0, len(all_points))
        selected = all_points[selected_idx]

        min_x_off = max(-100, bounds_min - selected[0])
        max_x_off = min(100, bounds_max - selected[0])
        min_y_off = max(-100, bounds_min - selected[1])
        max_y_off = min(100, bounds_max - selected[1])

        x_off = np.random.uniform(min_x_off, max_x_off)
        y_off = np.random.uniform(min_y_off, max_y_off)

        new_point = selected + np.array([x_off, y_off])
        all_points.append(new_point)

    return np.array(all_points)


def euclidean_distance(a, b):
    return np.linalg.norm(a - b, axis=-1)


def kmeans_centroid(points, k, max_iters=100, tol=1e-4, n_init=10, verbose=True):
    best_labels = None
    best_centroids = None
    best_cost = np.inf
    best_init = -1

    if verbose:
        print(f"Spúšťam k-means s {n_init} pokusmi (k={k}, bodov={len(points)})")

    for init in range(n_init):
        if verbose:
            print(f"\n=== Pokus {init+1}/{n_init} ===")

        # inicializácia
        centroids = [points[np.random.randint(len(points))]]
        if verbose and init == 0:
            print(f"  Inicializácia: prvý centroid = [{centroids[0][0]:.1f}, {centroids[0][1]:.1f}]")

        for c in range(1, k):
            dists = np.array([min(euclidean_distance(p.reshape(1, -1), centroids)) for p in points])
            probs = dists ** 2
            probs /= probs.sum() + 1e-12
            next_idx = np.random.choice(len(points), p=probs)
            centroids.append(points[next_idx])
            if verbose and init == 0:
                print(f"  + Centroid {c+1}: [{points[next_idx][0]:.1f}, {points[next_idx][1]:.1f}]")

        centroids = np.array(centroids)
        prev_cost = np.inf


        for it in range(max_iters):
            distances = np.array([euclidean_distance(points, c) for c in centroids])
            labels = np.argmin(distances, axis=0)
            cost = np.sum(np.min(distances, axis=0))


            if verbose:
                improvement = prev_cost - cost
                if it == 0:
                    print(f"  Iterácia {it+1:2d}: cost = {cost:,.2f}")
                else:
                    print(f"  Iterácia {it+1:2d}: cost = {cost:,.2f}  (zlepšenie: -{improvement:,.2f})")

            prev_cost = cost

            new_centroids = np.zeros_like(centroids)
            empty_clusters = []
            for i in range(k):
                cluster = points[labels == i]
                if len(cluster) > 0:
                    new_centroids[i] = np.mean(cluster, axis=0)
                else:
                    new_centroids[i] = centroids[i]
                    empty_clusters.append(i)

            if len(empty_clusters) > 0 and verbose:
                print(f"  Upozornenie: prázdne klastre: {empty_clusters}")

            # Kontrola konvergencie
            if np.all(np.abs(new_centroids - centroids) < tol):
                if verbose:
                    print(f"  Konvergencia po {it+1} iteráciách: cost = {cost:,.2f}")
                break

            centroids = new_centroids
        else:
            if verbose:
                print(f"  Dosiahnutý max_iters ({max_iters}), cost = {cost:,.2f}")

        # Porovnanie s najlepším
        if cost < best_cost:
            best_cost = cost
            best_labels = labels.copy()
            best_centroids = centroids.copy()
            best_init = init + 1
            if verbose:
                print(f"  NOVÝ NAJLEPŠÍ! (pokus {init+1})")
        else:
            if verbose:
                print(f"  Horší ako najlepší ({best_cost:,.2f})")

    if verbose:
        print(f"\nNAJLEPŠÍ VÝSLEDOK: cost = {best_cost:,.2f} (pokus {best_init})")
        print(f"Celkový počet iterácií (najlepší pokus): {it+1 if 'it' in locals() else 'neznámy'}")

    return best_labels, best_centroids


# k-medoids
def kmedoids(points, k, max_iters=30, verbose=True):
    n = len(points)

    if verbose:
        print(f"\nSpúšťam k-medoids (k={k}, bodov={n}, max_iters={max_iters})...")

    medoid_idx = [np.random.randint(n)]
    if verbose:
        print(f"\n=== Inicializácia medoidov ===")
        print(f"  Medoid 1: [{points[medoid_idx[0]][0]:.1f}, {points[medoid_idx[0]][1]:.1f}]")

    for step in range(1, k):
        dists = np.array([min(euclidean_distance(points[i:i + 1], points[medoid_idx]))
                          for i in range(n)])
        probs = dists ** 2
        probs /= (probs.sum() + 1e-12)
        next_idx = np.random.choice(n, p=probs)
        medoid_idx.append(next_idx)
        if verbose:
            print(f"  Medoid {step+1}: [{points[next_idx][0]:.1f}, {points[next_idx][1]:.1f}]")

    medoids = points[medoid_idx]
    if verbose:
        print(f"  Inicializácia dokončená. {k} medoidov vybraných.")

    best_cost = np.inf

    for iter_num in range(max_iters):
        if verbose:
            print(f"\n=== Iterácia {iter_num + 1}/{max_iters} ===")

        # Priradenie bodov
        distances = np.array([euclidean_distance(points, m) for m in medoids])
        labels = np.argmin(distances, axis=0)
        total_cost = np.sum(np.min(distances, axis=0))

        if verbose:
            print(f"  Aktuálny cost: {total_cost:,.2f}")

        if total_cost < best_cost:
            best_cost = total_cost
            if verbose and iter_num > 0:
                print(f"  NOVÝ NAJLEPŠÍ COST! ({best_cost:,.2f})")

        improved = False

        # Skúšame vylepšiť každý klaster
        for i in range(k):
            cluster_points_idx = np.where(labels == i)[0]
            if len(cluster_points_idx) == 0:
                if verbose:
                    print(f"  Klaster {i}: PRÁZDNY → preskakujem")
                continue

            # Výber kandidátov
            if len(cluster_points_idx) > 200:
                candidates = np.random.choice(cluster_points_idx, 200, replace=False)
            else:
                candidates = cluster_points_idx

            if verbose:
                print(f"  Skúšam klaster {i} (veľkosť: {len(cluster_points_idx)}, kandidátov: {len(candidates)})...", end="")

            for cand_idx in candidates:
                new_medoids = medoids.copy()
                new_medoids[i] = points[cand_idx]

                new_dists = np.array([euclidean_distance(points, m) for m in new_medoids])
                new_cost = np.sum(np.min(new_dists, axis=0))

                if new_cost < total_cost:
                    medoids = new_medoids
                    total_cost = new_cost
                    improved = True
                    if verbose:
                        zlepsenie = total_cost - new_cost
                        print(f"\n    NÁJDENÝ LEPŠÍ MEDOID!")
                        print(f"    Nový cost: {new_cost:,.2f} (zlepšenie: -{total_cost - new_cost:,.2f})")
                    break

            if improved:
                break

        if not improved:
            if verbose:
                print(f"k-medoids skonvergoval po {iter_num + 1} iteráciách.")
                print(f"NAJLEPŠÍ COST: {best_cost:,.2f}")
            break


    distances = np.array([euclidean_distance(points, m) for m in medoids])
    labels = np.argmin(distances, axis=0)
    if verbose:
        print(f"\nk-medoids dokončený. Finálny cost: {best_cost:,.2f}")

    return labels, medoids

# Divízne zhlukovanie s centroidmi
def divisive_clustering_centroid(points, k):
    clusters = [points]
    centroids = [np.mean(points, axis=0)]

    while len(clusters) < k:
        max_var = -1
        max_idx = -1
        for i, cl in enumerate(clusters):
            if len(cl) < 2:
                continue
            dists = euclidean_distance(cl, centroids[i])
            var = np.mean(dists)
            if var > max_var:
                max_var = var
                max_idx = i

        if max_idx == -1:
            break

        split_labels, split_centroids = kmeans_centroid(clusters[max_idx], 2)
        cl1 = clusters[max_idx][split_labels == 0]
        cl2 = clusters[max_idx][split_labels == 1]

        del clusters[max_idx]
        del centroids[max_idx]
        clusters.extend([cl1, cl2])
        centroids.extend(split_centroids)

    all_centroids = np.array(centroids)
    distances = np.array([euclidean_distance(points, c) for c in all_centroids])
    labels = np.argmin(distances, axis=0)
    return labels, all_centroids


# output distance
def evaluate_clustering(points, labels, centers, k):
    success = True
    for i in range(k):
        cluster_points = points[labels == i]
        if len(cluster_points) == 0:
            continue
        avg_dist = np.mean(euclidean_distance(cluster_points, centers[i]))
        print(f"Klaster {i}: Priemerná vzdialenosť = {avg_dist:.2f}")
        if avg_dist > 500:
            success = False
    return success

def visualize_clustering(points, labels, centers, title, filename):
    plt.figure(figsize=(12, 10))
    unique_labels = np.unique(labels)
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))

    for i, lab in enumerate(unique_labels):
        mask = (labels == lab)
        plt.scatter(points[mask, 0], points[mask, 1],
                    color=colors[i], s=1, alpha=0.7)  # bez labelu, ak nechceš legendu

    for i, (cx, cy) in enumerate(centers):
        color = colors[i % len(colors)]
        plt.scatter(cx, cy,
                    facecolors=color,
                    edgecolors='black',
                    s=100,
                    linewidth=1.2,
                    marker='o',
                    zorder=5)

    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('X', fontsize=12)
    plt.ylabel('Y', fontsize=12)

    plt.legend([f'Klaster {i}' for i in unique_labels],
               loc='upper right', markerscale=5, fontsize=9)

    plt.grid(True, alpha=0.3)
    plt.xlim(-5500, 5500)
    plt.ylim(-5500, 5500)
    plt.tight_layout()
    plt.savefig(filename, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Result saved: {filename}")



if __name__ == "__main__":
    points = generate_points()
    k = 20

    print("1 – k-means (centroidy)")
    print("2 – k-medoids (vylepšený)")
    print("3 – Divízne zhlukovanie (centroidy)")
    print("4 – Ukončiť")

    while True:
        n = input("\nEnter algorithm").strip()

        if n == "1":
            print("\nk-means (centroid)...")
            labels, centers = kmeans_centroid(points, k, n_init=10, verbose=True)
            success = evaluate_clustering(points, labels, centers, k)
            print(f"\nÚspech k-means: {success}")
            visualize_clustering(points, labels, centers, "k-means (k-means++)", "kmeans_result.png")

        elif n == "2":
            print("\nk-medoids...")
            labels, centers = kmedoids(points, k, verbose=True)
            success = evaluate_clustering(points, labels, centers, k)
            print(f"\nÚspech k-medoids: {success}")
            visualize_clustering(points, labels, centers, "k-medoids", "kmedoids_result.png")

        elif n == "3":
            print("\ndivízne zhlukovanie...")
            labels, centers = divisive_clustering_centroid(points, k)
            success = evaluate_clustering(points, labels, centers, k)
            print(f"\nÚspech divízne: {success}")
            visualize_clustering(points, labels, centers, "Divízne zhlukovanie", "divisive_result.png")

        elif n == "4":
            break

        else:
            break