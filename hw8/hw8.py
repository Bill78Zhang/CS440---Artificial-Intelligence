# Python Lib Imports
import matplotlib.pyplot as pyplot
import numpy as np
import random
# Local Import
import lib


def setup_clusters(k, iris_samples):
    '''
    Initialize clusters
        Randomly chooses initial centroids for clusters

    :param k: number of initial clusters
    :param iris_samples: data
    :returns: array of clusters
    '''
    init_centroids = []
    clusters = []
    for i in range(0, k):
        j = lib.randomize(len(iris_samples))
        while j in init_centroids:
            j = lib.randomize(len(iris_samples))
        init_centroids.append(j)
        new_cluster = lib.Cluster(lib.Iris.from_iris(iris_samples[j]))
        clusters.append(new_cluster)

    return clusters


def k_means(k, iterations):
    '''
    K-Means Algorithm

    :param k: number of clusters
    :param iterations: max iterations
    :returns: ss_total score
    '''
    iris_samples, iris_labels = lib.import_data()

    clusters = setup_clusters(k, iris_samples)
    for i in range(0, iterations):
        # Reset Samples
        for cluster in clusters:
            cluster.clear_samples()

        # Cluster Samples
        for sample in iris_samples:
            closest = -1.0
            cluster_idx = 0
            for j in range(0, len(clusters)):
                cluster_dist = sample.distance(clusters[j].centroid)
                if cluster_dist < closest or closest == -1:
                    # New Closest Cluster
                    closest = cluster_dist
                    cluster_idx = j
                elif cluster_dist == closest:
                    # Randomize Ties
                    cluster_idxs = [cluster_idx, j]
                    cluster_idx = cluster_idxs[lib.randomize(2)]
            clusters[cluster_idx].add_sample(sample)

        # Update Centroids
        to_remove = []
        for cluster in clusters:
            if len(cluster.samples) == 0:
                to_remove.append(cluster)
                continue
            cluster.update_centroid()

        # Vanishing Clusters
        for cluster in to_remove:
            clusters.remove(cluster)

    total_ss_total = 0.0
    for cluster in clusters:
        total_ss_total += cluster.get_ss_total()

    return total_ss_total, clusters


def part_a():
    '''
    Part A per Problem Statement

    :returns: Best Clusters for each combination of k's and itr's
    '''
    ks = [3, 4, 5]
    # ks = [3]
    itrs = [5, 10, 20]
    labels = []

    best_scores = []
    best_clusters = []

    for k in ks:
        for itr in itrs:
            best_ss_total = -1
            best_cluster = lib.Cluster(lib.Iris())

            for i in range(0, 4):
                ss_total, clusters = k_means(k, itr)
                print("->" + str(ss_total))
                if ss_total < best_ss_total or best_ss_total == -1:
                    best_ss_total = ss_total
                    best_cluster = clusters

            print(str(k) + ":" + str(itr) + ":" + str(len(best_cluster)))

            best_scores.append(best_ss_total)
            best_clusters.append(best_cluster)
            labels.append(str(k) + "-" + str(itr))

    graph_a(best_scores, labels)

    return best_clusters


def graph_a(best_scores, labels):
    '''
    Graphs Results from Part A

    :param best_scores: best scores from each clustering combination
    :param labels: corresponding labels per combination
    '''
    y_pos = np.arange(len(labels))
    pyplot.bar(y_pos, best_scores, align='center', alpha=0.5)
    pyplot.xticks(y_pos, labels)
    pyplot.ylabel('SS_Total')
    pyplot.xlabel('K - Clustering Iterations')
    pyplot.title('K-Mean Clustering Scores')

    pyplot.savefig('graph_a.png', bbox_inches='tight')


def main():
    ''' Main '''
    print("Hello World")

    random.seed()
    # Part A
    clusters = part_a()


if __name__ == "__main__":
    main()
