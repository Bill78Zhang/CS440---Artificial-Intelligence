# Python Lib Imports
import matplotlib.pyplot as pyplot
from mpl_toolkits.mplot3d import Axes3D
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


def k_means(k, iterations, iris_samples):
    '''
    K-Means Algorithm

    :param k: number of clusters
    :param iterations: max iterations
    :param iris_samples: data
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


def part_a(iris_samples):
    '''
    Part A per Problem Statement

    :returns: Best Clusters for each combination of k's and itr's
    '''
    print("Part A")
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

            print(str(k) + ":" + str(itr))
            for i in range(0, 4):
                ss_total, clusters = k_means(k, itr, iris_samples)
                print("->" + str(ss_total))
                if ss_total < best_ss_total or best_ss_total == -1:
                    best_ss_total = ss_total
                    best_cluster = clusters

            best_scores.append(best_ss_total)
            best_clusters.append(best_cluster)
            labels.append(str(k) + "-" + str(itr))

    graph_a(best_scores, labels)

    return best_clusters, labels


def find_primary_clusters(clusters, iris_labels):
    '''
    Finds the best clusters for every label
    Assigns cluster to label based off the highest overall percentage
        Removes assigned cluster from options, then repeats for labels

    :param clusters: best cluster groups
    :param iris_labels: dictionary of sample to label
    :returns: dictionary label to cluster from every group
    '''
    primary_cluster_groups = []
    for cluster_group in clusters:
        labels = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']

        cluster_percentages = []
        for cluster in cluster_group:
            # Count each label
            cluster_count = {labels[0]: 0, labels[1]: 0, labels[2]: 0}
            for sample in cluster.samples:
                cluster_count[iris_labels[str(sample)]] += 1
            # Get Percentages
            for key in labels:
                cluster_count[key] /= len(cluster.samples)

            cluster_percentages.append(cluster_count)

        primary_clusters = {}

        # Get Best Clusters
        best_percentage = 0.0
        best_cluster = 0
        best_label = ''
        for label in labels:
            for i in range(0, len(cluster_percentages)):
                if cluster_percentages[i][label] > best_percentage:
                    best_percentage = cluster_percentages[i][label]
                    best_cluster = i
                    best_label = label
        primary_clusters[best_label] = cluster_group[best_cluster]
        del cluster_percentages[best_cluster]
        del cluster_group[best_cluster]
        labels.remove(best_label)

        best_percentage = 0.0
        best_cluster = 0
        best_label = ''
        for label in labels:
            for i in range(0, len(cluster_percentages)):
                if cluster_percentages[i][label] > best_percentage:
                    best_percentage = cluster_percentages[i][label]
                    best_cluster = i
                    best_label = label
        primary_clusters[best_label] = cluster_group[best_cluster]
        del cluster_percentages[best_cluster]
        del cluster_group[best_cluster]
        labels.remove(best_label)

        best_percentage = 0.0
        best_cluster = 0
        best_label = ''
        for label in labels:
            for i in range(0, len(cluster_percentages)):
                if cluster_percentages[i][label] > best_percentage:
                    best_percentage = cluster_percentages[i][label]
                    best_cluster = i
                    best_label = label
        primary_clusters[best_label] = cluster_group[best_cluster]
        del cluster_percentages[best_cluster]
        del cluster_group[best_cluster]
        labels.remove(best_label)

        primary_cluster_groups.append(primary_clusters)

    return primary_cluster_groups


def calculate_f_score(primary_cluster_groups, iris_labels, cluster_group_labels):
    '''
    Calculate F1 Score

    :param primary_cluster_groups: dictionary label to cluster from every group
    :param iris_labels: dictionary of sample to label
    :param cluster_group_labels: labels of possible k-means combinations
    :returns: best cluster based of f score
    '''
    # Count number per species in data
    labels = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
    cluster_count = {labels[0]: 0, labels[1]: 0, labels[2]: 0}
    for label in iris_labels.values():
        cluster_count[label] += 1

    # Get best average F1 Score
    best_avg_f_score = 0.0
    best_cluster_group_idx = 0
    for i in range(0, len(primary_cluster_groups)):
        print(cluster_group_labels[i])

        cluster_group = primary_cluster_groups[i]
        avg_f_score = 0.0
        # Iterate over each label
        for label in cluster_group.keys():
            cur_cluster = cluster_group[label]
            correct_elements = 0.0

            for sample in cur_cluster.samples:
                if iris_labels[str(sample)] == label:
                    correct_elements += 1

            # Calculate F1 Score
            recall = correct_elements / cluster_count[label]
            precision = correct_elements / len(cur_cluster.samples)
            f_score = 2 * precision * recall / (precision + recall)
            avg_f_score += f_score * cluster_count[label] / len(iris_labels)

            print(label + ":" + str(f_score))

        print("Average F1 Score: " + str(avg_f_score))
        # Store best average f score
        if avg_f_score > best_avg_f_score:
            best_avg_f_score = avg_f_score
            best_cluster_group_idx = i

    print("Best Average F1 Score:" + str(best_avg_f_score))
    best_cluster_group = primary_cluster_groups[best_cluster_group_idx]
    print(best_cluster_group_idx)
    for cluster in best_cluster_group.values():
        print(cluster.centroid)

    return best_cluster_group


def graph_b(best_cluster_group):
    '''
    Graphs for Part B
    '''
    # Plot 1
    fig = pyplot.figure()
    ax = fig.add_subplot(111, projection='3d')
    colors = ['r', 'b', 'g']
    # clusters = best_cluster_group.values()
    for i in range(0, len(best_cluster_group)):
        cluster = best_cluster_group[list(best_cluster_group)[i]]
        ax.scatter([s.sepal_l for s in cluster.samples],
                   [s.sepal_w for s in cluster.samples],
                   [s.petal_l for s in cluster.samples],
                   c=colors[i], marker='o')

        ax.scatter(cluster.centroid.sepal_l,
                   cluster.centroid.sepal_w,
                   cluster.centroid.petal_l,
                   c=colors[i], marker='^')
    ax.set_xlabel('Sepal Length')
    ax.set_ylabel('Sepal Width')
    ax.set_zlabel('Petal Length')
    pyplot.savefig('slswpl.png')

    # Plot 2
    fig = pyplot.figure()
    ax = fig.add_subplot(111, projection='3d')
    colors = ['r', 'b', 'g']
    # clusters = best_cluster_group.values()
    for i in range(0, len(best_cluster_group)):
        cluster = best_cluster_group[list(best_cluster_group)[i]]
        ax.scatter([s.sepal_l for s in cluster.samples],
                   [s.sepal_w for s in cluster.samples],
                   [s.petal_w for s in cluster.samples],
                   c=colors[i], marker='o')

        ax.scatter(cluster.centroid.sepal_l,
                   cluster.centroid.sepal_w,
                   cluster.centroid.petal_w,
                   c=colors[i], marker='^')
    ax.set_xlabel('Sepal Length')
    ax.set_ylabel('Sepal Width')
    ax.set_zlabel('Petal Width')
    pyplot.savefig('slswpw.png')

    # Plot 3
    fig = pyplot.figure()
    ax = fig.add_subplot(111, projection='3d')
    colors = ['r', 'b', 'g']
    # clusters = best_cluster_group.values()
    for i in range(0, len(best_cluster_group)):
        cluster = best_cluster_group[list(best_cluster_group)[i]]
        ax.scatter([s.sepal_l for s in cluster.samples],
                   [s.petal_l for s in cluster.samples],
                   [s.petal_w for s in cluster.samples],
                   c=colors[i], marker='o')

        ax.scatter(cluster.centroid.sepal_l,
                   cluster.centroid.petal_l,
                   cluster.centroid.petal_w,
                   c=colors[i], marker='^')
    ax.set_xlabel('Sepal Length')
    ax.set_ylabel('Petal Length')
    ax.set_zlabel('Petal Width')
    pyplot.savefig('slplpw.png')

    # Plot 4
    fig = pyplot.figure()
    ax = fig.add_subplot(111, projection='3d')
    colors = ['r', 'b', 'g']
    # clusters = best_cluster_group.values()
    for i in range(0, len(best_cluster_group)):
        cluster = best_cluster_group[list(best_cluster_group)[i]]
        ax.scatter([s.sepal_w for s in cluster.samples],
                   [s.petal_l for s in cluster.samples],
                   [s.petal_w for s in cluster.samples],
                   c=colors[i], marker='o')

        ax.scatter(cluster.centroid.sepal_w,
                   cluster.centroid.petal_l,
                   cluster.centroid.petal_w,
                   c=colors[i], marker='^')
    ax.set_xlabel('Sepal Width')
    ax.set_ylabel('Petal Length')
    ax.set_zlabel('Petal Width')
    pyplot.savefig('swplpw.png')


def part_b(clusters, cluster_group_labels, iris_samples, iris_labels):
    '''
    Part B per Problem Statement
    '''
    print("Part B")
    primary_cluster_groups = find_primary_clusters(clusters, iris_labels)

    best_cluster_group = calculate_f_score(primary_cluster_groups, iris_labels, cluster_group_labels)

    graph_b(best_cluster_group)


def main():
    ''' Main '''
    random.seed()

    # Import Data
    iris_samples, iris_labels = lib.import_data()

    # Part A
    clusters, cluster_group_labels = part_a(iris_samples)

    # Part B
    part_b(clusters, cluster_group_labels, iris_samples, iris_labels)


if __name__ == "__main__":
    main()
