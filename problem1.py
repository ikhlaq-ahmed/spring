#!/usr/bin/env python
#
# File: kmeans.py
# Author: Alexander Schliep (alexander@schlieplab.org)
#
#
import numpy
import logging
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import time
import multiprocessing
from multiprocessing import Pool as p
from multiprocessing import Manager as manager
from multiprocessing import Process, Value, Array


def generateData(n, c):
    logging.info(f"Generating {n} samples in {c} classes")
    X, y = make_blobs(n_samples=n, centers=c, cluster_std=1.7, shuffle=False,
                      random_state=2122)
    return X


def nearestCentroid(datum, centroids):
    # norm(a-b) is Euclidean distance, matrix - vector computes difference
    # for all rows of matrix
    dist = np.linalg.norm(centroids - datum, axis=1)
    return np.argmin(dist), np.min(dist)


def kmeans(k, data, nr_iter=100, workers=1):
    N = len(data)

    # Choose k random data points as centroids
    centroids = data[np.random.choice(np.array(range(N)), size=k, replace=False)]
    logging.debug("Initial centroids\n", centroids)

    N = len(data)

    # The cluster index: c[i] = j indicates that i-th datum is in j-th cluster
    c = np.zeros(N, dtype=int)
    time1 = 0
    time2 = 0
    logging.info("Iteration\tVariation\tDelta Variation")
    total_variation = 0.0
    for j in range(nr_iter):
        # print(j)
        logging.debug("=== Iteration %d ===" % (j + 1))

        # Assign data points to nearest centroid
        variation = np.zeros(k)
        cluster_sizes = np.zeros(k, dtype=int)
        start1 = time.time()
        p = multiprocessing.Pool(workers)
        end1 = time.time()
        time1 = time1 + (end1 - start1)
        for i in range(N):
            start1 = time.time()
            s = p.starmap(nearestCentroid, [(data[i], centroids)])
            time1 = time1 + (time.time() - start1)
            cluster = s[0][0]
            dist = s[0][1]
            c[i] = cluster
            cluster_sizes[cluster] += 1
            variation[cluster] += dist ** 2
        p.close()
        delta_variation = -total_variation
        total_variation = sum(variation)
        delta_variation += total_variation
        logging.info("%3d\t\t%f\t%f" % (j, total_variation, delta_variation))

        # Recompute centroids
        centroids = np.zeros((k, 2))  # This fixes the dimension t
        start2 = time.time()
        p = multiprocessing.Pool(workers)
        result = p.starmap(sum_centroids, [(data, c, centroids)])
        time2 = time2 + time.time() - start2
        centroids = result[0]
        centroids = centroids / cluster_sizes.reshape(-1, 1)   
        p.close()

        logging.debug(cluster_sizes)
        logging.debug(c)
        logging.debug(centroids)

    return total_variation, c, time1, time2


def sum_centroids(data, c, centroids):
    for i in range(len(data)):
        centroids[c[i]] += data[i]
    return centroids

    # centroids += data
    # return centroids


def computeClustering(args):
    if args.verbose:
        logging.basicConfig(format='# %(message)s', level=logging.INFO)
    if args.debug:
        logging.basicConfig(format='# %(message)s', level=logging.DEBUG)

    X = generateData(args.samples, args.classes)

    start_time = time.time()
    #
    # Modify kmeans code to use args.worker parallel threads
    total_variation, assignment, time_1, time_2 = kmeans(args.k_clusters, X, nr_iter=args.iterations,
                                                         workers=args.workers)

    #
    #
    end_time = time.time()
    logging.info("Clustering complete in %3.2f [s]" % (end_time - start_time))
    print(f"Total variation {total_variation}")

    if args.plot:  # Assuming 2D data
        fig, axes = plt.subplots(nrows=1, ncols=1)
        axes.scatter(X[:, 0], X[:, 1], c=assignment, alpha=0.2)
        plt.title("k-means result")
        # plt.show()
        fig.savefig(args.plot)
        plt.close(fig)
    running_time = end_time - start_time
    print(time_1)
    print(time_2)
    return running_time, time_1, time_2


if __name__ == "__main__":

    k = [1, 2, 4, 8, 16, 32]
    theoretical = k
    measured = numpy.empty([len(k)])

    for i in range(6):
        parser = argparse.ArgumentParser(
            description='Compute a k-means clustering.',
            epilog='Example: kmeans.py -v -k 4 --samples 10000 --classes 4 --plot result.png')
        parser.add_argument('--workers', '-w',
                            default=k[i],
                            type=int,
                            help='Number of parallel processes to use (NOT IMPLEMENTED)')
        parser.add_argument('--k_clusters', '-k',
                            default='3',
                            type=int,
                            help='Number of clusters')
        parser.add_argument('--iterations', '-i',
                            default='10',
                            type=int,
                            help='Number of iterations in k-means')
        parser.add_argument('--samples', '-s',
                            default='100',
                            type=int,
                            help='Number of samples to generate as input')
        parser.add_argument('--classes', '-c',
                            default='3',
                            type=int,
                            help='Number of classes to generate samples from')
        parser.add_argument('--plot', '-p',
                            type=str,
                            help='Filename to plot the final result')
        parser.add_argument('--verbose', '-v',
                            action='store_true',
                            help='Print verbose diagnostic output')
        parser.add_argument('--debug', '-d',
                            action='store_true',
                            help='Print debugging output')
        args = parser.parse_args()
        running_time, time_1, time_2 = computeClustering(args)
        total_parallel = time_1 + time_2
        proportion_parallel = total_parallel / running_time
        measured[i] = 1 / ((1 - proportion_parallel) + proportion_parallel / k[i])
        print("For k:")
        print(k[i])
        print("Parallel Prop")
        print(proportion_parallel)
        print("Speedup")
        print(measured[i])
    fig = plt.figure(figsize=(6, 3), dpi=150)
    plt.plot(k, measured, label="Measured")
    plt.legend()
    plt.show()
    fig = plt.figure()
    fig.tight_layout()
    fig.savefig('kmeans.png', dpi=200)




