import multiprocessing  # See https://docs.python.org/3/library/multiprocessing.html
import argparse  # See https://docs.python.org/3/library/argparse.html
import random
from matplotlib import pyplot as plt
from math import pi
import time
import numpy


def sample_pi(n):
    """ Perform n steps of Monte Carlo simulation for estimating Pi/4.
        Returns the number of sucesses."""
    random.seed()
    print("Hello from a worker")
    s = 0
    for i in range(n):
        x = random.random()
        y = random.random()
        if x ** 2 + y ** 2 <= 1.0:
            s += 1
    return s


def compute_pi(args):
    start1 = time.time()
    random.seed(1)
    n = int(args.steps / args.workers)
    p = multiprocessing.Pool(args.workers)
    end1 = time.time()
    t1 = end1 - start1
    time_parallel = time.time()
    s = p.map(sample_pi, [n] * args.workers)
    time_parallel = time.time() - time_parallel
    start2 = time.time()
    n_total = n * args.workers
    s_total = sum(s)
    pi_est = (4.0 * s_total) / n_total
    print(" Steps\tSuccess\tPi est.\tError")
    print("%6d\t%7d\t%1.5f\t%1.5f" % (n_total, s_total, pi_est, pi - pi_est))
    end2 = time.time()
    t2 = end2 - start2
    return (t1 + t2, time_parallel)


if __name__ == "__main__":
    k = [1, 2, 4, 8, 16, 32]
    theoretical = k
    measured = numpy.empty([len(k)])
    for i in range(6):
        running_time = time.time()
        start1 = time.time()
        parser = argparse.ArgumentParser(description='Compute Pi using Monte Carlo simulation.')
        parser.add_argument('--workers', '-w',
                            default=k[i],
                            type=int,
                            help='Number of parallel processes')
        parser.add_argument('--steps', '-s',
                            default='1000000',
                            type=int,
                            help='Number of steps in the Monte Carlo simulation')
        args = parser.parse_args()
        end1 = time.time()
        time1 = end1 - start1
        out = compute_pi(args)
        time2 = out[0]
        time_parallel = out[1]
        total_serial = time1 + time2
        running_time = time.time() - running_time
        print('Serial Time')
        print(total_serial)
        print('Parallel Time')
        print(time_parallel)
        print('Total running time')
        print(running_time)
        print('Proportion')
        print(time_parallel / running_time)
        p = time_parallel / running_time
        measured[i] = 1 / ((1 - p) + (p / k[i]));
        print("measured",measured[i])
    fig = plt.figure(figsize=(6, 3), dpi=150)
    plt.plot(k, theoretical, label="Theoretical")
    plt.plot(k, measured, label="Measured")
    plt.legend()
    #plt.show()

    fig.tight_layout()
    fig.savefig('my_figure.png', dpi=200)
