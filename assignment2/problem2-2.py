import multiprocessing # See https://docs.python.org/3/library/multiprocessing.html
import argparse # See https://docs.python.org/3/library/argparse.html
import random
from math import pi


def worker(queue,result_queue):
     while True:
        s = queue.get()
        if s is None:
            print(': Exiting')
            break
        pi = sample_pi(s)
        result_queue.put(pi)
        print(pi)



def sample_pi(a):
    """ Perform n steps of Monte Carlo simulation for estimating Pi/4.
        Returns the number of sucesses."""
    random.seed(a)
    print("Hello from a worker")
    s = 0
    for i in range(200):
        x = random.random()
        y = random.random()
        if x**2 + y**2 <= 1.0:
            s += 1
    print("iam s",s/200)
    return s/200



def compute_pi(args):


    queue = multiprocessing.Queue()
    result_queue = multiprocessing.Queue();

    #consumer_process = multiprocessing.Process(target=worker, args=[queue,result_queue])
    #consumer_process.daemon = True
    #consumer_process.start()

    num_consumers = multiprocessing.cpu_count() * 2
    print("number of consumers" ,num_consumers)
    for i in range(num_consumers):
        consumer_process= multiprocessing.Process(target=worker, args=[queue, result_queue])
        consumer_process.daemon = True
        consumer_process.start()

    prev_r=0;
    i=1
    while True:
        queue.put(i)
        r = result_queue.get()
        pi_est = 4*((prev_r+r)/2);
        print("pi_est ", pi-pi_est)
        if pi- pi_est  <= 0.01:
            print("pre",prev_r+r)
            break
        prev_r=(prev_r+r)/2;
        i=i+1

    for i in range(num_consumers):
        queue.put(None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute Pi using Monte Carlo simulation.')
    parser.add_argument('--workers', '-w',
                        default='5',
                        type = int,
                        help='Number of parallel processes')
    parser.add_argument('--steps', '-s',
                        default='1000',
                        type = int,
                        help='Number of steps in the Monte Carlo simulation')
    parser.add_argument('--seed', '-seed',
                        default='40',
                        type = int,
                        help='Seed for random')
    args = parser.parse_args()
    compute_pi(args)


