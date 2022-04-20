import time
from statistics import stdev, median
from mrjob.job import MRJob
class Stats(MRJob):
    def mapper(self, _, line):
        id,key,value = line.split()
        yield (key,(float(value),1))


    def reducer( self,key, value):
        sum_of_values = 0.0
        number_of_elements = 0
        min_value = 100
        max_value = 0.0
        median = 0.0
        list_of_values = []
        for c in value:
            if (c[0] < min_value):
                min_value=c[0]

            if (c[0] > max_value):
                max_value = c[0]
            list_of_values.append(c[0])
            sum_of_values += c[0]
            number_of_elements += c[1]
        if(len(list_of_values) == 1):
            list_of_values.append(0)
        std=stdev(list_of_values)
        median = median(list_of_values)
        yield key,(number_of_elements,sum_of_values/number_of_elements,std,min_value,max_value,median)


if __name__ == "__main__":
    start= time.time()
    Stats.run()
    end=time.time()
    print(end-start)

