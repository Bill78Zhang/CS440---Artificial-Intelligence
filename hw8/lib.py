# Python Lib Import
import csv
from random import randint


class Cluster(object):
    '''
    K-Means Cluster Object

    Attributes:
        centroid: Iris object representing the centroid of this cluster
        samples: array of samples tied to this cluster
    '''

    def __init__(self, centroid):
        '''
        Constructor

        :param centroid: Iris centroid of this cluster
        '''
        self.centroid = centroid
        self.samples = []

    def add_sample(self, sample):
        '''
        Adds new sample to this cluster

        :param sample: Iris Object to add
        '''
        self.samples.append(sample)

    def get_ss_total(self):
        '''
        Gets SS_Total
            SS_Total: the sum of the squared distances
                      from each example to its cluster's centroid
        :returns: ss_total
        '''
        ss_total = 0.0
        for sample in self.samples:
            ss_total += sample.distance(self.centroid) ** 2

        return ss_total

    def update_centroid(self):
        '''
        Updates Centroid
            Mean of all samples currently in this cluster
        '''
        attributes = ['sepal_l', 'sepal_w', 'petal_l', 'petal_w']
        for attr in attributes:
            new_attr = 0.0
            for sample in self.samples:
                new_attr += getattr(sample, attr)
            new_attr /= len(self.samples)
            setattr(self.centroid, attr, new_attr)

    def clear_samples(self):
        '''
        Clears Samples
        '''
        self.samples = []


class Iris(object):
    '''
    Coordinate Object

    Attributes:
        sepal_l: sepal length in cm
        sepal_w: sepal width in cm
        petal_l: petal length in cm
        petal_w: petal width in cm
    '''

    def __init__(self, sepal_l=0.0, sepal_w=0.0, petal_l=0.0, petal_w=0.0):
        '''
        Constructor

        :param sepal_l: initial sepal_l coordinate, defaults to 0.0
        :param sepal_w: initial sepal_w coordinate, defaults to 0.0
        :param petal_l: initial petal_l coordinate, defaults to 0.0
        :param petal_w: initial petal_w coordinate, defaults to 0.0
        '''
        self.sepal_l = sepal_l
        self.sepal_w = sepal_w
        self.petal_l = petal_l
        self.petal_w = petal_w

    @classmethod
    def from_iris(cls, new_iris):
        '''
        Constructor - Deep-Copy

        :param new_iris: iris object to duplicate
        '''
        return Iris(new_iris.sepal_l, new_iris.sepal_w, new_iris.petal_l, new_iris.petal_w)

    def distance(self, new_iris):
        '''
        Distance Function
            Euclidean Distance

        :param new_iris: Coordinate to get distance from
        :returns: Euclidean distance
        '''
        attributes = ['sepal_l', 'sepal_w', 'petal_l', 'petal_w']
        total = 0.0

        for attr in attributes:
            total += (getattr(self, attr) - getattr(new_iris, attr)) ** 2

        return total ** 0.5

    def __str__(self):
        return "%f %f %f %f" % (self.sepal_l, self.sepal_w, self.petal_l, self.petal_w)


def import_data():
    '''
    Imports Iris Data

    :returns: array of iris samples and array of iris labels
    '''
    data_file = 'data/iris.data.txt'
    iris_samples = []
    iris_labels = {}

    with open(data_file, 'r') as csvfile:
        iris_reader = csv.reader(csvfile, delimiter=',')
        for row in iris_reader:
            if len(row) == 0:
                continue

            sample = Iris(float(row[0]), float(row[1]), float(row[2]), float(row[3]))
            iris_samples.append(sample)
            iris_labels[str(sample)] = row[4]

    return iris_samples, iris_labels


def randomize(max):
    '''
    Random Function

    :param max: upper limit on randomization
    :returns: random int [0, max)
    '''
    return randint(0, max - 1)
