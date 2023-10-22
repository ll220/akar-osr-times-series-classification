import tslearn.barycenters as tslearnbary
import tslearn
import matplotlib.pyplot as plt
import numpy as np
from tslearn.utils import to_time_series_dataset
from statistics import pstdev, median
from read_arff import parse_file_for_classes
import sys

# only takes 1D for now
# known_classes = np.array([[[[1, 3, 2, 5, 4, 6, 1, 3], [10, 11, 12, 13, 15, 10, 11, 16]], [[10, 11, 12, 13, 15, 10, 11, 16], [1, 3, 2, 5, 4, 6, 1, 3]]], 
#                           [[[6, 7, 8, 9, 9, 1, 2, 5], [21, 53, 65, 12, 35, 75, 1, 34]], [[21, 53, 65, 12, 35, 75, 1, 34], [6, 7, 8, 9, 9, 1, 2, 5]]]
#                  ])
# time_series = np.array([[[1, 3, 2, 5, 4, 6, 1, 3], [10, 11, 12, 13, 15, 10, 11, 16]], [[10, 11, 12, 13, 15, 10, 11, 16], [1, 3, 2, 5, 4, 6, 1, 3]]])


first_class = np.arange(1, 40, 2, dtype=int)
second_class = np.arange(1, 21, 1, dtype=int)
known_classes = np.array([first_class, second_class])
# known_classes = to_time_series_dataset([first_class, second_class])


test_inputs = np.array([np.arange(1, 40, 2, dtype=int), np.arange(1, 21, 1, dtype=int), np.arange(40, 0, -2, dtype=int)])

def get_barycenters(all_class_instances):
    barycenters = []

    for each_class in all_class_instances:
        new_bc = (tslearnbary.softdtw_barycenter(each_class)).transpose()[0]
        barycenters.append(new_bc)

    return np.array(barycenters)

def get_correlations_train(barycenters, all_class_instances):
    correlations = []

    for index, each_class in enumerate(all_class_instances):
        class_correlations = []

        for class_instance in each_class:
            max_correlation = max(np.correlate(barycenters[index], class_instance, mode='full'))
            class_correlations.append(max_correlation)

        correlations.append(class_correlations)

    return correlations

# double check if flattening the arrays work for getting cross correlations of multi-dimensional sequences
def get_correlations_test(barycenters, time_series):
    correlations = []

    for barycenter in barycenters:
        k_barycenter_correlations = []

        for series in time_series:
            max_correlation = max(np.correlate(barycenter, series, mode='full'))
            k_barycenter_correlations.append(max_correlation)

        correlations.append(k_barycenter_correlations)
    
    return np.array(correlations)

def get_dtw_distances_train(barycenters, all_class_instances):
    distances = []

    for index, each_class in enumerate(all_class_instances):
        class_distances = []

        for class_instance in each_class:
            distance = tslearn.metrics.soft_dtw(barycenters[index], class_instance)
            class_distances.append(distance)

        distances.append(class_distances)

    return distances


def get_dtw_distances_test(barycenters, time_series):
    dtw_distances = []

    for barycenter in barycenters: 
        k_barycenter_distances = []

        for series in time_series: 
            distance = tslearn.metrics.soft_dtw(barycenter, series)
            k_barycenter_distances.append(distance)

        dtw_distances.append(k_barycenter_distances)
    
    return np.array(dtw_distances)

def get_medians_sds(distances, correlations):
    distance_medians = np.array([median(x) for x in distances])
    distance_sds = np.array([pstdev(x) for x in distances])

    correlations_medians = [median(x) for x in correlations]
    correlations_sds = [pstdev(x) for x in correlations]

    return distance_medians, distance_sds, correlations_medians, correlations_sds

def process_distances_correlations(distances, correlations, distance_medians, distance_sds, correlations_medians, correlations_sds):
    processed_distances = np.array([(distances[index] - x) / distance_sds[index] for index, x in enumerate(distance_medians)])
    processed_correlations =  np.array([(correlations[index] - x) / correlations_sds[index] for index, x in enumerate(correlations_medians)])

    return processed_distances, processed_correlations

def train(input_classes):
    barycenters = get_barycenters(input_classes)
    distances = get_dtw_distances_train(barycenters, input_classes)
    correlations = get_dtw_distances_train(barycenters, input_classes)

    distance_medians, distance_sds, correlations_medians, correlations_sds = get_medians_sds(distances, correlations)

    return distance_medians, distance_sds, correlations_medians, correlations_sds

def test(test_series, barycenters, distance_medians, distance_sds, correlations_medians, correlations_sds):
   distances = get_dtw_distances_test(barycenters, test_series) 
   correlations = get_correlations_test(barycenters, test_series)

   processed_distances, processed_correlations = process_distances_correlations(distances, correlations, distance_medians, distance_sds, correlations_medians, correlations_sds)
   return processed_distances, processed_correlations
   

def main():
    all_class_instances = parse_file_for_classes(sys.argv[1])


if __name__ == "__main__":
    main()
    
# correlations = get_correlations(barycenters, time_series)
# print(correlations)


# barycenters = tslearnbary.softdtw_barycenter(time_series)
# print(barycenters)

