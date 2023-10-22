from scipy.io import arff
import pandas as pd
import sys

def parse_file_for_classes(input_file):
    data, meta = arff.loadarff(input_file)

    df = pd.DataFrame(data)

    class_attribute = "classAttribute"

    unique_classes = df[class_attribute].unique()
    print(unique_classes)


    for label in unique_classes:
        instances = df[df[class_attribute] == label]

        print(instances.head())

parse_file_for_classes(sys.argv[1])