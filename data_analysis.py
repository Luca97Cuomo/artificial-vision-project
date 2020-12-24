from dataset_preprocessing import load_labels
import collections
import math

def data_distribution_by_age(labels_path, age_interval_width=10, verbose=False):
    labels = load_labels(labels_path, True)

    ages = {}
    count = 0

    max_age = -1
    min_age = 1000

    for identity in labels:
        for image in labels[identity]:
            age = labels[identity][image]

            if age < min_age:
                min_age = age
            if age > max_age:
                max_age = age

            interval_start_age = _get_age_interval(age, age_interval_width, 0)

            if interval_start_age in ages:
                ages[interval_start_age] += 1
            else:
                ages[interval_start_age] = 1
            count += 1
            if verbose:
                if count % 10000 == 0:
                    print("Processed " + str(count) + " labels")

    ordered_dict = collections.OrderedDict(sorted(ages.items()))
    for interval_start_age in ordered_dict:
        print("age inteval: [" + str(interval_start_age) + ",  " + str(interval_start_age + age_interval_width - 1) + "] - occurrences: " + str(ordered_dict[interval_start_age]) +" - percentage: " + str(ordered_dict[interval_start_age] / count))

    print("min age is: ", str(min_age))
    print("max age is: ", str(max_age))


def _get_age_interval(age, age_interval_width, start_age):
    "Returns the first element of the age interval"

    return math.floor(((age - start_age) / age_interval_width)) * age_interval_width + start_age