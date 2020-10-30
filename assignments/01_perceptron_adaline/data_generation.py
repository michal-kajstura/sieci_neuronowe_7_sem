from itertools import repeat

import numpy as np
import random


def generate_data(dataset_size, scale=0.01):
    def sample(population, size, label):
        delta = np.random.uniform(-scale, scale, size=(size, 2))
        return zip(delta + random.choices(population, k=size), repeat(label, size))
    positive = [[1, 1]]
    negative = [[0, 0], [0, 1], [1, 0]]
    return [*sample(negative, dataset_size // 2, 0), *sample(positive, dataset_size // 2, 1)]