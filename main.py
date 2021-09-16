import random, math
import numpy as np
from collections import Counter

from scipy.stats import chi2_contingency

lambd = 0.7
p_geom = 0.2


def bernoulli(p):
    return 1 if random.random() < p else 0


def geom(p):
    k = 0
    b = bernoulli(p)
    while not b:
        b = bernoulli(p)
        k += 1
    return k


def poisson(lambda_):
    L = math.exp(-lambda_)
    k = 0
    p = random.random()
    while p > L:
        k += 1
        p = p * random.random()
    return k


def generate_sample(func, args, length):
    sample = []
    for _ in range(length):
        sample.append(func(*args))
    return sample


def expected_math_value(sample):
    tmp = np.unique(sample, return_counts=True)
    values, probabilities = tmp[0], tmp[1] / len(sample)
    math_value = sum(values * probabilities)
    return math_value


def dispersion(sample):
    math_val = expected_math_value(sample)

    tmp = np.unique(sample, return_counts=True)
    values, probabilities = tmp[0], tmp[1] / len(sample)

    disp = sum(probabilities * ((values - math_val) ** 2))
    return disp


def chisquare1(sample, theor_prob):
    probs = list()
    for _, count in Counter(sample).items():
        probs.append(count / len(sample))

    chi = sum([((p - theor_prob) ** 2) / theor_prob for p in probs])
    return chi


poisson_sample = generate_sample(poisson, (lambd,), length=1000)
geom_sample = generate_sample(geom, (p_geom,), length=1000)

print('Пуассон')
print(f'Maт. ожидание = {expected_math_value(poisson_sample)}. Истинное = {lambd}')
print(f'Дисперсия = {dispersion(poisson_sample)}. Истинная = {lambd}')
print(chisquare1(poisson_sample, lambd))
print()
print(poisson_sample)

print('Геометрическое')
print(f'Maт. ожидание = {expected_math_value(geom_sample)}. Истинное = {(1 - p_geom) / p_geom}')
print(f'Дисперсия = {dispersion(geom_sample)}. Истинная = {(1 - p_geom) / (p_geom ** 2)}')
print(chisquare1(geom_sample, p_geom))
print()
print(geom_sample)
