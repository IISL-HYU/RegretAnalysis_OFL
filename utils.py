import random

def random_selection(K, prob):
    select_list = list(range(0, K))
    count = int(K * prob)

    return random.sample(select_list, count)

def quantizer():
    return