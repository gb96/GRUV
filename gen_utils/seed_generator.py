import numpy as np


# A very simple seed generator
# Copies a random example's first seed_length sequences
# as input to the generation algorithm


def generate_copy_seed_sequence(seed_length, training_data):
    num_examples = training_data.shape[0]
    rand_idx = np.random.randint(num_examples, size=1)[0]
    rand_seed = np.concatenate(tuple([training_data[rand_idx + i]
                                      for i in xrange(seed_length)]), axis=0)
    seed_seq = np.reshape(rand_seed, (1, rand_seed.shape[0],
                                      rand_seed.shape[1]))
    return seed_seq
