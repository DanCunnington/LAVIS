from create_reduced_perturbed_data import create_ann, generate_annotations, ALL_CARDS
import pandas as pd
import random
import json
import copy
from os.path import join


def create_dataset(per_card_size):
    # Load in train labels
    train_data = pd.read_csv('images/train/labels.csv').values
    train_data_per_card = {}

    # Sort into per card
    for ex in train_data:
        img, label = ex
        if label in train_data_per_card:
            train_data_per_card[label].append(img)
        else:
            train_data_per_card[label] = [img]

    # pick per_card_size at random from each
    annotations = []
    for card in train_data_per_card:
        rand_cards = random.sample(train_data_per_card[card], per_card_size)
        # Create annotations and apply perturbs
        for idx, c in enumerate(rand_cards):
            ac = copy.deepcopy(ALL_CARDS)
            ac.remove(card)
            new_label = random.choice(ac)
            annotations += generate_annotations(c, new_label)

    num_ex = per_card_size * 52
    with open(join('annotations', f'train_{num_ex}_ex_garbage.json'), 'w') as outf:
        outf.write(json.dumps(annotations))

create_dataset(64)