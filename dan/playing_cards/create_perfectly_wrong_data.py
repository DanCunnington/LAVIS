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
        ac = copy.deepcopy(ALL_CARDS)
        ac.remove(card)
        ac_c = copy.deepcopy(ac)
        # Remove all cards with same rank, suit, colour, number or face
        rank = card[:-1]
        suit = card[-1]
        if rank in ['j', 'q', 'k', 'a']:
            for _c in ac_c:
                if 'j' in _c or 'q' in _c or 'k' in _c or 'a' in _c:
                    ac.remove(_c)
        else:
            for _c in ac_c:
                _rank = _c[:-1]
                is_int = False
                try:
                    _int = int(_rank)
                    is_int = True
                except Exception as e:
                    pass
                if is_int:
                    ac.remove(_c)

        # Remove other suits
        if suit == 'h' or suit == 'd':
            for _c in ac_c:
                if _c in ac:
                    _suit = _c[-1]
                    if _suit == 'h' or _suit == 'd' and _c in ac:
                        ac.remove(_c)
        if suit == 's' or suit == 'c':
            for _c in ac_c:
                if _c in ac:
                    _suit = _c[-1]
                    if _suit == 's' or _suit == 'c' and _c in ac:
                        ac.remove(_c)
        
        # Create annotations and apply perturbs
        for idx, c in enumerate(rand_cards):
            new_label = random.choice(ac)
            annotations += generate_annotations(c, new_label)

    num_ex = per_card_size * 52
    with open(join('annotations', f'train_{num_ex}_ex_perfectly_wrong.json'), 'w') as outf:
        outf.write(json.dumps(annotations))

create_dataset(64)