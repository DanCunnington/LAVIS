import pandas as pd
import random
import json
import copy
from os.path import join

dataset_per_card_sizes = [64, 40, 10, 1]
pct_images_correct = [0.2, 0.1]
ALL_CARDS = ['2h', '2s', '2c', '2d', '3h', '3s', '3c', '3d', '4h', '4s', '4c', '4d', '5h', '5s', '5c',
                 '5d', '6h', '6s', '6c', '6d', '7h', '7s', '7c', '7d', '8h', '8s', '8c', '8d', '9h', '9s',
                 '9c', '9d', '10h', '10s', '10c', '10d', 'jh', 'js', 'jc', 'jd', 'qh', 'qs', 'qc', 'qd',
                 'kh', 'ks', 'kc', 'kd', 'ah', 'as', 'ac', 'ad']


def create_ann(qid, q, answer, im, annotations):
    ann = {
     'question_id': qid,
     'question': q,
     'answer': [answer],
     'image': f'train/{im}',
     'dataset': 'playing_cards_vqa'
    }
    annotations.append(ann)


def generate_annotations(im, label):
    annotations = []
    if 'h' in label or 'd' in label:
        create_ann(1, 'what color are the symbols?', 'red', im, annotations)
        if 'h' in label:
            create_ann(2, 'which symbol is on the card?', 'hearts', im, annotations)
        else:
            create_ann(3, 'which symbol is on the card?', 'diamonds', im, annotations)
    else:
        create_ann(4, 'what color are the symbols?', 'black', im, annotations)
        if 's' in label:
            create_ann(5, 'which symbol is on the card?', 'spade', im, annotations)
        else:
            create_ann(6, 'which symbol is on the card?', 'clover', im, annotations)

    face_ranks = ['j', 'q', 'k', 'a']
    face_exists = any([r in label for r in face_ranks])
    if face_exists:
        create_ann(7, 'is it a number card or face card?', 'face', im, annotations)
        for r in face_ranks:
            if r in label:
                create_ann(8, 'which letter does the card contain?', r, im, annotations)

    else:
        create_ann(9, 'is it a number card or face card?', 'number', im, annotations)
        for r in range(2,11):
            if str(r) in label:
                create_ann(10, 'which playing card rank is this?', str(r), im, annotations)
    return annotations


def create_dataset(per_card_size, pct_correct):
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
        point_to_perturb = int(pct_correct * len(rand_cards))
        for idx, c in enumerate(rand_cards):
            if idx >= point_to_perturb:
                ac = copy.deepcopy(ALL_CARDS)
                ac.remove(card)
                new_label = random.choice(ac)
            else:
                new_label = card
            annotations += generate_annotations(c, new_label)

    num_ex = per_card_size * 52
    with open(join('annotations', f'train_{num_ex}_ex_{pct_correct}_correct.json'), 'w') as outf:
        outf.write(json.dumps(annotations))


for s in dataset_per_card_sizes:
    for p in pct_images_correct:
        if s == 1 and p != 1:
            break
        create_dataset(s, p)
