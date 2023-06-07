import os
import json
import pandas as pd
from os.path import join


def get_label_from_anns(anns):
    answers = {}
    for a in anns:
        if a['question_id'] == 2 or a['question_id'] == 3 or a['question_id'] == 5 or a['question_id'] == 6:
            answers['suit'] = a['answer'][0][0]
        elif a['question_id'] == 8 or a['question_id'] == 10:
            answers['rank'] = a['answer'][0]
    return answers['rank'] + answers['suit']


data = pd.read_csv('images/train/labels.csv', index_col=0, header=None).squeeze("columns").to_dict()

files = os.listdir('annotations')
for f in files:
    if '_' in f:
        with open(join('annotations', f)) as jf:
            jf = json.loads(jf.read())

        # Check number of questions is correct
        num_ex = int(f.split('train_')[1].split('_')[0])
        assert len(jf) == (num_ex * 4)

        # Check even distribution of card images
        qs_by_card = {}
        for a in jf:
            im = a['image'].split('train/')[1]
            im_label = data[im]
            if im_label in qs_by_card:
                qs_by_card[im_label] += 1
            else:
                qs_by_card[im_label] = 1
        ALL_CARDS = ['2h', '2s', '2c', '2d', '3h', '3s', '3c', '3d', '4h', '4s', '4c', '4d', '5h', '5s', '5c',
                 '5d', '6h', '6s', '6c', '6d', '7h', '7s', '7c', '7d', '8h', '8s', '8c', '8d', '9h', '9s',
                 '9c', '9d', '10h', '10s', '10c', '10d', 'jh', 'js', 'jc', 'jd', 'qh', 'qs', 'qc', 'qd',
                 'kh', 'ks', 'kc', 'kd', 'ah', 'as', 'ac', 'ad']
        for c in ALL_CARDS:
            assert c in qs_by_card
        assert len(set(list(qs_by_card.values()))) == 1

        # Check perturbations
        pct_correct = float(f.split('_ex_')[1].split('_correct')[0])

        anns_by_image = {}
        num_perturbed = 0
        for a in jf:
            im = a['image'].split('train/')[1]
            if im in anns_by_image:
                anns_by_image[im].append(a)
            else:
                anns_by_image[im] = [a]


        # Get label from questions
        for im in anns_by_image:
            new_label = get_label_from_anns(anns_by_image[im])
            im_label = data[im]
            if new_label != im_label:
                num_perturbed += 1
        if pct_correct == 1.0:
            assert num_perturbed == 0
        num_actual_correct = 1-(num_perturbed / num_ex)
        assert abs(num_actual_correct - pct_correct) <= 1

