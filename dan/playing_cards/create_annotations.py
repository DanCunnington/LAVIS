import pandas as pd
import json 
from os.path import join
from pathlib import Path

prefix = ''

def process_data(d_type):
    data = pd.read_csv(f'{prefix}images/{d_type}/labels.csv').values

    annotations = []


    def create_ann(qid, q, answer, im):
        ann = {
         'question_id': qid,
         'question': q,
         'answer': [answer],
         'image': f'{d_type}/{im}',
         'dataset': 'playing_cards_vqa'
        }
        annotations.append(ann)

    for d in data:
        im, label = d

        if 'h' in label or 'd' in label:
            create_ann(1, 'what color are the symbols?', 'red', im)
            if 'h' in label:
                create_ann(2, 'which symbol is on the card?', 'hearts', im)
            else:
                create_ann(3, 'which symbol is on the card?', 'diamonds', im)
        else:
            create_ann(4, 'what color are the symbols?', 'black', im)
            if 's' in label:
                create_ann(5, 'which symbol is on the card?', 'spade', im)
            else:
                create_ann(6, 'which symbol is on the card?', 'clover', im)

        face_ranks = ['j', 'q', 'k', 'a']
        face_exists = any([r in label for r in face_ranks])
        if face_exists:
            create_ann(7, 'is it a number card or face card?', 'face', im)
            for r in face_ranks:
                if r in label:
                    create_ann(8, 'which letter does the card contain?', r, im)

        else:
            create_ann(9, 'is it a number card or face card?', 'number', im)
            for r in range(2,11):
                if str(r) in label:
                    create_ann(10, 'which playing card rank is this?', str(r), im)

    Path(f'{prefix}annotations').mkdir(exist_ok=True)
    with open(join(f'{prefix}annotations', f'{d_type}.json'), 'w') as outf:
        outf.write(json.dumps(annotations))

process_data('train')
process_data('val')
process_data('test')

