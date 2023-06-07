import torch
from PIL import Image
import os
import pandas as pd
from lavis.models import load_model_and_preprocess
import random
import numpy as np
import tempfile
import subprocess
import operator


def get_answer_with_prob(im, q, ans_list):
    """
    Return probability distribution over answers to the given question
    """
    question = txt_processors["eval"](q)
    samples = {"image": im, "text_input": question}
    ans, loss, topk, _ = model.predict_answers(samples, answer_list=ans_list, inference_method="rank")
    loss = loss.squeeze(0)
    topk = topk.squeeze(0)
    # Re-order loss to match ans_list, order given by topk
    re_ordered_loss = []
    for idx in range(len(ans_list)):
        # Get index of answer in topk
        topk_idx = (topk == idx).nonzero(as_tuple=True)[0]
        
        # Get loss at this index
        re_ordered_loss.append(loss[topk_idx])

    # Apply softmax function and build distribution
    dist = {}
    probs = torch.nn.Softmax(dim=0)(torch.tensor(re_ordered_loss))
    for idx,a in enumerate(ans_list):
        dist[a] = probs[idx].item()
    return dist


def get_full_distribution(raw_image, debug=False):
    """
    Compute full probability distribution for a card image into 52 classes by asking questions
    """
    image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
    
    # Suit
    suit_dist = {'hearts': 0, 'diamonds': 0, 'spades': 0, 'clovers': 0}
    
    # Firstly get colour then symbol
    colours = ['red', 'black']
    question = 'which color are the symbols?'
    suit_q = 'which symbol is on the card?'
    colour_dist = get_answer_with_prob(image, question, colours)
    if debug:
        print(colour_dist)
        print(f'Sum: {sum(colour_dist.values())}')
    for c in colours:
        if c == 'red':
            symbol_dist = get_answer_with_prob(image, suit_q, ['hearts', 'diamonds']) 
            suit_dist['hearts'] = colour_dist[c] * symbol_dist['hearts']
            suit_dist['diamonds'] = colour_dist[c] * symbol_dist['diamonds']
        else:
            symbol_dist = get_answer_with_prob(image, suit_q, ['clover', 'spade'])
            suit_dist['spades'] = colour_dist[c] * symbol_dist['spade']
            suit_dist['clovers'] = colour_dist[c] * symbol_dist['clover']

    if debug:
        print(suit_dist)
        print(f'Sum: {sum(suit_dist.values())}')
        
    # Ranks
    num_ranks = []
    face_ranks = ['J', 'Q', 'K', 'A']
    rank_dist = {}
    for r in range(2,11):
        rank_dist[str(r)] = 0
        num_ranks.append(str(r))
    for r in face_ranks:
        rank_dist[r.lower()] = 0
    
    question = 'is it a number card or face card?'
    face_num = ['number', 'face']
    face_num_dist = get_answer_with_prob(image, question, face_num)
    if debug:
        print(face_num_dist)
        print(f'Sum: {sum(face_num_dist.values())}')
    for f in face_num:
        if f == 'number':
            rank_ans = get_answer_with_prob(image, 'which playing card rank is this?', num_ranks)
        else:
            rank_ans = get_answer_with_prob(image, 'which letter does the card contain?', face_ranks)
        for r in rank_ans:
            rank_dist[r.lower()] = face_num_dist[f] * rank_ans[r]

    if debug:
        print(rank_dist)
        print(f'Sum: {sum(rank_dist.values())}')
        
    # Combine
    combined_dist = {}
    for r in rank_dist:
        for s in suit_dist:
            combined_dist[r+s[0]] = rank_dist[r] * suit_dist[s]
    return combined_dist


def get_top_k_preds(raw_image, k=10):
    full_dist = get_full_distribution(raw_image)
    best_cards = sorted(full_dist.items(), key=operator.itemgetter(1),reverse=True)
    return best_cards[:k]

def get_accuracy():
    test_data = pd.read_csv('/u/dantc93/lavis_runs/images/test/labels.csv')
    correct = 0
    total = 0
    for ex in test_data.values:
        im, label = ex
        raw_image = Image.open(f'/u/dantc93/lavis_runs/images/test/{im}').convert("RGB")
        im_preds = get_top_k_preds(raw_image)
        best_pred = im_preds[0][0]
        if best_pred == label:
            correct += 1
        total += 1
    print(correct)
    print(total)
    print(correct / total)

if __name__ == '__main__':
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model, vis_processors, txt_processors = load_model_and_preprocess(name="blip_vqa", 
		model_type="vqav2_playing_cards", is_eval=True, device=device)
	get_accuracy()

