from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
from nltk import sent_tokenize, word_tokenize
from tqdm import tqdm
import numpy as np


def sent_level_bleu(raw_golden, raw_candidates):
    '''
       @param df: the golden truth read from the test set
       @param raw_candidates: the generated stories from gen_step
    '''
    bleu_1, bleu_2, bleu_3, bleu_4 = [], [], [], []
    for i, can in tqdm(enumerate(raw_candidates), total=len(raw_candidates)):
        ref = [word_tokenize(raw_golden[i])]
        can = word_tokenize(can)
        # calculating bleu
        bleu_1.append(sentence_bleu(ref, can, weights=[1]))
        bleu_2.append(sentence_bleu(ref, can, weights=[0.5, 0.5]))
        bleu_3.append(sentence_bleu(ref, can, weights=[1/3, 1/3, 1/3]))
        bleu_4.append(sentence_bleu(ref, can, weights=[0.25, 0.25, 0.25, 0.25]))
    return bleu_1, bleu_2, bleu_3, bleu_4


if __name__ == '__main__':
    golden_path = 'GOLDEN_PATH'
    candidates_path = 'CANDIDATES_PATH'

    with open(golden_path) as f:
        data = f.readlines()
    raw_golden = [line[line.index('.')+1:].strip() for line in data]
    with open(candidates_path) as f:
        data = f.readlines()
    raw_candidates = [line[line.index('.')+1:].strip() for line in data]

    sent_bleu_1, sent_bleu_2, sent_bleu_3, sent_bleu_4 = sent_level_bleu(raw_golden, raw_candidates)

    bleu_1 = np.array(sent_bleu_1).mean()
    bleu_2 = np.array(sent_bleu_2).mean()
    bleu_3 = np.array(sent_bleu_3).mean()
    bleu_4 = np.array(sent_bleu_4).mean()
    print(f'bleu_1: {bleu_1}, bleu_2: {bleu_2}, bleu_3: {bleu_3}, bleu_4: {bleu_4}')
