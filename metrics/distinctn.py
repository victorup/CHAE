#!/usr/bin/python
# -*- coding:utf-8 -*-
import nltk


def distinctn(text, n):
    '''text: str of text
       return: unique n-gram / all n-gram'''
    tokens = nltk.word_tokenize(text)
    uni_ngram_list = []
    n_gram_num = len(tokens) - n + 1
    for i in range(n_gram_num):
        n_gram = tokens[i:i+n]
        uni_ngram_list.append(' '.join(n_gram))
    uni_ngram_list = set(uni_ngram_list)
    if n_gram_num == 0:
        return 1
    return len(uni_ngram_list) / n_gram_num


if __name__ == '__main__':
    pred_path = 'PRED_PATH'

    with open(pred_path) as f:
        data = f.readlines()

    examples = [line[line.index('.')+1:].strip() for line in data]
    d_1 = 0
    d_2 = 0
    for eg in examples:
        d_1 += distinctn(eg, 1)
        d_2 += distinctn(eg, 2)

    print('d-1:', d_1 / len(examples), 'd-2:', d_2 / len(examples))
