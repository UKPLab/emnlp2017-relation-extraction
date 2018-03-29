# coding: utf-8
# Copyright (C) 2016 UKP lab
#
# Author: Daniil Sorokin (ukp.tu-darmstadt.de/ukp-home/)
# 
# Embeddings and vocabulary utility methods

import numpy as np
import re
import logging
logger = logging.getLogger()
logger.setLevel(logging.ERROR)

all_zeroes = "ALL_ZERO"
unknown = "_UNKNOWN"

special_tokens = {"&ndash;": "–",
                  "&mdash;": "—",
                  "@card@": "0"
                  }


def load_word_index(path):
    """
    Loads only the word index from the embeddings file

    @return word to index dictionary
    """
    word2idx = {}  # Maps a word to the index in the embeddings matrix

    with open(path, 'r', encoding='utf8') as fIn:
        idx = 1
        for line in fIn:
            split = line.strip().split(' ')
            word2idx[split[0]] = idx
            idx += 1

    word2idx[all_zeroes] = 0
    word2idx[unknown] = idx

    return word2idx


def load(path):
    """
    Loads pre-trained embeddings from the specified path.
    
    @return (embeddings as an numpy array, word to index dictionary)
    """
    word2idx = {}  # Maps a word to the index in the embeddings matrix
    embeddings = []

    with open(path, 'r', encoding='utf8') as fIn:
        idx = 1               
        for line in fIn:
            split = line.strip().split(' ')                
            embeddings.append(np.array([float(num) for num in split[1:]]))
            word2idx[split[0]] = idx
            idx += 1
    
    word2idx[all_zeroes] = 0
    embedding_size = embeddings[0].shape[0]
    logger.debug("Emb. size: {}".format(embedding_size))
    embeddings = np.asarray([[0.0]*embedding_size] + embeddings, dtype='float32')
    
    rare_w_ids = list(range(idx-101,idx-1))
    unknown_emb = np.average(embeddings[rare_w_ids,:], axis=0)
    embeddings = np.append(embeddings, [unknown_emb], axis=0)
    word2idx[unknown] = idx
    idx += 1

    logger.debug("Loaded: {}".format(embeddings.shape))
    
    return embeddings, word2idx


def get_idx(word, word2idx):
    """
    Get the word index for the given word. Maps all numbers to 0, lowercases if necessary.
    
    :param word: the word in question
    :param word2idx: dictionary constructed from an embeddings file
    :return: integer index of the word
    """
    unknown_idx = word2idx[unknown]
    word = word.strip()
    if word in word2idx:
        return word2idx[word]
    elif word.lower() in word2idx:
        return word2idx[word.lower()]
    elif word in special_tokens:
        return word2idx[special_tokens[word]]
    trimmed = re.sub("(^\W|\W$)", "", word)
    if trimmed in word2idx:
        return word2idx[trimmed]
    elif trimmed.lower() in word2idx:
        return word2idx[trimmed.lower()]
    no_digits = re.sub("([0-9][0-9.,]*)", '0', word)
    if no_digits in word2idx:
        return word2idx[no_digits]
    return unknown_idx


def get_idx_sequence(word_sequence, word2idx):
    """
    Get embedding indices for the given word sequence.

    :param word_sequence: sequence of words to process
    :param word2idx: dictionary of word mapped to their embedding indices
    :return: a sequence of embedding indices
    """
    vector = []
    for word in word_sequence:
        word_idx = get_idx(word, word2idx)
        vector.append(word_idx)
    return vector


def init_random(elements_to_embed, embedding_size, add_all_zeroes=False, add_unknown=False):
    """
    Initialize a random embedding matrix for a collection of elements. Elements are sorted in order to ensure
    the same mapping from indices to elements each time.

    :param elements_to_embed: collection of elements to construct the embedding matrix for
    :param embedding_size: size of the embedding
    :param add_all_zeroes: add a all_zero embedding at index 0
    :param add_unknown: add unknown embedding at the last index
    :return: an embedding matrix and a dictionary mapping elements to rows in the matrix
    """
    elements_to_embed = sorted(elements_to_embed)
    element2idx = {all_zeroes: 0} if add_all_zeroes else {}
    element2idx.update({el: idx for idx, el in enumerate(elements_to_embed, start=len(element2idx))})
    if add_unknown:
        element2idx[unknown] = len(element2idx)

    embeddings = np.random.random((len(element2idx),embedding_size)).astype('f')
    if add_all_zeroes:
        embeddings[0] = np.zeros([embedding_size])

    return embeddings, element2idx


def timedistributed_to_one_hot(y, nb_classes):
    """
    Encodes the categorical input as on-hot vectors using the second axis of the array. The input is a two-dimensional
    array where the first dimension corresponds to time steps and the second is associated with the same input.

    :param y: the two-dimensional array.
    :param nb_classes: number of classes.
    :return: three-dimensional array where teh last dimension corresponds to the one-hot class encodings.
    """
    Y = np.zeros((y.shape + (nb_classes,)), dtype="int8")
    for i in range(len(y)):
        for j in range(len(y[i])):
            Y[i,j,y[i,j]] = 1.
    return Y


def load_blacklist(path_to_list):
    try:
        with open(path_to_list) as f:
            return_list = {l.strip() for l in f.readlines()}
        return return_list
    except Exception as ex:
        logger.error("No list found. {}".format(ex))
    try:
        with open("../" + path_to_list) as f:
            return_list = {l.strip() for l in f.readlines()}
        return return_list
    except Exception as ex:
        logger.error("No list found. {}".format(ex))
        return set()
