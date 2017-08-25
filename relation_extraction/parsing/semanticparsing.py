# coding: utf-8
# Copyright (C) 2016 UKP lab
#
# Author: Daniil Sorokin (ukp.tu-darmstadt.de/ukp-home/)
#
import numpy as np
np.random.seed(1)

from parsing import sp_models
from semanticgraph import graph_utils
import keras.models
import ast
from utils import embedding_utils


class RelParser:

    def __init__(self, relext_model_name, data_folder="../../data/", embeddings_location="glove/glove.6B.50d.txt"):

        self._model = keras.models.load_model(data_folder + "keras-models/" + relext_model_name + ".kerasmodel")
        self._property2idx = {}

        with open(data_folder + "keras-models/" + relext_model_name + ".property2idx") as f:
            self._property2idx = ast.literal_eval(f.read())

        self._max_sent_len = self._model.get_input_shape_at(0)[0][1]

        self._embeddings, self._word2idx = embedding_utils.load(data_folder + embeddings_location)
        print("Loaded embeddings:", self._embeddings.shape)
        self._idx2word = {v: k for k, v in self._word2idx.items()}

        with open(data_folder + "properties-with-labels.txt") as infile:
            self._property2label = {l.split("\t")[0] : l.split("\t")[1].strip() for l in infile.readlines()}
        self._idx2property = {v: k for k, v in self._property2idx.items()}

        self._graphs_to_indices = sp_models.to_indices
        if "Ghost" in relext_model_name:
            self._graphs_to_indices = sp_models.to_indices_with_ghost_entities
        elif "CNN" in relext_model_name:
            self._graphs_to_indices = sp_models.to_indices_with_relative_positions

    def sem_parse(self, g, verbose=False):
        if verbose:
            print(" ".join(g['tokens']))

        data_as_indices = self._graphs_to_indices([g], self._word2idx, self._property2idx, self._max_sent_len, mode="test")
        probabilities = self._model.predict(data_as_indices[:-1], verbose=0)
        classes = np.argmax(probabilities, axis = 1)
        edge_set = []
        for i, e in enumerate(g['edgeSet']):
            e['kbID'] = self._idx2property[classes[i]]
            e["lexicalInput"] = self._property2label[e['kbID']] if e['kbID'] in self._property2label else embedding_utils.all_zeroes
            edge_set.append(e)
            if verbose:
                graph_utils.print_edge(e, g)
                sorted_probabilities = sorted(enumerate(probabilities[i]), key=lambda x: x[1], reverse=True)
                print("{} ({:.4}), {}".format(e['kbID'], np.max(probabilities[i]),
                                              [(self._idx2property[p_id], p_prob) for p_id, p_prob in sorted_probabilities[1:4]]))
        g['edgeSet'] = edge_set
        return g
