# coding: utf-8
# Copyright (C) 2016 UKP lab
#
# Author: Daniil Sorokin (ukp.tu-darmstadt.de/ukp-home/)
#
import numpy as np
np.random.seed(1)

import os
import codecs

from core import keras_models
from core import embeddings


class RelParser:

    def __init__(self, relext_model_name, models_foldes="../trainedmodels/"):

        module_location = os.path.abspath(__file__)
        module_location = os.path.dirname(module_location)

        model_params = keras_models.model_params
        max_sent_len = keras_models.model_params['max_sent_len']
        self._embeddings, self._word2idx = embeddings.load(keras_models.model_params['wordembeddings'])
        print("Loaded embeddings:", self._embeddings.shape)
        self._idx2word = {v: k for k, v in self._word2idx.items()}

        self._model = getattr(keras_models, relext_model_name)(model_params,
                                                         np.zeros((len(self._word2idx), 50), dtype='float32'),
                                                         max_sent_len, len(keras_models.property2idx))

        self._model.load_weights(models_foldes + relext_model_name + ".kerasmodel")

        with codecs.open(os.path.join(module_location, "../../resources/properties-with-labels.txt"), encoding='utf-8') as infile:
            self._property2label = {l.split("\t")[0]: l.split("\t")[1].strip() for l in infile.readlines()}

        self._graphs_to_indices = keras_models.to_indices
        if "Context" in relext_model_name:
            self._graphs_to_indices = keras_models.to_indices_with_extracted_entities
        elif "CNN" in relext_model_name:
            self._graphs_to_indices = keras_models.to_indices_with_relative_positions

    def classify_graph_relations(self, g):
        data_as_indices = list(self._graphs_to_indices([g], self._word2idx))
        probabilities = self._model.predict(data_as_indices[:-1], verbose=0)
        if len(probabilities) == 0:
            return None
        probabilities = probabilities[0]
        classes = np.argmax(probabilities, axis=1)
        for i, e in enumerate(g['edgeSet']):
            if i < len(probabilities):
                e['kbID'] = keras_models.idx2property[classes[i]]
                e["lexicalInput"] = self._property2label[e['kbID']] if e['kbID'] in self._property2label else embeddings.all_zeroes
            else:
                e['kbID'] = "P0"
                e["lexicalInput"] = embeddings.all_zeroes
        return g
