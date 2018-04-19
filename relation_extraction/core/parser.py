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

    def __init__(self, relext_model_name, models_folder="../trainedmodels/"):
        """
        Initialize a new relation parser with the given model type. This class simplifies the loading of models and
        encapsulates encoding sentences into the correct format for the given model.

        :param relext_model_name: The name of the model type that should correspond to the correct model class and
        the name of the model file
        :param models_folder: location of pre-trained model files
        """

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

        self._model.load_weights(models_folder + relext_model_name + ".kerasmodel")

        with codecs.open(os.path.join(module_location, "../../resources/properties-with-labels.txt"), encoding='utf-8') as infile:
            self._property2label = {l.split("\t")[0]: l.split("\t")[1].strip() for l in infile.readlines()}

        self._graphs_to_indices = keras_models.to_indices
        if "Context" in relext_model_name:
            self._graphs_to_indices = keras_models.to_indices_with_extracted_entities
        elif "CNN" in relext_model_name:
            self._graphs_to_indices = keras_models.to_indices_with_relative_positions

    def classify_graph_relations(self, graphs):
        """
        Classify graph relation in the given list of sentences. Each sentence should be a dictionary that has a "tokens"
        and a "edgeSet" fields. The edge set encodes pairs of entities in the sentence that would be assigned either a
        relation type or en empty relation.

        :param graphs: input as a list of dictionaries
        :return: the input graphs with labeled edges
        """
        graphs = keras_models.split_graphs(graphs)
        data_as_indices = list(self._graphs_to_indices(graphs, self._word2idx))
        probabilities = self._model.predict(data_as_indices[:-1], verbose=0)
        if len(probabilities) == 0:
            return None
        classes = np.argmax(probabilities, axis=-1)
        assert len(classes) == len(graphs)
        for gi, g in enumerate(graphs):
            if gi < len(classes):
                g_classes = classes[gi]
                for i, e in enumerate(g['edgeSet']):
                    if i < len(g_classes):
                        e['kbID'] = keras_models.idx2property[g_classes[i]]
                        e["lexicalInput"] = self._property2label[e['kbID']] if e['kbID'] in self._property2label else embeddings.all_zeroes
                    else:
                        e['kbID'] = "P0"
                        e["lexicalInput"] = embeddings.all_zeroes
        return graphs
