# coding: utf-8
# Copyright (C) 2016 UKP lab
#
# Author: Daniil Sorokin (ukp.tu-darmstadt.de/ukp-home/)
#

from utils import embedding_utils
import numpy as np
from parsing import keras_models
from semanticgraph import io
from keras.models import load_model
import json
import ast
np.random.seed(1)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('model_name')
    parser.add_argument('val_set')
    parser.add_argument('save_to')
    parser.add_argument('--data_folder', default="../../../data/")
    parser.add_argument('--word_embeddings', default="glove/glove.6B.50d.txt")

    args = parser.parse_args()

    data_folder = args.data_folder
    model_name = args.model_name

    word2idx = embedding_utils.load_word_index(data_folder + args.word_embeddings)

    val_data, _ = io.load_relation_graphs_from_file(
        data_folder + args.val_set, load_vertices=True)

    print("Applying the model to a dataset of size: {}".format(len(val_data)))

    print("Reading the property index")
    with open(data_folder + "keras-models/" + model_name + ".property2idx") as f:
        property2idx = ast.literal_eval(f.read())
    n_out = len(property2idx)
    print("N_out:", n_out)
    idx2property = {v: k for k, v in property2idx.items()}
    with open(data_folder + "properties-with-labels.txt") as infile:
        property2label = {l.split("\t")[0] : l.split("\t")[1].strip() for l in infile.readlines()}

    max_sent_len = 36
    print("Max sentence length set to: {}".format(max_sent_len))

    max_graph_size = 7
    print("Max graph size set to: {}".format(max_graph_size))

    val_sentence_matrix, val_entity_matrix, _ = keras_models.graphs_to_indices(val_data, word2idx, property2idx,
                                                                               max_sent_len, max_graph_size, mode="test")
    print("Loading the model")
    model = load_model(data_folder + "keras-models/" + model_name + ".kerasmodel")

    print("Predict")
    predictions = model.predict([val_sentence_matrix, val_entity_matrix], batch_size=200, verbose=1)
    predictions_classes = np.argmax(predictions, axis=2)
    for g_i, g in enumerate(val_data):
        edge_set = []
        for i, e in enumerate(g['edgeSet'][:max_graph_size]):
            e['kbID'] = idx2property[predictions_classes[g_i][i]]
            e["lexicalInput"] = property2label[e['kbID']] if e['kbID'] in property2label else embedding_utils.all_zeroes
            edge_set.append(e)
        g['edgeSet'] = edge_set

    print("Saving data to {}".format(data_folder + args.save_to))
    with open(data_folder + args.save_to, "w") as f:
        json.dump(val_data, f)