# coding: utf-8
# Copyright (C) 2016 UKP lab
#
# Author: Daniil Sorokin (ukp.tu-darmstadt.de/ukp-home/)
#
import numpy as np
np.random.seed(1)

from keras import callbacks
from keras.utils import np_utils
import hyperopt as hy
import json

from evaluation import metrics
from core import keras_models, embeddings
from graph import io


def f_train(params):
    model = getattr(keras_models, model_name)(params, embedding_matrix, max_sent_len, n_out)
    callback_history = model.fit(train_as_indices[:-1],
                                 [train_y_properties_one_hot],
                                 epochs=20, batch_size=keras_models.model_params['batch_size'], verbose=1,
                                 validation_data=(
                                     val_as_indices[:-1], val_y_properties_one_hot),
                                 callbacks=[callbacks.EarlyStopping(monitor="val_loss", patience=1, verbose=1)])

    predictions = model.predict(val_as_indices[:-1], batch_size=16, verbose=1)
    predictions_classes = np.argmax(predictions, axis=1)
    _, _, acc = metrics.compute_micro_PRF(predictions_classes, val_as_indices[-1])
    return {'loss': -acc, 'status': hy.STATUS_OK}


def evaluate(model, data_input, gold_output):
    predictions = model.predict(data_input, batch_size=keras_models.model_params['batch_size'], verbose=1)
    if len(predictions.shape) == 3:
        predictions_classes = np.argmax(predictions, axis=2)
        train_batch_f1 = metrics.accuracy_per_sentence(predictions_classes, gold_output)
        print("Results (per sentence): ", train_batch_f1)
        train_y_properties_stream = gold_output.reshape(gold_output.shape[0] * gold_output.shape[1])
        predictions_classes = predictions_classes.reshape(predictions_classes.shape[0] * predictions_classes.shape[1])
        class_mask = train_y_properties_stream != 0
        train_y_properties_stream = train_y_properties_stream[class_mask]
        predictions_classes = predictions_classes[class_mask]
    else:
        predictions_classes = np.argmax(predictions, axis=1)
        train_y_properties_stream = gold_output

    accuracy = metrics.accuracy(predictions_classes, train_y_properties_stream)
    micro_scores = metrics.compute_micro_PRF(predictions_classes, train_y_properties_stream, empty_label=keras_models.p0_index)
    print("Results: Accuracy: ", accuracy)
    print("Results: Micro-Average F1: ", micro_scores)
    return predictions_classes, predictions


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('model_name')
    parser.add_argument('mode', choices=['train', 'optimize', 'train-continue'])
    parser.add_argument('train_set')
    parser.add_argument('val_set')
    parser.add_argument('--models_folder', default="../trainedmodels/")

    args = parser.parse_args()

    model_name = args.model_name
    mode = args.mode

    embedding_matrix, word2idx = embeddings.load(keras_models.model_params['wordembeddings'])
    print("Loaded embeddings:", embedding_matrix.shape)

    training_data, _ = io.load_relation_graphs_from_file(args.train_set, load_vertices=True)
    val_data, _ = io.load_relation_graphs_from_file(args.val_set, load_vertices=True)

    print("Training data size: {}".format(len(training_data)))
    print("Validation data size: {}".format(len(val_data)))

    max_sent_len = keras_models.model_params['max_sent_len']
    print("Max sentence length set to: {}".format(max_sent_len))

    to_one_hot = np_utils.to_categorical
    graphs_to_indices = keras_models.to_indices
    if "Context" in model_name:
        to_one_hot = embeddings.timedistributed_to_one_hot
        graphs_to_indices = keras_models.to_indices_with_extracted_entities
    elif "CNN" in model_name:
        graphs_to_indices = keras_models.to_indices_with_relative_positions

    train_as_indices = list(graphs_to_indices(training_data, word2idx))
    print("Dataset shapes: {}".format([d.shape for d in train_as_indices]))
    training_data = None
    n_out = len(keras_models.property2idx)
    print("N_out:", n_out)

    val_as_indices = list(graphs_to_indices(val_data, word2idx))
    val_data = None

    if "train" in mode:
        print("Training the model")
        print("Initialize the model")
        model = getattr(keras_models, model_name)(keras_models.model_params, embedding_matrix, max_sent_len, n_out)
        if "continue" in mode:
            print("Load pre-trained weights")
            model.load_weights(args.models_folder + model_name + ".kerasmodel")

        train_y_properties_one_hot = to_one_hot(train_as_indices[-1], n_out)
        val_y_properties_one_hot = to_one_hot(val_as_indices[-1], n_out)

        callback_history = model.fit(train_as_indices[:-1],
                                     [train_y_properties_one_hot],
                                     epochs=50, batch_size=keras_models.model_params['batch_size'], verbose=1,
                                     validation_data=(
                                         val_as_indices[:-1], val_y_properties_one_hot),
                                     callbacks=[callbacks.EarlyStopping(monitor="val_loss", patience=5, verbose=1),
                                                callbacks.ModelCheckpoint(
                                                    args.models_folder + model_name + ".kerasmodel",
                                                    monitor='val_loss', verbose=1, save_best_only=True)])
    elif mode == "optimize":
        import optimization_space

        space = optimization_space.space

        train_y_properties_one_hot = to_one_hot(train_as_indices[-1], n_out)
        val_y_properties_one_hot = to_one_hot(val_as_indices[-1], n_out)

        trials = hy.Trials()
        best = hy.fmin(f_train, space, algo=hy.rand.suggest, max_evals=10, trials=trials)
        print("Best trial:", best)
        print("Details:", trials.best_trial)
        print("Saving trials.")
        with open("../data/trials/" + model_name + "_final_trails.json", 'w') as ftf:
            json.dump([(t['misc']['vals'], t['result']) for t in trials.trials], ftf)

    print("Loading the best model")
    model = getattr(keras_models, model_name)(keras_models.model_params, embedding_matrix, max_sent_len, n_out)
    model.load_weights(args.models_folder + model_name + ".kerasmodel")


    print("Results on the training set")
    evaluate(model, train_as_indices[:-1], train_as_indices[-1])
    print("Results on the validation set")
    evaluate(model, val_as_indices[:-1], val_as_indices[-1])
