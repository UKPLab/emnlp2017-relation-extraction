# coding: utf-8
# Copyright (C) 2016 UKP lab
#
# Author: Daniil Sorokin (ukp.tu-darmstadt.de/ukp-home/)
#

from utils import evaluation_utils, embedding_utils
import numpy as np
np.random.seed(1)

from parsing import keras_models
from keras import callbacks
from keras.utils import np_utils
from semanticgraph import io
import hyperopt as hy
import json
import ast


p0_index = 1


def f_train(params):
    model = getattr(keras_models, model_name)(params, embeddings, max_sent_len, n_out)
    callback_history = model.fit(train_as_indices[:-1],
                                 [train_y_properties_one_hot],
                                 nb_epoch=20, batch_size=256 if params['gpu'] else 200, verbose=1,
                                 validation_data=(
                                     val_as_indices[:-1], val_y_properties_one_hot),
                                 callbacks=[callbacks.EarlyStopping(monitor="val_loss", patience=1, verbose=1)])

    predictions = model.predict(val_as_indices[:-1], batch_size=256, verbose=1)
    predictions_classes = np.argmax(predictions, axis=1)
    _,_,acc = evaluation_utils.evaluate_instance_based(predictions_classes,
                                                       val_as_indices[-1])
    print('Acc:', acc)
    with open(data_folder + "trials/" + model_name + "_current_trails.log", 'a') as ctf:
        ctf.write("{}\t{}\n".format(params, acc))
    return {'loss': -acc, 'status': hy.STATUS_OK}


def evaluate(model, input, gold_output):
    predictions = model.predict(input, batch_size=256, verbose=1)
    if len(predictions.shape) == 3:
        predictions_classes = np.argmax(predictions, axis=2)
        train_batch_f1 = evaluation_utils.evaluate_batch_based(predictions_classes,
                                                               gold_output)
        print("Results (per batch): ", train_batch_f1)
        train_y_properties_stream = gold_output.reshape(gold_output.shape[0] * gold_output.shape[1])
        predictions_classes = predictions_classes.reshape(predictions_classes.shape[0] * predictions_classes.shape[1])
        p_indices = train_y_properties_stream != 0
        train_y_properties_stream = train_y_properties_stream[p_indices]
        predictions_classes = predictions_classes[p_indices]
    else:
        predictions_classes = np.argmax(predictions, axis=1)
        train_y_properties_stream = gold_output

    train_f1 = evaluation_utils.evaluate_instance_based(predictions_classes,
                                                        train_y_properties_stream, empty_label=p0_index)
    print("Results (per relation): ", train_f1)
    return predictions_classes


def load_the_model(model_name, model_params, embeddings, max_sent_len, n_out):
    import h5py
    print("Loading the best model")
    model = getattr(keras_models, model_name)(model_params, embeddings, max_sent_len, n_out)
    f = h5py.File(args.models_folder + model_name + ".kerasmodel", mode='r')
    model.load_weights_from_hdf5_group(f['model_weights'])
    return model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('model_name')
    parser.add_argument('--mode', default="train-test", choices=['train-test', 'train', 'test', 'optimize',
                                                                 'train-plus-test'])
    parser.add_argument('--models_folder', default="../trainedmodels/")
    parser.add_argument('--model_params', default="model_params.json")
    parser.add_argument('--word_embeddings', default="../resources/glove/glove.6B.50d.txt")
    parser.add_argument('--train_set', default="../data/wikipedia-wikidata/enwiki-20160501/semantic-graphs-filtered-training.02_06.json")
    parser.add_argument('--val_set', default="../data/wikipedia-wikidata/enwiki-20160501/semantic-graphs-filtered-validation.02_06.json")
    parser.add_argument('--test_set', default="../data/wikipedia-wikidata/enwiki-20160501/semantic-graphs-filtered-held-out.02_06.json")
    parser.add_argument('--property_index')
    parser.add_argument('-s', action='store_true', help="Use only a portion of the training and validation sets.")

    args = parser.parse_args()

    model_name = args.model_name
    mode = args.mode

    with open(args.model_params) as f:
        model_params = json.load(f)

    embeddings, word2idx = embedding_utils.load(args.word_embeddings)
    print("Loaded embeddings:", embeddings.shape)

    training_data, _ = io.load_relation_graphs_from_file(args.train_set, load_vertices=True)

    val_data, _ = io.load_relation_graphs_from_file(args.val_set, load_vertices=True)

    if args.s:
        training_data = training_data[:len(training_data) // 3]
        print("Training data size set to: {}".format(len(training_data)))
        val_data = val_data[:len(val_data) // 3]
        print("Validation data size set to: {}".format(len(val_data)))

    if mode in ['test', 'train-plus-test']:
        print("Reading the property index")
        with open(args.models_folder + model_name + ".property2idx") as f:
            property2idx = ast.literal_eval(f.read())
    elif args.property_index:
        print("Reading the property index from parameter")
        with open(args.property_index) as f:
            property2idx = ast.literal_eval(f.read())
    else:
        _, property2idx = embedding_utils.init_random({e["kbID"] for g in training_data
                                                       for e in g["edgeSet"]} | {"P0"}, 1, add_all_zeroes=True, add_unknown=True)

    max_sent_len = max(len(g["tokens"]) for g in training_data)
    print("Max sentence length:", max_sent_len)
    max_sent_len = 36
    print("Max sentence length set to: {}".format(max_sent_len))

    
    to_one_hot = np_utils.to_categorical
    graphs_to_indices = keras_models.to_indices
    if "Context" in model_name:
        to_one_hot = embedding_utils.timedistributed_to_one_hot
        graphs_to_indices = keras_models.to_indices_with_real_entities
    elif "CNN" in model_name:
        graphs_to_indices = keras_models.to_indices_with_relative_positions

    _, position2idx = embedding_utils.init_random(np.arange(-max_sent_len, max_sent_len), 1, add_all_zeroes=True)
    train_as_indices = list(graphs_to_indices(training_data, word2idx, property2idx, max_sent_len,
                                         embeddings=embeddings, position2idx=position2idx))
    print("Dataset shapes: {}".format([d.shape for d in train_as_indices]))
    training_data = None
    n_out = len(property2idx)
    print("N_out:", n_out)

    val_as_indices = list(graphs_to_indices(val_data, word2idx, property2idx, max_sent_len,
                                       embeddings=embeddings, position2idx=position2idx))
    val_data = None

    if "train" in mode:
        print("Save property dictionary.")
        with open(args.models_folder + model_name + ".property2idx", 'w') as outfile:
            outfile.write(str(property2idx))

        print("Training the model")
        if "plus" in mode:
            print("Load model")
            model = load_the_model(model_name, model_params, embeddings, max_sent_len, n_out)

        else:
            print("Initialize the model")
            model = getattr(keras_models, model_name)(model_params, embeddings, max_sent_len, n_out)

        train_y_properties_one_hot = to_one_hot(train_as_indices[-1], n_out)
        val_y_properties_one_hot = to_one_hot(val_as_indices[-1], n_out)

        callback_history = model.fit(train_as_indices[:-1],
                                     [train_y_properties_one_hot],
                                     nb_epoch=100, batch_size=256 if model_params['gpu'] else 200, verbose=1,
                                     validation_data=(
                                         val_as_indices[:-1], val_y_properties_one_hot),
                                     callbacks=[callbacks.EarlyStopping(monitor="val_loss", patience=3, verbose=1),
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

    if "test" in mode:
        model = load_the_model(model_name, model_params, embeddings, max_sent_len, n_out)

        print("Testing")
        print("Results on the training set")
        evaluate(model, train_as_indices[:-1], train_as_indices[-1])
        print("Results on the validation set")
        evaluate(model, val_as_indices[:-1], val_as_indices[-1])

        print("Results on the test set")
        test_set, _ = io.load_relation_graphs_from_file(args.test_set)
        test_as_indices = list(graphs_to_indices(
            test_set, word2idx, property2idx, max_sent_len, embeddings=embeddings, position2idx=position2idx))
        evaluate(model, test_as_indices[:-1], test_as_indices[-1])
