# coding: utf-8
# Copyright (C) 2016 UKP lab
#
# Author: Daniil Sorokin (ukp.tu-darmstadt.de/ukp-home/)
#
import itertools
import numpy as np
np.random.seed(1)

from keras import layers, models
from keras import backend as K
from keras import regularizers

from utils import embedding_utils
from utils import graph

import tqdm

from semanticgraph import graph_utils

RESOURCES_FOLDER = "../resources/"
property_blacklist = embedding_utils.load_blacklist(RESOURCES_FOLDER + "property_blacklist.txt")


def model_RnnMarkers(p, embeddings, max_sent_len, n_out):
    print("Parameters:", p)
    # Take sentence encoded as indices and convert it to embeddings
    sentence_input = layers.Input(shape=(max_sent_len,), dtype='int32', name='sentence_input')
    word_embeddings = layers.Embedding(output_dim=embeddings.shape[1], input_dim=embeddings.shape[0],
                         input_length=max_sent_len, weights=[embeddings],
                         mask_zero=True, trainable=False)(sentence_input)
    word_embeddings = layers.Dropout(p['dropout1'])(word_embeddings)

    # Take token markers that identify entity positions, convert to position embeddings
    entity_markers = layers.Input(shape=(max_sent_len,), dtype='int8', name='entity_markers')
    pos_embeddings = layers.Embedding(output_dim=p['position_emb'], input_dim=4, input_length=max_sent_len,
                         mask_zero=True, W_regularizer = regularizers.l2(), trainable=True)(entity_markers)

    # Merge word and position embeddings and apply the specified amount of RNN layers
    x = layers.merge([word_embeddings, pos_embeddings], mode="concat")
    for i in range(p["rnn1_layers"]-1):
        x = getattr(layers, p['rnn1'])(p['units1'], return_sequences=True, consume_less='gpu' if p['gpu'] else "cpu")(x)
    sentence_vector = getattr(layers, p['rnn1'])(p['units1'], consume_less='gpu' if p['gpu'] else "cpu")(x)

    # # If specified apply a set of fully-connected layers
    # for i in range(p["penultimate_layers"]):
    #     sentence_vector = layers.Dense(p['units2'], activation="tanh")(sentence_vector)

    # Apply softmax
    sentence_vector = layers.Dropout(p['dropout1'])(sentence_vector)
    main_output = layers.Dense(n_out, activation = "softmax", name='main_output')(sentence_vector)

    model = models.Model(input=[sentence_input, entity_markers], output=[main_output])
    model.compile(optimizer=p['optimizer'], loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def model_CNN(p, embeddings, max_sent_len, n_out):
    print("Parameters:", p)
    # Take sentence encoded as indices split in three parts and convert it to embeddings
    sentence_input = layers.Input(shape=(max_sent_len,), dtype='int32', name='sentence_input')
    word_embeddings = layers.Embedding(output_dim=embeddings.shape[1],
                                       input_dim=embeddings.shape[0],
                                       input_length=max_sent_len, weights=[embeddings],
                                       mask_zero=True, trainable=False)(sentence_input)
    word_embeddings = layers.Dropout(p['dropout1'])(word_embeddings)

    # Take token markers that identify entity positions, convert to position embeddings
    entity_markers = layers.Input(shape=(2, max_sent_len,), dtype='int8', name='entity_markers')

    pos_embeddings = layers.wrappers.TimeDistributed(layers.Embedding(output_dim=p['position_emb'], input_dim=(max_sent_len*2)+1, input_length=max_sent_len,
                                                                      mask_zero=False, W_regularizer = regularizers.l2(), trainable=True),  name='pos_embedding')(entity_markers)

    pos_embeddings = layers.Permute((2,1,3))(pos_embeddings)
    pos_embeddings = layers.Reshape((max_sent_len, p['position_emb']*2))(pos_embeddings)

    # Merge word and position embeddings and apply the specified amount of CNN layers
    x = layers.merge([word_embeddings, pos_embeddings], mode="concat")

    x = MaskedConvolution1D(nb_filter=p['units1'], filter_length=p['window_size'], border_mode='same')(x)
    sentence_vector = MaskedGlobalMaxPooling1D()(x)

    sentence_vector = layers.Lambda(lambda l: K.tanh(l))(sentence_vector)

    # Apply softmax
    sentence_vector = layers.Dropout(p['dropout1'])(sentence_vector)
    main_output = layers.Dense(n_out, activation = "softmax", name='main_output')(sentence_vector)

    model = models.Model(input=[sentence_input, entity_markers], output=[main_output])
    model.compile(optimizer=p['optimizer'], loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def model_PCNN(p, embeddings, max_sent_len, n_out):
    print("Parameters:", p)
    # Take sentence encoded as indices and convert it to embeddings
    sentence_input = layers.Input(shape=(max_sent_len,), dtype='int32', name='sentence_input')
    word_embeddings = layers.Embedding(output_dim=embeddings.shape[1], input_dim=embeddings.shape[0],
                                       input_length=max_sent_len, weights=[embeddings],
                                       mask_zero=True, trainable=False)(sentence_input)
    word_embeddings = layers.Dropout(p['dropout1'])(word_embeddings)

    # Take token markers that identify entity positions, convert to position embeddings
    entity_markers = layers.Input(shape=(max_sent_len,), dtype='int8', name='entity_markers')
    pos_embeddings = layers.Embedding(output_dim=p['position_emb'], input_dim=4, input_length=max_sent_len,
                                      mask_zero=True, W_regularizer = regularizers.l2(), trainable=True)(entity_markers)

    # Merge word and position embeddings and apply the specified amount of CNN layers
    x = layers.merge([word_embeddings, pos_embeddings], mode="concat")

    for i in range(p["rnn1_layers"]-1):
        x = getattr(layers, p['rnn1'])(p['units1'], return_sequences=True, consume_less='gpu' if p['gpu'] else "cpu")(x)
    sentence_vector = getattr(layers, p['rnn1'])(p['units1'], consume_less='gpu' if p['gpu'] else "cpu")(x)

    # If specified apply a set of fully-connected layers
    for i in range(p["penultimate_layers"]):
        sentence_vector = layers.Dense(p['units2'], activation="tanh")(sentence_vector)

    # Apply softmax
    sentence_vector = layers.Dropout(p['dropout1'])(sentence_vector)
    main_output = layers.Dense(n_out, activation = "softmax", name='main_output')(sentence_vector)

    model = models.Model(input=[sentence_input, entity_markers], output=[main_output])
    model.compile(optimizer=p['optimizer'], loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def model_RnnMarkersAttention(p, embeddings, max_sent_len, n_out):
    print("Parameters:", p)

    # Take sentence encoded as indices and convert it to embeddings
    sentence_input = layers.Input(shape=(max_sent_len,), dtype='int32', name='sentence_input')
    word_embeddings = layers.Embedding(output_dim=embeddings.shape[1], input_dim=embeddings.shape[0],
                         input_length=max_sent_len, weights=[embeddings],
                         mask_zero=True, trainable=False)(sentence_input)
    word_embeddings = layers.Dropout(p['dropout1'])(word_embeddings)

    # Take token markers that identify entity positions, convert to position embeddings
    entity_markers = layers.Input(shape=(max_sent_len,), dtype='int8', name='entity_markers')
    pos_embeddings = layers.Embedding(output_dim=p['position_emb'], input_dim=4, input_length=max_sent_len,
                         mask_zero=True, W_regularizer = regularizers.l2(),  trainable=True)(entity_markers)

    # Merge word and position embeddings and apply the specified amount of RNN layers
    x = layers.merge([word_embeddings, pos_embeddings], mode="concat")
    for i in range(p["rnn1_layers"]-1):
        x = getattr(layers, p['rnn1'])(p['units1'], return_sequences=True, consume_less='gpu' if p['gpu'] else "cpu")(x)
    sentence_vector = getattr(layers, p['rnn1'])(p['units1'], return_sequences=False, consume_less='gpu' if p['gpu'] else "cpu")(x)

    ### Attention ###
    # Compute memory vectors for each word and then compute score by multiplying each with the sentence vector
    word_memories = layers.wrappers.TimeDistributed(layers.Dense(p['units1'], activation="relu"),
                                                    name = "word_memories")(word_embeddings)
    word_scores = layers.Merge(mode=lambda inputs: K.batch_dot(inputs[0], inputs[1], axes=(1,2)),
                     output_shape=(max_sent_len,), name = "scores")([sentence_vector, word_memories])
    word_scores = layers.Activation('softmax')(word_scores)
    # Compute context vectors for each word and take a sum of them weighted by the computed scores
    context_vectors = layers.wrappers.TimeDistributed(layers.Dense(p['units1'], activation="relu"),
                                                      name = "context_vectors")(word_embeddings)
    memory_output = layers.Merge(mode=lambda inputs: K.batch_dot(inputs[0], inputs[1], axes=(1,1)),
                     output_shape=(p['units1'],), name = "memory_output")([context_vectors, word_scores])

    # Merge the sentence vector and the memory output
    x = layers.merge([memory_output, sentence_vector], mode=p['memory_merge'])

    # Apply softmax
    x = layers.Dropout(p['dropout1'])(x)
    main_output = layers.Dense(n_out, activation = "softmax", name='main_output')(x)

    model = models.Model(input=[sentence_input, entity_markers], output=[main_output])
    model.compile(optimizer=p['optimizer'], loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def model_RnnMarkersAttentionV2(p, embeddings, max_sent_len, n_out):
    print("Parameters:", p)

    # Take sentence encoded as indices and convert it to embeddings
    sentence_input = layers.Input(shape=(max_sent_len,), dtype='int32', name='sentence_input')
    word_embeddings = layers.Embedding(output_dim=embeddings.shape[1], input_dim=embeddings.shape[0],
                               input_length=max_sent_len, weights=[embeddings],
                               mask_zero=True, trainable=False)(sentence_input)
    word_embeddings = layers.Dropout(p['dropout1'])(word_embeddings)

    # Take token markers that identify entity positions, convert to position embeddings
    entity_markers = layers.Input(shape=(max_sent_len,), dtype='int8', name='entity_markers')
    pos_embeddings = layers.Embedding(output_dim=p['position_emb'], input_dim=4, input_length=max_sent_len,
                         mask_zero=True, W_regularizer = regularizers.l2(),  trainable=True)(entity_markers)

    # Merge word and position embeddings and apply the specified amount of RNN layers
    x = layers.merge([word_embeddings, pos_embeddings], mode="concat")
    for i in range(p["rnn1_layers"]-1):
        x = getattr(layers, p['rnn1'])(p['units1'], return_sequences=True, consume_less='gpu' if p['gpu'] else "cpu")(x)
    sentence_matrix = getattr(layers, p['rnn1'])(p['units1'], return_sequences=True, consume_less='gpu' if p['gpu'] else "cpu")(x)
    # Store all output vectors of the RNN and the final one as the sentence vector
    sentence_vector = GetOutput()(x)

    ### Attention ###
    # Compute memory vectors for each word and then compute score by multiplying each with the sentence vector
    word_memories = layers.wrappers.TimeDistributed(layers.Dense(p['units1'], activation="relu"),
                                                    name = "word_memories")(word_embeddings)
    word_scores = layers.Merge(mode=lambda inputs: K.batch_dot(inputs[0], inputs[1], axes=(1,2)),
                     output_shape=(max_sent_len,), name = "word_scores")([sentence_vector, word_memories])
    word_scores = layers.Activation('softmax')(word_scores)
    # Take a sum of teh RNN outputs at each step weighted by the computed scores
    memory_output = layers.Merge(mode=lambda inputs: K.batch_dot(inputs[0], inputs[1], axes=(1,1)),
                     output_shape=(p['units1'],), name = "memory_output")([sentence_matrix, word_scores])

    # Merge the sentence vector and the memory output
    x = layers.merge([memory_output, sentence_vector], mode=p['memory_merge'])

    # Apply softmax
    x = layers.Dropout(p['dropout1'])(x)
    main_output = layers.Dense(n_out, activation = "softmax", name='main_output')(x)

    model = models.Model(input=[sentence_input, entity_markers], output=[main_output])
    model.compile(optimizer=p['optimizer'], loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def model_RnnMarkersAttentionEmb(p, embeddings, max_sent_len, n_out):
    print("Parameters:", p)

    # Take sentence encoded as indices and convert it to embeddings
    sentence_input = layers.Input(shape=(max_sent_len,), dtype='int32', name='sentence_input')
    word_embeddings = layers.Embedding(output_dim=embeddings.shape[1], input_dim=embeddings.shape[0],
                                input_length=max_sent_len, weights=[embeddings],
                                mask_zero=True, trainable=False)(sentence_input)
    word_embeddings = layers.Dropout(p['dropout1'])(word_embeddings)

    # Convert each token to a memory embedding that are leanred
    memory_embeddings = layers.Embedding(output_dim=embeddings.shape[1], input_dim=embeddings.shape[0],
                                input_length=max_sent_len, mask_zero=True, trainable=True,
                                  W_regularizer = regularizers.l2())(sentence_input)
    memory_embeddings = layers.Dropout(p['dropout1'])(memory_embeddings)

    # Take token markers that identify entity positions, convert to position embeddings
    entity_markers = layers.Input(shape=(max_sent_len,), dtype='int8', name='entity_markers')
    pos_embeddings = layers.Embedding(output_dim=p['position_emb'], input_dim=4, input_length=max_sent_len,
                         mask_zero=True, W_regularizer = regularizers.l2(), trainable=True)(entity_markers)

    # Merge word and position embeddings and apply the specified amount of RNN layers
    x = layers.merge([word_embeddings, pos_embeddings], mode="concat")
    for i in range(p["rnn1_layers"]-1):
        x = getattr(layers, p['rnn1'])(p['units1'], return_sequences=True, consume_less='gpu' if p['gpu'] else "cpu")(x)
    sentence_vector = getattr(layers, p['rnn1'])(p['units1'], return_sequences=False, consume_less='gpu' if p['gpu'] else "cpu")(x)

    ### Attention ###
    # Compute a memory vector for a sentence and compute a score for each word by multiplying it with word memory embeddings
    sentence_memory = layers.Dense(embeddings.shape[1], activation="tanh")(sentence_vector)
    word_scores = layers.Merge(mode=lambda inputs: K.batch_dot(inputs[0], inputs[1], axes=(1,2)),
                     output_shape=(max_sent_len,), name = "word_scores")([sentence_memory, memory_embeddings])
    word_scores = layers.Activation('softmax')(word_scores)
    memory_output = layers.Merge(mode=lambda inputs: K.batch_dot(inputs[0], inputs[1], axes=(1,1)),
                     output_shape=(embeddings.shape[1],), name = "memory_output")([word_embeddings, word_scores])

    # Merge the memory output and the sentence vector
    x = layers.merge([memory_output, sentence_vector], mode=p['memory_merge'])

    # Apply softmax
    x = layers.Dropout(p['dropout1'])(x)
    main_output = layers.Dense(n_out, activation = "softmax", name='main_output')(x)

    model = models.Model(input=[sentence_input, entity_markers], output=[main_output])
    model.compile(optimizer=p['optimizer'], loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def model_RnnMarkersSumGhosts(p, embeddings, max_sent_len, n_out):
    print("Parameters:", p)

    # Take sentence encoded as indices and convert it to embeddings
    sentence_input = layers.Input(shape=(max_sent_len,), dtype='int32', name='sentence_input')
    # Repeat the input 3 times as will need it once for the target entity pair and twice for the ghost pairs
    x = layers.RepeatVector(MAX_EDGES_PER_GRAPH)(sentence_input)
    word_embeddings = layers.wrappers.TimeDistributed(layers.Embedding(output_dim=embeddings.shape[1], input_dim=embeddings.shape[0],
                                                                input_length=max_sent_len, weights=[embeddings],
                                                                mask_zero=True, trainable=False))(x)
    word_embeddings = layers.Dropout(p['dropout1'])(word_embeddings)

    # Take token markers that identify entity positions, convert to position embeddings
    entity_markers = layers.Input(shape=(MAX_EDGES_PER_GRAPH, max_sent_len,), dtype='int8', name='entity_markers')
    pos_embeddings = layers.wrappers.TimeDistributed(layers.Embedding(output_dim=p['position_emb'],
                                                         input_dim=4, input_length=max_sent_len,
                                                         mask_zero=True, W_regularizer = regularizers.l2(),
                                                         trainable=True))(entity_markers)

    # Merge word and position embeddings and apply the specified amount of RNN layers
    x = layers.merge([word_embeddings, pos_embeddings], mode="concat")
    for i in range(p["rnn1_layers"]-1):
        x = layers.wrappers.TimeDistributed(
            getattr(layers, p['rnn1'])(p['units1'], return_sequences=True,
                                       consume_less='gpu' if p['gpu'] else "cpu"))(x)
    sentence_matrix = layers.wrappers.TimeDistributed(
        getattr(layers, p['rnn1'])(p['units1'],
                                   return_sequences=False, consume_less='gpu' if p['gpu'] else "cpu"))(x)

    # Take the vector of the sentences with the target entity pair
    layers_to_concat = []
    for i in range(MAX_EDGES_PER_GRAPH):
        sentence_vector = layers.Lambda(lambda l: l[:, i], output_shape=(p['units1'],))(sentence_matrix)
        if i == 0:
            context_vectors = layers.Lambda(lambda l: l[:, i+1:], output_shape=(MAX_EDGES_PER_GRAPH-1, p['units1']))(sentence_matrix)
        elif i == MAX_EDGES_PER_GRAPH - 1:
            context_vectors = layers.Lambda(lambda l: l[:, :i], output_shape=(MAX_EDGES_PER_GRAPH-1, p['units1']))(sentence_matrix)
        else:
            context_vectors = layers.Lambda(lambda l: K.concatenate([l[:, :i], l[:, i+1:]], axis=1), output_shape=(MAX_EDGES_PER_GRAPH-1, p['units1']))(sentence_matrix)
        if p['contex.sum'] == 'max':
            context_vector = layers.GlobalMaxPooling1D()(context_vectors)
        elif p['contex.sum'] == 'avg':
            context_vector = layers.GlobalAveragePooling1D()(context_vectors)
        else:
            context_vector = GlobalSumPooling1D()(context_vectors)
        edge_vector = layers.merge([sentence_vector, context_vector], mode="concat")
        edge_vector = layers.Reshape((1, p['units1']*2))(edge_vector)
        layers_to_concat.append(edge_vector)
    # edge_vectors = layers.Lambda(lambda l: K.stack(l), output_shape=(MAX_EDGES_PER_GRAPH-1, p['units1']*2))(layers_to_concat)
    edge_vectors = layers.Merge(mode='concat', concat_axis=1)(layers_to_concat)

    # Apply softmax
    edge_vectors = layers.Dropout(p['dropout1'])(edge_vectors)
    main_output = layers.wrappers.TimeDistributed(layers.Dense(n_out, activation = "softmax", name='main_output'))(edge_vectors)

    model = models.Model(input=[sentence_input, entity_markers], output=[main_output])
    model.compile(optimizer=p['optimizer'], loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def model_RnnMarkersSumGhostsV2(p, embeddings, max_sent_len, n_out):
    print("Parameters:", p)

    # Take sentence encoded as indices and convert it to embeddings
    sentence_input = layers.Input(shape=(max_sent_len,), dtype='int32', name='sentence_input')
    # Repeat the input 3 times as will need it once for the target entity pair and twice for the ghost pairs
    x = layers.RepeatVector(3)(sentence_input)
    word_embeddings = layers.wrappers.TimeDistributed(layers.Embedding(output_dim=embeddings.shape[1], input_dim=embeddings.shape[0],
                                                                       input_length=max_sent_len, weights=[embeddings],
                                                                       mask_zero=True, trainable=False))(x)
    word_embeddings = layers.Dropout(p['dropout1'])(word_embeddings)

    # Take token markers that identify entity positions, convert to position embeddings
    entity_markers = layers.Input(shape=(3, max_sent_len,), dtype='int8', name='entity_markers')
    pos_embeddings = layers.wrappers.TimeDistributed(layers.Embedding(output_dim=p['position_emb'],
                                                                      input_dim=4, input_length=max_sent_len,
                                                                      mask_zero=True, W_regularizer = regularizers.l2(),
                                                                      trainable=True))(entity_markers)

    # Merge word and position embeddings and apply the specified amount of RNN layers
    x = layers.merge([word_embeddings, pos_embeddings], mode="concat")
    for i in range(p["rnn1_layers"]-1):
        x = layers.wrappers.TimeDistributed(
            getattr(layers, p['rnn1'])(p['units1'], return_sequences=True,
                                       consume_less='gpu' if p['gpu'] else "cpu"))(x)
    sentence_matrix = layers.wrappers.TimeDistributed(
        getattr(layers, p['rnn1'])(p['units1'],
                                   return_sequences=False, consume_less='gpu' if p['gpu'] else "cpu"))(x)

    # Take the vector of the sentences with the target entity pair
    sentence_vector = GetOutput(target_positions=[0])(sentence_matrix)
    sentence_vector = layers.Flatten()(sentence_vector)

    # Take the two vectors of the ghost entity pairs and sum them
    ghost_vectors = GetOutput(target_positions=[1,2])(sentence_matrix)
    ghost_vector = GlobalSumPooling1D()(ghost_vectors)

    # Merge the sentence and ghost vectors
    x = layers.merge([sentence_vector,ghost_vector], mode="concat")

    # Apply softmax
    x = layers.Dropout(p['dropout1'])(x)
    main_output = layers.Dense(n_out, activation = "softmax", name='main_output')(x)

    model = models.Model(input=[sentence_input, entity_markers], output=[main_output])
    model.compile(optimizer=p['optimizer'], loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def model_RnnMarkersSumGhostsV3(p, embeddings, max_sent_len, n_out):
    print("Parameters:", p)

    # Take sentence encoded as indices and convert it to embeddings
    sentence_input = layers.Input(shape=(max_sent_len,), dtype='int32', name='sentence_input')
    # Repeat the input 3 times as will need it once for the target entity pair and twice for the ghost pairs
    x = layers.RepeatVector(3)(sentence_input)
    word_embeddings = layers.wrappers.TimeDistributed(layers.Embedding(output_dim=embeddings.shape[1], input_dim=embeddings.shape[0],
                                                                       input_length=max_sent_len, weights=[embeddings],
                                                                       mask_zero=True, trainable=False))(x)
    word_embeddings = layers.Dropout(p['dropout1'])(word_embeddings)

    # Take token markers that identify entity positions, convert to position embeddings
    entity_markers = layers.Input(shape=(3, max_sent_len,), dtype='int8', name='entity_markers')
    pos_embeddings = layers.wrappers.TimeDistributed(layers.Embedding(output_dim=p['position_emb'],
                                                                      input_dim=4, input_length=max_sent_len,
                                                                      mask_zero=True, W_regularizer = regularizers.l2(),
                                                                      trainable=True))(entity_markers)

    # Merge word and position embeddings and apply the specified amount of RNN layers
    x = layers.merge([word_embeddings, pos_embeddings], mode="concat")
    for i in range(p["rnn1_layers"]-1):
        x = layers.wrappers.TimeDistributed(
            getattr(layers, p['rnn1'])(p['units1'], return_sequences=True,
                                       consume_less='gpu' if p['gpu'] else "cpu"))(x)
    sentence_matrix = layers.wrappers.TimeDistributed(
        getattr(layers, p['rnn1'])(p['units1'],
                                   return_sequences=False, consume_less='gpu' if p['gpu'] else "cpu"))(x)

    # Take the vector of the sentences with the target entity pair
    sentence_vector = GetOutput(target_positions=[0])(sentence_matrix)
    sentence_vector = layers.Flatten()(sentence_vector)

    # Take the two vectors of the ghost entity pairs and sum them
    ghost_vectors = GetOutput(target_positions=[1,2])(sentence_matrix)
    ghost_vector = GlobalSumPooling1D()(ghost_vectors)

    ghost_vector = layers.Dense(p['units2'], activation = "tanh")(ghost_vector)

    # Merge the sentence and ghost vectors
    x = layers.merge([sentence_vector,ghost_vector], mode="concat")

    # Apply softmax
    x = layers.Dropout(p['dropout1'])(x)
    main_output = layers.Dense(n_out, activation = "softmax", name='main_output')(x)

    model = models.Model(input=[sentence_input, entity_markers], output=[main_output])
    model.compile(optimizer=p['optimizer'], loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def model_RnnMarkersScoredGhosts(p, embeddings, max_sent_len, n_out):
    print("Parameters:", p)

    # Take sentence encoded as indices and convert it to embeddings
    sentence_input = layers.Input(shape=(max_sent_len,), dtype='int32', name='sentence_input')
    # Repeat the input 3 times as will need it once for the target entity pair and twice for the ghost pairs
    x = layers.RepeatVector(MAX_EDGES_PER_GRAPH)(sentence_input)
    word_embeddings = layers.wrappers.TimeDistributed(layers.Embedding(output_dim=embeddings.shape[1], input_dim=embeddings.shape[0],
                                                                       input_length=max_sent_len, weights=[embeddings],
                                                                       mask_zero=True, trainable=False))(x)
    word_embeddings = layers.Dropout(p['dropout1'])(word_embeddings)

    # Take token markers that identify entity positions, convert to position embeddings
    entity_markers = layers.Input(shape=(MAX_EDGES_PER_GRAPH, max_sent_len,), dtype='int8', name='entity_markers')
    pos_embeddings = layers.wrappers.TimeDistributed(layers.Embedding(output_dim=p['position_emb'],
                                                                      input_dim=4, input_length=max_sent_len,
                                                                      mask_zero=True, W_regularizer = regularizers.l2(),
                                                                      trainable=True))(entity_markers)

    # Merge word and position embeddings and apply the specified amount of RNN layers
    x = layers.merge([word_embeddings, pos_embeddings], mode="concat")
    for i in range(p["rnn1_layers"]-1):
        x = layers.wrappers.TimeDistributed(
            getattr(layers, p['rnn1'])(p['units1'], return_sequences=True,
                                       consume_less='gpu' if p['gpu'] else "cpu"))(x)
    sentence_matrix = layers.wrappers.TimeDistributed(
        getattr(layers, p['rnn1'])(p['units1'],
                                   return_sequences=False, consume_less='gpu' if p['gpu'] else "cpu"))(x)

    ### Attention over ghosts ###
    layers_to_concat = []
    for i in range(MAX_EDGES_PER_GRAPH):
        # Compute a memory vector for the target entity pair
        sentence_vector = layers.Lambda(lambda l: l[:, i], output_shape=(p['units1'],))(sentence_matrix)
        target_sentence_memory = layers.Dense(p['units1'], activation="linear", bias=False)(sentence_vector)
        if i == 0:
            context_vectors = layers.Lambda(lambda l: l[:, i+1:], output_shape=(MAX_EDGES_PER_GRAPH-1, p['units1']))(sentence_matrix)
        elif i == MAX_EDGES_PER_GRAPH - 1:
            context_vectors = layers.Lambda(lambda l: l[:, :i], output_shape=(MAX_EDGES_PER_GRAPH-1, p['units1']))(sentence_matrix)
        else:
            context_vectors = layers.Lambda(lambda l: K.concatenate([l[:, :i], l[:, i+1:]], axis=1), output_shape=(MAX_EDGES_PER_GRAPH-1, p['units1']))(sentence_matrix)
        # Compute the score between each memory and the memory of the target entity pair
        sentence_scores = layers.Merge(mode=lambda inputs: K.batch_dot(inputs[0],
                                                                       inputs[1], axes=(1,2)),
                                       output_shape=(MAX_EDGES_PER_GRAPH,))([target_sentence_memory, context_vectors])
        sentence_scores = layers.Activation('softmax')(sentence_scores)

        # Compute the final vector by taking the weighted sum of context vectors and the target entity vector
        context_vector = layers.Merge(mode=lambda inputs: K.batch_dot(inputs[0], inputs[1], axes=(1,1)),
                                      output_shape=(p['units1'],))([context_vectors, sentence_scores])
        edge_vector = layers.merge([sentence_vector, context_vector], mode="concat")
        edge_vector = layers.Reshape((1, p['units1']*2))(edge_vector)
        layers_to_concat.append(edge_vector)

    edge_vectors = layers.Merge(mode='concat', concat_axis=1)(layers_to_concat)

    # Apply softmax
    edge_vectors = layers.Dropout(p['dropout1'])(edge_vectors)
    main_output = layers.wrappers.TimeDistributed(layers.Dense(n_out, activation = "softmax", name='main_output'))(edge_vectors)

    model = models.Model(input=[sentence_input, entity_markers], output=[main_output])
    model.compile(optimizer=p['optimizer'], loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def model_RnnMarkersScoredGhostsV2(p, embeddings, max_sent_len, n_out):
    print("Parameters:", p)

    # Take sentence encoded as indices and convert it to embeddings
    sentence_input = layers.Input(shape=(max_sent_len,), dtype='int32', name='sentence_input')
    # Repeat the input 3 times as will need it once for the target entity pair and twice for the ghost pairs
    x = layers.RepeatVector(3)(sentence_input)
    word_embeddings = layers.wrappers.TimeDistributed(layers.Embedding(output_dim=embeddings.shape[1], input_dim=embeddings.shape[0],
                                                                       input_length=max_sent_len, weights=[embeddings],
                                                                       mask_zero=True, trainable=False))(x)
    word_embeddings = layers.Dropout(p['dropout1'])(word_embeddings)

    # Take token markers that identify entity positions, convert to position embeddings
    entity_markers = layers.Input(shape=(3, max_sent_len,), dtype='int8', name='ghost_entity_markers')
    pos_embeddings = layers.wrappers.TimeDistributed(layers.Embedding(output_dim=p['position_emb'],
                                                                      input_dim=4, input_length=max_sent_len,
                                                                      mask_zero=True, W_regularizer = regularizers.l2(),
                                                                      trainable=True))(entity_markers)

    # Merge word and position embeddings and apply the specified amount of RNN layers
    x = layers.merge([word_embeddings, pos_embeddings], mode="concat")
    for i in range(p["rnn1_layers"]-1):
        x = layers.wrappers.TimeDistributed(
            getattr(layers, p['rnn1'])(p['units1'], return_sequences=True,
                                       consume_less='gpu' if p['gpu'] else "cpu"))(x)
    sentence_matrix = layers.wrappers.TimeDistributed(
        getattr(layers, p['rnn1'])(p['units1'],
                                   return_sequences=False, consume_less='gpu' if p['gpu'] else "cpu"))(x)

    ### Attention over ghosts ###
    # Compute a memory vector for the target entity pair
    target_sentence_vector = layers.Lambda(lambda l: K.gather(K.permute_dimensions(l, (1,0,2)), 0), output_shape=(p['units1'],))(sentence_matrix)
    target_sentence_vector = layers.Dense(p['units1'], activation="linear", bias=False)(target_sentence_vector)

    # Compute the score between the sentence vectors and the memory of the target entity pair
    sentence_scores = layers.Merge(mode=lambda inputs: K.batch_dot(inputs[0],
                                                          inputs[1], axes=(1,2)),
                                   name = "sentence_scores", output_shape=(3,))([target_sentence_vector, sentence_matrix])
    sentence_scores = layers.Activation('softmax')(sentence_scores)

    # Compute the final vector by taking the weighted sum of ghost vectors and the target entity vector
    x = layers.Merge(mode=lambda inputs: K.batch_dot(inputs[0], inputs[1], axes=(1,1)),
                     output_shape=(p['units1'],), name = "o")([sentence_matrix, sentence_scores])

    # Apply softmax
    x = layers.Dropout(p['dropout1'])(x)
    main_output = layers.Dense(n_out, activation = "softmax", name='main_output')(x)

    model = models.Model(input=[sentence_input, entity_markers], output=[main_output])
    model.compile(optimizer=p['optimizer'], loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def model_RnnMarkersScoredGhostsV3(p, embeddings, max_sent_len, n_out):
    print("Parameters:", p)

    # Take sentence encoded as indices and convert it to embeddings
    sentence_input = layers.Input(shape=(max_sent_len,), dtype='int32', name='sentence_input')
    # Repeat the input 3 times as will need it once for the target entity pair and twice for the ghost pairs
    x = layers.RepeatVector(3)(sentence_input)
    word_embeddings = layers.wrappers.TimeDistributed(layers.Embedding(output_dim=embeddings.shape[1], input_dim=embeddings.shape[0],
                                                                       input_length=max_sent_len, weights=[embeddings],
                                                                       mask_zero=True, trainable=False))(x)
    word_embeddings = layers.Dropout(p['dropout1'])(word_embeddings)

    # Take token markers that identify entity positions, convert to position embeddings
    entity_markers = layers.Input(shape=(3, max_sent_len,), dtype='int8', name='ghost_entity_markers')
    pos_embeddings = layers.wrappers.TimeDistributed(layers.Embedding(output_dim=p['position_emb'],
                                                                      input_dim=4, input_length=max_sent_len,
                                                                      mask_zero=True, W_regularizer = regularizers.l2(),
                                                                      trainable=True))(entity_markers)

    # Merge word and position embeddings and apply the specified amount of RNN layers
    x = layers.merge([word_embeddings, pos_embeddings], mode="concat")
    for i in range(p["rnn1_layers"]-1):
        x = layers.wrappers.TimeDistributed(
            getattr(layers, p['rnn1'])(p['units1'], return_sequences=True,
                                       consume_less='gpu' if p['gpu'] else "cpu"))(x)
    sentence_matrix = layers.wrappers.TimeDistributed(
        getattr(layers, p['rnn1'])(p['units1'],
                                   return_sequences=False, consume_less='gpu' if p['gpu'] else "cpu"))(x)

    ### Attention over ghosts ###
    # Compute a memory vector for the target entity pair
    target_sentence_vector = GetOutput([0])(sentence_matrix)
    target_sentence_vector = layers.Flatten()(target_sentence_vector)
    target_sentence_memory = layers.Dense(p['units1'], activation="linear", bias=False)(target_sentence_vector)

    ghost_vectors = GetOutput(target_positions=[1,2])(sentence_matrix)

    # Compute the score between each memory and the memory of the target entity pair
    sentence_scores = layers.Merge(mode=lambda inputs: K.batch_dot(inputs[0],
                                                                   inputs[1], axes=(1,2)),
                                   name = "sentence_scores", output_shape=(2,))([target_sentence_memory, ghost_vectors])
    sentence_scores = layers.Activation('softmax')(sentence_scores)

    # Compute the final vector by taking the weighted sum of ghost vectors and the target entity vector
    ghost_vector = layers.Merge(mode=lambda inputs: K.batch_dot(inputs[0], inputs[1], axes=(1,1)),
                     output_shape=(p['units1'],), name = "o")([ghost_vectors, sentence_scores])

    x = layers.merge([target_sentence_vector,ghost_vector], mode="concat")

    # Apply softmax
    x = layers.Dropout(p['dropout1'])(x)
    main_output = layers.Dense(n_out, activation = "softmax", name='main_output')(x)

    model = models.Model(input=[sentence_input, entity_markers], output=[main_output])
    model.compile(optimizer=p['optimizer'], loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def model_RnnMarkersScoredGhostsV4(p, embeddings, max_sent_len, n_out):
    print("Parameters:", p)

    # Take sentence encoded as indices and convert it to embeddings
    sentence_input = layers.Input(shape=(max_sent_len,), dtype='int32', name='sentence_input')
    # Repeat the input 3 times as will need it once for the target entity pair and twice for the ghost pairs
    x = layers.RepeatVector(3)(sentence_input)
    word_embeddings = layers.wrappers.TimeDistributed(layers.Embedding(output_dim=embeddings.shape[1], input_dim=embeddings.shape[0],
                                                                       input_length=max_sent_len, weights=[embeddings],
                                                                       mask_zero=True, trainable=False))(x)
    word_embeddings = layers.Dropout(p['dropout1'])(word_embeddings)

    # Take token markers that identify entity positions, convert to position embeddings
    entity_markers = layers.Input(shape=(3, max_sent_len,), dtype='int8', name='ghost_entity_markers')
    pos_embeddings = layers.wrappers.TimeDistributed(layers.Embedding(output_dim=p['position_emb'],
                                                                      input_dim=4, input_length=max_sent_len,
                                                                      mask_zero=True, W_regularizer = regularizers.l2(),
                                                                      trainable=True))(entity_markers)

    # Merge word and position embeddings and apply the specified amount of RNN layers
    x = layers.merge([word_embeddings, pos_embeddings], mode="concat")
    for i in range(p["rnn1_layers"]-1):
        x = layers.wrappers.TimeDistributed(
            getattr(layers, p['rnn1'])(p['units1'], return_sequences=True,
                                       consume_less='gpu' if p['gpu'] else "cpu"))(x)
    sentence_matrix = layers.wrappers.TimeDistributed(
        getattr(layers, p['rnn1'])(p['units1'],
                                   return_sequences=False, consume_less='gpu' if p['gpu'] else "cpu"))(x)

    ### Attention over ghosts ###
    # Compute a memory vector for the target entity pair
    target_sentence_vector = GetOutput([0])(sentence_matrix)
    target_sentence_vector = layers.Flatten()(target_sentence_vector)
    target_sentence_memory = layers.Dense(p['units1'], activation="linear", bias=False)(target_sentence_vector)

    ghost_vectors = GetOutput(target_positions=[1,2])(sentence_matrix)

    # Compute the score between each memory and the memory of the target entity pair
    sentence_scores = layers.Merge(mode=lambda inputs: K.batch_dot(inputs[0],
                                                                   inputs[1], axes=(1,2)),
                                   name = "sentence_scores", output_shape=(2,))([target_sentence_memory, ghost_vectors])
    sentence_scores = layers.Activation('softmax')(sentence_scores)

    # Compute the final vector by taking the weighted sum of ghost vectors and the target entity vector
    ghost_vector = layers.Merge(mode=lambda inputs: K.batch_dot(inputs[0], inputs[1], axes=(1,1)),
                                output_shape=(p['units1'],), name = "o")([ghost_vectors, sentence_scores])

    x = layers.merge([target_sentence_vector,ghost_vector], mode="concat")

    x = layers.Dense(p['units1'], activation = "tanh")(x)

    # Apply softmax
    x = layers.Dropout(p['dropout1'])(x)
    main_output = layers.Dense(n_out, activation = "softmax", name='main_output')(x)

    model = models.Model(input=[sentence_input, entity_markers], output=[main_output])
    model.compile(optimizer=p['optimizer'], loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def model_RnnMarkersAttentionEmbScoredGhosts(p, embeddings, max_sent_len, n_out):
    print("Parameters:", p)

    # Take sentence encoded as indices and convert it to embeddings
    sentence_input = layers.Input(shape=(max_sent_len,), dtype='int32', name='sentence_input')
    # Repeat the input 3 times as will need it once for the target entity pair and twice for the ghost pairs
    x = layers.RepeatVector(3)(sentence_input)
    word_embeddings = layers.wrappers.TimeDistributed(layers.Embedding(output_dim=embeddings.shape[1], input_dim=embeddings.shape[0],
                                                                input_length=max_sent_len, weights=[embeddings],
                                                                mask_zero=True, trainable=False))(x)
    word_embeddings = layers.Dropout(p['dropout1'])(word_embeddings)

    # Convert each token to a memory embedding that are learned
    memory_embeddings = layers.Embedding(output_dim=embeddings.shape[1], input_dim=embeddings.shape[0],
                                  input_length=max_sent_len, mask_zero=True, trainable=True,
                                  W_regularizer = regularizers.l2())(sentence_input)
    memory_embeddings = layers.Dropout(p['dropout1'])(memory_embeddings)

    # Take token markers that identify entity positions, convert to position embeddings
    entity_markers = layers.Input(shape=(3, max_sent_len,), dtype='int8', name='entity_markers')
    pos_embeddings = layers.wrappers.TimeDistributed(layers.Embedding(output_dim=p['position_emb'],
                                                         input_dim=4, input_length=max_sent_len,
                                                         mask_zero=True, W_regularizer = regularizers.l2(),
                                                         trainable=True))(entity_markers)

    # Merge word and position embeddings and apply the specified amount of RNN layers
    x = layers.merge([word_embeddings, pos_embeddings], mode="concat")
    for i in range(p["rnn1_layers"]-1):
        x = layers.wrappers.TimeDistributed(
            getattr(layers, p['rnn1'])(p['units1'], return_sequences=True,
                                       consume_less='gpu' if p['gpu'] else "cpu"))(x)
    sentence_matrix = layers.wrappers.TimeDistributed(
        getattr(layers, p['rnn1'])(p['units1'],
                                   return_sequences=False, consume_less='gpu' if p['gpu'] else "cpu"))(x)

    ### Attention ###
    # Take the vector of the sentences with the target entity pair
    sentence_vector = GetOutput(target_positions=[0])(sentence_matrix)
    sentence_vector = layers.Flatten()(sentence_vector)

    # Compute a memory vector for a sentence and compute a score for each word by multiplying it with word memory embeddings
    sentence_memory = layers.Dense(embeddings.shape[1], activation="tanh")(sentence_vector)
    word_scores = layers.Merge(mode=lambda inputs: K.batch_dot(inputs[0], inputs[1], axes=(1,2)),
                 output_shape=(max_sent_len,))([sentence_memory, memory_embeddings])
    word_scores = layers.Activation('softmax')(word_scores)
    # We have to map tokens to embeddings here again (doesn't affect anything) due to a Theano bug/our bug
    word_embeddings = layers.Embedding(output_dim=embeddings.shape[1], input_dim=embeddings.shape[0],
                          input_length=max_sent_len, weights=[embeddings],
                          mask_zero=True, trainable=False)(sentence_input)
    # Take a weighted sum of words here
    memory_output = layers.Merge(mode=lambda inputs: K.batch_dot(inputs[0], inputs[1], axes=(1,1)),
                     output_shape=(embeddings.shape[1],))([word_embeddings, word_scores])

    ### Attention over ghosts ###
    # Compute a memory vector for each entity pair
    sentence_memories = layers.wrappers.TimeDistributed(
        getattr(layers, p['rnn1'])(50,
                                   return_sequences=False, consume_less='gpu' if p['gpu'] else "cpu"))(x)
    # Compute the score between each memory and the memory of the target entity pair
    sentence_scores = layers.Lambda(lambda l: K.batch_dot(K.gather(K.permute_dimensions(l, (1,0,2)),0), l, axes=(1,2)),
                      name = "a", output_shape=(3,))(sentence_memories)
    sentence_scores = layers.Activation('softmax')(sentence_scores)

    # Compute the final vector by taking the weighted sum of ghost vectors and the target entity vector
    ghost_vector = layers.Merge(mode=lambda inputs: K.batch_dot(inputs[0], inputs[1], axes=(1,1)),
                     output_shape=(p['units1'],), name = "o")([sentence_vector, sentence_scores])

    # merge the output of the weighted sum of ghost vectors and the weighted sum of words
    x = layers.merge([memory_output, ghost_vector], mode='concat')

    # Apply softmax
    x = layers.Dropout(p['dropout1'])(x)
    main_output = layers.Dense(n_out, activation = "softmax", name='main_output')(x)

    model = models.Model(input=[sentence_input, entity_markers], output=[main_output])
    model.compile(optimizer=p['optimizer'], loss='categorical_crossentropy', metrics=['accuracy'])

    return model


class GetOutput(layers.Layer):
    def __init__(self, target_positions=[], **kwargs):
        self.supports_masking = False
        self.target_positions = target_positions
        super(GetOutput, self).__init__(**kwargs)

    def call(self, x, mask=None):
        if not self.target_positions:
            return x
        return K.permute_dimensions(K.gather(K.permute_dimensions(x, [1,0,2]), self.target_positions), [1,0,2])

    def compute_mask(self, x, mask=None):
        return None

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], len(self.target_positions) ,input_shape[2])

    def get_config(self):
        config = {'target_positions': self.target_positions}
        base_config = super(GetOutput, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class GlobalSumPooling1D(layers.Layer):

    def __init__(self, **kwargs):
        super(GlobalSumPooling1D, self).__init__(**kwargs)
        self.input_spec = [layers.InputSpec(ndim=3)]

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[2])

    def call(self, x, mask=None):
        return K.sum(x, axis=1)


class MaskedConvolution1D(layers.Convolution1D):
    def __init__(self, **kwargs):
        self.supports_masking = True
        super(MaskedConvolution1D, self).__init__(**kwargs)

    def compute_mask(self, x, mask=None):
        return mask


class MaskedGlobalMaxPooling1D(layers.pooling._GlobalPooling1D):
    def __init__(self, **kwargs):
        self.supports_masking = True
        super(MaskedGlobalMaxPooling1D, self).__init__(**kwargs)

    def call(self, x, mask=None):
        if mask == None:
            return K.max(x, axis = 1)
        else:
            return K.max(K.switch(mask[:,:,np.newaxis], x, -np.inf ), axis = 1)

    def compute_mask(self, x, mask=None):
        return None


def get_negative_edges(g, limit=1):
    """

    :param g: graphs a dictionary
    :return: a list of negative edges
    >>> get_negative_edges({'edgeSet': [{'kbID': 'P397', 'left': [8], 'right': [23]}, \
    {'kbID': 'P376', 'left': [80], 'right': [8]}], 'vertexSet': [{'tokenpositions': [8]}, {'tokenpositions': [23]}, {'tokenpositions': [80]}]}) \
    == [{'left': [23], 'kbID': 'P0', 'right': [80]}]
    True
    """
    vertex_pairs = itertools.combinations(g["vertexSet"], 2)
    existing_edges = [p for e in g["edgeSet"] for p in [(e['left'], e['right']), (e['right'], e['left'])]]
    negative_edges = []
    for vertex_pair in vertex_pairs:
        left_right = (vertex_pair[0]['tokenpositions'], vertex_pair[1]['tokenpositions'])
        if left_right not in existing_edges:
            negative_edges.append({'kbID': 'P0', 'left': left_right[0], 'right': left_right[1]})
    if len(negative_edges) > limit:
        negative_edges = np.random.choice(negative_edges, limit, replace=False)
    return list(negative_edges)


def to_indices(graphs, word2idx, property2idx, max_sent_len, replace_entities_with_unkown = False, mode='train', **kwargs):
    """
    :param graphs:
    :param word2idx:
    :param property2idx:
    :param max_sent_len:
    :return:
    """
    num_edges = len([e for g in graphs for e in g['edgeSet'] if e['kbID'] not in property_blacklist])
    print("Dataset number of edges: {}".format(num_edges))
    sentences_matrix = np.zeros((num_edges, max_sent_len), dtype="int32")
    entity_matrix = np.zeros((num_edges, max_sent_len), dtype="int8")
    y_matrix = np.zeros(num_edges, dtype="int16")
    index = 0
    for g in tqdm.tqdm(graphs, ascii=True):
        token_ids = embedding_utils.get_idx_sequence(g["tokens"], word2idx)
        if len(token_ids) > max_sent_len:
            token_ids = token_ids[:max_sent_len]
        for edge in g["edgeSet"]:
            if edge['kbID'] not in property_blacklist:
                sentences_matrix[index, :len(token_ids)] = \
                    [word2idx[embedding_utils.unknown] if i in edge["left"] + edge["right"] else t for i, t in enumerate(token_ids)] \
                        if replace_entities_with_unkown else token_ids
                entity_matrix[index, :len(token_ids)] = \
                    [m for _, m in graph_utils.get_entity_indexed_vector(token_ids, edge, mode="mark-bi")]
                if mode == "train":
                    _, property_kbid, _ = graph_utils.edge_to_kb_ids(edge, g)
                    property_kbid = property2idx.get(property_kbid, property2idx[embedding_utils.unknown])
                    y_matrix[index] = property_kbid
                index += 1
    return [sentences_matrix, entity_matrix, y_matrix]


MAX_EDGES_PER_GRAPH = 7


def to_indices_with_real_entities(graphs, word2idx, property2idx, max_sent_len, mode='train', **kwargs):
    """
    :param graphs:
    :param word2idx:
    :param property2idx:
    :param max_sent_len:
    :return:
    """
    graphs_to_process = []
    for g in graphs:
        if len(g['edgeSet']) > 0:
            if len(g['edgeSet']) <= MAX_EDGES_PER_GRAPH:
                graphs_to_process.append(g)
            else:
                for i in range(0, len(g['edgeSet']), MAX_EDGES_PER_GRAPH):
                    graphs_to_process.append({"tokens": g["tokens"], "edgeSet": g["edgeSet"][i:i+ MAX_EDGES_PER_GRAPH]})
    graphs = graphs_to_process
    sentences_matrix = np.zeros((len(graphs), max_sent_len), dtype="int32")
    entity_matrix = np.zeros((len(graphs), MAX_EDGES_PER_GRAPH, max_sent_len), dtype="int8")
    y_matrix = np.zeros((len(graphs), MAX_EDGES_PER_GRAPH), dtype="int16")
    for index, g in enumerate(tqdm.tqdm(graphs, ascii=True)):
        token_ids = embedding_utils.get_idx_sequence(g["tokens"], word2idx)
        if len(token_ids) > max_sent_len:
            token_ids = token_ids[:max_sent_len]
        sentences_matrix[index, :len(token_ids)] = token_ids
        for j, edge in enumerate(g["edgeSet"][:MAX_EDGES_PER_GRAPH]):
            entity_matrix[index, j, :len(token_ids)] = \
                [m for _, m in graph_utils.get_entity_indexed_vector(token_ids, edge, mode="mark-bi")]
            _, property_kbid, _ = graph_utils.edge_to_kb_ids(edge, g)
            property_kbid = property2idx.get(property_kbid, property2idx[embedding_utils.unknown])
            y_matrix[index, j] = property_kbid
    return sentences_matrix, entity_matrix, y_matrix


def graphs_for_evaluation(graphs, graphs_tagged):
    for_evaluation = []
    for i, g in enumerate(tqdm.tqdm(graphs, ascii=True, ncols=100)):
        for edge in g["edgeSet"]:
            new_g = {"edgeSet": [edge], "tokens": g['tokens']}
            entities = [ne for ne, t in graph.extract_entities(graphs_tagged[i])]
            entities += [edge['left'], edge['right']]
            new_g['vertexSet'] = [{'tokenpositions': ne} for ne in entities]
            new_g['edgeSet'].extend(get_negative_edges(new_g, limit=6))
            for_evaluation.append(new_g)
    return for_evaluation


def to_indices_with_ghost_entities(graphs, word2idx, property2idx, max_sent_len, embeddings, **kwargs):
    sentences_matrix, entity_matrix, y_matrix = to_indices(graphs, word2idx, property2idx, max_sent_len, **kwargs)
    ghost_entity_matrix = create_ghost_edges(sentences_matrix, entity_matrix, embeddings)
    entity_matrix = entity_matrix.reshape((entity_matrix.shape[0], 1, entity_matrix.shape[1]))
    entity_matrix = np.concatenate([entity_matrix, ghost_entity_matrix], axis = 1)
    return [sentences_matrix, entity_matrix, y_matrix]


def to_indices_with_relative_positions(graphs, word2idx, property2idx, max_sent_len, position2idx, **kwargs):
    num_edges = len([e for g in graphs for e in g['edgeSet']])
    sentences_matrix = np.zeros((num_edges, max_sent_len), dtype="int32")
    entity_matrix = np.zeros((num_edges, 2, max_sent_len), dtype="int8")
    y_matrix = np.zeros(num_edges, dtype="int16")
    index = 0
    max_entity_index = max_sent_len - 1
    for g in tqdm.tqdm(graphs, ascii=True):
        token_ids = embedding_utils.get_idx_sequence(g["tokens"], word2idx)
        if len(token_ids) > max_sent_len:
            token_ids = token_ids[:max_sent_len]
        for edge in g["edgeSet"]:
            sentences_matrix[index, :len(token_ids)] = token_ids
            _, property_kbid, _ = graph_utils.edge_to_kb_ids(edge, g)
            property_kbid = property2idx[property_kbid]
            entity_vector = graph_utils.get_entity_indexed_vector(token_ids, edge, mode="position")
            entity_vector = [(-max_entity_index if m1 < -max_entity_index else max_entity_index if m1 > max_entity_index else m1,
                              -max_entity_index if m2 < -max_entity_index else max_entity_index if m2 > max_entity_index else m2) for _, m1,m2  in entity_vector]
            entity_matrix[index, :, :len(token_ids)] = [[position2idx[m] for m,_  in entity_vector],[position2idx[m] for _, m  in entity_vector]]

            y_matrix[index] = property_kbid
            index += 1
    return [sentences_matrix, entity_matrix, y_matrix]


def softmax(x):
    e_x = np.exp(x)
    return e_x / e_x.sum(axis=0)


def create_ghost_edges(sentences_matrix, entity_matrix, embeddings):
    ghost_matrix = np.zeros((entity_matrix.shape[0], 2, entity_matrix.shape[1]))
    for i in range(sentences_matrix.shape[0]):
        entity_vector = entity_matrix[i][entity_matrix[i].nonzero()]
        sentence_vector = sentences_matrix[i][sentences_matrix[i].nonzero()]
        e1_one_hot = entity_vector == 2
        e2_one_hot = entity_vector == 3
        entity_embs = np.dot(np.asarray([e1_one_hot,e2_one_hot]), embeddings[sentence_vector])

        e1_index = np.nonzero(e1_one_hot)[0]
        e2_index = np.nonzero(e2_one_hot)[0]
        entity_attention = np.dot(entity_embs, embeddings[sentence_vector].T)
        entity_attention = softmax(entity_attention.T).T

        entity_attention[:,[np.concatenate([e1_index, e2_index])]] = -np.Inf
        ghost_markers = np.tile(entity_vector, (2,1))
        ghost_markers[0][e1_index] =  1
        ghost_markers[1][e2_index] =  1
        if entity_attention.shape[-1] > 0:
            selected_entities = np.argmax(entity_attention, axis=-1)
            ghost_markers[0][selected_entities[0]] = 2
            ghost_markers[1][selected_entities[1]] = 3
            ghost_matrix[i,:,:entity_vector.shape[0]] = ghost_markers

    return ghost_matrix


if __name__ == "__main__":
    # Testing
    import doctest
    print(doctest.testmod())
