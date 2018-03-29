from core import embeddings, keras_models
from evaluation import metrics
from graph import io
from model_train import evaluate

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('model_name')
    parser.add_argument('test_set')
    parser.add_argument('--models_folder', default="../trainedmodels/")

    args = parser.parse_args()

    embedding_matrix, word2idx = embeddings.load(keras_models.model_params['wordembeddings'])
    print("Loaded embeddings:", embedding_matrix.shape)
    max_sent_len = keras_models.model_params['max_sent_len']
    n_out = len(keras_models.property2idx)

    model = getattr(keras_models, args.model_name)(keras_models.model_params, embedding_matrix, max_sent_len, n_out)
    model.load_weights(args.models_folder + args.model_name + ".kerasmodel")
    print("Loaded the model")
    test_data, _ = io.load_relation_graphs_from_file(args.test_set, load_vertices=True)
    print("Loaded the test set")

    graphs_to_indices = keras_models.to_indices
    if "Context" in args.model_name:
        to_one_hot = embeddings.timedistributed_to_one_hot
        graphs_to_indices = keras_models.to_indices_with_extracted_entities
    elif "CNN" in args.model_name:
        graphs_to_indices = keras_models.to_indices_with_relative_positions

    test_as_indices = list(graphs_to_indices(test_data, word2idx))
    test_data = None

    print("Results on the test set")
    predictions_classes, predictions = evaluate(model, test_as_indices[:-1], test_as_indices[-1])

    if len(predictions.shape) == 3:
        test_gold_stream = test_as_indices[-1].reshape(test_as_indices[-1].shape[0] * test_as_indices[-1].shape[1])
        class_mask = test_gold_stream != 0
        test_gold_stream = test_gold_stream[class_mask]
        predictions = predictions.reshape(predictions.shape[0] * predictions.shape[1], predictions.shape[2])
        predictions = predictions[class_mask]
    else:
        test_gold_stream = test_as_indices[-1]

    micro_curve = metrics.compute_precision_recall_curve(predictions, test_gold_stream, micro=True, empty_label=keras_models.p0_index)
    with open("../data/curves/micro_curve.dat", 'w') as out:
        out.write("\n".join(["{}\t{}".format(*t) for t in micro_curve]))
    print("Micro precision-recall-curve stored in:", "../data/curves/micro_curve.dat")

    macro_curve = metrics.compute_precision_recall_curve(predictions, test_gold_stream, micro=False, empty_label=keras_models.p0_index)
    with open("../data/curves/macro_curve.dat", 'w') as out:
        out.write("\n".join(["{}\t{}".format(*t) for t in macro_curve]))
    print("Macro precision-recall-curve stored in:", "../data/curves/macro_curve.dat")
