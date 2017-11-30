# coding: utf-8
# Copyright (C) 2016 UKP lab
#
# Author: Daniil Sorokin (ukp.tu-darmstadt.de/ukp-home/)
#

import nltk
import json
import logging
from semanticgraph import io
import tqdm

logging.basicConfig(level=logging.DEBUG)

if __name__ == "__main__":

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.ERROR)

    data_folder = "../data/"

    relations_data, _ = io.load_relation_graphs_from_file(
        "../data/wikipedia-wikidata/enwiki-20160501/semantic-graphs-filtered-training.02_06.json", load_vertices=False)
    logging.debug('Loaded, size: {}'.format(len(relations_data)))

    ne_tagger = nltk.tag.stanford.StanfordNERTagger("../resources/stanfordcorenlp/models-3.7.0/edu/stanford/nlp/models/ner/english.all.3class.caseless.distsim.crf.ser.gz",
                                                    path_to_jar = "../resources/stanfordcorenlp/stanford-ner-2015-12-09/stanford-ner-3.6.0.jar")
    pos_tagger = nltk.tag.stanford.StanfordPOSTagger("../resources/stanfordcorenlp/models-3.7.0/edu/stanford/nlp/models/pos-tagger/english-caseless-left3words-distsim.tagger",
                                                     path_to_jar = "../resources/stanfordcorenlp/stanford-postagger-full-2015-12-09/stanford-postagger-3.6.0.jar")
    webquestions_utterances_tokens = [q_obj['tokens'] for q_obj in relations_data]
    logging.debug('Tokenized')
    webquestions_utterances_nes = []
    webquestions_utterances_poss = []
    for i in tqdm.tqdm(range((len(relations_data) // 3000) + 1), ncols=100, ascii=True):
        webquestions_utterances_nes += ne_tagger.tag_sents(webquestions_utterances_tokens[i*3000:(i+1)*3000])
        webquestions_utterances_poss += pos_tagger.tag_sents(webquestions_utterances_tokens[i*3000:(i+1)*3000])

    logging.debug('Finished, size: {}'.format(len(webquestions_utterances_nes)))
    webquestions_utterances_alltagged = [list(zip(tokens,
                                                  list(zip(*webquestions_utterances_nes[i]))[1],
                                                  list(zip(*webquestions_utterances_poss[i]))[1])) for i, tokens in enumerate(webquestions_utterances_tokens)]

    with open("../data/wikipedia-wikidata/enwiki-20160501/semantic-graphs-filtered-training.tagged.json", "w") as out:
        json.dump(webquestions_utterances_alltagged, out, indent=4)
    logger.debug("Saved")
