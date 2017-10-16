from collections import deque

from flask import Flask, request, url_for, redirect
from flask_cors import CORS
import json
import logging
from datetime import datetime

import numpy as np
from pycorenlp import StanfordCoreNLP

from keras.models import load_model
from utils import embedding_utils, graph

from parsing import sp_models

from parsing.semanticparsing import RelParser

app = Flask(__name__)
CORS(app)
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.ERROR)
logger.setLevel(logging.DEBUG)

relparser = RelParser("model_ContextWeighted")

corenlp = StanfordCoreNLP('http://localhost:9000')
corenlp_properties = {
    'annotators': 'tokenize, pos, ner',
    'outputFormat': 'json'
}


@app.route("/")
def hello():
    return redirect(url_for('static', filename='index.html'))


@app.route("/parse/", methods=['GET', 'POST'])
def answer_question():
    if request.method == 'POST':
        input_text = request.json.get("inputtext")
        logger.debug("Processing the request")
        log = {}
        logger.debug("Prase")
        log['relation_graph'] = construct_relations_graph(input_text)
        return json.dumps(log)
    return "No answer"


def construct_relations_graph(input_text):
    logger.debug("Tagging: {}".format(input_text))
    tagged = get_tagged_from_server(input_text)
    logger.debug("Tagged: {}".format(tagged))
    logger.debug("Extract entities")
    entity_fragments = graph.extract_entities(tagged)
    edges = graph.generate_edges(entity_fragments)
    non_parsed_graph = {'tokens': [t for t, _, _ in tagged],
                        'edgeSet': edges}
    parsed_graph = relparser.sem_parse(non_parsed_graph, verbose=False)
    return parsed_graph


def get_tagged_from_server(input_text):
    corenlp_output = corenlp.annotate(input_text,properties=corenlp_properties).get("sentences", [])[0]
    tagged = [(t['originalText'], t['ner'], t['pos']) for t in corenlp_output['tokens']]
    return tagged