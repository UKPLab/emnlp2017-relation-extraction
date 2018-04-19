from flask import Blueprint, request, url_for, redirect
import json
import logging

from pycorenlp import StanfordCoreNLP

from core import entity_extraction
from relation_extraction.core.parser import RelParser

relext = Blueprint("relext_server", __name__, static_folder='static')

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.ERROR)
logger.setLevel(logging.DEBUG)

relparser = RelParser("model_ContextWeighted", models_folder="relation-extraction/trainedmodels/")

corenlp = StanfordCoreNLP('http://semanticparsing:9000')
corenlp_properties = {
    'annotators': 'tokenize, pos, ner',
    'outputFormat': 'json'
}


@relext.route("/")
def hello():
    return redirect(url_for('relext_server.static', filename='index.html'))


@relext.route("/parse/", methods=['GET', 'POST'])
def parse_sentence():
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
    entity_fragments = entity_extraction.extract_entities(tagged)
    edges = entity_extraction.generate_edges(entity_fragments)
    non_parsed_graph = {'tokens': [t for t, _, _ in tagged],
                        'edgeSet': edges}
    parsed_graph = relparser.classify_graph_relations([non_parsed_graph])[0]
    return parsed_graph


def get_tagged_from_server(input_text):
    corenlp_output = corenlp.annotate(input_text,properties=corenlp_properties).get("sentences", [])[0]
    tagged = [(t['originalText'], t['ner'], t['pos']) for t in corenlp_output['tokens']]
    return tagged
