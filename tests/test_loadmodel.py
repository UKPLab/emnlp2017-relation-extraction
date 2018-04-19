import pytest

from core import parser, entity_extraction


def test_load_relationparser():
    relparser = parser.RelParser("model_ContextWeighted", models_folder="../trainedmodels/")
    tagged = [('Star', 'O', 'NNP'), ('Wars', 'O', 'NNP'), ('VII', 'O', 'NNP'), ('is', 'O', 'VBZ'), ('an', 'O', 'DT'),
              ('American', 'MISC', 'JJ'), ('space', 'O', 'NN'), ('opera', 'O', 'NN'), ('epic', 'O', 'NN'),
              ('film', 'O', 'NN'), ('directed', 'O', 'VBN'), ('by', 'O', 'IN'), ('J.', 'PERSON', 'NNP'),
              ('J.', 'PERSON', 'NNP'), ('Abrams', 'PERSON', 'NNP'), ('.', 'O', '.')]
    entity_fragments = entity_extraction.extract_entities(tagged)
    edges = entity_extraction.generate_edges(entity_fragments)
    non_parsed_graph = {'tokens': [t for t, _, _ in tagged],
                    'edgeSet': edges}
    graph_with_relations = relparser.classify_graph_relations([non_parsed_graph])[0]
    assert [e['kbID'] for e in graph_with_relations['edgeSet']] == ['P800', 'P136', 'P136']


if __name__ == '__main__':
    pytest.main([__file__])
