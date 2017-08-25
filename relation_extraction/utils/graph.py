# coding: utf-8
# Copyright (C) 2016 UKP lab
#
# Author: Daniil Sorokin (ukp.tu-darmstadt.de/ukp-home/)
#

import nltk
import copy
import re


def replace_entities(g):
    """
    Replaces an entity participating in the first relation in the graph with <e>

    :param g: graph as a dictionary
    :return: the original graph modified
    >>> replace_entities({'edgeSet': [{'right': ['Nfl', 'Redskins']}], 'tokens': ['where', 'are', 'the', 'nfl', 'redskins', 'from', '?']}) == {'edgeSet': [{'right': ['Nfl', 'Redskins']}], 'tokens': ['where', 'are', 'the', '<e>', 'from', '?']}
    True
    >>> replace_entities({'edgeSet': [{'canonical_right': 'Vasco Núñez de Balboa', 'right': ['Vasco', 'Nunez', 'De', 'Balboa'], 'kbID': 'P106v', 'type': 'reverse'},\
   {'canonical_right': 'Reise', 'hopUp': 'P279v', 'kbID': 'P425v', 'type': 'direct', 'right': ['journey']}], \
   'tokens': ['what', 'was', 'vasco', 'nunez', 'de', 'balboa', 'original', 'purpose', 'of', 'his', 'journey', '?']})['tokens']
    ['what', 'was', '<e>', 'original', 'purpose', 'of', 'his', 'journey', '?']
    >>> replace_entities({'edgeSet': [{'right': ['House', 'Of', "Representatives"]}], 'tokens': "what is the upper house of the house of representatives ?".split()})['tokens']
    ['what', 'is', 'the', 'upper', 'house', 'of', 'the', '<e>', '?']
    """
    tokens = g.get('tokens', [])
    edge = get_graph_first_edge(g)
    entity = [t.lower() for t in edge.get('right', [])]
    new_tokens = []
    entity_pos = 0
    for i, t in enumerate(tokens):
        if entity_pos == len(entity) or t != entity[entity_pos]:
            if entity_pos > 0:
                if entity_pos == len(entity):
                    new_tokens.append("<e>")
                else:
                    new_tokens.extend(entity[:entity_pos])
                entity_pos = 0
            new_tokens.append(t)
        else:
            entity_pos += 1
    g['tokens'] = new_tokens
    return g


def normalize_tokens(g):
    """
    Normalize a tokens of the graph by setting it to lower case and removing any numbers.

    :param g: graph to normalize
    :return: graph with normalized tokens
    >>> normalize_tokens({'tokens':["Upper", "Case"]})
    {'tokens': ['upper', 'case']}
    >>> normalize_tokens({'tokens':["He", "started", "in", "1995"]})
    {'tokens': ['he', 'started', 'in', '0']}
    """
    tokens = g.get('tokens', [])
    g['tokens'] = [re.sub(r"\d+", "0", t.lower()) for t in tokens]
    return g


def get_graph_first_edge(g):
    """
    Get the first edge of the graph or an empty edge if there is non

    :param g: a graph as a dictionary
    :return: an edge as a dictionary
    >>> get_graph_first_edge({'edgeSet': [{'right':[4,5,6]}], 'entities': []}) == {'right':[4,5,6]}
    True
    >>> get_graph_first_edge({})
    {}
    >>> get_graph_first_edge({'edgeSet':[]})
    {}
    >>> get_graph_first_edge({'edgeSet': [{'right':[4,5,6]}, {'right':[8]}], 'entities': []}) == {'right':[4,5,6]}
    True
    """
    return g["edgeSet"][0] if 'edgeSet' in g and g["edgeSet"] else {}


def get_graph_last_edge(g):
    """
    Get the last edge of the graph or an empty edge if there is non

    :param g: a graph as a dictionary
    :return: an edge as a dictionary
    """
    return g["edgeSet"][-1] if 'edgeSet' in g and g["edgeSet"] else {}


def copy_graph(g):
    """
    Create a copy of the given graph.

    :param g: input graph as dictionary
    :return: a copy of the graph
    >>> copy_graph({'edgeSet': [{'right':[4,5,6]}], 'entities': [], 'tokens':[]}) == {'edgeSet': [{'right':[4,5,6]}], 'entities': [], 'tokens':[]}
    True
    >>> copy_graph({}) == {'edgeSet':[], 'entities':[]}
    True
    """
    new_g = {'edgeSet': copy.deepcopy(g.get('edgeSet', [])),
             'entities': copy.copy(g.get('entities', []))}
    if 'tokens' in g:
        new_g['tokens'] = g.get('tokens', [])
    if 'filter' in g:
        new_g['filter'] = g['filter']
    return new_g

np_grammar = r"""
    NP:
    {(<NN|NNS>|<NNP|NNPS>)<NNP|NN|NNS|NNPS>+}
    {(<NN|NNS>+|<NNP|NNPS>+)<IN|CC>(<PRP\$|DT><NN|NNS>+|<NNP|NNPS>+)}
    {<JJ|RB|CD>*<NNP|NN|NNS|NNPS>+}
    {<NNP|NN|NNS|NNPS>+}
    CD:
    {<CD>+}
    """
np_parser = nltk.RegexpParser(np_grammar)


def extract_entities_from_tagged(annotated_tokens, tags):
    """
    The method takes a list of tokens annotated with the Stanford NE annotation scheme and produces a list of entites.

    :param annotated_tokens: list of tupels where the first element is a token and the second is the annotation
    :return: list of entities each represented by the corresponding token ids

    Tests:
    >>> extract_entities_from_tagged([('what', 'O'), ('character', 'O'), ('did', 'O'), ('natalie', 'PERSON'), ('portman', 'PERSON'), ('play', 'O'), ('in', 'O'), ('star', 'O'), ('wars', 'O'), ('?', 'O')], tags={'PERSON'})
    [[3, 4]]
    >>> extract_entities_from_tagged([('Who', 'O'), ('was', 'O'), ('john', 'PERSON'), ('noble', 'PERSON')], tags={'PERSON'})
    [[2, 3]]
    >>> extract_entities_from_tagged([(w, 'NE' if t != 'O' else 'O') for w, t in [('Who', 'O'), ('played', 'O'), ('Aragorn', 'PERSON'), ('in', 'O'), ('the', 'ORG'), ('Hobbit', 'ORG'), ('?', 'O')]], tags={'NE'})
    [[2], [4, 5]]
    """
    vertices = []
    current_vertex = []
    for i, (w, t) in enumerate(annotated_tokens):
        if t in tags:
            current_vertex.append(i)
        elif len(current_vertex) > 0:
            vertices.append(current_vertex)
            current_vertex = []
    if len(current_vertex) > 0:
        vertices.append(current_vertex)
    return vertices


def extract_entities(tokens_ne_pos):
    """
    Extract entities from the NE tags and POS tags of a sentence. Regular nouns are lemmatized to get rid of plurals.

    :param tokens_ne_pos: list of POS and NE tags.
    :return: list of entities in the order: NE>NNP>NN
    >>> extract_entities([('who', 'O', 'WP'), ('are', 'O', 'VBP'), ('the', 'O', 'DT'), ('current', 'O', 'JJ'), ('senators', 'O', 'NNS'), ('from', 'O', 'IN'), ('missouri', 'LOCATION', 'NNP'), ('?', 'O', '.')])
    [(['Missouri'], 'LOCATION'), (['current', 'senators'], 'NN')]
    >>> extract_entities([('what', 'O', 'WDT'), ('awards', 'O', 'NNS'), ('has', 'O', 'VBZ'), ('louis', 'PERSON', 'NNP'), ('sachar', 'PERSON', 'NNP'), ('won', 'O', 'NNP'), ('?', 'O', '.')])
    [(['Louis', 'Sachar'], 'PERSON'), (['Won'], 'NNP'), (['awards'], 'NN')]
    >>> extract_entities([('who', 'O', 'WP'), ('was', 'O', 'VBD'), ('the', 'O', 'DT'), ('president', 'O', 'NN'), ('after', 'O', 'IN'), ('jfk', 'O', 'NNP'), ('died', 'O', 'VBD'), ('?', 'O', '.')])
    [(['president', 'after', 'jfk'], 'NN')]
    >>> extract_entities([('who', 'O', 'WP'), ('natalie', 'PERSON', 'NN'), ('likes', 'O', 'VBP')])
    [(['Natalie'], 'PERSON')]
    >>> extract_entities([('what', 'O', 'WDT'), ('character', 'O', 'NN'), ('did', 'O', 'VBD'), ('john', 'O', 'NNP'), \
    ('noble', 'O', 'NNP'), ('play', 'O', 'VB'), ('in', 'O', 'IN'), ('lord', 'O', 'NNP'), ('of', 'O', 'IN'), ('the', 'O', 'DT'), ('rings', 'O', 'NNS'), ('?', 'O', '.')])
    [(['John', 'Noble'], 'NNP'), (['character'], 'NN'), (['lord', 'of', 'the', 'rings'], 'NN')]
    >>> extract_entities([['who', 'O', 'WP'], ['plays', 'O', 'VBZ'], ['lois', 'PERSON', 'NNP'], ['lane', 'PERSON', 'NNP'], ['in', 'O', 'IN'], ['superman', 'O', 'NNP'], ['returns', 'O', 'NNS'], ['?', 'O', '.']])
    [(['Lois', 'Lane'], 'PERSON'), (['superman', 'returns'], 'NN')]
    >>> extract_entities([('the', 'O', 'DT'), ('empire', 'O', 'NN'), ('strikes', 'O', 'VBZ'), ('back', 'O', 'RB'), ('is', 'O', 'VBZ'), ('the', 'O', 'DT'), ('second', 'O', 'JJ'), ('movie', 'O', 'NN'), ('in', 'O', 'IN'), ('the', 'O', 'DT'), ('star', 'O', 'NN'), ('wars', 'O', 'NNS'), ('franchise', 'O', 'VBP')])
    [(['empire'], 'NN'), (['second', 'movie'], 'NN'), (['star', 'wars'], 'NN')]
    >>> extract_entities([['who', 'O', 'WP'], ['played', 'O', 'VBD'], ['cruella', 'LOCATION', 'NNP'], ['deville', 'LOCATION', 'NNP'], ['in', 'O', 'IN'], ['102', 'O', 'CD'], ['dalmatians', 'O', 'NNS'], ['?', 'O', '.']])
    [(['Cruella', 'Deville'], 'LOCATION'), (['102', 'dalmatians'], 'NN')]
    >>> extract_entities([['who', 'O', 'WP'], ['was', 'O', 'VBD'], ['the', 'O', 'DT'], ['winner', 'O', 'NN'], ['of', 'O', 'IN'], ['the', 'O', 'DT'], ['2009', 'O', 'CD'], ['nobel', 'O', 'NNP'], ['peace', 'O', 'NNP'], ['prize', 'O', 'NNP'], ['?', 'O', '.']])
    [(['Nobel', 'Peace', 'Prize'], 'NNP'), (['winner'], 'NN'), (['2009'], 'CD')]
    >>> extract_entities([['who', 'O', 'WP'], ['is', 'O', 'VBZ'], ['the', 'O', 'DT'], ['senator', 'O', 'NN'], ['of', 'O', 'IN'], ['connecticut', 'LOCATION', 'NNP'], ['2010', 'O', 'CD'], ['?', 'O', '.']])
    [(['Connecticut'], 'LOCATION'), (['senator'], 'NN'), (['2010'], 'CD')]
    """
    persons = extract_entities_from_tagged([(w, t) for w, t, _ in tokens_ne_pos], ['PERSON'])
    locations = extract_entities_from_tagged([(w, t) for w, t, _ in tokens_ne_pos], ['LOCATION'])
    orgs = extract_entities_from_tagged([(w, t) for w, t, _ in tokens_ne_pos], ['ORGANIZATION'])

    chunks = np_parser.parse([(w, t if p == "O" else "O") for w, p, t in tokens_ne_pos])
    nps = []
    index = 0
    for el in chunks:
        if type(el) == nltk.tree.Tree and (el.label() == "NP" or el.label() == "CD"):
            nps.append(list(range(index, index+ len(el.leaves()))))
            index += len(el.leaves())
        else:
            index += 1
    ne_vertices = [(ne, 'PERSON') for ne in persons] + [(ne, 'LOCATION') for ne in locations] + [(ne, 'ORGANIZATION') for ne in orgs]
    vertices = []
    for nn in nps:
        if not ne_vertices or not all(n in v for n in nn for v, _ in ne_vertices):
            ne_vertices.append((nn, 'NNP'))
    return ne_vertices + vertices


if __name__ == "__main__":
    # Testing
    import doctest
    print(doctest.testmod())


