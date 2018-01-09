# coding: utf-8
# Copyright (C) 2016 UKP lab
#
# Author: Daniil Sorokin (ukp.tu-darmstadt.de/ukp-home/)
#

import nltk


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


def generate_edges(vertices):
    edges = []
    for i, v1 in enumerate(vertices):
        for v2 in vertices[i+1:]:
            edges.append({'left': v1[0], 'right': v2[0]})
    return edges


if __name__ == "__main__":
    # Testing
    import doctest
    print(doctest.testmod())


