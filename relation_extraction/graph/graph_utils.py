# -*- coding: utf-8 -*-
# Copyright (C) 2016 UKP lab
#
# Author: Daniil Sorokin (ukp.tu-darmstadt.de/ukp-home/)
#
# Note: Throughout the code we refer to a conjunction of relations in the same sentence as '(relation) graph'

import numpy as np

kbid_numerical = "_NUMERICAL"
kbid_date = "_DATE"
kbid_empty = "_EMPTY"

LEGACY_MODE = True


def vertex_by_token_position(graph, token_positions):
    """
    Get a vertex form the graph by the index (position) of a token/tokens.
    If vertex spans over multiple tokens a reference to any of the positions would retrieve the vertex.
    If a vertex spans a smaller but overlapping set of tokens than requested with this method, it won't be retrieved.

    :param graph: the graph to extract the vertex from encoded as a dictionary with a "vertexSet"
    :param token_positions: a collection of token positions
    :return: vertex

    Tests:
    >>> g = {'tokens': ['The', 'International', 'Socialist', 'Group', ',', 'part', 'of', 'Respect', 'Renewal', ',', \
    'called', 'for', 'a', 'first', 'preference', 'vote', 'for', 'the', 'Green', 'Party', 'candidate', ',', 'Siân', \
    'Berry', ',', 'rather', 'than', 'Lindsey', 'German', '.'], \
    'vertexSet': [{'tokenpositions': [18, 19], 'kbID': 'Q9669', 'lexicalInput': 'Green Party'}, \
    {'tokenpositions': [22, 23], 'kbID': 'Q7533470', 'lexicalInput': 'Siân Berry'}]}
    >>> vertex_by_token_position(g, [1])
    {}
    >>> vertex_by_token_position(g, [22])['kbID']
    'Q7533470'
    >>> vertex_by_token_position({}, [22])
    {}
    """
    if "vertexSet" not in graph:
        return {}
    token_positions_set = set(token_positions)
    for v in graph["vertexSet"]:
        if "tokenpositions" in v and token_positions_set <= set(v["tokenpositions"]):
            return v
    return {}


def get_vertex_kbid(vertex):
    """
    Get the KbId of the vertex if it is present. Otherwise return a placeholder based on the type of the vertex/entity.

    :param vertex:
    :return: KbId or "_NUMERICAL" or "_DATE" or "_EMPTY"
    """
    if 'kbID' not in vertex:
        return kbid_empty
    if vertex.get("type") is "NUMERICAL":
        return kbid_numerical
    if vertex.get("type") is "DATE":
        return kbid_date
    return vertex['kbID']


def edge_to_kb_ids(edge, g):
    """
    Convert the given edge from the given graph to a triple of KbIds.

    :param edge: input edge
    :param g: input graph
    :return: a triple of the form (left vertex KbId, property KbId, right vertex KbId)
    """

    left_vertex = vertex_by_token_position(g, edge['left']) if len(edge['left']) > 0 else {}
    right_vertex = vertex_by_token_position(g, edge['right']) if len(edge['right']) > 0 else {}
    left_kbid = get_vertex_kbid(left_vertex)
    right_kbid = get_vertex_kbid(right_vertex)
    property_kbid = kbid_empty if 'kbID' not in edge else edge['kbID']
    return left_kbid, property_kbid, right_kbid


def get_sentence_boundaries(tokens, edge):
    sent_left_border = len(tokens)
    sent_right_border = len(tokens)
    if edge["left"] + edge["right"]:
        entities_left = min(edge["left"] + edge["right"])
        entities_right = max(edge["left"] + edge["right"])
        if 0 < entities_left < len(tokens):
            sent_left_border = max(i for i in range(entities_left) if tokens[i] == '.' or i == 0)
        if 0 < entities_right < len(tokens):
            sent_right_border = min(i for i in range(entities_right, len(tokens))
                                    if tokens[i] == '.' or i == len(tokens) - 1)
    return sent_left_border, sent_right_border


def get_entity_indexed_vector(tokens, edge, mode="mark-bi"):
    """
    Incorporates the current edge right and left entities into the sentence index representation.
    Default mode: Each token that belongs to an entities is augmented with an 2 marker, the rest of the
     entities are marked with 1

    :param tokens: list of tokens or token indices, whatever values are supplied,
    they are copied over to the first position in the tuple
    :param edge: edge from a graph
    :param mode: incorporation mode: {"mark","position","bio","mark-bi", "bio-bi"}
    :return: list of tuples where the second values is a binary variable indicating entities
    >>> get_entity_indexed_vector(["The", "MIT", "ATIS", "System", ".", "This", "paper", "describes", "the", "status", "."], {"left": [6], "right": [9]})
    [('The', 4), ('MIT', 4), ('ATIS', 4), ('System', 4), ('.', 4), ('This', 1), ('paper', 2), ('describes', 1), ('the', 1), ('status', 3), ('.', 1)]
    """
    sent_left_border, sent_right_border = get_sentence_boundaries(tokens, edge)
    if mode == "mark":
        return [(t, 2) if i in edge["left"] + edge["right"]
                else (t, 1) if sent_left_border <= i <= sent_right_border else (t, 3 if not LEGACY_MODE else 1) for i, t in enumerate(tokens)]
    if mode == "mark-bi":
        return [(t, 2) if i in edge["left"]
                else (t, 3) if i in edge["right"]
                else (t, 1) if sent_left_border <= i <= sent_right_border else (t, 4 if not LEGACY_MODE else 1) for i, t in enumerate(tokens)]
    if mode == "bio":
        b_tokens = edge["left"][:1] + edge["right"][:1]
        i_tokens = edge["left"][1:] + edge["right"][1:]
        return [(t, 2) if i in b_tokens else (t, 3) if i in i_tokens else (t, 1) for i, t in enumerate(tokens)]
    if mode == "bio-bi":
        b_tokens_v1 = edge["left"][:1]
        b_tokens_v2 = edge["right"][:1]
        i_tokens_v1 = edge["left"][1:]
        i_tokens_v2 = edge["right"][1:]
        return [(t, 2) if i in b_tokens_v1 else (t, 3) if i in b_tokens_v2 else
                (t, 4) if i in i_tokens_v1 else (t, 5) if i in i_tokens_v2 else (t, 1)
                for i, t in enumerate(tokens)]
    elif mode == "position":
        left = np.asarray(edge["left"])
        right = np.asarray(edge["right"])
        return [(t, token_to_entity_distance(left, i), token_to_entity_distance(right, i)) for i, t in enumerate(tokens)]


def token_to_entity_distance(entity_token_positions, token_position):
    """
    Positional distance from the given token to the given entity.

    :param entity_token_positions: list of token position for the entity.
    :param token_position: target token position
    :return: distance

    >>> token_to_entity_distance([4,5], 0)
    -4
    >>> token_to_entity_distance([4,5], 4)
    0
    >>> token_to_entity_distance([4,5], 5)
    0
    >>> token_to_entity_distance([4,5], 7)
    2
    """
    if len(entity_token_positions) < 1:
        entity_token_positions = np.asarray([-1])
    return (token_position - np.asarray(entity_token_positions))[np.abs(token_position - np.asarray(entity_token_positions)).argmin()]


def edge_to_str(edge, g):
    """
    Represent a single edge form the given graph as a string for printing

    :param edge: the edge to print
    :param g: graph that contains the edge and its vertices
    :return: string representation of the edge
    """
    l = vertex_by_token_position(g, edge['left'])
    if l == {}:
        l['lexicalInput'] = " ".join([g['tokens'][idx] for idx in edge['left']])
    r = vertex_by_token_position(g, edge['right'])
    if r == {}:
        r['lexicalInput'] = " ".join([g['tokens'][idx] for idx in edge['right']])
    return "{}:{} -{}:{}-> {}:{}".format(l['lexicalInput'],
                                        l['kbID'] if 'kbID' in l else kbid_empty,
                                        edge.get('lexicalInput', ""),
                                        edge['kbID']  if 'kbID' in edge else kbid_empty,
                                        r['lexicalInput'],
                                        r['kbID'] if 'kbID' in r else kbid_empty)


def print_edge(edge, g):
    """
    Print a single edge form the given graph.

    :param edge: the edge to print
    :param g: graph that contains the edge and its vertices
    """
    print(edge_to_str(edge, g))


if __name__ == "__main__":
    # Testing
    import doctest
    print(doctest.testmod())
