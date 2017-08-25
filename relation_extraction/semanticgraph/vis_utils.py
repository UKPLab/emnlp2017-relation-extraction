# -*- coding: utf-8 -*-
# Copyright (C) 2016 UKP lab
#
# Author: Daniil Sorokin (ukp.tu-darmstadt.de/ukp-home/)
#

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

import numpy as np
from .graph_utils import vertex_by_token_position


def show_relation_graph(g):
    """
    Displays the relation graph using matplotlib.

    :param g: input graph.
    """
    if "vertexSet" not in g:
        vertex_indices = {str(indices):indices for e in g["edgeSet"] for indices in [e["left"]] + [e["right"]] }
        g["vertexSet"] = []
        for k, v in vertex_indices.items():
            g["vertexSet"].append({"lexicalInput": " ".join([g['tokens'][idx] for idx in v])})
    fig, ax = plt.subplots()
    step = np.pi*2 / float(len(g["vertexSet"]))
    print(step, len(g["vertexSet"]))
    x, y = 0.0, 0.0
    vertex_coordinates = {}

    for i, vertex in enumerate(g["vertexSet"]):
        x, y = 1 - np.cos(step*i)*2, 1 - np.sin(step*i)
        vertex_coordinates[vertex["lexicalInput"]] = x, y
        circle = mpatches.Circle([x,y], 0.1, fc = "none")
        ax.add_patch(circle)
        x, y = 1 - np.cos(step*i)*2.5, 1 - np.sin(step*i)*1.25
        plt.text(x, y, vertex["lexicalInput"], ha="center", family='sans-serif', size=10)

    for edge in g["edgeSet"]:
        left_vertex = vertex_by_token_position(g, edge['left']) if len(edge['left']) > 0 else {}
        right_vertex = vertex_by_token_position(g, edge['right']) if len(edge['right']) > 0 else {}
        if left_vertex == {}:
            left_vertex['lexicalInput'] = " ".join([g['tokens'][idx] for idx in edge['left']])
        if right_vertex == {}:
            right_vertex['lexicalInput'] = " ".join([g['tokens'][idx] for idx in edge['right']])

        x, y = list(zip(vertex_coordinates[left_vertex["lexicalInput"]], vertex_coordinates[right_vertex["lexicalInput"]]))
        line = mlines.Line2D(x, y, lw=1., alpha=1)
        ax.add_line(line)
        property_kbid = "" if 'kbID' not in edge else edge['kbID']
        property_label = "" if 'lexicalInput' not in edge else edge['lexicalInput']
        plt.text(np.average(x), np.average(y), property_kbid + ":" + property_label, ha="center", family='sans-serif', size=10)

    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    plt.axis('equal')
    plt.axis('off')

    plt.show()