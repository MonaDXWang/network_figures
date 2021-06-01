import os
import sys
import json

from itertools import compress
from collections import defaultdict

import numpy as np
import scipy as sc
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
import plotter

from neuron_info import ntype, is_neuron, is_postemb

def make_network():

    with open('./input/daf2-dauer.json') as file:
        data = json.load(file)
    
    edges = {}

    for i, con in enumerate(data):

        pre = data[i]['partners'][0]
        post = data[i]['partners'][1]

        #making the edges
        key = (pre, post)

        if key not in edges.keys():
            edges[key] = 1
        
        else:
            edges[key] += 1

    return edges


def edge_classification_graph(network):

    graph = nx.DiGraph()

    for key in network.keys():
        pre = key[0]
        post = key[1]
        syns = network[key]

        graph.add_edge(
            pre, post, weight=syns
        )
    
    #number of nodes
    nodelist = list(graph.nodes())
    A = np.array(nx.adjacency_matrix(graph, nodelist=nodelist, weight='weight').todense()).astype(float)

    #symmetrize the adjacency matrix
    c = (A + np.transpose(A))/2.0
    
    #degree matrix
    d = np.diag(np.sum(c, axis=0))
    df = sc.linalg.fractional_matrix_power(d, -0.5)

    #Laplacian matrix
    l = d - c
    
    #compute the vertical coordinates
    b = np.sum(c * np.sign(A - np.transpose(A)), 1)
    z = np.matmul(np.linalg.pinv(l), b)
    
    #degree-normalized graph Laplacian
    q = np.matmul(np.matmul(df, l), df)

    #coordinates in plane are eigenvectors of degree-normalized graph Laplacian
    _, vx = np.linalg.eig(q)
    x = np.matmul(df, vx[:,1])
    y = np.matmul(df, vx[:,2])
    
    for n in graph.nodes():
        i = nodelist.index(n)
        typ = ntype(n)
        graph.nodes[n]['type'] = ntype(n)
        graph.nodes[n]['celltype'] = 'neuron' if is_neuron(n) else typ
        graph.nodes[n]['is_postemb'] = int(is_postemb(n))
        graph.nodes[n]['is_neuron'] = int(is_neuron(n))
        graph.nodes[n]['x'] = x[i]
        graph.nodes[n]['y'] = y[i]
        graph.nodes[n]['z'] = z[i]
    
    graph.add_node('marker1', type='m', celltype='m', is_postemb=0, is_neuron=0, x=-0.04, z= 2.5, y=0)
    graph.add_node('marker2', type='m', celltype='m', is_postemb=0, is_neuron=0, x=-0.04, z=-2.0, y=0)
    graph.add_node('marker3', type='m', celltype='m', is_postemb=0, is_neuron=0, x= 0.02, z=-2.0, y=0)
    
    print(min(x), max(x))
    print(min(z), max(z))
    
    for n in graph.nodes():
        graph.nodes[n]['x'] *= 11250.0
        graph.nodes[n]['y'] *= 11250.0
        graph.nodes[n]['z'] *= -150.0
    
    nx.write_graphml(graph, './graph.graphml')
    
    #some bug is making the marker coordinates disappear, so below is a hack to fix that.
    with open('./graph.graphml') as fil:
        graphml = fil.read()
    graphml = graphml.replace('"d21"', '"d5"')
    graphml = graphml.replace('"d22"', '"d3"')
    graphml = graphml.replace('"d23"', '"d4"')
    
    with open('./graph.graphml', 'w') as fil:
        fil.write(graphml)
        

    size = (0.35, 0.35)
    margin = {'left': 0.1, 'right': 0.05, 'top': 0.02, 'bottom': 0.08}
    
    p = plotter.Plotter(output_path='figure', page_size= 7)

    p.plot(
        'empty', [], size=size, 
        margin=margin,
        x_label='Network similarity', y_label='Processing depth',
        xlim=(-0.04, 0.02), ylim=(-2.0, 2.5), ypad=0,
        xticks=(-0.04, -0.02, 0.0, 0.02),
        yticks=(-2.0, -0.5, 1.0, 2.5),
        save='_coordinate_system'
    )

edge_classification_graph(make_network())

