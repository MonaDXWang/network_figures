import os
import sys
import json

from itertools import compress
from collections import defaultdict

import numpy as np
import scipy as sc
import networkx as nx

def make_network():

    with open('./daf2-dauer.json') as file:
        data = json.load(file)
        print(data)

make_network()
#  def edge_classification_graph(self, f, small=False):
        
#         classifications = self.data.edge_classifications
#         G = self.data.G
#         G_normalized = self.data.G_normalized
        

#         graph = nx.DiGraph()
        
#         for (pre, post), syns in G.items():
#             classification = categories[classifications[pre, post]]
#             if classification == 'Variable' and syns[-1] == 0:
#                 continue
#             graph.add_edge(
#                 pre, post, weight=max(syns), classification=classification,
#                 weight_normalized=max(G_normalized[(pre, post)]), 
#                 transparency_stable=max(syns) if classification == 'Stable' else 0, 
#                 transparency_variable=max(syns) if classification == 'Variable' else 0, 
#                 transparency_changing=max(syns) if classification == 'Developmental change' else 0
#             )
#             for i, d in enumerate(all_datasets):
#                 graph[pre][post][d] = syns[i]
        
#         for n in ('CANL', 'CANR', 'excgl'):
#             if graph.has_node(n):
#                 graph.remove_node(n)
#         if n in ('CEPshDL', 'CEPshDR', 'CEPshVL', 'CEPshVR', 'GLRDL', 'GLRDR', 'GLRL', 'GLRR', 'GLRVL', 'GLRVR'):
#             if not graph.has_node(n):
#                 graph.add_node(n)
                


#         #number of nodes
#         nodelist = list(graph.nodes())
#         A = np.array(nx.adjacency_matrix(graph, nodelist=nodelist, weight='weight_normalized').todense()).astype(float)
                

#         #symmetrize the adjacency matrix
#         c = (A + np.transpose(A))/2.0
        
#         #degree matrix
#         d = np.diag(np.sum(c, axis=0))
#         df = sc.linalg.fractional_matrix_power(d, -0.5)

#         #Laplacian matrix
#         l = d - c
        
#         #compute the vertical coordinates
#         b = np.sum(c * np.sign(A - np.transpose(A)), 1)
#         z = np.matmul(np.linalg.pinv(l), b)
        
#         #degree-normalized graph Laplacian
#         q = np.matmul(np.matmul(df, l), df)

#         #coordinates in plane are eigenvectors of degree-normalized graph Laplacian
#         _, vx = np.linalg.eig(q)
#         x = np.matmul(df, vx[:,1])
#         y = np.matmul(df, vx[:,2])
        


#         for n in graph.nodes():
#             i = nodelist.index(n)
#             typ = ntype(n)
#             graph.node[n]['type'] = ntype(n)
#             graph.node[n]['celltype'] = 'neuron' if is_neuron(n) else typ
#             graph.node[n]['is_postemb'] = int(is_postemb(n))
#             graph.node[n]['is_neuron'] = int(is_neuron(n))
#             graph.node[n]['x'] = x[i]
#             graph.node[n]['y'] = y[i]
#             graph.node[n]['z'] = z[i]
        
#         graph.add_node('marker1', type='m', celltype='m', is_postemb=0, is_neuron=0, x=-0.04, z= 2.5, y=0)
#         graph.add_node('marker2', type='m', celltype='m', is_postemb=0, is_neuron=0, x=-0.04, z=-2.0, y=0)
#         graph.add_node('marker3', type='m', celltype='m', is_postemb=0, is_neuron=0, x= 0.02, z=-2.0, y=0)
        
#         print min(x), max(x)
#         print min(z), max(z)
        
#         for n in graph.nodes():
#             graph.node[n]['x'] *= 11250.0
#             graph.node[n]['y'] *= 11250.0
#             graph.node[n]['z'] *= -150.0
        
#         nx.write_graphml(graph, os.path.join(self.plt.output_path, f + '_graph.graphml'))
        
#         #some bug is making the marker coordinates disappear, so below is a hack to fix that.
#         with open(os.path.join(self.plt.output_path, f + '_graph.graphml')) as fil:
#             graphml = fil.read()
#         graphml = graphml.replace('"d21"', '"d5"')
#         graphml = graphml.replace('"d22"', '"d3"')
#         graphml = graphml.replace('"d23"', '"d4"')
#         with open(os.path.join(self.plt.output_path, f + '_graph.graphml'), 'w') as fil:
#             fil.write(graphml)
            
#         if small:
#             size = (0.2, 0.2)
#             margin = {'left': 0.04, 'right': 0.01, 'top': 0.02, 'bottom': 0.08}
#         else:
#             size = (0.35, 0.35)
#             margin = {'left': 0.1, 'right': 0.05, 'top': 0.02, 'bottom': 0.08}
        
#         self.plt.plot(
#             'empty', [], size=size, 
#             margin=margin,
#             x_label='Network similarity', y_label='Processing depth',
#             xlim=(-0.04, 0.02), ylim=(-2.0, 2.5), ypad=0,
#             xticks=(-0.04, -0.02, 0.0, 0.02),
#             yticks=(-2.0, -0.5, 1.0, 2.5),
#             save=f+'_coordinate_system'
#         )