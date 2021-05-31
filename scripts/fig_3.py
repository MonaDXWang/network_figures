# -*- coding: utf-8 -*-

import os
import sys
from itertools import compress
from collections import defaultdict

import numpy as np
import scipy as sc
import networkx as nx
from statsmodels.sandbox.stats.multicomp import fdrcorrection0
from statsmodels.stats.proportion import proportions_ztest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from _data.neuron_info import ntype, is_neuron, is_postemb, dark_colors
from _data.dataset_info import all_datasets

from _data.plot import Plotter

categories = {
    'increase': 'Developmental change', 
    'decrease': 'Developmental change', 
    'stable': 'Stable', 
    'noise': 'Variable', 
    'remainder': 'Variable'
}

class Figure(object):
    
    def __init__(self, fig_num, data, page_size):
        self.plt = Plotter(output_path=fig_num, page_size=page_size)
        self.data = data



    def edge_classification_graph(self, f, small=False):
        
        classifications = self.data.edge_classifications
        G = self.data.G
        G_normalized = self.data.G_normalized
        
        

        graph = nx.DiGraph()
        
        for (pre, post), syns in G.items():
            classification = categories[classifications[pre, post]]
            if classification == 'Variable' and syns[-1] == 0:
                continue
            graph.add_edge(
                pre, post, weight=max(syns), classification=classification,
                weight_normalized=max(G_normalized[(pre, post)]), 
                transparency_stable=max(syns) if classification == 'Stable' else 0, 
                transparency_variable=max(syns) if classification == 'Variable' else 0, 
                transparency_changing=max(syns) if classification == 'Developmental change' else 0
            )
            for i, d in enumerate(all_datasets):
                graph[pre][post][d] = syns[i]
        
        for n in ('CANL', 'CANR', 'excgl'):
            if graph.has_node(n):
                graph.remove_node(n)
        if n in ('CEPshDL', 'CEPshDR', 'CEPshVL', 'CEPshVR', 'GLRDL', 'GLRDR', 'GLRL', 'GLRR', 'GLRVL', 'GLRVR'):
            if not graph.has_node(n):
                graph.add_node(n)
                


        #number of nodes
        nodelist = list(graph.nodes())
        A = np.array(nx.adjacency_matrix(graph, nodelist=nodelist, weight='weight_normalized').todense()).astype(float)
                

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
            graph.node[n]['type'] = ntype(n)
            graph.node[n]['celltype'] = 'neuron' if is_neuron(n) else typ
            graph.node[n]['is_postemb'] = int(is_postemb(n))
            graph.node[n]['is_neuron'] = int(is_neuron(n))
            graph.node[n]['x'] = x[i]
            graph.node[n]['y'] = y[i]
            graph.node[n]['z'] = z[i]
        
        graph.add_node('marker1', type='m', celltype='m', is_postemb=0, is_neuron=0, x=-0.04, z= 2.5, y=0)
        graph.add_node('marker2', type='m', celltype='m', is_postemb=0, is_neuron=0, x=-0.04, z=-2.0, y=0)
        graph.add_node('marker3', type='m', celltype='m', is_postemb=0, is_neuron=0, x= 0.02, z=-2.0, y=0)
        
        print min(x), max(x)
        print min(z), max(z)
        
        for n in graph.nodes():
            graph.node[n]['x'] *= 11250.0
            graph.node[n]['y'] *= 11250.0
            graph.node[n]['z'] *= -150.0
        
        nx.write_graphml(graph, os.path.join(self.plt.output_path, f + '_graph.graphml'))
        
        #some bug is making the marker coordinates disappear, so below is a hack to fix that.
        with open(os.path.join(self.plt.output_path, f + '_graph.graphml')) as fil:
            graphml = fil.read()
        graphml = graphml.replace('"d21"', '"d5"')
        graphml = graphml.replace('"d22"', '"d3"')
        graphml = graphml.replace('"d23"', '"d4"')
        with open(os.path.join(self.plt.output_path, f + '_graph.graphml'), 'w') as fil:
            fil.write(graphml)
            
        if small:
            size = (0.2, 0.2)
            margin = {'left': 0.04, 'right': 0.01, 'top': 0.02, 'bottom': 0.08}
        else:
            size = (0.35, 0.35)
            margin = {'left': 0.1, 'right': 0.05, 'top': 0.02, 'bottom': 0.08}
        
        self.plt.plot(
            'empty', [], size=size, 
            margin=margin,
            x_label='Network similarity', y_label='Processing depth',
            xlim=(-0.04, 0.02), ylim=(-2.0, 2.5), ypad=0,
            xticks=(-0.04, -0.02, 0.0, 0.02),
            yticks=(-2.0, -0.5, 1.0, 2.5),
            save=f+'_coordinate_system'
        )

        
    def _edge_types_per_neuron(self, inputs=False):
        
        classifications = self.data.edge_classifications
        G = self.data.G
        
        stable = defaultdict(float)
        variable = defaultdict(float)
        
        for (pre, post), syns in G.items():

            classification = classifications[(pre, post)]
            if classification in ('stable', 'increase', 'decrease'):
                if inputs:
                    stable[post] += 1
                else:
                    stable[pre] += 1
                
            if classification in ('remainder', 'noise'):
                if inputs:
                    variable[post] += 1
                else:
                    variable[pre] += 1
        
        return (stable, variable)


    def variable_edge_prevalence(self, f, inputs=False):

        x_label='Non-variable connection'
        y_label='Variable connection'
        xlim = (0, 20)
        ylim = (0, 40)
        fname = '_variable_edge_prevalence'
        
        if inputs:
            x_label += ' inputs'
            y_label += ' inputs'
        else:
            x_label += ' outputs'
            y_label += ' outputs'
                      
        stable, variable = self._edge_types_per_neuron(inputs=inputs)
        ns = list(set(stable.keys() + variable.keys()))
        
        x, y, c, l = [], [], [], []
        types = ['sensory', 'inter', 'modulatory', 'motor']
        if inputs:
            types += ['muscle']
            fname += '_input'
            ylim = (0, 30)
        for typ in types:
            x.append([stable[n] for n in ns if ntype(n) == typ])
            y.append([variable[n] for n in ns if ntype(n) == typ])
            c.append(dark_colors[typ])
            l.append(typ.capitalize() + (((' ' if typ != 'inter' else '') + 'neuron') if typ != 'muscle' else ''))

        self.plt.plot(
            'scatter', [x, y], size=(0.4, 0.4), 
            margin={'left': 0.08, 'right': 0.02, 'top': 0.15, 'bottom': 0.08},
            x_label=x_label, y_label=y_label, colors=c, legend=l, 
            legendpos='floatright',
            xlim=xlim, ylim=ylim, crop=True, 
            yticks=(range(0, 50, 10)),
            save=f+fname, 
        )
        

    
    def variable_edge_prevalence_quantified(self, f, inputs=False):
        
        y_label='Variable/non-variable'
        
        if inputs:
            y_label += ' inputs'
            size = (0.2*1.2, 0.17)
            margin_top = 0.1
            ylim = None
        else:
            y_label += ' outputs'
            size = (0.2, 0.12)
            ylim = (0, 7.1)
            margin_top = 0.05
                      
        stable, variable = self._edge_types_per_neuron(inputs=inputs)
        ns = list(set(stable.keys() + variable.keys()))
        

        data, l, c = [], [], []
        types = ['sensory', 'inter', 'modulatory', 'motor',]
        if inputs:
            types += ['muscle']
        for typ in types:
            data.append([variable[n]/stable[n] for n in ns if ntype(n) == typ and stable[n]])
            c.append(dark_colors[typ])
            l.append(typ.capitalize())

        
        stats = ((2,4), (4, 5), (3,5), (2,5), (5, 1), ) if inputs else ((3, 2), (3, 1), (3, 4), (2, 4), (1, 4))
        self.plt.plot(
            'box_plot', data, 
            size=(0.4, 0.3), 
            margin={'left': 0.09, 'right': 0.01, 'top': 0.15, 'bottom': 0.08},
            colors=c, xticklabels=l, y_label=y_label, ylim=ylim,
            show_outliers=False, stats=stats, 
            save=f+'_variable_edge_prevalence_quantified', darkmedian=True
        )
        
        

    def changes_type_overrepresentation(self, f):
    
        
        classifcations = self.data.edge_classifications
        G = self.data.G
        
        edges_sample = [e for e, t in classifcations.items() if t in ('increase', 'decrease')]
        edges_population = [e for e, t in classifcations.items() if t in ('increase', 'decrease', 'stable')]
        
        G_type_sample = defaultdict(int)
        G_type_population = defaultdict(int)
        for (pre, post) in G:
            edge = (pre, post)
            edge_t = (ntype(pre), ntype(post))
            if edge in edges_sample:
                G_type_sample[edge_t] += 1
            if edge in edges_population:
                G_type_population[edge_t] += 1

        p_values = []
        labels = []
        signs = []

        type_count_change = defaultdict(int)
        type_count_stable = defaultdict(int)
        
        for edge_t in G_type_population:
#            n = G_type_population[edge_t]
#            if n < 3:
#                continue
            if edge_t[1] == 'other':
                continue
            
            for typ in [edge_t[0]]:
                type_count_change[typ] += G_type_sample[edge_t]
                type_count_stable[typ] += G_type_population[edge_t]-G_type_sample[edge_t]

            count = G_type_sample[edge_t]
            average = G_type_population[edge_t]*float(sum(G_type_sample.values()))/sum(G_type_population.values())
            total = G_type_population[edge_t]
            
            _, p = proportions_ztest([count, average], [total, total])

            signs.append('under' if count < average else 'over')
            p_values.append(p)
            labels.append(edge_t)

  
        fdr = fdrcorrection0(p_values)
        
        adjusted_p_values = fdr[1]
    
        significant_labels = list(compress(labels, fdr[0]))
        significant_p_values = list(compress(adjusted_p_values, fdr[0]))
        significant_signs = list(compress(signs, fdr[0]))
        
        
        for l, p, s in zip(significant_labels, significant_p_values, significant_signs):
            print '_'.join(l), s, p
        
        

        data = []
        types = ['sensory', 'modulatory', 'inter', 'motor', 'muscle']
        
        output = []

        for i, post in enumerate(types):
            x, y = [], []
            for j, pre in enumerate(types):
                edge = (pre, post)
                
                sample = G_type_sample[edge]
                population = G_type_population[edge]
                
                if population < 10:
                    continue
                
                proportion = 0
                if population:
                    proportion = sample/(population+0.0)
#                    proportion = sample/(population+0.0)*3-1.3 for variable
                red = min(255, 255*proportion*2)
                color = '#%02x%02x%02x' % (red, 0, 0)
                color = '#%02x%02x%02x' % (255-red, 255-red, 255-red)
                
                output.append((pre,  post, population, color))


                if pre == 'muscle':
                    continue
                if population == 0:
                    continue
                
                x.append(j*(len(types)+1)+i)
                y.append(sample/float(population))

            data.append((post, (x, y)))



        edge_to_p = dict(zip(labels, adjusted_p_values))
        output = sorted(output, key=lambda x: (edge_to_p[(x[0], x[1])] < 0.05, x[2]))
        

        with open(os.path.join(self.plt.output_path, f + '_changes_type_overrepresentation.txt'), 'w') as f:
            f.write('"rowname"\t"key"\t"value"\t"color"\n')
            for (pre,  post, population, color) in output:
                if pre in ('sensory', 'modulatory', 'inter', 'motor'):
                    if pre != 'inter':
                        pre += ' '
                    pre += 'neuron'
                if post in ('sensory', 'modulatory', 'inter', 'motor'):
                    if post != 'inter':
                        post += ' '
                    post += 'neuron'
                pre = pre.capitalize()
                post = post.capitalize()
                    
                f.write('"{}"\t"{}"\t{}\t"{}"\n'.format(pre,  post, population, color))