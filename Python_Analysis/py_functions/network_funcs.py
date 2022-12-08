

# libraries
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import nxviz
from nxviz.plots import CircosPlot, MatrixPlot, ArcPlot, BasePlot


def make_graph(nodes_df, edges_df):
    # make graph from nodes and edges
    g = nx.DiGraph()
    for i, row in nodes_df.iterrows():
        keys = row.index.tolist()

        values = row.values

        # The dict contains all attributes

        g.add_node(row['ID'], **dict(zip(keys, values)))

    for i, row in edges_df.iterrows():
        keys = row.index.tolist()

        values = row.values

        g.add_edge(row['source'], row['target'], weight=row['LL_peak'], **dict(zip(keys, values)))

    return g

def get_graph(data, nodes, feature='LL_peak', labels_all=labels_all):
    graph_dat = data.groupby(['Stim', 'Chan'], as_index=False)['LL_peak'].mean()#summ[summ.Sig_block>3]
    graph_dat.insert(0, 'source',graph_dat.Stim )
    graph_dat.insert(0, 'target',graph_dat.Chan )
    for i in range(len(labels_all)):
        graph_dat.loc[graph_dat.Stim==i, 'source'] = labels_all[i]
        graph_dat.loc[graph_dat.Chan==i, 'target'] = labels_all[i]
    #G       = nx.from_pandas_edgelist(graph_dat, source="S", target="C",edge_attr="LL_peak")
    edges   = graph_dat.drop(columns=['Stim', 'Chan'])
    edges =edges.reset_index(drop=True)
    print(edges.LL_peak.values[0])
    G       = make_graph(nodes, edges)
    print(nx.info(G))
    return G
