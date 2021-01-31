#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 22:39:29 2020

@author: simransetia
This program is used to calculate the network parameters of all the merged networks.
"""
import networkx as nx
from union_graph import Hu,Huw
from union_graph1 import Hud,H1
from xlwt import Workbook

import numpy as np
wb = Workbook()
sheet1 = wb.add_sheet('Graph_Parameters')
sheet1.write(0,0,'Diameter')
sheet1.write(0,1,nx.diameter(Huw))
sheet1.write(0,2,nx.diameter(Hu))
sheet1.write(0,3,nx.diameter(Hud))
sheet1.write(0,4,nx.diameter(H1))
sheet1.write(1,0,'Average Shortest Path')
sheet1.write(1,1,nx.average_shortest_path_length(Huw))
sheet1.write(1,2,nx.average_shortest_path_length(Hu))
sheet1.write(1,3,nx.average_shortest_path_length(Hud))
sheet1.write(1,4,nx.average_shortest_path_length(H1))
sheet1.write(2,0,'Eccentricity')
x=nx.eccentricity(Huw)
sheet1.write(2,1,np.mean(list(x.values())))
x=nx.eccentricity(Hu)
sheet1.write(2,2,np.mean(list(x.values())))
x=nx.eccentricity(Hud)
sheet1.write(2,3,np.mean(list(x.values())))
x=nx.eccentricity(H1)
sheet1.write(2,4,np.mean(list(x.values())))
sheet1.write(3,0,'Harmonic Centrality')
x=nx.harmonic_centrality(Huw)
sheet1.write(3,1,np.mean(list(x.values())))
x=nx.harmonic_centrality(Hu)
sheet1.write(3,2,np.mean(list(x.values())))
x=nx.harmonic_centrality(Hud)
sheet1.write(3,3,np.mean(list(x.values())))
x=nx.harmonic_centrality(H1)
sheet1.write(3,4,np.mean(list(x.values())))
sheet1.write(4,0,'Eigenvector Centrality')
x=nx.eigenvector_centrality(Huw)
sheet1.write(4,1,np.mean(list(x.values())))
x=nx.eigenvector_centrality(Hu)
sheet1.write(4,2,np.mean(list(x.values())))
x=nx.eigenvector_centrality(Hud)
sheet1.write(4,3,np.mean(list(x.values())))
x=nx.eigenvector_centrality(H1)
sheet1.write(4,4,np.mean(list(x.values())))
sheet1.write(5,0,'Betweenness Centrality')
x=nx.betweenness_centrality(Huw)
sheet1.write(5,1,np.mean(list(x.values())))
x=nx.betweenness_centrality(Hu)
sheet1.write(5,2,np.mean(list(x.values())))
x=nx.betweenness_centrality(Hud)
sheet1.write(5,3,np.mean(list(x.values())))
x=nx.betweenness_centrality(H1)
sheet1.write(5,4,np.mean(list(x.values())))
sheet1.write(6,0,'Edge Betweenness Centrality')
x=nx.edge_betweenness_centrality(Huw)
sheet1.write(6,1,np.mean(list(x.values())))
x=nx.edge_betweenness_centrality(Hu)
sheet1.write(6,2,np.mean(list(x.values())))
x=nx.edge_betweenness_centrality(Hud)
sheet1.write(6,3,np.mean(list(x.values())))
x=nx.edge_betweenness_centrality(H1)
sheet1.write(6,4,np.mean(list(x.values())))
sheet1.write(7,0,'Closeness Centrality')
x=nx.closeness_centrality(Huw)
sheet1.write(7,1,np.mean(list(x.values())))
x=nx.closeness_centrality(Hu)
sheet1.write(7,2,np.mean(list(x.values())))
x=nx.closeness_centrality(Hud)
sheet1.write(7,3,np.mean(list(x.values())))
x=nx.closeness_centrality(H1)
sheet1.write(7,4,np.mean(list(x.values())))
sheet1.write(8,0,'Degree Centrality')
x=nx.degree_centrality(Huw)
sheet1.write(8,1,np.mean(list(x.values())))
x=nx.degree_centrality(Hu)
sheet1.write(8,2,np.mean(list(x.values())))
x=nx.degree_centrality(Hud)
sheet1.write(8,3,np.mean(list(x.values())))
x=nx.degree_centrality(H1)
sheet1.write(8,4,np.mean(list(x.values())))
wb.save("week11_parameters.xls")