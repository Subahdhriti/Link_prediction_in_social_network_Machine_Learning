import networkx as nx
import matplotlib.pyplot as plt
import csv
import numpy as np
import pylab

k = 3

#ef refreshGraph():
	


#def onClick(event):
	

#fig, ax = plt.subplots()    
#fig.canvas.mpl_connect('button_press_event', onClick)

csv_file = open('total_edgelist.csv','r')
csv_reader = csv.reader(csv_file)
g = nx.Graph()
for line in csv_reader:
	g.add_edge(line[0],line[1],color='y',weight=2)

csv_file2 = open('total_recommendations.csv','r')
csv_reader2 = csv.reader(csv_file2,delimiter=',')
for line2 in csv_reader2:
	for i in range(k):
	    g.add_edge(line2[0],line2[i+1],color='b',weight=1)
#for line in csv_reader2:
 #   print(line[1][1])

edges = g.edges()
colors = [g[u][v]['color'] for u,v in edges]
weights = [g[u][v]['weight'] for u,v in edges]

nx.draw(g,edges=edges, edge_color=colors, width=weights,with_labels = True)
plt.show(g)	