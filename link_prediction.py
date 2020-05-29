from igraph import *
from math import *
import sys
import numpy as np
import os
import glob
import random
from random import shuffle
from random import seed
import matplotlib.pyplot as plt
import time
import datetime
import collections
import csv
import json

k = 3 		# Top k recomendations for a target user
maxl = 2 	# Number of iterations for Katz Algorithm
beta = 0.1 	# The damping factor for Katz Algorithm



node_data = json.load(open('node_char.json'))



###################################
######## Helper Functions #########
###################################

# load edge-list from file
def get_edge_list(dataset_path):
	data_file = open(dataset_path)
	edge_list = map(lambda x:tuple(map(int,x.split())),data_file.read().split("\n")[:-1])
	data_file.close()
	return edge_list





# Get the similarity product for a path
# (product of path-step similarities)
def get_sim_product(sim, shortest_path):
	prod = 1
	for i in range(len(shortest_path) - 1):
		prod *= sim[shortest_path[i]][shortest_path[i+1]]
	return round(prod,3)






# Filter out, Sort and Get top-K predictions
def get_top_k_recommendations(graph,sim,i,k):
	return  sorted(filter(lambda x: i!=x and graph[i,x] != 1,range(len(sim[i]))) , key=lambda x: sim[i][x],reverse=True)[0:k]





# Convert edge_list into a set of constituent edges
def get_vertices_set(edge_list):
	res = set()
	for x,y in edge_list:
		res.add(x)
		res.add(y)
	return res





# Split the dataset into two parts (50-50 split)
# Create 2 graphs, 1 used for training and the other for testing
def split_data(edge_list):
	random.seed(350)
	indexes = range(len(edge_list))
	test_indexes = set(random.sample(indexes, len(indexes)/2)) # removing 50% edges from test data
	train_indexes = set(indexes).difference(test_indexes)
	test_list = [edge_list[i] for i in test_indexes]
	train_list = [edge_list[i] for i in train_indexes]
	csv_file = open('test.csv',"w")
	fields = ['Node1','Node2']
	thewriter = csv.DictWriter(csv_file,fieldnames=fields)
	for i in range(len(test_list)):
		thewriter.writerow({'Node1' : str(test_list[i][0]) , 'Node2' : str(test_list[i][1])})
	return train_list,test_list





# Calculates accuracy metrics (Precision & Recall),
# for a given similarity-model against a test-graph.
def print_precision_and_recall(sim,graph,train_graph,test_graph,test_vertices_set,train_vertices_set,esim):
	precision = recall = c = 0
	for i in test_vertices_set:
		if i in train_vertices_set:
			actual_friends_of_i = set(test_graph.neighbors(i))

			# Handles case where test-data < k
			if len(actual_friends_of_i) < k:
				k2 = len(actual_friends_of_i)
			else:
				k2 = k

			top_k = set(get_top_k_recommendations(train_graph,esim,i,k2))
			
			precision += len(top_k.intersection(actual_friends_of_i))/float(k2)
			recall += len(top_k.intersection(actual_friends_of_i))/float(len(actual_friends_of_i))
			c += 1
	#print(esim)
	print "Precision is : " + str(precision/c)
	print "Recall is : " + str(recall/c)
	a = str(raw_input("Want to know Similarities between nodes?(y/n) : "))
	if a == 'y':
		show_common(sim,graph)




def show_common(sim,graph):
	node1 = int(raw_input("Node1: "))
	node2 = int(raw_input("Node2: "))

	print "Mutual Frindr---------------------: " + str(len(set(graph.neighbors(node1)).intersection(set(graph.neighbors(node2)))))
	print "Common Groups---------------------: " + str(list(set(node_data['Nodes'][node1]['groups']).intersection(set(node_data['Nodes'][node2]['groups']))))
	print "Common Subjects-------------------: " + str(list(set(node_data['Nodes'][node1]['education']).intersection(set(node_data['Nodes'][node2]['education']))))
	print "Common Places Lived---------------: " + str(list(set(node_data['Nodes'][node1]['lives']).intersection(set(node_data['Nodes'][node2]['lives']))))
	print "Common Interests and Hobbies------: " + str(list(set(node_data['Nodes'][node1]['interests']).intersection(set(node_data['Nodes'][node2]['interests']))))
	print "Common School,College or Companies: " + str(list(set(node_data['Nodes'][node1]['workplaces']).intersection(set(node_data['Nodes'][node2]['workplaces']))))
	print "Common Liked Pages----------------: " + str(list(set(node_data['Nodes'][node1]['likes']).intersection(set(node_data['Nodes'][node2]['likes']))))
	print "Similarity------------------------: " + str(sim[node1][node2])
	a = str(raw_input("Want to know Similarities between nodes?(y/n) : "))
	if a == 'y':
		show_common(sim,graph)


def get_recomemendations(edge_list,sim,name):
	graph = Graph(edge_list)
	edge_vertices_set = get_vertices_set(edge_list)

	#output to a file name output.txt
	if name == "train":
		csv_file = open('train_recommendations.csv',"w")
		fields = ['Node','R1','R2','R3']
		thewriter = csv.DictWriter(csv_file,fieldnames=fields)
		csv_file2 = open('train_edgelist.csv',"w")
		fields2 = ['Node1','Node2','Weight']
		thewriter2 = csv.DictWriter(csv_file2,fieldnames=fields2)

		csv_file3 = open('train_graph_recommendations.csv',"w")
		fields3 = ['Node1','Node2']
		thewriter3 = csv.DictWriter(csv_file3,fieldnames=fields3)

	if name == "total":
		csv_file = open('total_recommendations.csv',"w")
		fields = ['Node','R1','R2','R3']
		thewriter = csv.DictWriter(csv_file,fieldnames=fields)
		csv_file2 = open('total_edgelist.csv',"w")
		fields2 = ['Node1','Node2','Weight']
		thewriter2 = csv.DictWriter(csv_file2,fieldnames=fields2)


	for i in edge_vertices_set:
		if i in edge_vertices_set:
			actual_friends_of_i = set(graph.neighbors(i))

			# Handles case where test-data < k
			if len(actual_friends_of_i) < k:
				k2 = len(actual_friends_of_i)
			else:
				k2 = k

			top_k = get_top_k_recommendations(graph,sim,i,k2)
			if len(top_k) == 3:
				thewriter.writerow({'Node' : str(i) , 'R1' : top_k[0] , 'R2' : top_k[1] ,'R3' : top_k[2]}) #write to csv file
				if name == "train":
					thewriter3.writerow({'Node1' : str(i) , 'Node2' : top_k[0]})
					thewriter3.writerow({'Node1' : str(i) , 'Node2' : top_k[1]})
					thewriter3.writerow({'Node1' : str(i) , 'Node2' : top_k[2]})
			elif len(top_k) == 2:
				thewriter.writerow({'Node' : str(i) , 'R1' : top_k[0] , 'R2' : top_k[1]}) #write to csv file
				if name == "train":
					thewriter3.writerow({'Node1' : str(i) , 'Node2' : top_k[0]})
					thewriter3.writerow({'Node1' : str(i) , 'Node2' : top_k[1]})
			elif len(top_k) == 1:
				thewriter.writerow({'Node' : str(i) , 'R1' : top_k[0]}) #write to csv file
				if name == "train":
					thewriter3.writerow({'Node1' : str(i) , 'Node2' : top_k[0]})
	i=0
	for i in range(len(edge_list)):
		thewriter2.writerow({'Node1' : str(edge_list[i][0]) , 'Node2' : str(edge_list[i][1]) ,'Weight' : str(sim[edge_list[i][0]][edge_list[i][1]])})





def characteristics_similarity(i,j):
	gr_len = max(len(set(node_data['Nodes'][i]['groups'])) , len(set(node_data['Nodes'][j]['groups'])))
	edu_len = max(len(set(node_data['Nodes'][i]['education'])) , len(set(node_data['Nodes'][j]['education'])))
	interests_len = max(len(set(node_data['Nodes'][i]['interests'])) , len(set(node_data['Nodes'][j]['interests'])))
	work_len = max(len(set(node_data['Nodes'][i]['workplaces'])) , len(set(node_data['Nodes'][j]['workplaces'])))
	likes_len = max(len(set(node_data['Nodes'][i]['likes'])) , len(set(node_data['Nodes'][j]['likes'])))
	lives_len = max(len(set(node_data['Nodes'][i]['lives'])) , len(set(node_data['Nodes'][j]['lives'])))


	gr = len(set(node_data['Nodes'][i]['groups']).intersection(set(node_data['Nodes'][j]['groups'])))
	edu = len(set(node_data['Nodes'][i]['education']).intersection(set(node_data['Nodes'][j]['education'])))
	interests = len(set(node_data['Nodes'][i]['interests']).intersection(set(node_data['Nodes'][j]['interests'])))
	work = len(set(node_data['Nodes'][i]['workplaces']).intersection(set(node_data['Nodes'][j]['workplaces'])))
	likes = len(set(node_data['Nodes'][i]['likes']).intersection(set(node_data['Nodes'][j]['likes'])))
	lives = len(set(node_data['Nodes'][i]['lives']).intersection(set(node_data['Nodes'][j]['lives'])))
	
	csim_max = float((6*gr_len + 5*edu_len + 3*interests_len + 2*work_len + 1*likes_len + 4*lives_len))

	csim = float((6*gr + 5*edu + 3*interests + 2*work + 1*likes + 4*lives))/csim_max
	#print str(6*gr)
	#print str(i) + ":" + str(j) + ":" + str(csim)
	return csim





def similarity(graph, i, j, method):
	if method == "common_neighbors":
		csim = characteristics_similarity(i,j)
		algo_sim = len(set(graph.neighbors(i)).intersection(set(graph.neighbors(j))))
		sim = float(csim + algo_sim)/2
		return sim
	elif method == "jaccard":
		csim = characteristics_similarity(i,j)
		algo_sim = len(set(graph.neighbors(i)).intersection(set(graph.neighbors(j))))/float(len(set(graph.neighbors(i)).union(set(graph.neighbors(j)))))
		sim = float(csim + algo_sim)/2
		#print str(i) + ":" + str(j) + ":" + str(sim)
		return sim
	elif method == "adamic_adar":
		csim = characteristics_similarity(i,j)
		algo_sim = sum([1.0/math.log(graph.degree(v)) for v in set(graph.neighbors(i)).intersection(set(graph.neighbors(j)))])
		sim = float(csim + algo_sim)/2
		return sim
	elif method == "preferential_attachment":
		csim = characteristics_similarity(i,j)
		algo_sim = graph.degree(i) * graph.degree(j)
		sim = float(csim + algo_sim)/2
		return sim
	elif method == "friendtns":
		csim = characteristics_similarity(i,j)
		algo_sim = round((1.0/(graph.degree(i) + graph.degree(j) - 1.0)),3)
		sim = float(csim + algo_sim)/2
		return sim
		












###################################
### Methods for Link Prediction ###
###################################

def local_methods(edge_list,method):
	
	graph = Graph(edge_list)
	edge_n = graph.vcount()
	edge_vertices_set = get_vertices_set(edge_list)



	train_list, test_list = split_data(edge_list)
	train_graph = Graph(train_list)
	test_graph = Graph(test_list)
	train_n =  train_graph.vcount() # This is maximum of the vertex id + 1
	train_vertices_set = get_vertices_set(train_list) # Need this because we have to only consider target users who are present in this train_vertices_set
	test_vertices_set = get_vertices_set(test_list) # Set of target users

	sim = [[0 for i in range(train_n)] for j in range(train_n)]
	for i in range(train_n):
		for j in range(train_n):
			if i!=j and i in train_vertices_set and j in train_vertices_set:
				sim[i][j] = similarity(train_graph,i,j,method)
			elif i == j and i in train_vertices_set and j in train_vertices_set:
				sim[i][j] = 1

	get_recomemendations(train_list,sim,'train')



	sim1 = [[0 for i in range(edge_n)] for j in range(edge_n)]
	for i in range(edge_n):
		for j in range(edge_n):
			if i!=j and i in edge_vertices_set and j in edge_vertices_set:
				sim1[i][j] = similarity(graph,i,j,method)
			elif i == j and i in edge_vertices_set and j in edge_vertices_set:
				sim1[i][j] = 1

	get_recomemendations(edge_list,sim1,'total')

	print_precision_and_recall(sim1,graph,train_graph,test_graph,test_vertices_set,train_vertices_set,sim)








# Calculates the Katz Similarity measure for a node pair (i,j)
def katz_similarity(katzDict,i,j):
	l = 1
	neighbors = katzDict[i]
	score = 0

	while l <= maxl:
		numberOfPaths = neighbors.count(j)
		if numberOfPaths > 0:
			score += (beta**l)*numberOfPaths

		neighborsForNextLoop = []
		for k in neighbors:
			neighborsForNextLoop += katzDict[k]
		neighbors = neighborsForNextLoop
		l += 1

	return score






# Implementation of the Katz algorithm
def katz(edge_list,method):
	graph = Graph(edge_list)
	train_list, test_list = split_data(edge_list)
	train_graph = Graph(train_list)
	test_graph = Graph(test_list)
	train_n = train_graph.vcount()
	train_vertices_set = get_vertices_set(train_list) # Need this because we have to only consider target users who are present in this train_vertices_set
	test_vertices_set = get_vertices_set(test_list) # Set of target users

	# build a special dict that is like an adjacency list
	katzDict = {}
	adjList = train_graph.get_adjlist()

	for i, l in enumerate(adjList):
		katzDict[i] = l

	sim = [[0 for i in xrange(train_n)] for j in xrange(train_n)]
	for i in xrange(train_n):
		if i not in train_vertices_set:
			continue

		for j in xrange(i+1, train_n):
			if j in train_vertices_set:		# TODO: check if we need this
				sim[i][j] = sim[j][i] = katz_similarity(katzDict,i,j)

	print_precision_and_recall(sim,graph,train_graph,test_graph,test_vertices_set,train_vertices_set,sim)







# Implementation of the friendTNS algorithm
def friendtns(edge_list, method):
	graph = Graph(edge_list)
	edge_n = graph.vcount()
	edge_vertices_set = get_vertices_set(edge_list)

	train_list, test_list = split_data(edge_list)
	train_graph = Graph(train_list)
	test_graph = Graph(test_list)
	train_n =  train_graph.vcount() # This is maximum of the vertex id + 1
	train_vertices_set = get_vertices_set(train_list) # Need this because we have to only consider target users who are present in this train_vertices_set
	test_vertices_set = get_vertices_set(test_list) # Set of target users

	sim = [[0 for i in range(train_n)] for j in range(train_n)]
	for i in range(train_n):
		for j in range(train_n):
			if i!=j and i in train_vertices_set and j in train_vertices_set and train_graph[i,j] != 0:
				sim[i][j] = similarity(train_graph,i,j,method)
			elif i == j and i in train_vertices_set and j in train_vertices_set:
				sim[i][j] = 1

	# Calculate Shortest Paths from each vertex to every other vertex in the train_vertices_set
	sp = {}
	for i in train_vertices_set:
		sp[i] = train_graph.get_shortest_paths(i)

	# Extended Sim matrix
	esim = [[0 for i in range(train_n)] for j in range(train_n)]
	for i in range(train_n):
		for j in range(train_n):
			if i!=j and i in train_vertices_set and j in train_vertices_set:
				if len(sp[i][j]) == 0: # no path exists
					esim[i][j] = 0
				elif train_graph[i,j] == 1 and train_graph[j,i] == 1: # are neighbors
					esim[i][j] = sim[i][j]
				else:
					esim[i][j] = get_sim_product(sim,sp[i][j])
	get_recomemendations(train_list,esim,'train')



	sim1 = [[0 for i in range(edge_n)] for j in range(edge_n)]
	for i in range(edge_n):
		for j in range(edge_n):
			if i!=j and i in edge_vertices_set and j in edge_vertices_set:
				sim1[i][j] = similarity(graph,i,j,method)
			elif i == j and i in edge_vertices_set and j in edge_vertices_set:
				sim1[i][j] = 1

	sp1 = {}
	for i in edge_vertices_set:
		sp1[i] = graph.get_shortest_paths(i)

	# Extended Sim matrix
	esim1 = [[0 for i in range(edge_n)] for j in range(edge_n)]
	for i in range(edge_n):
		for j in range(edge_n):
			if i!=j and i in edge_vertices_set and j in edge_vertices_set:
				if len(sp1[i][j]) == 0: # no path exists
					esim1[i][j] = 0
				elif graph[i,j] == 1 and graph[j,i] == 1: # are neighbors
					esim1[i][j] = sim1[i][j]
				else:
					esim1[i][j] = get_sim_product(sim1,sp1[i][j])
			elif i == j and i in edge_vertices_set and j in edge_vertices_set:
				esim1[i][j] = 1
	get_recomemendations(edge_list,esim1,'total')
	print_precision_and_recall(esim1,graph,train_graph,test_graph,test_vertices_set,train_vertices_set,esim)








###################################
############# Main ################
###################################

def main():
	# default-case/ help
	if len(sys.argv) < 3 :
		print "python link_prediction.py <common_neighbors/jaccard/adamic_adar/preferential_attachment/katz/friendtns> data_file_path"
		exit(1)

	# Command line argument parsing
	method = sys.argv[1].strip()
	dataset_path = sys.argv[2].strip()
	edge_list = get_edge_list(dataset_path)

	if method == "common_neighbors" or method == "jaccard" or method == "adamic_adar" or method == "preferential_attachment":
		local_methods(edge_list,method)
	elif method == "katz":
		katz(edge_list,method)
	elif method == "friendtns":
		friendtns(edge_list,method)
	else:
		print "python link_prediction.py <common_neighbors/jaccard/adamic_adar/preferential_attachment/katz/friendtns> data_file_path"

if __name__ == "__main__":
	main()