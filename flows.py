#!/usr/local/bin/python3
############################################################
# Starter code for solving flow problems in graphs
# April 2022
# 
############################################################
from importlib.resources import path
import sys
import os
import numpy as np
import simplegraphs as sg
import math
import random

MAX_WIDTH = 2000
MAX_HEIGHT = 2000

def gold(coords):
    # write your code here
    return []


def rounding(matrix):
    # write your code here
    for i in range(0, len(matrix)):
        for j in range(0, len(matrix[0])):
            pass
    return matrix


def maxflow(G, s, t):
    Gf = sg.copyGraph(G)
    # Run the BFS algorithm to determine the parent pointers in the graph
    _, parents, _ = sg.BFS(Gf, s)

    # Check if an augmenting path exists in Gf from source to sink
    augmenting_path, path_present = find_path(parents, s, t)
    
    # Return and exit program if no augmenting path exists
    if (not path_present):
        print('No Solution Exists!')
        return

    # If a path exists, compute the flow values for the current and subsequent augmenting paths
    flowlist = {}
    while path_present:
        # Compute flow value for the current augmenting path
        flow = compute_flow(Gf, augmenting_path)

        # Append the path and its corresponding flow value to the flowlist
        ''' Reference: https://stackoverflow.com/questions/7257588/why-cant-i-use-a-list-as-a-dict-key-in-python '''
        flowlist[tuple(augmenting_path)] = flow     # Convert list to tuple (lists not hashable in Python)

        # Send flow from Source to Sink
        for i in range(0, len(augmenting_path) - 1):
            # Check if edge has remaining capacity
            if Gf["adj"][augmenting_path[i]][augmenting_path[i + 1]] > flow:
                Gf["adj"][augmenting_path[i]][augmenting_path[i + 1]] -= flow      # TODO: Check/Create flow update functions
            else:
                # Edge has no remaining capacity or capacity <= flow. 
                # Either case, edge becomes critical. Remove this edge from the graph (adjacency list)
                Gf["adj"][augmenting_path[i]].pop(augmenting_path[i + 1])

            # Add Residual edges (REVERSE) if flow has been sent along this edge
            # If there is no residual edge from v to u, add one and set the residual capacity as the flow value
            if augmenting_path[i] not in Gf["adj"][augmenting_path[i + 1]].keys():
                Gf["adj"][augmenting_path[i + 1]][augmenting_path[i]] = flow
            else:
                # Update the residual capacity
                Gf["adj"][augmenting_path[i + 1]][augmenting_path[i]] += flow

        # Run BFS again
        _, parents, _ = sg.BFS(Gf, s)
        # Find the next augmenting path through which flow can be sent
        augmenting_path, path_present = find_path(parents, s, t)

    return flowlist     # Flows: dict[tuple(<path>), flow value]

def find_path(parents: dict, source: int, sink: int):
    '''
        "After determining if an augmenting path exists..."
        Helper function to determine whether an augmenting path exists from Source to Sink.
        IF yes, return the augmenting path
    '''
    # If the Source or Sink node is not in the graph, no augmenting path exists
    if source not in parents.keys() or sink not in parents.keys():
        return [], False

    # Check for an augmenting path
    augmenting_path = []    # Maintain list for augmenting path nodes
    curr = sink             # Temporary variable for iterating through the sink node
    while parents[curr] is not None:        # Loop until we reach the source node
        augmenting_path.insert(0, curr)     # Append vertex to the augmenting path
        curr = parents[curr]                # Update the sink vertex
    augmenting_path.insert(0, source)       # Append the source node

    return augmenting_path, True

def compute_flow(Gf: dict, augmenting_path: list):
    '''
        Helper function to compute the flow value in an augmenting path (bottleneck capacity).
    '''
    flow = np.Inf   # Flow value (Defalut: INFINITY)
    
    for i in range (0, len(augmenting_path) - 1):
        edge_capacity = Gf["adj"][augmenting_path[i]][augmenting_path[i + 1]]
        # Compute flow (bottleneck capacity)
        flow = min(flow, edge_capacity)

    return flow     # The bottleneck capacity of the path

def residual_edge_exists(Gf: dict, augmenting_path: list, u: int, v: int):
    '''
        Helper method to determine whether a residual edge exists from any vertex v to u (u, v in V)
    '''
    return augmenting_path[u] in Gf["adj"][augmenting_path[v]].keys()

def capacity_left(Gf: dict, augmenting_path: list, u: int, v: int, flow: float):
    '''
        Helper method to determine whether there is capacity left on the edge(u, v)
    '''
    return Gf["adj"][augmenting_path[u]][augmenting_path[v]] > flow

############################################################
#
# The remaining functions are for reading and writing outputs, and processing
# the command line arguments. You shouldn't have to modify them.  You can use them to 
# help you test
# 
############################################################

def main(args = []):
    # Expects 2 command-line arguments:
    # 1) name of a file describing the graph
    if len(args) < 2:
        print("Too few arguments! There should be at least 4.")
        print("flows.py <cmd> <file>")
        return

    task = args[0]
    if task == "gold":
        coords = read_input(args[1])
        gold(coords)   
    elif task == "rounding":
        matrix = read_input(args[1])
        nm = rounding(matrix)
        if compare_matrix(matrix, nm):
            print_matrix(nm)
    elif task == "maxflow":
        # the following may help you test your maxflow solution
        graph_file = args[1]
        s = int(args[2])
        t = int(args[3])
        G = sg.readGraph(graph_file) # Read the graph from disk
        flow = maxflow(G, s, t)
        print(flow)

    return


def read_input(filename):
    with open(filename, 'r') as f:
        blocks = [[int(x) for x in s.split()] for s in f.read().splitlines()]
    return blocks

def print_matrix(matrix):
    for r in matrix:
        print(*r)

# verifies that two matricies have the same size, same row and column sums
def compare_matrix(m1,m2):
    r1 = len(m1)
    r2 = len(m2)
    c1 = len(m1[0])
    c2 = len(m2[0])
    if r1!=r2 or c1!=c2:
        print('Sizes are different')
        return False

    for ri in range(0,r1):
        rs1 = sum(m1[ri])
        rs2 = sum(m2[ri])
        if rs1 != rs2:
            print('Row sum {ri} differ: {rs1} != {rs2}')
            return False

    for cj in range(0,c1):
        cs1 = 0
        cs2 = 0
        for i in range(0,r1):
            cs1 += m1[i][cj]
            cs2 += m2[i][cj]
        if cs1 != cs2:
            print('Col sum {cj} differ: {cs1} != {cs2}')
            return False

    return True

if __name__ == "__main__":
    main(sys.argv[1:])    