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
    print(parents)
    ## write your code below this point
    augmenting_paths, paths_present = augment(Gf, parents, s, t)
    if (not paths_present):
        print('No Solution Exists!')
        return

    print(augmenting_paths)
    flows = find_path(Gf, augmenting_paths)
    # Flows: dict[tuple(<path>), flow value]
    return flows

def augment(Gf: dict, parents: dict, source: int, sink: int):
    # If the Sink node is not in the graph
    if sink not in parents:
        return [], False

    augmenting_paths = []
    temp = sink
    while sink in parents:
        path = []
        curr = sink
        while (curr is not source):
            path.append(curr)
            temp = curr
            curr = parents[curr]
            # Remove this path from node to the sink
            if (temp is not sink):
                sg.delEdge(Gf, curr, temp)
        
        sg.delEdge(Gf, source, temp)
        path.append(source)
        augmenting_paths.append(path.reverse())
        _, parents, _ = sg.BFS(Gf, source)
        print(parents)

    return augmenting_paths, True


def find_path(Gf, augmenting_paths):
    pass

def is_residual(capacity: int):
    return capacity == 0

def capacity_left(capacity: int, flow: int):
    return capacity - flow



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