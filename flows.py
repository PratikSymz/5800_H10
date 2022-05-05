#!/usr/local/bin/python3
############################################################
# Starter code for solving flow problems in graphs
# April 2022
# 
############################################################
import sys
import os
import numpy as np
import simplegraphs as sg

MAX_WIDTH = 2000
MAX_HEIGHT = 2000

def gold(coords):
    # write your code here
    
    print(coords)


def rounding(matrix):
    # write your code here
    # print(matrix)
    # print(type(matrix))

    sums_rows = [sum([item%10 for item in row]) for row in matrix]
    sums_cols = [sum([item%10 for item in col]) for col in np.array(matrix).T]

    # print(sums_rows)
    # print(sums_cols)

    filtered = []

    for row in matrix:
        a = []
        for item in row:
            if item % 10 > 0:
                a.append(10)
            else:
                a.append(0)
        filtered.append(a) 

    #print(filtered)

    G = sg.emptyGraph(0)
    p = 1

    for i,row in enumerate(sums_rows):
        sg.addDirEdge(G, 0, i+1, row)
        p += 1

    for i,col in enumerate(sums_cols):
        sg.addDirEdge(G, p, len(matrix[0]) + len(matrix) + 1, col)
        p += 1

    for i, col_capacities in enumerate(filtered):
        for j, cap in enumerate(col_capacities):
            if cap:
                sg.addDirEdge(G, i+1, len(matrix[0]) + j+1, cap)


    mf = maxflow(G, 0, len(matrix[0]) + len(matrix) + 1)

    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            matrix [i][j] -= matrix[i][j] % 10


    l = []

    for x in mf.keys():
        #print(x)

        if len(x) > 4:
            #print('x is larger than 3')

            y = list(x[1:-1])
            #print(y)

            while len(y)>1:
                p = y.pop(0)
                q = y[0]

                #print(p,q,y)

                if p > q:
                    matrix[q-1][p - len(matrix) - 1] -= 10
                    print(matrix)
                else:
                    if (p,q) not in l:
                        matrix[p-1][q - len(matrix) - 1] += 10
                        print(matrix)

        else:
            a,b = x[-3:-1]
            l.append((a,b))
            #print(a,b, len(matrix))
            matrix[a-1][b - len(matrix) - 1] += 10
            print(matrix)

    return matrix


def maxflow(G:dict, s:int, t:int) -> dict: 
    Gf = sg.copyGraph(G)
    ## write your code below this point

    # for i in Gf['adj']:
    #         print(f"{i} : {Gf['adj'][i]}")

    flows = {}

    _, parents, _ = sg.BFS(Gf,s)

    augment_exists, augmenting_path = augment(parents, s, t)

    while augment_exists:

        # print('flows.py::L41: augmenting path:',augmenting_path)

        flow = find_flow(Gf,augmenting_path)
        flows[tuple(augmenting_path)] = flow

        # print('flows.py::L43: flow:',flow)
        # for g in Gf['adj']:
        #     print(g,":",Gf['adj'][g])

        for i,_ in enumerate(augmenting_path[:-1]):
            # print(f'flows.py::L46: capacity from {augmenting_path[i]} to {augmenting_path[i+1]}:', Gf['adj'][augmenting_path[i]][augmenting_path[i+1]])
            #Forward:
            if Gf['adj'][augmenting_path[i]][augmenting_path[i+1]] > flow:
                Gf['adj'][augmenting_path[i]][augmenting_path[i+1]] -= flow
            else:
                Gf['adj'][augmenting_path[i]].pop(augmenting_path[i+1])
            
            #Residual:
            if augmenting_path[i] in Gf['adj'][augmenting_path[i+1]].keys():
                Gf['adj'][augmenting_path[i+1]][augmenting_path[i]] += flow
            else:
                Gf['adj'][augmenting_path[i+1]][augmenting_path[i]] = flow

        _, parents, _ = sg.BFS(Gf,s)

        augment_exists, augmenting_path = augment(parents, s, t)

    return flows

############################################################
# Helpers
############################################################

def augment(parents:dict, s:int, t:int) -> tuple:
    path = []
    if t not in parents.keys():
        return False, []

    while parents[t] is not None:
        path.append(t)
        if parents[t] == s:
            path.append(parents[t])
            path.reverse()
            return True, path
        t = parents[t]
    return False, []

def find_flow(G:dict, path:list) -> float:
    flow = np.inf
    while len(path) > 1:
        # print(f'Edge between {path[0]} and {path[1]}')
        # print('flows.py::L80: path',path)
        # print('flows.py::L81: flow',flow)
        cap = G['adj'][path[0]][path[1]]
        # print('flows.py::L83: Capacity:',cap)
        flow = min(flow, cap)
        path = path[1:]

    # print(f'flows.py::L87: final flow: {flow}')

    return flow

def createKey(x, y):
    return str(x) + ":" + str(y)

def determine_color(x, y):
    if (x % 2 == 0) and (y % 2 == 0):
        return True
    else:
        return False 

############################################################

############################################################
#
# The remaining functions are for reading and writing outputs, and processing
# the command line arguments. You shouldn't have to modify them.  You can use them to 
# help you test
# 
############################################################

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

if __name__ == "__main__":
    main(sys.argv[1:])    
