# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 14:29:21 2020

@author: mi
"""

import pandas as pd

def convert_csv(matrix):
    i = 0
    df = pd.DataFrame(columns=['Source','Target','Weight'])
    for row in range(1, len(matrix)):
        for col in range(1, len(matrix)):
            if col >= row:
                if matrix[col][row] != '0':
                    df.loc[i] = [matrix[0][row],matrix[col][0],matrix[col][row]]
                    i += 1
    df.to_csv('Gephi数据_170.csv', index=False)
    print (df)

def buildmatrix(x, y):
    return [[0 for j in range(y)] for i in range(x)]

def inimatrix(matrix, lines, length):
    i=0
    for line in lines:
        items=line.strip().split('\t')
        for j in range(0,length):
            matrix[i][j] = items[j]
        i=i+1
    return matrix

with open('m_170.txt','r',encoding='utf-8') as m:
    lines=m.readlines()

for line in lines:
    print(line.strip())

length = len(lines)

matrix=buildmatrix(length, length)
matrix=inimatrix(matrix, lines, length)
convert_csv(matrix)

