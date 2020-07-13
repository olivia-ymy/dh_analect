# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 16:46:23 2020

@author: mi
"""


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


def showmatrix(matrix,path):
    count = 0
    for i in range(0, len(matrix)):
        matrixtxt = ''
        for j in range(0, len(matrix)):
            matrixtxt = matrixtxt+str(matrix[i][j])+'\t'
        matrixtxt = matrixtxt[:-1]+'\n'
        count = count+1
        savedata(path, matrixtxt)
        print('Line No.'+str(count)+' had been done!')


def savedata(path, text):
    with open(path, 'a', encoding='utf-8') as f:
        f.write(text)
    return path+' write succeeded!'


with open('m1.txt','r',encoding='utf-8') as m:
    lines=m.readlines()

for line in lines:
    print(line.strip().replace(',','\t'))

length = len(lines)

matrix=buildmatrix(length, length)
matrix=inimatrix(matrix, lines, length)

concept_list='concepts_170.txt'

with open(concept_list,'r', encoding='utf-8') as concept_list_file:
    concept_lines=concept_list_file.readlines()

selection=['+']

for concept_line in concept_lines:
    selection.append(concept_line.strip().split('\t')[1])
 
#selection=['+','仁','义','礼','信']
   
print(selection)

row_matrix=[]

for line in matrix:
    if line[0] in selection:
        row_matrix.append(line)

col_matrix=[]
for row in row_matrix:
    col_list=[]
    col_list.append(row[0])
    col_matrix.append(col_list)


j=0
for row in row_matrix[0]:
    print(row)
    if row in selection[1:]:
        print(row)
        i=0
        for col in col_matrix:
            col.append(row_matrix[i][j])
            i=i+1
    j=j+1


matrixtxt = showmatrix(col_matrix,'m_4.txt')

    