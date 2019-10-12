# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 10:56:42 2019

@author: y
"""
import re


#导入人物字典{人物序号：[姓名列表]}
def get_charadict(infile):
    chara_dict = {}
    for each_line in open(infile,encoding='utf-8').readlines():
        line_content = each_line.strip('\t').split()
        #print(line_content)
        
        chara_dict[int(line_content[0])] = []
        for i in range(1, len(line_content)):
            chara_dict[int(line_content[0])].append(line_content[i])
    return(chara_dict)

#转换人物字典为姓名字典{姓名：人物序号}
def name_to_index(chara_dict):
    name_dict = {}
    for index in chara_dict.keys():
        for name in chara_dict[index]:
            name_dict[name] = index
    return(name_dict)

#替换人名
def replaceName(string,text):
    return re.sub(string, string.replace('/',''), text);

index2name = get_charadict("character.txt")
name2index = name_to_index(index2name)

names=[]
for key in name2index.keys():
    names.append(key)
    print(key)

names_split=[]
for name in names:
    name='/'.join(name)
    names_split.append(name)
    print(name)
#
#
for each_line in open('analects_chapter_cut.txt','r',encoding='utf-8').readlines():
    for name_s in names_split:
        each_line=replaceName(name_s,each_line)
    print(each_line)

