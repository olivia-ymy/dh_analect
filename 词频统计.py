# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 11:46:54 2019

@author: y
"""
from collections import Counter
import re

# 创建停用词list
def stopwordslist(filepath):
    stopwords = [line.strip() for line in open(filepath, 'r', encoding="utf-8").readlines()]
    return stopwords

#每个字符后面加空格
def withspace(s):
    s=s.replace('/',' ')
    return s.split(' ')

# 对句子进行分词
def seg_sentence(sentence):
    sentence_seged = withspace(sentence.strip())
    stopwords = stopwordslist('标点符号.txt')  #将标点符号停用
    outstr = []
    for word in sentence_seged:
        if word not in stopwords:
            if word != '\t':
                outstr.append(word)
    return outstr

term_frequency={}
 
with open('analects_chapter_cut.txt', 'r', encoding="utf-8") as inputs:#加载要处理的文件的路径
    lineno=1
    for line in inputs.readlines():
        line=line.replace('【','')
        line=line.replace('】','')
        line=re.sub('\(\d+\.\d+\)','',line)
        line_seg = seg_sentence(line)  # 这里的返回值是字符串
        ac=Counter(line_seg)
        data = dict(ac.most_common())#排序操作。ac.most_common(5) # 按序输出出现次数top5的元素，如不指定数字，则按序输出全部元素
        term_frequency[lineno]=data
        lineno=lineno+1

#导入概念列表，转化为序号-概念辞典{序号：概念}
def get_charadict(infile):
    chara_dict = {}
    for each_line in open(infile,encoding='utf-8').readlines():
        line_content = each_line.strip('\t').split()
        #print(line_content)
        
        chara_dict[int(line_content[0])] = []
        for i in range(1, len(line_content)):
            chara_dict[int(line_content[0])].append(line_content[i])
    return(chara_dict)

#转换为概念-序号辞典{姓名：人物序号}
def name_to_index(chara_dict):
    name_dict = {}
    for index in chara_dict.keys():
        for name in chara_dict[index]:
            name_dict[name] = index
    return(name_dict)


#concept_list='onegram_3.txt'
#concept_list='onegram_5.txt'
#concept_list='onegram_10.txt'
#concept_list='concepts_5.txt'
#concept_list='concepts_60.txt'
concept_list='concepts_170.txt'

index2concept = get_charadict(concept_list)
#概念个数
concept_num = len(index2concept.keys())
print('共有%s个概念' % concept_num)
concept2index = name_to_index(index2concept)

    
content2concept = {}
for i in range(1, 21):
    index2count={}
    for j in range(1,concept_num+1):#初始化
        index2count[j]=0
    for key in concept2index.keys():
        frequency_dict=term_frequency[i]
        if key in frequency_dict:
#            print(key,frequency_dict[key])
            index2count[concept2index[key]]+=frequency_dict[key]
    line_count=[]
    for j in range(1,concept_num+1):
        line_count.append(index2count[j])
    content2concept[i]=line_count

#获取整部、前半部、后半部-全部概念文档词频向量
whole_part = [0] * concept_num
head_part = [0] * concept_num
tail_part = [0] * concept_num
for i in range(len(content2concept)):
    for j in range(concept_num):
        whole_part[j] += content2concept[i+1][j]
    if i<10:
        for j in range(concept_num):
            head_part[j] += content2concept[i+1][j]
    else:
        for j in range(concept_num):
            tail_part[j] += content2concept[i+1][j]
    
counts = []
for con in content2concept.keys():
    counts.append(content2concept[con])
counts.append(whole_part)
counts.append(head_part)
counts.append(tail_part)
len(counts[0])

core_concept = list(index2concept.keys())

    
#获得每一章节的tf-idf文档向量
from sklearn.feature_extraction.text import TfidfTransformer
transformer = TfidfTransformer()
tfidf = transformer.fit_transform(counts).toarray()
tfidf.shape



#每章节的概念向量计算相似度，寻找对应相似度最大的章节
def cos_simi(a, b):
    muti_sum = 0
    a_lensqur = 0
    b_lensqur = 0
    for i in range(len(a)):
        muti_sum += a[i] * b[i]
        a_lensqur += a[i]**2
        b_lensqur += b[i]**2
    base = a_lensqur**0.5 * b_lensqur**0.5
    return(muti_sum/ base)

#获得章节-章节相似度矩阵，交叉点为相似度
chapter2simi = {}
for i in range(20):
    chapter2simi[i] = [0]*20

for i in range(20):
    for j in range(i + 1, 20):
        chapter2simi[i][j] = cos_simi(tfidf[i], tfidf[j])
        chapter2simi[j][i] = cos_simi(tfidf[i], tfidf[j])


#获得每篇与整部的相似度
chapter_whole_simi = [0] * 20
for i in range(20):
    chapter_whole_simi[i] = cos_simi(tfidf[i], tfidf[-3])

#获得前半部、后半部与整部的相似度
head_whole_simi = cos_simi(tfidf[-2], tfidf[-3])
tail_whole_simi = cos_simi(tfidf[-1], tfidf[-3])



#每章与全书的相似度
print('每章与全书的相似度:')
key_value ={}
for i in range(20):
    key_value[str(i+1)]=chapter_whole_simi[i]

#排序
print(sorted(key_value.items(), key = lambda kv:(kv[1], kv[0]),reverse=True)) 

#前半部、后半部与全书的相似度
print('前半部', '后半部')
print(head_whole_simi, tail_whole_simi)





#获得后半部中与前半部章节最匹配的章节
head_tail_match = []
head_tail_simi = []
for i in range(10):
    simi = chapter2simi[i]
    head_tail_match.append(simi.index(max(simi[10:]))+1)
    head_tail_simi.append(max(simi[10:]))
head_tail_sort = sorted(head_tail_simi, reverse=True)

#获得前半部中与后半部章节最匹配的章节
tail_head_match = []
tail_head_simi = []
for i in range(10, 20):
    simi = chapter2simi[i]
    tail_head_match.append(simi.index(max(simi[:10]))+1)
    tail_head_simi.append(max(simi[:10]))
tail_head_sort = sorted(tail_head_simi, reverse=True)

#获得章节重复匹配
maxmatch = []
maxmatch_simi = []
for i in range(20):
    simi = chapter2simi[i]
    maxmatch.append(simi.index(max(simi))+1)
    maxmatch_simi.append(max(simi))
maxmatch_sort = sorted(maxmatch_simi, reverse=True)





#绘制前后部相似度匹配连线图
from pylab import *
import matplotlib.pyplot as plt
import numpy as np
mpl.rcParams['font.sans-serif'] = ['SimHei']

x = np.arange(1, 3)
c = ['#DCDCDC', '#87CEFA', '#4682B4', '#000080']

fig = plt.figure(figsize = (10, 7))

ax1 = fig.add_subplot(121)
ax1.set_ylim([0, 21])
ax1.set_yticks(np.arange(1, 21, 1))
ax1.set_xticks([])
ax1.set_ylabel('章节', fontsize=15)
ax1.set_title('前10章与后10章配对'+'('+str(concept_num)+'个概念)')
for i in range(1, 11):
    if head_tail_simi[i-1] == head_tail_sort[0]:
        ax1.plot(x, [i, head_tail_match[i-1]], c[-1])
    elif head_tail_simi[i-1] == head_tail_sort[1]:
        ax1.plot(x, [i, head_tail_match[i-1]], c[-2])
    elif head_tail_simi[i-1] == head_tail_sort[2]:
        ax1.plot(x, [i, head_tail_match[i-1]], c[-3])
    else:
        ax1.plot(x, [i, head_tail_match[i-1]], c[0])

ax2 = ax1.twinx()
ax2.set_ylim([0, 21])
ax2.set_yticks(np.arange(1, 21, 1))

ax3 = fig.add_subplot(122)
ax3.set_ylim([0, 21])
ax3.set_yticks(np.arange(1,21,1))
ax3.set_xticks([])
#ax3.set_ylabel('章节', fontsize=15)
ax3.set_title('后10章与前10章配对'+'('+str(concept_num)+'个概念)')
for i in range(1, 11):
    if tail_head_simi[i-1] == tail_head_sort[0]:
        ax3.plot(x, [i+10, tail_head_match[i-1]], c[-1])
    elif tail_head_simi[i-1] == tail_head_sort[1]:
        ax3.plot(x, [i+10, tail_head_match[i-1]], c[-2])
    elif tail_head_simi[i-1] == tail_head_sort[2]:
        ax3.plot(x, [i+10, tail_head_match[i-1]], c[-3])
    else:
        ax3.plot(x, [i+10, tail_head_match[i-1]], c[0])

ax4 = ax3.twinx()
ax4.set_ylim([0, 21])
ax4.set_yticks(np.arange(1, 21, 1))

plt.savefig('head_tail_similarity_onegram3.png')
plt.show()





#绘制章节重复匹配相似图
fig = plt.figure(figsize = (10, 7))

ax1 = fig.add_subplot(121)
ax1.set_ylim([0, 21])
ax1.set_yticks(np.arange(1, 21, 1))
ax1.set_xticks([])
ax1.set_ylabel('章节', fontsize=15)
ax1.set_title('章节之间配对'+'('+str(concept_num)+'个概念)')
for i in range(1, 21):
    if maxmatch_simi[i-1] == maxmatch_sort[0]:
        ax1.plot(x, [i, maxmatch[i-1]], c[-1])
    elif maxmatch_simi[i-1] == maxmatch_sort[1]:
        ax1.plot(x, [i, maxmatch[i-1]], c[-2])
    elif maxmatch_simi[i-1] == maxmatch_sort[2]:
        ax1.plot(x, [i, maxmatch[i-1]], c[-3])
    else:
        ax1.plot(x, [i, maxmatch[i-1]], c[0])

ax2 = ax1.twinx()
ax2.set_ylim([0, 21])
ax2.set_yticks(np.arange(1, 21, 1))


plt.savefig('chapter_match_onegram3.png')
plt.show()

