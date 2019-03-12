
# coding: utf-8

# In[1]:


from concept_match import get_charadict, name_to_index, clean_content, chara_match
from nltk.util import ngrams
from collections import defaultdict
import pandas as pd


# In[5]:


from nltk.util import ngrams


# In[6]:


#将论语文本中的人名替换为#
def replace_name(content):
    index2name = get_charadict("character.txt")
    chara_num = len(index2name.keys())
    name2index = name_to_index(index2name)
    
    chara2content = {}
    for i in range(1, chara_num + 1):
        chara2content[i] = []
    chara2content = chara_match(name2index, content, chara2content)
    
    return content

#将论语文本中的标点符号替换为#
def replace_punct(content):
    punct = ['：', '「', '，', '？', '」', '；', '。', '！', '《', '》', '『', '』', '、']
    for chapter in content.keys():
        for p in punct:
            content[chapter] = content[chapter].replace(p, '#')
    return content

#每一句获得ngram
def ngrams_freq(s, grams_freq, n):
    for i in list(ngrams(s, n)):
        if i not in grams_freq.keys():
            grams_freq[i] = 0
        grams_freq[i] += 1
    return grams_freq

#全文获得ngram
def get_ngrams(content):
    onegrams_freq = {}
    bigrams_freq = {}
    trigrams_freq = {}
    fourgrams_freq = {}
    fivegrams_freq = {}
    
    for chapter in content.keys():
        seqs = content[chapter].split('#')
        for seq in seqs:
            onegrams_freq = ngrams_freq(seq, onegrams_freq, 1)
            bigrams_freq = ngrams_freq(seq, bigrams_freq, 2)
            trigrams_freq = ngrams_freq(seq, trigrams_freq, 3)
            fourgrams_freq = ngrams_freq(seq, fourgrams_freq, 4)
            fivegrams_freq = ngrams_freq(seq, fivegrams_freq, 5)
    
    return onegrams_freq, bigrams_freq, trigrams_freq, fourgrams_freq, fivegrams_freq


# In[7]:


s = '子曰##学而时习之#不亦说乎#有朋自远方来#不亦乐乎#人不知而不愠#不亦君子乎##'
seqs = s.split('#')
for seq in seqs:
    print(list(ngrams(seq, 1)))


# In[8]:


#读入论语文本
content_dict = clean_content("analects.txt")
content_dict = replace_name(content_dict)
content_dict = replace_punct(content_dict)
onegrams, bigrams, trigrams, fourgrams, fivegrams = get_ngrams(content_dict)

onegrams_list = sorted(onegrams.items(), key=lambda x: x[1], reverse=True)
bigrams_list = sorted(bigrams.items(), key=lambda x: x[1], reverse=True)
trigrams_list = sorted(trigrams.items(), key=lambda x: x[1], reverse=True)
fourgrams_list = sorted(fourgrams.items(), key=lambda x: x[1], reverse=True)
fivegrams_list = sorted(fivegrams.items(), key=lambda x: x[1], reverse=True)

n_gram = {}
for i in range(1, 6):
    n_gram[i] = []
for i in onegrams_list:
    if i[1] > 5:
        n_gram[1].append(i[0][0])
for i in bigrams_list:
    if i[1] > 5:
        gram = ''
        for w in i[0]:
            gram = gram + w
        n_gram[2].append(gram)
for i in trigrams_list:
    if i[1] > 5:
        gram = ''
        for w in i[0]:
            gram = gram + w
        n_gram[3].append(gram)
for i in fourgrams_list:
    if i[1] > 3:
        gram = ''
        for w in i[0]:
            gram = gram + w
        n_gram[4].append(gram)
for i in fivegrams_list:
    if i[1] > 3:
        gram = ''
        for w in i[0]:
            gram = gram + w
        n_gram[5].append(gram)


# In[13]:


len(n_gram[2])


# In[6]:


grams = []
for i in n_gram[1]:
    grams.append(i)
for i in n_gram[2]:
    grams.append(i)
for i in n_gram[3]:
    grams.append(i)

df = pd.DataFrame(grams)
df.to_csv('ngrams.csv', index=False, sep=',', encoding='utf_8_sig')

