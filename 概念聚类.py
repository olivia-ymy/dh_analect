
#代码参考以下网页
#https://blog.csdn.net/alanconstantinelau/article/details/69258443


import re


def removesymbol(line):
    #【(1.2)】
    line=re.sub('(\d+.\d+)','',line)
    line=line.replace('【','')
    line=line.replace('】','')
    line=line.replace('】','')
    line=line.replace('】','')
    line=line.replace('//','/')
    return line


def readdata(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        lines=f.readlines()
    for line in lines:
        data.append(removesymbol(line.strip()))
    return data


def savedata(path, text):
    with open(path, 'a', encoding='utf-8') as f:
        f.write(text)
    return path+' write succeeded!'


def buildmatrix(x, y):
    return [[0 for j in range(y)] for i in range(x)]


def dic(path):
    keygroup = readdata(path)
    keytxt = '/'.join(keygroup)
    keylist = list(set([key for key in keytxt.split('/') if key != '']))
    keydic = {}
    pos = 0
    for i in keylist:
        pos = pos+1
        keydic[pos] = str(i)
    return keydic


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


def inimatrix(matrix, dic, length):
    matrix[0][0] = '+'
    for i in range(1, length):
        matrix[0][i] = dic[i]
    for i in range(1, length):
        matrix[i][0] = dic[i]
    return matrix


def countmatirx(matrix, dic, mlength, keylis):
    for i in range(1, mlength):
        for j in range(1, mlength):
            count = 0
            for k in keylis:
                ech = str(k).split('/')
                # print(ech)
                if str(matrix[0][i]) in ech and str(matrix[j][0]) in ech and str(matrix[0][i]) != str(matrix[j][0]):
                    count = count+1
                else:
                    continue
            matrix[i][j] = str(count)
    return matrix


def main():
    txtpath = r'analects_cut.txt'
    wrypath = r'1.txt'
    keylis = readdata(txtpath)
    keydic = dic(txtpath)
    length = len(keydic)+1
    matrix = buildmatrix(length, length)
    print('Matrix had been built successfully!')
    matrix = inimatrix(matrix, keydic, length)
    print('Columns and rows had been writen!')
    matrix = countmatirx(matrix, keydic, length, keylis)
    print('Matrix had been counted successfully!')
    
    matrixtxt = showmatrix(matrix, wrypath)

if __name__ == '__main__':
    main()

