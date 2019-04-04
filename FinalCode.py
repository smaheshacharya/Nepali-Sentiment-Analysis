import pandas as pd
import math
from functools import reduce
import numpy as np
df = pd.read_csv('merge.csv',delimiter = ',',names=['Data','Label']);
first_row = df.ix[1:,0]
second_row  = df.ix[1:,1]

data_with_split = []
each_docs = []
def split_doc():
	for data in first_row:
		each_docs = data.split()
		data_with_split.append(each_docs)
	return data_with_split



word_lists = split_doc()

my_set = set.union(*map(set,word_lists))

word_dict  = [[]]
for i in range(1,len(my_set)):
	word_dict = dict.fromkeys(my_set,0)
print("word dict")
print(word_dict)
### count each words from each docs
for count_word_value in word_lists:
	for word in count_word_value:
	    word_dict[word] += 1
	print(word_dict)

def computeTf(worddict,docs_list):
	tfDict = {}
	doc_word_count = len(docs_list)
	for word , count in worddict.items():
		tfDict[word] = count/float(doc_word_count)
	return tfDict
tf = {}
for each_line in word_lists:
	 tf = computeTf(word_dict,each_line)


def computeIdf(docList):
	idfDict = {}
	N = len(docList)
	print(N)
	idfDict = dict.fromkeys(docList[0].keys(),0)

	for doc in docList:
		for word,val in doc.items():
			if val >0:
				idfDict[word] += 1
	for word, val in idfDict.items():
		idfDict[word] = math.log(N/float(val))

	return idfDict

idf ={}

idf = computeIdf(word_lists)
print("idf")
print(idf)



# def computeTfidf(tfbow,idf):
# 	tfidf = {}
# 	for word , val in tfbow.items():
# 		tfidf[word] =val*idf[word]
# 	return tfidf

# tfidf1 = computeTfidf(tfBowa,idf)
# tfidf2 = computeTfidf(tfBowb,idf)
# print(pd.DataFrame([tfidf1,tfidf2]))
