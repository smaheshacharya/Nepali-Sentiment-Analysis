import pandas as pd
import math as mth
from functools import reduce

df = pd.read_csv('merge.csv',delimiter = ',',names=['Data','Label']);
first_row = df.ix[1:,0]
# print(first_row)
second_row  = df.ix[1:,1]
# print(first_row)
# print(second_row)
data_with_split = []
each_docs = {}
def split_doc():
	for data in first_row:
		each_docs = data.split()
		data_with_split.append(each_docs)
	return data_with_split # it returns arr of each docs with spleted words


word_lists = {}
word_lists = split_doc() # your original lists
length_of_docs = len(word_lists)
print(length_of_docs)
my_set = set.union(*map(set,word_lists))# seperate each individual words from data to make matrix
word_dict  = {}
for i in range(1,len(my_set)):
	word_dict = dict.fromkeys(my_set,0)



### count each words from each docs
for count_word_value in word_lists:
	for word in count_word_value:
	    word_dict[word] += 1
def computeTf(worddict,docs_list):
	tfDict = {}
	doc_word_count = len(docs_list)
	for word , count in word_dict.items():
		tfDict[word] = count/float(doc_word_count)
	return tfDict
tf = {}
for each_line in word_lists:
	 tf = computeTf(word_dict,each_line)
# convert_dict_list = []
# for keyin word_dict.items():
# 	list_dict = [key]
# 	convert_dict_list.append(list_dict)
# word_dict = convert_dict_list
# print(word_dict)
countIdfforwordvalue = {}

def computeCountDict(word_dict,word_lists):
	countIdfforword = {}
	for i in range(1,len(my_set)):
		countIdfforword = dict.fromkeys(my_set,0)
	for word,value in word_dict.items():
		for each_line_item in word_lists:
			if word in each_line_item:
				countIdfforword[word] += 1
			# else:
			# 	countIdfforword[word] = 1
	return countIdfforword
# print(type(countIdfforwordvalue))

countIdfforwordvalue = computeCountDict(word_dict,word_lists)

def computeTfidf(tf,idf):
	tfidf = {}
	for i in range(1,len(my_set)):
		tfidf = dict.fromkeys(my_set,0)
	for word , val in tf.items():
		tfidf[word] = val*mth.log(idf[word]/length_of_docs)
	return tfidf

tfidf_result = computeTfidf(tf,countIdfforwordvalue)
print(tfidf_result)
