import pandas as pd
import math as mth
from functools import reduce
from collections import Counter

df = pd.read_csv('merge.csv',delimiter = ',',names=['Data','Label']);
print(df)
first_row = df.ix[1:,0]
# print(first_row)
second_row  = df.ix[1:,1]
print(second_row)
data_with_split = []
each_docs = []
stop_words_split_final=[]
def split_doc():
	for data in first_row:
		each_docs = data.split()
		data_with_split.append(each_docs)
	return data_with_split # it returns arr of each docs with spleted words
word_lists = []
word_lists = split_doc() 
length_of_docs = len(word_lists)
def individual_words():
	my_set = set.union(*map(set,word_lists))# seperate each individual words from data to make matrix
	return my_set

def set_to_list():
	my_set_list = individual_words()
	convert_into_list = list(my_set_list)
	return convert_into_list
individual_word_list = set_to_list()

def count_occurence_of_word_vocab():
	my_set = individual_words()
	doc ={}
	word_dict  = {}
	for i in range(len(word_lists)):
		for word in word_lists[i]:
			word_dict = dict.fromkeys(my_set,0)

	for count_word_value in word_lists:
		for word in count_word_value:
			if word in word_dict:
				word_dict[word] += 1
	return word_dict

def vectorizer_docs(line):
	vectorizer_docs = []
	matrix_doc = []
	for word in individual_word_list:
		if word in line:
			vectorizer_docs.append(1)
		else:
			vectorizer_docs.append(0)
	return vectorizer_docs
	vectorizer_docs.clear()


doc_vec1 = []
doc_vec2 = []		
for line in word_lists:
	doc_vec1 = vectorizer_docs(line)
	doc_vec2.append(doc_vec1)
print(doc_vec2)


def computeTf(docs_list):
	tf_vec = []
	tf_each_doc_vec = []
	doc_word_count = len(docs_list)
	count_each_word = Counter(docs_list)
	print(count_each_word)
	for each_word,count in count_each_word.items():
		tf_vec.append(count/float(doc_word_count))
	tf_each_doc_vec.append(tf_vec)
	return tf_each_doc_vec

tf = []
tf_vec = []
for each_line in word_lists:
	 tf = computeTf(each_line)
	 tf_vec += tf
print(tf_vec)