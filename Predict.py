import pandas as pd
import math as mth
from functools import reduce
from collections import Counter
import numpy as np
from sklearn.naive_bayes import GaussianNB
import pickle
import datetime
import random
import sys
from sklearn import metrics
import re
from random import randrange
from sklearn.model_selection import train_test_split
now = str(datetime.datetime.now())

df = pd.read_csv('merge.csv', quotechar='"', names=['Data', 'Label']);
# print(df)
first_col = df.ix[1:, 0]
print(first_col)
second_col = df.ix[1:, 1]
print(second_col)
data_with_split = []
each_docs = []
stop_words_split_final = []



#data cleaning method
def data_preprocessing(string):
    text = re.sub('\,|\@|\-|\"|\'| \)|\(|\)| \{| \}| \[| \]|!|‘|’|“|”| \:-|\?|।|/|\—', '', string)
    return text
#hello

def split_doc():
    for data in first_col:
        return_string = data_preprocessing(data)
        each_docs = return_string.split()
        data_with_split.append(each_docs)
    return data_with_split  # it returns arr of each docs with spleted words


word_lists = []
word_lists = split_doc()
length_of_docs = len(word_lists)


#####################################
# print(word_lists)

def individual_words():
    my_set = set.union(*map(set, word_lists))  # seperate each individual words from data to make matrix
    return my_set


def set_to_list():
    my_set_list = individual_words()
    convert_into_list = list(my_set_list)
    return convert_into_list


individual_word_list = set_to_list()


def count_occurence_of_word_vocab():
    my_set = individual_words()
    doc = {}
    word_dict = {}
    for i in range(len(word_lists)):
        for word in word_lists[i]:
            word_dict = dict.fromkeys(my_set, 0)

    for count_word_value in word_lists:
        for word in count_word_value:
            if word in word_dict:
                word_dict[word] += 1
    return word_dict


word_dict = count_occurence_of_word_vocab()



length_word_dict = len(word_dict)

# def vectorizer_docs(line):
#     vectorizer_docs = []
#     matrix_doc = []
#     for word in individual_word_list:
#         if word in line:
#             vectorizer_docs.append(1)
#         else:
#             vectorizer_docs.append(0)
#     return vectorizer_docs
#     vectorizer_docs.clear()


# doc_vec1 = []
# doc_vec2 = []
# for line in word_lists:
#     doc_vec1 = vectorizer_docs(line)
#     doc_vec2.append(doc_vec1)
# print(doc_vec2)

dict1={}
def computeTf(docs_list):
    tf_vec = []
    tf_each_doc_vec = []
    doc_word_count = len(docs_list)
    count_each_word = Counter(docs_list)  # counter count the word in list how many times it occure
    for each_word,val in word_dict.items():
        if each_word in docs_list:
            count = count_each_word.get(each_word)
            tf_vec.append(count / float(doc_word_count))
        else:
            tf_vec.append(0)
    tf_each_doc_vec.append(tf_vec)
    return tf_each_doc_vec


tf = []
tf_vec = []
for each_line in word_lists:
    tf = computeTf(each_line)
    tf_vec += tf
    #######################################3
# print("Term Frequency")
# print(tf_vec)

countIdfforwordvalue = {}
word_dict = count_occurence_of_word_vocab()
my_set = individual_words()


def computeCountDict(word_dict, word_lists):
    countIdfforword = {}
    for i in range(1, len(my_set)):
        countIdfforword = dict.fromkeys(my_set, 0)
    for word, value in word_dict.items():
        for each_line_item in word_lists:
            if word in each_line_item:
                countIdfforword[word] += 1
        # else:
        # 	countIdfforword[word] = 1
    return countIdfforword


countIdfforwordvalue = computeCountDict(word_dict, word_lists)

#  #  return no of doc conatin word for each word
#  def doc_contain_word(parameter_word):
# 		word_value_in_each_doc = countIdfforwordvalue.get(parameter_word)
# 		return word_value_in_each_doc


def computeIdf(docs_list):
    idf_vec = []
    idf_each_doc_vec = []
    for each_word,val in word_dict.items():
        if each_word in docs_list:
            word_value_in_each_doc = countIdfforwordvalue.get(each_word)
            idf_vec.append(mth.log(length_of_docs / word_value_in_each_doc))
        else:
            idf_vec.append(0)
    idf_each_doc_vec.append(idf_vec)
    return idf_each_doc_vec


idf = []
idf_vec = []
for each_line in word_lists:
    idf = computeIdf(each_line)
    idf_vec += idf
################################################
# print("idf vector")
# print(idf_vec)

TfIdf_vec = []
def computeTfIdf(Tfvec, Idfvec):
    TfIdf_vec = [a * b for a, b in zip(Tfvec, Idfvec)]
    return TfIdf_vec





tfidf_vector_for_each_docs = []
tfidf_vector_collection = []
for tf_list, idf_list in zip(tf_vec, idf_vec):  # zip helps to iteration two different collection samultaneously
    tfidf_vector_for_each_docs = computeTfIdf(tf_list, idf_list)
    tfidf_vector_collection.append(tfidf_vector_for_each_docs)
# make model with sk-learn

features = np.array(tfidf_vector_collection)
print(type(second_col))
labels_string = np.array(second_col)
# print(labels_string)
labels_list = [int(int_labels) for int_labels in labels_string]
labels = np.array(labels_list)


array_length = len(features)
# print(type(features))

features_taken_len = int(array_length * 80 / 100)  # 80% of data make for train 20% remening data for testing
# feature_train_test = np.split(features, [features_taken_len,features_taken_len*2, features_taken_len*3,features_taken_len*4 ])
# label_train_test = np.split(labels, [features_taken_len,features_taken_len*2, features_taken_len*3,features_taken_len*4 ])
# feature_array_train, feature_array_test, labels_array_train, labels_array_test = train_test_split( features, labels, test_size=0.33, random_state=42)

feature_array_train = features[:features_taken_len]  # 80% of data make for train 20% remening data for testing
labels_array_train = labels[:features_taken_len]
feature_array_test = features[features_taken_len:]  # 80% of data make for train 20% remening data for testing
labels_array_test =  labels[features_taken_len:]

#############################cross_validation_split
# def cross_validation_split(dataset, labels, folds):
# 	dataset_split_features = list()
# 	dataset_split_labels = list()
# 	dataset_features = dataset
# 	dataset_labels = labels
# 	fold_size = int(len(dataset) / folds)
# 	for i in range(folds):
# 		fold_features = list()
# 		fold_labels = list()
# 		while len(fold_features) < fold_size:
# 			index_features = randrange(len(dataset_features))
# 			fold_features.append(dataset_features.pop(index_features))
# 			index_labels = randrange(len(dataset_labels))
# 			fold_labels.append(dataset_labels.pop(index_labels))
# 		dataset_split_features.append(fold_features)
# 		dataset_split_labels.append(fold_labels)

# 	return dataset_split_features,dataset_split_labels
# # Naive byes classifier sklearn
# #train model

# features,labels = cross_validation_split(word_lists,labels_list,4)


naive_byes = GaussianNB()  # create  object  from  GaussianNb  class
TrainData = naive_byes.fit(feature_array_train, labels_array_train)

classifier_data = open("classify_data.pickle", "ab")
pickle.dump(TrainData, classifier_data)
classifier_data.close()
	


with open('classify_data.pickle', 'rb') as pickle_saved_data:
    unpickled_data = pickle.load(pickle_saved_data)



predict_result = unpickled_data.predict(feature_array_test)


#calculate precision recall and f measure
# print(final_labels.shape)
# print(predict.shape)


precision = metrics.precision_score(predict_result,labels_array_test ,average='weighted')
print("precision")
print(precision)


recall = metrics.recall_score(predict_result,labels_array_test,average='weighted')
print("recall")
print(recall)



f_score = 2*(precision*recall)/(precision+recall)
print("f_score")
print(f_score)


#
####################for TFIDF of input data we need dict in bellow format####################

# {doc1:{word1:count1},{word2:count2}}

# >>> d = {}
# >>> d['dict1'] = {}
# >>> d['dict1']['innerkey'] = 'value'
# >>> d
# {'dict1': {'innerkey': 'value'}}
dict_for_idf = {}

def count_each_word_each_doc():
    i = 1
    for each_line_for_idf in word_lists:
        dict_for_idf[i] = {}
        count_each_word_for_idf = Counter(each_line_for_idf) 
        for each_word_of_line_for_idf in each_line_for_idf:
            count_for_idf = count_each_word_for_idf.get(each_word_of_line_for_idf)
            dict_for_idf[i][each_word_of_line_for_idf] = count_for_idf
        i = i+1 
    return dict_for_idf
dict_for_idf_final = count_each_word_each_doc()





print("***************************************")
input_data = input("Type Text For Prediction ")
# to make predict input value similar as our training sample we use reshape


def input_tf(input_data):

    each_input_word = []
# change into array of word
    each_input_word = input_data.split()

#input data from user
    length_input_data = len(each_input_word)

    count_each_inputword = Counter(each_input_word)
    input_data_tfvec = []
# tf_each_input_word = []
#TF computation of input data

    for word,val in word_dict.items():#where word_dict is all the word collection from data set
        if word in each_input_word:
            count = count_each_inputword.get(word)
            input_data_tfvec.append(count / float(length_input_data))
        else:
            input_data_tfvec.append(0)
    return input_data_tfvec

tf_value_of_input_data = input_tf(input_data)


def input_idf(input_data):
    idf_vec_input_data = []
    idf_each_doc_vec_input_data = []
    for each_word_input_data,val in word_dict.items():
        if each_word_input_data in input_data:
            word_value_in_each_doc_input_data = countIdfforwordvalue.get(each_word_input_data)
            idf_each_doc_vec_input_data.append(mth.log(length_of_docs / word_value_in_each_doc_input_data))
        else:
            idf_each_doc_vec_input_data.append(0)
    return idf_each_doc_vec_input_data


idf_value_of_input_data = input_idf(input_data)

def computeTfIdf_input(tf_value_of_input_data, idf_value_of_input_data):
    tfidf_input_vec = [a * b for a, b in zip(tf_value_of_input_data, idf_value_of_input_data)]
    return tfidf_input_vec

TfIdf_value_of_input_data = computeTfIdf_input(tf_value_of_input_data,idf_value_of_input_data)

value_for_predict = np.array(TfIdf_value_of_input_data).reshape(1,-1)
predict = unpickled_data.predict(value_for_predict)
print(predict)
