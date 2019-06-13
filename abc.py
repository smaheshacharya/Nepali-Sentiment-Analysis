import CodeCompile.py
with open('classify_data.pickle', 'rb') as pickle_saved_data:
    unpickled_data = pickle.load(pickle_saved_data)



predict_result = unpickled_data.predict(feature_array_test)
print(predict_result)


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
