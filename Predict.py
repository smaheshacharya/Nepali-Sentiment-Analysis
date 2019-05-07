from CodeCompile import *
#test model
naive_byes_test = GaussianNB()
TestData = naive_byes_test.partial_fit(feature_array_test, labels_array_test, classes=np.unique(labels_array_test))


with open('classify_data.pickle', 'rb') as pickle_saved_data:
    unpickled_data = pickle.load(pickle_saved_data)



predict_result = unpickled_data.predict(feature_array_test)
print("predict")
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
# #prediction after taking input from user
print("***************************************")
input_data = input("Type Text For Prediction ")
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
# to make predict input value similar as our training sample we use reshape
value_for_predict = np.array(input_data_tfvec).reshape(1,-1)
predict = unpickled_data.predict(value_for_predict)
print(predict)
