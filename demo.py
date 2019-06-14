import pandas as pd
import math
import numpy as np
text = "के तपाईं DFLP को बारेमा अलिकति बन्न सक्नुहुन्छ"
text1 = "ब्रिज मेकअप गर्नको लागि सबै सामान पाइन्छ hello"


bow1 = text.split()
bow2 = text1.split()
print("bow1")
print(bow1)

#
# def get_stopwords():
#     stopwords = ['अभियोग','छ','लागेका','अंगुर']
#     return stopwords
# def remove_stopwords(texts):
# 	result = []
# 	stopwords = get_stopwords()
# 	for word in texts:
# 		if word in stopwords:
# 		 texts.remove(word)
# 	return texts
#
# bow1 += remove_stopwords(bow1)
# bow2 += remove_stopwords(bow2)

text3 = set(bow1).union(set(bow2))
text1_dict = dict.fromkeys(text3,0)
text2_dict = dict.fromkeys(text3,0)

for word in bow1:
    text1_dict[word] += 1
for word in bow2:
    text2_dict[word] += 1
print(text1_dict)
print('\n')
print(text2_dict)
print()

print("here")
print(pd.DataFrame([text1_dict,text2_dict]))

print("finish")
def computeTf(worddict,bow):
	tfDict = {}
	bowCount = len(bow)
	for word , count in worddict.items():
		tfDict[word] = count/float(bowCount)

	return tfDict
tfBowa = computeTf(text1_dict,bow1)
tfBowb = computeTf(text2_dict,bow2)

print("tf")
print(tfBowa)
print(tfBowb)
# print(pd.DataFrame([tfBowa,tfBowb]))

def computeIdf(docList):
	idfDict = {}
	N = len(docList)
	idfDict = dict.fromkeys(docList[0].keys(),0)

	for doc in docList:
		for word,val in doc.items():
			if val >0:
				idfDict[word] += 1
	for word, val in idfDict.items():
		idfDict[word] = math.log(N/float(val))

	return idfDict


idf = computeIdf(text1_dict)
print("idf")
print(idf)

def computecompute_TfIdf(tfbow,idf):
	compute_TfIdf = {}
	for word , val in tfbow.items():
		compute_TfIdf[word] =val*idf[word]
	return compute_TfIdf

compute_TfIdf1 = computecompute_TfIdf(tfBowa,idf)
compute_TfIdf2 = computecompute_TfIdf(tfBowb,idf)
print(pd.DataFrame([compute_TfIdf1,compute_TfIdf2]))
