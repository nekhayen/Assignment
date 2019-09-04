import nltk
import pandas as pd
import numpy as np
import string
from nltk.corpus import reuters
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import LinearSVC
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, classification_report
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score



def tokenize(corpus):
    cachedStopWords = stopwords.words("english")
    min_length = 3
    words = map(lambda word: word.lower(), word_tokenize(corpus));
    words = [word for word in words
                  if word not in cachedStopWords]
    tokens =(list(map(lambda token: PorterStemmer().stem(token),
                  words)));
    p = re.compile('[a-zA-Z]+');
    filtered_tokens =list(filter(lambda token:
                  p.match(token) and len(token)>=min_length,
         tokens));
    return filtered_tokens



documents = reuters.fileids()

train_docs_id = list(filter(lambda doc: doc.startswith("train"), documents))
test_docs_id = list(filter(lambda doc: doc.startswith("test"), documents))
print(train_docs_id, file=open("train_docs_id.txt", "a"))
    
train_docs = [reuters.raw(doc_id) for doc_id in train_docs_id]
test_docs = [reuters.raw(doc_id) for doc_id in test_docs_id]
print(train_docs, file=open("train_docs.txt", "a"))


    # TfidfVectorizer uses the output of Preprocessing of def tokenize function
vectorizer = TfidfVectorizer(tokenizer=tokenize)
    
    # Learn and transform train documents
vectorised_train_documents = vectorizer.fit_transform(train_docs)
    
vectorised_test_documents = vectorizer.transform(test_docs)
    
   # Transform multilabel labels to binary system
mlb = MultiLabelBinarizer()
train_labels = mlb.fit_transform([reuters.categories(doc_id) for doc_id in train_docs_id]) 
test_labels = mlb.transform([reuters.categories(doc_id) for doc_id in test_docs_id])




svc_= OneVsRestClassifier(svm.SVC())
params =  {'estimator__C': [1, 10, 100, 1000], 'estimator__gamma': [10, 1, 0.1, 0.01, 0.001, 0.0001], 'estimator__kernel': ['rbf']}
classifier = GridSearchCV(svc_, param_grid=params,verbose=1, n_jobs=-1)
classifier.fit(vectorised_train_documents, train_labels)

print("Best Hyper Parameters from GridsearchCV:\n",classifier.best_params_, classifier.best_estimator_)


predictions = classifier.predict(vectorised_test_documents)

precision = precision_score(test_labels, predictions, average='micro')
recall = recall_score(test_labels, predictions, average='micro')
f1 = f1_score(test_labels, predictions, average='micro')
print("Micro-average quality numbers")
print("Precision: {:.4f}, Recall: {:.4f}, F1-measure: {:.4f}".format(precision, recall, f1))
precision = precision_score(test_labels, predictions, average='macro')
recall = recall_score(test_labels, predictions, average='macro')
f1 = f1_score(test_labels, predictions, average='macro')
print("Macro-average quality numbers")
print("Precision: {:.4f}, Recall: {:.4f}, F1-measure: {:.4f}".format(precision, recall, f1))
#the percentage of samples that have all their labels classified correctly
print("Subset accuracy: %1.4f" % (accuracy_score(test_labels, predictions)))
#precision, recall and f1 for every of the 90 labels
print(classification_report(test_labels, predictions))
    



    
