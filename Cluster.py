from nltk.corpus import reuters
from bert_serving.client import BertClient
from sklearn.cluster import KMeans
import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re

#input Reuters dataset as a string text
text=''
for index, i in  enumerate(reuters.fileids()):
    text += reuters.raw(fileids=[i])

#splitting strings into a list of strings for BERT    
text1=text.split('.')
#delete all empty strings
filtered1 = [x for x in text1 if len(x.strip()) > 0]
#delete all \n   
filtered2 = [i.replace('\n','') for i in filtered1]
    
#output preprocessed corpus into a text file  
print(filtered2, file=open("corpus.txt", "a"))

# initialization of BERT client in cmd: bert-serving-start -model_dir D:\uncased_L-12_H-768_A-12 –cpu -verbose - pooling_strategy = REDUCE_MEAN -max_seq_len=100
bc = BertClient()
BERT_embedding = bc.encode(filtered2)


#Kmeans K range 4-20
#Sum of squared distances of samples to their closest cluster center for K=4-20
Sum_of_squared_distances = []
K = range(4,20)
for k in K:
    km = KMeans(n_clusters=k)
    km.fit(BERT_embedding)
    Sum_of_squared_distances.append(km.inertia_)

print (Sum_of_squared_distances)    
clusters = km.labels_.tolist()
print(clusters)
    
#visualising Elbow method for optimal K
plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k')
plt.show()
