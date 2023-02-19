import pickle
from gensim.parsing.preprocessing import remove_stopwords
from gensim.utils import simple_preprocess
from gensim.parsing.porter import PorterStemmer
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import numpy as np
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.metrics import classification_report, accuracy_score
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
import gensim.downloader as api


def load_dataset(_file):

    with open(_file, 'rb') as file:
        X, y, z, t = pickle.load(file)

    return X, y, z, t

#Gets the pre-read dataset from pkl file to save time for trials
X_train, y_train, X_test, y_test = load_dataset('dataset.pkl')

for i in range(len(X_train)):
    X_train[i]=remove_stopwords(X_train[i])
    
#porter_stemmer=PorterStemmer()

reviews = [simple_preprocess(line, deacc=True) for line in X_train] # Word tokenizer
#reviews=[[porter_stemmer.stem(word) for word in review] for review in reviews]

for i in range(len(X_test)):
    X_test[i]=remove_stopwords(X_test[i])
    
test_reviews = [simple_preprocess(line, deacc=True) for line in X_test] 
#test_reviews=[[porter_stemmer.stem(word) for word in review] for review in test_reviews]



"""
https://github.com/RaRe-Technologies/gensim-data
Used models from api:
    glove-twitter-100
    glove-wiki-gigaword-50
    
Used corpus from api:
    text8
"""

#Loading pre-trained word embedding models from gensim downloader 
word_vectors = api.load("glove-wiki-gigaword-50")
word_vectors.save("glove-wiki-gigaword-50.model")
#word_vectors = KeyedVectors.load("glove-wiki-gigaword-50.model")
size = 50

""" 
Generation of a word2vec model from the text8 data of gensim
# Skip-gram model (sg = 1)
window = 5
min_count = 1
workers = 3
sg = 1
size=100
dataset = api.load("text8")
model = Word2Vec(dataset, vector_size=size, sg = sg)
model.save("text8.model")
word_vectors=model.wv
"""
print("Vector creation started...")
review_vectors=[]
for review in reviews:
    total=np.zeros((size),dtype='float32')
    word_count=0
    for word in review:
        if word in word_vectors.index_to_key :
            total =total+ word_vectors[word]
            word_count += 1
    if word_count != 0: 
        review_vectors.append(total/word_count)
    else:
        review_vectors.append(total)

print("Fitting Start...")
clf=svm.SVC()
clf.fit(review_vectors,y_train)
print("Fitting End...")


predictions_word2vec = clf.predict(review_vectors)
print("Model: ",str(clf))
print("IN SAMPLE:")
print(classification_report(y_train,predictions_word2vec))

test_review_vectors=[]
for review in test_reviews:
    total=np.zeros((size),dtype='float32')
    word_count=0
    for word in review:
        if word in word_vectors.index_to_key :
            total =total+ word_vectors[word]
            word_count += 1
    if word_count != 0: 
        test_review_vectors.append(total/word_count)
    else:
        test_review_vectors.append(total)
        
test_predictions_word2vec = clf.predict(test_review_vectors)
print("\nTEST:")
print(classification_report(y_test,test_predictions_word2vec))
print("Test Accuracy: ", accuracy_score(y_test,test_predictions_word2vec))
