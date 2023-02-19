import os
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import Perceptron
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.metrics import classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn import svm
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, chi2
import pickle
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn import metrics
from sklearn.model_selection import KFold
from gensim.parsing.porter import PorterStemmer
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import remove_stopwords
import re


def load_dataset(_file):

    with open(_file, 'rb') as file:
        X, y, z, t = pickle.load(file)

    return X, y, z, t

#Gets the pre-read dataset from pkl file to save time for trials
X_train, y_train, X_test, y_test = load_dataset('dataset.pkl')


#for i in range(len(X_train)):
#    X_train[i]=remove_stopwords(X_train[i])
    
porter_stemmer=PorterStemmer()

X_train = [simple_preprocess(line, deacc=True) for line in X_train] 
X_train=[[porter_stemmer.stem(word) for word in review] for review in X_train]

#for i in range(len(X_test)):
#   X_test[i]=remove_stopwords(X_test[i])
    
X_test = [simple_preprocess(line, deacc=True) for line in X_test] 
X_test=[[porter_stemmer.stem(word) for word in review] for review in X_test]

for i in range(len(X_train)):
    X_train[i]=' '.join(X_train[i])
    
for i in range(len(X_test)):
    X_test[i]=' '.join(X_test[i])




cv = TfidfVectorizer(min_df=10, max_df=0.5,ngram_range=(1, 2))
X_vect = cv.fit_transform(X_train) #Learns vocabulary and vectorizes
X_test_vec = cv.transform(X_test) #Transforms the test set 
print('LEN:', len(cv.get_feature_names())) #Control for the total features size 

bestfeatures = SelectKBest(chi2, k=1400) #Feature selection from chi2 stats
bestfeatures.fit(X_vect, y_train)

dfscores = pd.DataFrame(bestfeatures.scores_)  #A dataframe to observe high scored features
dfcolumns = pd.DataFrame(cv.get_feature_names())

featurescores = pd.concat([dfcolumns, dfscores], axis=1)
featurescores.columns = ['Specs', 'Score']
print(featurescores.nlargest(100, 'Score'))

X_vect = bestfeatures.transform(X_vect)
X_test_vec= bestfeatures.transform(X_test_vec)

model = LogisticRegression()
model.fit(X_vect, y_train)

test_predicted = model.predict(X_test_vec)
print('OUT TRAINING SET:')
print("Test Accuracy:", accuracy_score(y_test, test_predicted))
print(classification_report(y_test, test_predicted))
 
in_predict = model.predict(X_vect)
print('IN TRAINING SET:')
print(classification_report(y_train, in_predict))

with open('step2_model_Runtime.pkl', 'wb') as sf:
    pickle.dump((cv, bestfeatures, model), sf)



#-----------------Cross Validation--------------------
vectorizer = TfidfVectorizer(min_df=10, max_df=0.5,ngram_range=(1, 2))

kf = KFold(n_splits=5)

clf=model
crvalscores=[]

whole_X=np.concatenate((X_train, X_test),axis=0)
whole_y=np.concatenate((y_train, y_test),axis=0)

for train_index, test_index in kf.split(whole_X): 
    tr_start=train_index[0]
    X_tr, X_tst, y_tr, y_tst=[], [], [], []

    for index in train_index:
        X_tr.append(whole_X[index])
        y_tr.append(whole_y[index])
    for index in test_index:
        X_tst.append(whole_X[index])
        y_tst.append(whole_y[index])

    X_vect = vectorizer.fit_transform(X_tr)
    print('LEN:', len(vectorizer.get_feature_names()))
    X_test_vec = vectorizer.transform(X_tst)
    
    bestfeatures.fit(X_vect, y_tr)
    X_vect = bestfeatures.transform(X_vect)
    X_test_vec = bestfeatures.transform(X_test_vec)
    
    clf.fit(X_vect,y_tr)
    predicted=clf.predict(X_test_vec)
    
    crvalscores.append(metrics.accuracy_score(y_tst, predicted))
    #print(classification_report(y_tst, predicted))


print(np.mean(crvalscores),' ', np.std(crvalscores))
    
    
#----------------------------------------------------

#Visualisation of Results
cm= confusion_matrix(y_test, test_predicted ,labels=model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=model.classes_)
disp.plot() 
#scores = cross_val_score(model, X_train_f, y_train, cv=100)
#print('Mean Cross-validation Accuracy:', format(np.mean(scores)), 'Std:', np.std(scores))

