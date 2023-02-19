import pickle
#Group: Runtime 
#Efekan Kavalci & Erkin Simsek
import sys
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.naive_bayes import MultinomialNB
from gensim.parsing.porter import PorterStemmer
from gensim.utils import simple_preprocess
import warnings
warnings.filterwarnings('ignore')

def get_testdataset(path):
    X_test, y_test = ([], [])
    for r, d, f in os.walk(path):
        f = sorted(f, key=lambda x: int(x.split('_')[0]))
        for filename in f:
            if '.txt' in filename:
                f = open(path + '/' + filename, 'r', encoding='ascii', errors='ignore')                
                text = f.read()
                text = text.lower()
 
                X_test.append(text)
                y_test.append(filename[-5])

    return (X_test, y_test)


if __name__ == '__main__':
    args = sys.argv[1:]
    X_test, y_test = get_testdataset(args[1])
    pkl=args[0]
    with open(pkl, 'rb') as file:
        cv, bestfeatures, model = pickle.load(file)
        
    porter_stemmer=PorterStemmer()
    X_test = [simple_preprocess(line, deacc=True) for line in X_test] 
    X_test=[[porter_stemmer.stem(word) for word in review] for review in X_test]

    for i in range(len(X_test)):
        X_test[i]=' '.join(X_test[i])
        
    X_test_vec = cv.transform(X_test)
    X_test_f = bestfeatures.transform(X_test_vec)
    test_predicted = model.predict(X_test_f)
    print(''.join(test_predicted))


