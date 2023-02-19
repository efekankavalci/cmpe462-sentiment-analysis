import pickle
import os
import sys

def get_dataset(path):
    X_train, y_train = ([], [])
    for r, d, f in os.walk(path):
        f = sorted(f,key=lambda x: int(x.split('_')[0]))
        for filename in f:
            if '.txt' in filename:
                f = open(path+'/' + filename, 'r', encoding='ascii', errors='ignore')
                #print(filename)
                text = f.read()
                text = text.lower()
                X_train.append(text)
                y_train.append(filename[-5])
                

    return (X_train, y_train)

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

args = sys.argv[1:]
path1 , path2 = args[0], args[1]
X_train, y_train = get_dataset(path1)
X_test, y_test = get_testdataset(path2)


with open('dataset.pkl', 'wb') as f:
    pickle.dump((X_train, y_train, X_test, y_test), f)


