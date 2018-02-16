import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


def split_data(X, y, train_test_split=0.7, labeled_unlabeled_split=0.6):
    offset = int(X.shape[0] * train_test_split)
    X_train, y_train = X[:offset], y[:offset]
    X_test, y_test = X[offset:], y[offset:]

    offset = int(X_train.shape[0] * labeled_unlabeled_split)
    X_labeled, y_labeled = X_train[:offset], y[:offset]
    X_unlabeled = X_train[offset:]

    return X_labeled, X_unlabeled, y_labeled, X_test, y_test

def extract_data(file_path, file_name):
    if(file_name=="nba_train.csv"):
        data = np.genfromtxt(file_path, delimiter=',')
        X=data[1:,1:-1]
        X = X.astype(np.float32)
        y=data[1:,-1]

        view1_features=[0,1,2]
        view2_features=[i for i in range(len(X[0])) if i not in view1_features]

    if(file_name=="news_train.csv"):
        data = np.genfromtxt(file_path, delimiter=',')
        X=data[1:,1:-1]
        X = X.astype(np.float32)
        y=data[1:,-1]

        median=np.median(y)
        y[y>median] = 0
        y[y != 0] = 1

        view1_features=[0,1,2,3,4,5,6,7,10,18,19,20,21,22,23,24,25,26,27,28,29]
        view2_features = [i for i in range(len(X[0])) if i not in view1_features]

    if(file_name=="brest_cancer_train.csv"):
        data = np.genfromtxt(file_path, delimiter=',')
        X=data[:,1:-1]
        X = X.astype(np.float32)
        y=data[:,-1]
        y[y == 2]=0
        y[y == 4]=1
        view1_features=[0,1,2,3]
        view2_features = [i for i in range(len(X[0])) if i not in view1_features]

    if(file_name=="mushrooms_train.csv"):
        data = pd.read_csv(file_path)
        encoder = LabelEncoder()
        for column_name, column_type in data.dtypes.iteritems():
            data[column_name]=encoder.fit_transform(data[column_name])
        data=np.array(data)
        X=data[1:,1:]
        y=data[1:,0]

        view1_features=[i for i in range(0,len(X[0]),2)]
        view2_features = [i for i in range(len(X[0])) if i not in view1_features]

    if file_name == "income_train.csv":
        data = pd.read_csv(file_path)
        encoder = LabelEncoder()
        onehotencoder = OneHotEncoder()
        for column_name, column_type in data.dtypes.iteritems():
            labels=data[column_name].unique()
            new_col = encoder.fit_transform(data[column_name])
            if(column_name not in ['age', 'fnlwgt','occupation','education','capital.gain','capital.loss','education.num','hours.per.week','income']):
                new_col = onehotencoder.fit_transform(new_col.reshape(-1,1))
                labels[...]=column_name+'_'+labels[...]
                new_col = pd.DataFrame(new_col.toarray(),columns=labels)
                data= pd.concat([new_col,data],axis=1)
                del data[column_name]
            else:
                data[column_name]=new_col

        y=np.array(data['income'])
        y=y[:int((len(data))*0.25)]
        del data['income']
        data = np.array(data)

        X = data[:int((len(data))*0.25), :]
        view1_features = [i for i in range(0, len(X[0]), 2)]
        view2_features = [i for i in range(len(X[0])) if i not in view1_features]

    if file_name == "student-mat.csv" or file_name == "student-por.csv":
        data = pd.read_csv(file_path)
        encoder = LabelEncoder()

        # to classification
        for column_name, column_type in data.dtypes.iteritems():
            if(column_type==object):
                data[column_name]=encoder.fit_transform(data[column_name])

        data=np.array(data)

        X=data[:,:30]
        y=data[:,32]
        y[y<=10]=0
        y[y>10]=1

        view1_features=[1,2,3,4,5,6,7,8,9,11,17,18,19,22,23,28]
        view2_features = [i for i in range(30) if i not in view1_features]


        view1_features=[i for i in range(0,len(X[0]),2)]
        view2_features = [i for i in range(len(X[0])) if i not in view1_features]

    if file_name == "3_10principal_components.csv":
        data = pd.read_csv(file_path)

        data=np.array(data)

        X=data[1:,:-1]
        y=data[1:,-1]

        view1_features=[1,2,3,4,5]
        view2_features = [i for i in range(len(X[0])) if i not in view1_features]

    if file_name == "page-block.csv":
        data = pd.read_csv(file_path)

        data=np.array(data)

        X=data[:,:-1]
        y=data[:,-1]

        y[y==" positive"]=1
        y[y!=1]=0

        view1_features=[0,1,4,5]
        view2_features = [i for i in range(len(X[0])) if i not in view1_features]

    if file_name == "yeast1.csv":
        data = pd.read_csv(file_path)

        data=np.array(data)

        X=data[:,:-1]
        y=data[:,-1]

        y[y==" positive"]=1
        y[y!=1]=0

        view1_features=[0,1,2,3]
        view2_features = [i for i in range(len(X[0])) if i not in view1_features]

    X=np.nan_to_num(X)
    return X, y, view1_features, view2_features