from SP_coTrain import SP_coTrain
from Base_Model import Base
from Co_Training import Co_Training_Classifier

from math import ceil
import numpy as np
import pandas as pd
from copy import deepcopy

from sklearn.preprocessing import LabelEncoder, OneHotEncoder


def predict_proba(model, X_unlabeled):
    return model.predict_proba(X_unlabeled)


def extract_v_vector(probs, y, add_rate):
    v_vector = np.zeros(probs.shape[0])
    classes = np.unique(y)
    count_per_class = [sum(y == cls) for cls in classes]
    pred_y = np.argmax(probs, axis=1)
    for cls in range(len(classes)):
        indices = np.where(pred_y == cls)[0]
        cls_score = probs[indices, cls]
        idx_sort = np.argsort(cls_score)
        add_num = min(int(np.ceil(count_per_class[cls] * add_rate)), indices.shape[0])
        v_vector[indices[idx_sort[-add_num:]]] = 1
    return v_vector.astype('bool')


def extract_new_train(v_current, X_labeled, y, X_unlabeled, pred_y):
    to_add_X = []
    to_add_y = []
    for i, flag in enumerate(v_current):
        if flag:
            to_add_X.append(X_unlabeled[i])
            to_add_y.append(pred_y[i])
    new_X = np.append(X_labeled,np.array(to_add_X),axis=0)
    new_y = np.append(y , np.array(to_add_y),axis=0)

    return new_X, new_y


def train_model(base_model, X, y):
    model = deepcopy(base_model)
    model.fit(X, y)
    return model


# calculated G by a given K when the threshold sets the fraction of unlabeled data to be labeled
def calc_G_and_k(unlabeled_size):
    threshold = 0.5
    num_to_label = ceil((1 - threshold) * unlabeled_size)
    K = 10
    G = ceil(num_to_label / K)
    return K, G


# return the model instance by given model name
def get_model(base_line, base_model, X_unlabeled):
    model = None

    if base_line == "spaco":
        model = SP_coTrain([base_model, base_model], num_of_iters=10, add_rate=0.2)
    elif base_line == "co":
        K, G = calc_G_and_k(X_unlabeled.shape[0])
        model = Co_Training_Classifier(base_model=base_model, K=K, G=G)
    elif base_line == "base":
        model = Base([deepcopy(base_model), deepcopy(base_model)])

    return model


# Split the data first to train and test and then split the train to labeled and unlabeled
def split_data(X, y, train_test_split=0.7, labeled_unlabeled_split=0.6):
    offset = int(X.shape[0] * train_test_split)
    X_train, y_train = X[:offset], y[:offset]
    X_test, y_test = X[offset:], y[offset:]

    offset = int(X_train.shape[0] * labeled_unlabeled_split)
    X_labeled, y_labeled = X_train[:offset], y[:offset]
    X_unlabeled = X_train[offset:]

    return X_labeled, X_unlabeled, y_labeled, X_test, y_test


# Extracts the raw data and splits to train and test. sets the features for the different views by the given dataset
def extract_data(file_path, file_name):
    X = None
    view1_features = None
    view2_features = None
    y = None
    if (file_name == "nba_train.csv"):
        data = np.genfromtxt(file_path, delimiter=',')
        X = data[1:, 1:-1]
        X = X.astype(np.float32)
        y = data[1:, -1]

        view1_features = [0, 1, 2]
        view2_features = [i for i in range(len(X[0])) if i not in view1_features]

    if (file_name == "news_train.csv"):
        data = np.genfromtxt(file_path, delimiter=',')
        X = data[1:, 1:-1]
        X = X.astype(np.float32)
        y = data[1:, -1]

        X = X[:int(0.333 * len(X)), :]
        y = y[:int(0.333 * len(y))]

        median = np.median(y)

        y[y > median] = 0
        y[y != 0] = 1

        view1_features = [0, 1, 2, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
        view2_features = [i for i in range(len(X[0])) if i not in view1_features]

    if (file_name == "brest_cancer_train.csv"):
        data = np.genfromtxt(file_path, delimiter=',')
        X = data[:, 1:-1]
        X = X.astype(np.float32)
        y = data[:, -1]
        y[y == 2] = 0
        y[y == 4] = 1
        view1_features = [0, 1, 2, 3]
        view2_features = [i for i in range(len(X[0])) if i not in view1_features]

    if (file_name == "mushrooms_train.csv"):
        data = pd.read_csv(file_path)
        encoder = LabelEncoder()
        for column_name, column_type in data.dtypes.iteritems():
            data[column_name] = encoder.fit_transform(data[column_name])
        data = np.array(data)
        X = data[1:, 1:]
        y = data[1:, 0]

        view1_features = [i for i in range(0, len(X[0]), 2)]
        view2_features = [i for i in range(len(X[0])) if i not in view1_features]

    if (file_name == "income_train.csv"):
        data = pd.read_csv(file_path)
        encoder = LabelEncoder()
        onehotencoder = OneHotEncoder()
        for column_name, column_type in data.dtypes.iteritems():
            labels = data[column_name].unique()
            new_col = encoder.fit_transform(data[column_name])
            if (column_name not in ['age', 'fnlwgt', 'occupation', 'education', 'capital.gain', 'capital.loss',
                                    'education.num', 'hours.per.week', 'income']):
                new_col = onehotencoder.fit_transform(new_col.reshape(-1, 1))
                labels[...] = column_name + '_' + labels[...]
                new_col = pd.DataFrame(new_col.toarray(), columns=labels)
                data = pd.concat([new_col, data], axis=1)
                del data[column_name]
            else:
                data[column_name] = new_col

        y = np.array(data['income'])
        y = y[:int((len(data)) * 0.25)]
        del data['income']
        data = np.array(data)

        X = data[:int((len(data)) * 0.25), :]
        view1_features = [i for i in range(0, len(X[0]), 2)]
        view2_features = [i for i in range(len(X[0])) if i not in view1_features]

    X = np.nan_to_num(X)
    return X, y, view1_features, view2_features
