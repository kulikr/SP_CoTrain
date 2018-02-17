import SP_coTrain
from Base_Model import Base
from Co_Training import Co_Training_Classifier

from math import ceil
import numpy as np
import pandas as pd
from copy import deepcopy



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


def extract_v_vector_ours(probs, y, add_rate):
    v_vector = np.zeros(probs.shape[0])
    max_probs=np.max(probs,axis=1)
    indices = np.where(max_probs>(1 - add_rate))[0]
    v_vector[indices]=1
    return v_vector.astype('bool')


def extract_new_train(v_current, X_labeled, y, X_unlabeled, pred_y):
    to_add_X = []
    to_add_y = []
    for i, flag in enumerate(v_current):
        if flag:
            to_add_X.append(X_unlabeled[i])
            to_add_y.append(pred_y[i])
    if len(to_add_X)==0:
        return X_labeled,y
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

def get_add_rate(labeled_rate,numOfIters):
    return (1-labeled_rate)/((numOfIters+1)*labeled_rate)

# Return the model instance by given model name
def get_model(base_line, base_model, X_unlabeled,labeled_rate):
    model = None

    if base_line == "spaco":
        add_rate=get_add_rate(labeled_rate,10)
        model = SP_coTrain.SP_coTrain([base_model, base_model], num_of_iters=10, add_rate=0.2)
    elif base_line == "co":
        K, G = calc_G_and_k(X_unlabeled.shape[0])
        model = Co_Training_Classifier(base_model=base_model, K=K, G=G)
    elif base_line == "base":
        model = Base([deepcopy(base_model), deepcopy(base_model)])
    elif base_line == "reg_base":
        model = deepcopy(base_model)

    return model


