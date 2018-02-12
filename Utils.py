import numpy as np
from copy import deepcopy

def predict_proba(model,X_unlabeled):
    return model.predict_proba(X_unlabeled)


def extract_v_vector(probs, y, add_rate):
    v_vector = np.zeros(probs.shape[0])
    classes = np.unique(y)
    count_per_class = [sum(y == cls) for cls in classes]
    pred_y = np.argmax(probs,axis=1)
    for cls in range(len(classes)):
        indices = np.where(pred_y == cls)[0]
        cls_score = probs[indices,cls]
        idx_sort = np.argsort(cls_score)
        add_num = min(int(np.ceil(count_per_class[cls] * add_rate)), indices.shape[0])
        v_vector[indices[idx_sort[-add_num:]]] = 1
    return v_vector.astype('bool')


def extract_new_train(v_current, X_labeled, y, X_unlabeled, pred_y):
    to_add_X=[]
    to_add_y=[]
    for i, flag in enumerate(v_current):
        if flag:
            to_add_X.append(X_unlabeled[i])
            to_add_y.append(pred_y[i])
    new_X=X_labeled+to_add_X
    new_y=y+to_add_y

    return new_X, new_y




def train_model(base_model, X, y):
    model = deepcopy(base_model)
    model.fit(X,y)
    return model