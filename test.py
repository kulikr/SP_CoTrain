import SP_coTrain
import SP_coTrain_ours
import SP_coTrain_ours2
import numpy as np
from datetime import datetime
import data_utils as du
import Utils
from copy import deepcopy
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, accuracy_score

BASE_MODELS = {"SVC": SVC(probability=True),
               "RandomForest": RandomForestClassifier()}

dir_path="./data"
file_name="nba_train.csv"
base_model="RandomForest"
cv=1
labeled_split=0.2

def evaluate_model(base_line, base_model, X_labeled, X_unlabeled, y_labeled, X_test, y_test, view1, view2):
    model=None
    num_of_iters=10
    # get the model to evaluate
    if (base_line=="spaco"):
        model = SP_coTrain.SP_coTrain([BASE_MODELS[base_model],BASE_MODELS[base_model]],num_of_iters=num_of_iters)
    elif (base_line=="spaco_ours"):
        add_rate=Utils.get_add_rate(labeled_split,8)
        model = SP_coTrain_ours.SP_coTrain([BASE_MODELS[base_model], BASE_MODELS[base_model]],add_rate=add_rate,num_of_iters=num_of_iters,gamma=0.2)
    elif (base_line=="spaco_ours2"):
        model = SP_coTrain_ours2.SP_coTrain([BASE_MODELS[base_model], BASE_MODELS[base_model]],add_rate=0.2,num_of_iters=num_of_iters,gamma=0.2)
    elif (base_line=="reg"):
        model = deepcopy(BASE_MODELS[base_model])

    # train the  model
    start = datetime.now()
    if base_line=='reg':
        model.fit(X_labeled,y_labeled)
    else:
        model.fit(X_labeled, X_unlabeled, y_labeled, view1, view2)
    end = datetime.now()
    fit_time = (end - start).total_seconds()

    # predict
    start = datetime.now()
    y_pred = model.predict(X_test)
    end = datetime.now()
    predict_time = (end - start).total_seconds()

    # calculate accuracy and auc
    y_pred=np.array(y_pred)
    auc = roc_auc_score(y_test, y_pred)

    acc = accuracy_score(y_test, y_pred)

    return fit_time, predict_time, auc, acc


X, y, view1, view2 = du.extract_data(dir_path + '/' + file_name, file_name)

res_spaco = []
res_spaco_ours = []
res_spaco_ours2 = []
res_reg=[]

for i in range(cv):
    print("CV: " + str(i))
    X, y = shuffle(X, y)
    X_labeled, X_unlabeled, y_labeled, X_test, y_test = du.split_data(X, y, train_test_split=0.8,
                                                                         labeled_unlabeled_split=labeled_split)

    res_spaco.append(evaluate_model("spaco",base_model,X_labeled, X_unlabeled, y_labeled, X_test, y_test, view1, view2))

    res_spaco_ours.append(evaluate_model("spaco_ours",base_model,X_labeled, X_unlabeled, y_labeled, X_test, y_test, view1, view2))

    res_spaco_ours2.append(evaluate_model("spaco_ours2",base_model,X_labeled, X_unlabeled, y_labeled, X_test, y_test, view1, view2))

    res_reg.append(evaluate_model("reg",base_model,X_labeled, X_unlabeled, y_labeled, X_test, y_test, view1, view2))

f=open('./test_res.txt',mode='w')
f.write("spaco\n")
f.write(str(res_spaco[0]))
f.write("\n\n")
f.write("spaco_ours\n")
f.write(str(res_spaco_ours[0]))
f.write("\n\n")
f.write("spaco_ours2\n")
f.write(str(res_spaco_ours2[0]))
f.write("\n\n")
f.write("reg\n")
f.write(str(res_reg[0]))
f.write("\n\n")
f.close()


