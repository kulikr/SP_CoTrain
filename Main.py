import Utils
import re
import os
from datetime import datetime
import json

from sklearn.svm import SVC
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import auc, accuracy_score

BASE_MODELS = {"SVC": SVC(probability=True),
               "RandomForest": RandomForestClassifier()}

LABELED_RATIOS = [0.2, 0.4, 0.6]


def evaluate_model(base_line, base_model, X_labeled, X_unlabeled, y_labeled, X_test, y_test, view1, view2):
    # get the model to evaluate
    model = Utils.get_model(base_line, BASE_MODELS[base_model], X_unlabeled)

    # train the  model
    start = datetime.now()
    model.fit(X_labeled, X_unlabeled, y_labeled, view1, view2)
    end = datetime.now()
    fit_time = (end - start).total_seconds()

    # predict
    start = datetime.now()
    y_pred = model.predict(X_test)
    end = datetime.now()
    predict_time = (end - start).total_seconds()

    # calculate accuracy and auc
    auc_score = auc(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)

    return fit_time, predict_time, auc_score, acc


# Evaluates the algorithm with 10 fold cv each time the train is shuffled and splitted randomly to train and test.
def cross_validation(dir_path, file_name, cv=10, base_model="RandomForest", labeled_split=0.2):
    X, y, view1, view2 = Utils.extract_data(dir_path + '/' + file_name, file_name)

    res_spaco = []
    res_co = []
    res_base = []

    for i in range(cv):
        print("CV: " + str(i))
        X, y = shuffle(X, y)
        X_labeled, X_unlabeled, y_labeled, X_test, y_test = Utils.split_data(X, y, train_test_split=0.8,
                                                                             labeled_unlabeled_split=labeled_split)

        res_spaco.append(evaluate_model("spaco",base_model,X_labeled, X_unlabeled, y_labeled, X_test, y_test, view1, view2))

        res_co.append(evaluate_model("co",base_model,X_labeled, X_unlabeled, y_labeled, X_test, y_test, view1, view2))

        res_base.append(evaluate_model("base",base_model,X_labeled, X_unlabeled, y_labeled, X_test, y_test, view1, view2))

    return res_spaco, res_co, res_base


# Runs the experiment on the given data
def run_experiment(dir_path, file_name):
    res_spaco = {}
    res_co = {}
    res_base = {}

    for base_model in BASE_MODELS.keys():
        res_spaco[base_model] = {}
        res_co[base_model] = {}
        res_base[base_model] = {}

        for labled_ratio in LABELED_RATIOS:
            res_spaco[base_model][str(labled_ratio)], res_co[base_model][str(labled_ratio)], res_base[base_model][
                str(labled_ratio)] = cross_validation(dir_path, file_name, 10, base_model, labled_ratio)

    # Dump all the results
    internal_dir_name = re.sub(r".csv", "", file_name)

    f = open("./results/" + internal_dir_name + "/spaco.json", 'w')
    json.dump(res_spaco, f)
    f.close()

    f = open("./results/" + internal_dir_name + "/co.json", 'w')
    json.dump(res_spaco, f)
    f.close()

    f = open("./results/" + internal_dir_name + "/base.json", 'w')
    json.dump(res_spaco, f)
    f.close()

directory_in_str = "./data"
directory = os.fsencode(directory_in_str)

# loop on all the files
for file in os.listdir(directory):
    file_name = os.fsdecode(file)
    run_experiment(directory_in_str,file_name)