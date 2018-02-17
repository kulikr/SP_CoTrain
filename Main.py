import Utils
import data_utils as du
import re
import os
import numpy as np
from datetime import datetime
import json
import matplotlib.pyplot as plt
import SP_coTrain

from sklearn.svm import SVC
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_auc_score, accuracy_score


CONFIG = "ITERS"

BASE_MODELS = {"SVC": SVC(probability=True),
               "RandomForest": RandomForestClassifier(),
               "NaiveBayes": GaussianNB()}

LABELED_RATIOS = [0.2, 0.4, 0.6]

ITERS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
ITERS_BASE_MODEL = "RandomForest"


def evaluate_model(base_line, base_model, X_labeled, X_unlabeled, y_labeled, X_test, y_test, view1, view2,
                   labeled_rate):
    # get the model to evaluate
    model = Utils.get_model(base_line, BASE_MODELS[base_model], X_unlabeled, labeled_rate)

    # train the  model
    start = datetime.now()
    if base_line == "reg_base":
        model.fit(X_labeled, y_labeled)
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
    y_pred = np.array(y_pred)
    auc = roc_auc_score(y_test, y_pred)

    acc = accuracy_score(y_test, y_pred)

    return fit_time, predict_time, auc, acc


# Evaluates the algorithm with 10 fold cv each time the train is shuffled and splitted randomly to train and test.
def cross_validation(dir_path, file_name, cv=10, base_model="RandomForest", labeled_rate=0.2):
    X, y, view1, view2 = du.extract_data(dir_path + '/' + file_name, file_name)

    res_spaco = []
    res_co = []
    res_base = []
    res__reg_base = []

    for i in range(cv):
        print("CV: " + str(i))
        X, y = shuffle(X, y)
        X_labeled, X_unlabeled, y_labeled, X_test, y_test = du.split_data(X, y, train_test_split=0.8,
                                                                          labeled_unlabeled_split=labeled_rate)

        res_spaco.append(
            evaluate_model("spaco", base_model, X_labeled, X_unlabeled, y_labeled, X_test, y_test, view1, view2,
                           labeled_rate))

        res_co.append(evaluate_model("co", base_model, X_labeled, X_unlabeled, y_labeled, X_test, y_test, view1, view2,
                                     labeled_rate))

        res_base.append(
            evaluate_model("base", base_model, X_labeled, X_unlabeled, y_labeled, X_test, y_test, view1, view2,
                           labeled_rate))

        res_base.append(
            evaluate_model("reg_base", base_model, X_labeled, X_unlabeled, y_labeled, X_test, y_test, view1, view2,
                           labeled_rate))

    return res_spaco, res_co, res_base, res__reg_base


# Runs the experiment on the given data
def run_experiment(dir_path, file_name):
    res_spaco = {}
    res_co = {}
    res_base = {}
    res_reg_base = {}

    for base_model in BASE_MODELS.keys():
        res_spaco[base_model] = {}
        res_co[base_model] = {}
        res_base[base_model] = {}
        res_reg_base[base_model] = {}

        for labled_ratio in LABELED_RATIOS:
            res_spaco[base_model][str(labled_ratio)], res_co[base_model][str(labled_ratio)], res_base[base_model][
                str(labled_ratio)], res_spaco[base_model][str(labled_ratio)] = cross_validation(dir_path, file_name, 10,
                                                                                                base_model,
                                                                                                labled_ratio)

    # Dump all the results
    internal_dir_name = re.sub(r".csv", "", file_name)

    f = open("./results/" + internal_dir_name + "/spaco.json", 'w')
    json.dump(res_spaco, f)
    f.close()

    f = open("./results/" + internal_dir_name + "/co.json", 'w')
    json.dump(res_co, f)
    f.close()

    f = open("./results/" + internal_dir_name + "/base.json", 'w')
    json.dump(res_base, f)
    f.close()

    f = open("./results/" + internal_dir_name + "/reg_base.json", 'w')
    json.dump(res_reg_base, f)
    f.close()


directory_in_str = "./data"
directory = os.fsencode(directory_in_str)


def run_iters_experiment(dir_path, file_name):
    res = {}
    res['x']=[]
    res['y']=[]
    for num_of_iters in ITERS:
        X, y, view1, view2 = du.extract_data(dir_path + '/' + file_name, file_name)
        acc = []
        for i in range(10):
            print("CV: " + str(i))
            X, y = shuffle(X, y)
            X_labeled, X_unlabeled, y_labeled, X_test, y_test = du.split_data(X, y, train_test_split=0.8,
                                                                              labeled_unlabeled_split=0.2)
            base_models = [BASE_MODELS[ITERS_BASE_MODEL], BASE_MODELS[ITERS_BASE_MODEL]]
            model = SP_coTrain.SP_coTrain(base_models, num_of_iters, add_rate=0.1, gamma=0.5)
            model.fit(X_labeled, X_unlabeled, y_labeled, view1, view2)
            y_pred = model.predict(X_test)
            acc.append(accuracy_score(y_test, y_pred))

        res['x'].append(num_of_iters)
        res['y'].append(sum(acc) / len(acc))
    plot_graph(res)

def plot_graph(results,title="Accuracy per Num OF Iterations"):
    plt.title(title)
    plt.xlabel("Num OF Iters")
    plt.ylabel("Accuracy")

    plt.plot(results['x'],results['y'])

    plt.savefig("./charts/" + title + ".png")
    plt.close()

if CONFIG != "ITERS":

    # loop on all the files
    for file in os.listdir(directory):
        file_name = os.fsdecode(file)
        run_experiment(directory_in_str, file_name)
else:

    # loop on all the files
    for file in os.listdir(directory):
        file_name = os.fsdecode(file)
        run_iters_experiment(directory_in_str, file_name)
