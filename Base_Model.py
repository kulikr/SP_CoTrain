import numpy as np
import Utils
from copy import deepcopy


class Base:

    def __init__(self, base_models):
        self.models=[None,None]
        self.models[0] = deepcopy(base_models[0])
        self.models[1] = deepcopy(base_models[1])

    def fit(self, X_labeled, X_unlabeled,y_labeled, view1, view2):
        # set views (the features for each view)
        self.views = [view1,view2]

        for view in range(len(self.views)):
            self.models[view]=self.models[view].fit(X_labeled[:, self.views[view]],y_labeled)


    def predict(self,X):
        res1 = self.models[0].predict_proba(X[:, self.views[0]])
        res2 = self.models[1].predict_proba(X[:, self.views[1]])
        classes = self.models[0].classes_
        res = []
        for i in range(len(res1)):
            if res1[i][0] + res2[i][0] > res1[i][1] + res2[i][1]:
                res.append(classes[0])
            else:
                res.append(classes[1])
        return res


