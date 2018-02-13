import numpy as np
import Utils


class Base:

    def __init__(self, base_models):
        self.models = base_models

    def fit(self, X_labeled, y_labeled, X_unlabeled, view1, view2):
        # set views (the features for each view)
        self.views = [view1,view2]

        for view in range(len(views)):
            self.models[view]=self.models[view].fit(X_labeled[:,views[view]],y)


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


