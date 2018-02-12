from sklearn.svm import SVC
from copy import deepcopy
import numpy as np


class Co_Training_Classifier:

    def __init__(self, base_model=SVC(probability=True), K=0, G=0):
        self.base_model = base_model
        self.K = K
        self.G = G
        self.view1_features = None
        self.view2_features = None
        self.h1=None
        self.h2=None

    def fit(self, labeled_X, labeled_y, unlabeled_X, view1_features, view2_features):

        self.view1_features = view1_features
        self.view2_features = view2_features

        # seperate the labeled features of instances by the different views
        L1_X = labeled_X[:,view1_features]
        L2_X = labeled_X[:,view2_features]

        L1_y = labeled_y
        L2_y = labeled_y

        # seperate the unlabeled features of the instances by the different views
        U1 = unlabeled_X[:,view1_features]
        U2 = unlabeled_X[:,view2_features]

        # create list of instances indices for each of classifiers unlabeled data
        U1_indices = [i for i in range(len(U1))]
        U2_indices = [i for i in range(len(U2))]

        # iteratively add more labeled data
        for k in range(self.K):
            # create deep copy of the base model
            h1 = deepcopy(self.base_model)
            h2 = deepcopy(self.base_model)
            # train both of the classifiers
            h1.fit(L1_X, L1_y)
            h2.fit(L2_X, L2_y)
            # predict and calculate the 'confidence level' for unlabeled data
            prob1 = h1.predict_proba(U1)
            prob2 = h2.predict_proba(U2)
            # predict the class of the unlabeled data
            res1 = h1.predict(U1)
            res2 = h2.predict(U2)
            top_1 = self.extract_topG(res1, prob1)
            top_2 = self.extract_topG(res2, prob2)
            L1_new_X, L1_new_y,U2,U2_indices = self.extract_from_unlabeled(U2, top_2, view1_features, U2_indices, unlabeled_X)
            L2_new_X, L2_new_y ,U1,U1_indices= self.extract_from_unlabeled(U1, top_1, view2_features, U1_indices, unlabeled_X)

            # append the new labeled instances
            L1_X = np.append(L1_X,L1_new_X,axis=0)
            L2_X = np.append(L2_X,L2_new_X,axis=0)

            L1_y =np.append(L1_y,L1_new_y)
            L2_y = np.append(L2_y,L2_new_y)

        self.h1 = deepcopy(self.base_model)
        self.h2 = deepcopy(self.base_model)

        self.h1.fit(L1_X, L1_y)
        self.h2.fit(L2_X, L2_y)

    def predict(self, X):
        res1 = self.h1.predict_proba(X[:, self.view1_features])
        res2 = self.h2.predict_proba(X[:, self.view2_features])
        classes = self.h1.classes_
        res = []
        for i in range(len(res1)):
            if res1[i][0] + res2[i][0] > res1[i][1] + res2[i][1]:
                res.append(classes[0])
            else:
                res.append(classes[1])
        return res

    # The function gets for each instance the probabilities distribution predicted and return top G tuples of index
    # in the data and class
    def extract_topG(self, labels, prob):
        tuples = []

        for i in range(len(prob)):
            max_prob = np.max(prob[i])
            tuples.append([i, max_prob])

        tuples = np.array(tuples)
        tuples = tuples[tuples[:, 1].argsort()]
        indices = tuples[-self.G:, 0]

        indices=indices.astype(np.int)
        res = []
        for index in indices:
            res.append([index, labels[index]])
        return np.array(res)

    # Removes the indices of the new labeled instances from the unlabeled indices list and extracts the instances
    # other view features
    def extract_from_unlabeled(self, U, top, view_features, U_indices, unlabeled_X):

        L_X = None
        L_y = np.array([])
        indices_to_del=[]
        for tuple in top:  # tuple -> (index,label)
            index=U_indices[int(tuple[0])]
            instance = unlabeled_X[index,view_features]
            if L_X is None:
                L_X=np.array([instance])
            else:
                L_X = np.append(L_X, [instance], axis=0)
            L_y = np.append(L_y, np.array([tuple[1]]), axis=0)
            indices_to_del.append(int(tuple[0]))

        U=np.delete(U,indices_to_del,0)
        U_indices = [index for index in U_indices if index not in top[:, 0]]
        return L_X, L_y , U ,U_indices


