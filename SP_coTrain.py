import numpy as np
import Utils

class SP_coTrain:

    def __init__(self, base_models, num_of_iters=8, add_rate=0.5, gamma=0.5):
        """
        C'tor

        :param base_models: The base models to use for the different views
        :param num_of_iters: The number of iterations to train the model
        :param add_rate: The lambda hyperparameter for the sp_coTrain model
        :param gamma: The gamma hyperparameter for the sp_coTrain model
        """
        self.base_models=base_models
        self.trained_models=[None, None]
        self.num_of_iters=num_of_iters
        self.add_rate=add_rate
        self.gamma=gamma
        self.views=None


    def fit(self, X_labeled, X_unlabeled, y, view1,view2):
        """
        Trains the model

        :param X_labeled: The labeled train data
        :param X_unlabeled: The unlabeled train data
        :param y: The labels for the labeled train data
        :param views: Array of features list for each of the views
        """
        #set views (the features for each view)
        self.views=[view1,view2]

        # Extract number of classes
        num_classes = np.unique(y)

        # init the models
        pred_probs = []
        v_vectors = []
        for view in range(2):
            self.trained_models[view] = Utils.train_model(self.base_models[view], X_labeled[:,self.views[view]] ,y)
            pred_probs.append(Utils.predict_proba(self.trained_models[view],X_unlabeled[:,self.views[view]]))
            v_vectors.append(Utils.extract_v_vector(pred_probs[view], y, self.add_rate))
        pred_y = np.argmax(sum(pred_probs), axis=1)

        for i in range(self.num_of_iters):
            for view in range(2):

                # update the current view with the other view v-vector
                v_other = v_vectors[1 - view]
                pred_probs[view][v_other, pred_y[v_other]] += self.gamma
                v_current = Utils.extract_v_vector(pred_probs[view], y, self.add_rate)

                # update current view and current model
                new_train_data,new_y = Utils.extract_new_train(v_current, X_labeled, y, X_unlabeled, pred_y)
                self.trained_models[view] = Utils.train_model(self.base_models[view], new_train_data[:,self.views[view]], new_y)

                # update y
                pred_probs[view] = Utils.predict_proba(self.trained_models[view],X_unlabeled[:,self.views[view]])
                pred_y = np.argmax(sum(pred_probs), axis=1)

                # update current view for next view
                self.add_rate += 0.5 ########******######
                v_vectors[view] = Utils.extract_v_vector(pred_probs[view], y, self.add_rate)

    def predict(self):
        pass

    def get_params(self):
        pass

    def set_params(self):
        pass

    def spaco(model_names,data,save_paths,iter_step=1,gamma=0.3,train_ratio=0.2):
        pass

