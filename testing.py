import torch
import numpy as np
from training import corrcoef
from training import RMSELoss
from torch.nn import L1Loss
from torch.nn import MSELoss
from sklearn.metrics.pairwise import euclidean_distances
from training import pad_batch_online

def isnotebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter

if isnotebook():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm
tqdm.pandas()

rmse = RMSELoss(eps=0)
l1 = L1Loss()
l2 = MSELoss()


class Testing:
    """
        Create Testing Instance
        :param model: torch model
            model to train
        :param inps: pd.Series
            series containing inputs
        :param tgts: pd.Series
            series containing the corresponding targets

        :param criterion: torch.Loss
            criterion to calculate the loss between targets and predictions
            - if using cross correlation the function must accept predicted correlatios vs. true correlation and predicted vectors vs. target vectors

        :param use_cross_corr: bool
            specify whether to calculate the cross correlation

        (necessary for calculating cross correlation)
        :param labels: pd.Series
            series containing the label for each input sample
        :param cross_corr_matrix: pd.DataFrame
            pd.DataFrame containing the pairwise correlation between all unique labels
            - labels are used and row and column index
        :param  label_vectors: pd.DataFrame
            pd.DataFrame containing the semantic vectors for each label
            - labels are used as row index

        """


    def __init__(self, model, inps, tgts, criterion,
                 with_onset_dim=False,
                 with_time_dim=False,
                 use_cross_corr = False,
                 labels = [],
                 label_vectors=None,
                 cross_corr_matrix = None):

        self.model = model
        self.inps = inps
        self.tgts = tgts
        self.labels = labels

        self.predictions = None
        self.losses = None
        self.sublosses = None

        self.device = next(model.parameters()).device # get device model is located at

        self.lens_input = torch.Tensor(np.array(inps.apply(len), dtype=np.int)).to(self.device)
        self.lens_output = torch.Tensor(np.array(tgts.apply(len), dtype=np.int)).to(self.device)

        self.criterion = criterion

        self.use_cross_corr = use_cross_corr

        self.with_onset_dim = with_onset_dim
        self.with_time_dim = with_time_dim


        # necessary for predicting labels
        self.label_vectors = label_vectors
        if not label_vectors is None:
            self.label_vectors_np = np.array(list(label_vectors.vector))

        # predicting labbels
        self.top_10_predicted_labels_euclidean = None # top 10 predicted labels reffering to the distance to the prediction
        self.top_10_predicted_labels_euclidean_distance = None # distance of top 10 predicted labels to the prediction
        self.euclidean_dist_to_true_labels = None # euclidean distance of true label to the prediction
        self.rank_of_true_label_in_predictions_euclidean = None # the rank of the true label in the distance to the prediction

        self.top_10_predicted_labels_cross_correlation = None
        self.top_10_predicted_labels_cross_correlation_correlation = None
        self.cross_correlation_with_true_labels = None
        self.rank_of_true_label_in_predictions_cross_correlation = None

        if use_cross_corr:
            assert len(labels) > 0, "In order to use cross correlation please provide labels!"
            assert not cross_corr_matrix is None, "Please provide a precomputed Cross Correlation Matrix!"
            assert not label_vectors is None, "Please provide a lookup df with labels and corresponding embedding vectors!"
            self.cross_corr_matrix = cross_corr_matrix
            self.label_vectors_torch = torch.from_numpy(np.asarray(list(label_vectors.vector))).to(self.device)



    def score(self):
        """
        Function for scoring model on inputs

        :return :
            -
            - predictions made by the model for each sample are stored in self.predictions
            - the loss/subloss for each prediction with respect to the provided criterion is stored in self.losses/self.sub_losses

        """

        test_predictions = []
        test_losses = []
        test_sublosses = []

        with torch.no_grad():  # no gradient calculation
            for idxs in tqdm(range(len(self.inps)), desc="Predicting..."):
                #lens_input_jj = [self.lens_input[idxs]]
                lens_input_jj = self.lens_input[idxs:(idxs+1)]
                batch_input = self.inps.iloc[idxs:(idxs+1)]
                batch_input = pad_batch_online(lens_input_jj, batch_input, self.device,self.with_onset_dim, self.with_time_dim)
                #lens_output_jj = [self.lens_output[idxs]]
                #batch_input = torch.tensor(list([self.inps.iloc[idxs]]), device = self.device)


                Y_hat = self.model(batch_input,lens_input_jj)
                batch_output = torch.tensor(list([self.tgts.iloc[idxs]]), device=self.device)

                if self.use_cross_corr:
                    batch_output_cross_corr = self.labels.iloc[idxs]
                    batch_output_cross_corr = torch.from_numpy(np.asarray([self.cross_corr_matrix.loc[batch_output_cross_corr]])).to(self.device)
                    Y_hat_cross_corr = corrcoef(torch.cat((Y_hat, self.label_vectors_torch)))[:1,1:]
                    loss = self.criterion(Y_hat_cross_corr, batch_output_cross_corr, Y_hat, batch_output)
                else:
                    loss = self.criterion(Y_hat, batch_output)


                if isinstance(loss, tuple):
                    sub_losses = loss[1:]
                    loss = loss[0]
                    test_losses += [loss.item()]
                    test_sublosses += [[sub_loss.item() for sub_loss in sub_losses]] # for each sample [subloss1_i,subloss2_i,subloss3_i]


                else:
                    test_losses += [loss.item()]

                prediction = Y_hat.cpu().detach().numpy()[0]
                test_predictions +=[prediction]

            self.predictions = test_predictions
            self.losses = test_losses

            if len(test_sublosses) > 0:
                test_sublosses = np.asarray(test_sublosses)
                self.sublosses = [test_sublosses[:, i] for i in range(test_sublosses.shape[1])] # for each subloss [subloss1_i, subloss1_j,subloss1_k]


    def predict_top10_labels(self,prediction, true_label, metric):
        """
        :param prediction: np.array
            model prediciton of semantic vector
        :param true_label: str
            true label name
        :param metric: str
            metric to predict labels (one of: "cross-correlation" or "euclidean")
        :return label_pred, dist[top_10_closest], dist_of_true_label, rank_of_true_label_in_dist: np.array, np.array, float, int
            - np.array of labels for top 10 closest vectors to predicted vector
            - np.array with corresponding distances of these top 10 vectors to prediction
            - distance of true vector to predicted
            - rank of true vector with respect to the distance to the predicted vector
        """

        if metric == "cross-correlation":
            prediction = torch.from_numpy(np.asarray([prediction])).to(self.device)
            cross_corr = np.array(corrcoef(torch.cat((prediction, self.label_vectors_torch)))[:1, 1:].cpu())[0] # cross correlation of prediction with all others

            true_index = int(self.label_vectors[self.label_vectors.label == true_label].index[0]) # index of true label
            cross_corr_of_true_label = cross_corr[true_index] # true cross correlation

            cross_corr_argsort = np.argsort(cross_corr)[::-1] # sort descending (highes correlation)
            top_10_cross_corr = cross_corr_argsort[:10] # top 10 correlation indices
            rank_of_true_label_in_cross_corr = np.where(cross_corr_argsort == true_index)[0][0] + 1 # check true index in sorted correlation

            label_pred = np.array(self.label_vectors.loc[top_10_cross_corr].label) # labels with highest correlations

            return label_pred, cross_corr[top_10_cross_corr], cross_corr_of_true_label, rank_of_true_label_in_cross_corr

        elif metric == "euclidean":
            prediction = np.asarray([prediction])
            dist = euclidean_distances(prediction, self.label_vectors_np)[0]

            true_index = int(self.label_vectors[self.label_vectors.label == true_label].index[0])
            dist_of_true_label = dist[true_index]

            dist_argsort = np.argsort(dist)
            top_10_closest = dist_argsort[:10]
            rank_of_true_label_in_dist = np.where(dist_argsort == true_index)[0][0] + 1

            label_pred = np.array(self.label_vectors.loc[top_10_closest].label)

            return label_pred, dist[top_10_closest], dist_of_true_label, rank_of_true_label_in_dist

    def predict_labels(self, metric):
        """
        :param metric: str
            - metric to get labels from predicted semantic vectors
        :return:
            sets self.top_10_predicted_labels_...
            self.top_10_predicted_labels_..
            self..._to_true_labels
            self.rank_of_true_label_in_predictions_...
            by using provided metric
        """
        assert not self.predictions is None, "In order to predict labels please call the score function first!"
        assert metric in ["euclidean", "cross-correlation"], "only euclidean distance and cross-correlation are implemnted to get the label of the closest target vector form the prediction!"
        assert not self.label_vectors is None, "In order to compare the predicted vectors to the true vectors, please provide a lookup df with labels and corresponding embedding vectors!"
        assert len(self.labels) > 0, "In order to compare the prediction to the true label, true labels must be provided!"


        top_10_predicted_labels = []
        top_10_predicted_labels_distance = []
        dist_to_true_labels = []
        rank_of_true_label_in_predictions = []

        for i, prediction in enumerate(tqdm(self.predictions, desc="Predicting Labels...")):
            true_label = self.labels.iloc[i]
            top10, top10_distance, true_label_distance, true_label_rank = self.predict_top10_labels(prediction,
                                                                                                     true_label,
                                                                                                     metric=metric)
            top_10_predicted_labels.append(top10)
            top_10_predicted_labels_distance.append(top10_distance)
            dist_to_true_labels.append(true_label_distance)
            rank_of_true_label_in_predictions.append(true_label_rank)


        if metric == "cross-correlation":
            self.top_10_predicted_labels_cross_correlation = top_10_predicted_labels
            self.top_10_predicted_labels_cross_correlation_correlation = top_10_predicted_labels_distance
            self.cross_correlation_with_true_labels = dist_to_true_labels
            self.rank_of_true_label_in_predictions_cross_correlation = rank_of_true_label_in_predictions

        else:
            self.top_10_predicted_labels_euclidean = top_10_predicted_labels
            self.top_10_predicted_labels_euclidean_distance = top_10_predicted_labels_distance
            self.euclidean_dist_to_true_labels = dist_to_true_labels
            self.rank_of_true_label_in_predictions_euclidean = rank_of_true_label_in_predictions

