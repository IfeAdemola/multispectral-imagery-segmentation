import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

"""
Old version is in data/notebooks/metrics.py
Invaid class is still calculated in the confusion matrix there.
"""


"num_classes - 1"
"07_10: changing to binary, background and deforested, so no need to remove 1"
"09_10: returning to n-1"

"""
Confusion matrix:
    state: a matrix of size (num_classes, num_classes), where each element (i, j) represents
           the number of samples with groundtruth i and prediction j.
    add(gt, pred): adds a pair of groundtruth and prediction to the state.
    add_batch(gt, pred): adds a batch of groundtruth and prediction to the state.
    get_aa(): calculates the average accuracy, which is the diagonal of the confusion matrix divided by the number of classes.
    get_IoU(): calculates the Intersection over Union (IoU) for each class.
    plot_cm(labels): plots the confusion matrix using seaborn heatmap.
"""
class ConfusionMatrix:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.state = np.zeros((self.num_classes, self.num_classes))

    def calc(self, gt, pred):
        return confusion_matrix(gt.flatten(), pred.flatten(), labels=np.arange(self.num_classes))
    
    def plot_cm(self, labels):
        """
        Plots the confusion matrix using seaborn heatmap.
        """
        cm = self.norm_on_lines()

        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt=".1f", cmap="Blues", xticklabels=labels,
                     yticklabels=labels)
        plt.xlabel('Prediction')
        plt.ylabel('Groundtruth')
        plt.title('')
        plt.show()

    def get_existing_classes(self):
        return sum(np.sum(self.state, axis=1) > 0)

    def add(self, gt, pred):
        assert gt.shape == pred.shape
        gt, pred = gt.flatten(), pred.flatten()

        if gt.size != 0:
            self.state += confusion_matrix(gt, pred, labels=np.arange(self.num_classes))

    def add_batch(self, gt, pred):
        if not isinstance(gt, np.ndarray):
            gt = gt.cpu().numpy()
            pred = pred.cpu().numpy()

        assert len(gt.shape) == 3
        noc = gt.shape[0]    # number of channels
        for batch_idx in range(noc):
            self.add(gt[batch_idx], pred[batch_idx])

        return None

    def norm_on_lines(self):
        a = self.state[1:, 1:] #self.state[1:, 1:]  # Exclude class 0
        b = np.sum(self.state[1:, 1:], axis=1)[:, None]      #  np.sum(self.state[1:, 1:], axis=1)[:, None]
        norm = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
        return norm*100

    def get_aa(self):
        confmatrix = self.norm_on_lines()
        return np.diagonal(confmatrix).sum() / (self.num_classes - 1)

    def get_IoU(self):
        IoU = np.zeros(self.num_classes - 1)
        cm = self.state 
        for i in range(1, self.num_classes):
            a = cm[i, i]
            b = (cm[i, 1:].sum() + cm[1:, i].sum() - cm[i, i])
            IoU[i-1] = np.divide(a, b, out=np.zeros_like(a), where=b != 0)   
        return IoU

    def get_mIoU(self):
        return np.mean(self.get_IoU())
    
    def get_F1_score(self):
        f1_scores = np.zeros(self.num_classes - 1)
        cm = self.state
        for i in range(1, self.num_classes):
            TP = cm[i, i]
            FP = cm[1:, i].sum() - TP
            FN = cm[i, 1:].sum() - TP
            precision = np.divide(TP, (TP + FP), out=np.zeros_like(TP), where=(TP + FP) != 0)
            recall = np.divide(TP, (TP + FN), out=np.zeros_like(TP), where=(TP + FN) != 0)
            f1_scores[i-1] = np.divide(2 * precision * recall, (precision + recall), out=np.zeros_like(precision), where=(precision + recall) != 0)   # f1_score[i-1]
        return f1_scores
 
    def get_mF1_score(self):
        return np.mean(self.get_F1_score())
    
    def get_dice_coefficient(self):
        """
        Computes the Dice coefficient for each class (excluding class 0).
        """
        dice_coeff = np.zeros(self.num_classes - 1)
        cm = self.state
        for i in range(1, self.num_classes):
            TP = cm[i, i]
            FP = cm[1:, i].sum() - TP
            FN = cm[i, 1:].sum() - TP
        # Compute Dice coefficient for all classes
        dice_coeff[i-1] = np.divide(2 * TP, 2 * TP + FP + FN, out=np.zeros_like(TP), where=(2 * TP + FP + FN) != 0)
        
        return dice_coeff

    def get_precision(self):
        """
        Computes precision for each class (excluding class 0).
        """
        precision = np.zeros(self.num_classes - 1)
        cm = self.state
        for i in range(1, self.num_classes):
            TP = cm[i, i]
            FP = cm[1:, i].sum() - TP
            FN = cm[i, 1:].sum() - TP
            precision[i-1] = np.divide(TP, (TP + FP), out=np.zeros_like(TP), where=(TP + FP) != 0)        
        return precision

    def get_recall(self):
        """
        Computes recall for each class (excluding class 0).
        """
        recall = np.zeros(self.num_classes - 1)
        cm = self.state
        for i in range(1, self.num_classes):
            TP = cm[i, i]
            FP = cm[1:, i].sum() - TP
            FN = cm[i, 1:].sum() - TP
            recall[i-1] = np.divide(TP, (TP + FN), out=np.zeros_like(TP), where=(TP + FN) != 0)
        
        return recall
    
    def get_mean_dice_coefficient(self):
        """
        Computes the mean Dice coefficient across all classes (excluding class 0).
        """
        dice_coeff = self.get_dice_coefficient()
        return np.mean(dice_coeff)

    def get_mean_precision(self):
        """
        Computes the mean precision across all classes (excluding class 0).
        """
        precision = self.get_precision()
        return np.mean(precision)

    def get_mean_recall(self):
        """
        Computes the mean recall across all classes (excluding class 0).
        """
        recall = self.get_recall()
        return np.mean(recall)
    
    def get_weighted_dice_coefficient(self, class_weights):
        """
        Computes the weighted Dice coefficient across all classes (excluding class 0).
        
        Args:
        - class_weights: A list or array of weights for each class, length should be num_classes - 1.
        
        Returns:
        - Weighted Dice coefficient.
        """
        dice_coeff = self.get_dice_coefficient()
        weighted_dice = np.average(dice_coeff, weights=class_weights)
        
        return weighted_dice
    
    def get_weighted_precision(self, class_weights):
        """
        Computes the weighted precision across all classes (excluding class 0).
        
        Args:
        - class_weights: A list or array of weights for each class, length should be num_classes - 1.
        
        Returns:
        - Weighted precision.
        """
        precision = self.get_precision()
        weighted_precision = np.average(precision, weights=class_weights)
        
        return weighted_precision

    def get_weighted_recall(self, class_weights):
        """
        Computes the weighted recall across all classes (excluding class 0).
        
        Args:
        - class_weights: A list or array of weights for each class, length should be num_classes - 1.
        
        Returns:
        - Weighted recall.
        """
        recall = self.get_recall()
        weighted_recall = np.average(recall, weights=class_weights)
        
        return weighted_recall

    def get_weighted_f1_score(self, class_weights):
        """
        Computes the weighted F1 score across all classes (excluding class 0).
        
        Args:
        - class_weights: A list or array of weights for each class, length should be num_classes - 1.
        
        Returns:
        - Weighted F1 score.
        """
        f1_scores = self.get_F1_score()
        weighted_f1 = np.average(f1_scores, weights=class_weights)
        
        return weighted_f1
 
# def AA(gt, pred, num_classes):
#     cm = ConfusionMatrix(num_classes)
#     cm.add(gt, pred)
#     confmatrix = cm.norm_on_lines()
#     return np.mean(np.diagonal(confmatrix))

# def IoU(gt, pred, num_classes):
#     cm = ConfusionMatrix(num_classes).calc(gt, pred)
#     IoU = np.zeros(num_classes - 1)
#     for i in range(1, num_classes):
#         TP = cm[i, i]
#         FP = cm[1:, i].sum() - TP
#         FN = cm[i, 1:].sum() - TP
#         IoU[i - 1] = TP / (TP + FP + FN) if (TP + FP + FN) != 0 else 0
#     return IoU

# def mIou(gt, pred, num_classes):
#     return np.mean(IoU(gt, pred, num_classes))

# def F1_score(gt, pred, num_classes):
#     cm = ConfusionMatrix(num_classes).calc(gt, pred)
#     f1_scores = np.zeros(num_classes)
#     for i in range(num_classes):
#         TP = cm[i, i]
#         FP = cm[1:, i].sum() - TP
#         FN = cm[i, 1:].sum() - TP
#         precision = TP / (TP + FP) if (TP + FP) != 0 else 0
#         recall = TP / (TP + FN) if (TP + FN) != 0 else 0
#         f1_scores[i] = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

#     return f1_scores
