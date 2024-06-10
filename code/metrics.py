import numpy as np
from sklearn.metrics import confusion_matrix

"""
Old version is in data/notebooks/metrics.py
Invaid class is still calculated in the confusion matrix there.
"""


"num_classes - 1"
class ConfusionMatrix:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.state = np.zeros((self.num_classes, self.num_classes))

    def calc(self, gt, pred):
        return confusion_matrix(gt.flatten(), pred.flatten(), labels=np.arange(self.num_classes))

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
        nob = gt.shape[0]
        for batch_idx in range(nob):
            self.add(gt[batch_idx], pred[batch_idx])

    def norm_on_lines(self):
        a = self.state[1:, 1:]  # Exclude class 0
        b = np.sum(self.state[1:, 1:], axis=1)[:, None]
        return np.divide(a, b, out=np.zeros_like(a), where=b != 0)

    def get_aa(self):
        confmatrix = self.norm_on_lines()
        return np.diagonal(confmatrix).sum() / (self.num_classes - 1)

    def get_IoU(self):
        IoU = np.zeros(self.num_classes - 1)
        for i in range(1, self.num_classes):
            cm = self.state
            a = cm[i, i]
            b = (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
            IoU[i - 1] = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
        return IoU

    def get_mIoU(self):
        return np.mean(self.get_IoU())
    
def AA(gt, pred, num_classes):
    cm = ConfusionMatrix(num_classes)
    cm.add(gt, pred)
    confmatrix = cm.norm_on_lines()
    return np.mean(np.diagonal(confmatrix))

def IoU(gt, pred, num_classes):
    cm = ConfusionMatrix(num_classes).calc(gt, pred)
    IoU = np.zeros(num_classes - 1)
    for i in range(1, num_classes):
        TP = cm[i, i]
        FP = cm[:, i].sum() - TP
        FN = cm[i, :].sum() - TP
        IoU[i - 1] = TP / (TP + FP + FN) if (TP + FP + FN) != 0 else 0
    return IoU

def mIou(gt, pred, num_classes):
    return np.mean(IoU(gt, pred, num_classes))

