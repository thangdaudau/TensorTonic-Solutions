import numpy as np

def compute_monitoring_metrics(system_type, y_true, y_pred):
    """
    Compute the appropriate monitoring metrics for the given system type.
    """
    # Write code here
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    n = y_true.shape[0]
    def classification():
        TP = np.sum((y_true == 1) & (y_pred == 1)).item()
        TN = np.sum((y_true == 0) & (y_pred == 0)).item()
        FP = np.sum((y_true == 0) & (y_pred == 1)).item()
        FN = np.sum((y_true == 1) & (y_pred == 0)).item()
        acc = (TP + TN) / n
        prec = TP / (TP + FP) if TP + FP else 0
        recall = TP / (TP + FN) if TP + FN else 0
        f1 = 2 * prec * recall / (prec + recall) if prec + recall else 0
        return [("accuracy", acc), ("f1", f1), ("precision", prec), ("recall", recall)]
    def regression():
        dy = y_true - y_pred
        mae = np.fabs(dy).mean()
        rmse = np.sqrt((dy * dy).mean())
        return [("mae", mae), ("rmse", rmse)]
    def ranking():
        idx = np.argsort(y_pred)
        total = np.sum(y_true)
        k = 3
        relevant = y_true[idx[-k:]].sum().item()
        precAtK = relevant / k
        recallAtK = relevant / total if total else 0
        return [("precision_at_3", precAtK), ("recall_at_3", recallAtK)]
    
    func = {"classification": classification, "regression": regression, "ranking": ranking}
    return func[system_type]()