import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc

def evaluate_metrics(egt, detected_edges):
    # Ensure the images are binary
    egt = (egt > 128).astype(np.uint8)
    detected_edges = (detected_edges > 128).astype(np.uint8)

    # Flatten the arrays to calculate the metrics
    egt_flat = egt.flatten()
    detected_edges_flat = detected_edges.flatten()

    # Calculate Precision, Recall, and F1 Score
    precision = precision_score(egt_flat, detected_edges_flat)
    recall = recall_score(egt_flat, detected_edges_flat)
    f1 = f1_score(egt_flat, detected_edges_flat)

    # Calculate ROC AUC
    fpr, tpr, _ = roc_curve(egt_flat, detected_edges_flat)
    roc_auc = auc(fpr, tpr)

    return precision, recall, f1, roc_auc