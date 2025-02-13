import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, cohen_kappa_score

def evaluate_model(model, test_generator):
    """ Evaluate the model on the test dataset and compute evaluation metrics """
    # Get true labels and predictions
    y_true = test_generator.classes
    y_pred_probs = model.predict(test_generator)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Compute metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro')
    sensitivity = recall_score(y_true, y_pred, average='macro')  # Sensitivity = Recall
    specificity = compute_specificity(y_true, y_pred, test_generator.num_classes)
    f1 = f1_score(y_true, y_pred, average='macro')
    kappa = cohen_kappa_score(y_true, y_pred)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Sensitivity (Recall): {sensitivity:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"Kappa Score: {kappa:.4f}")

    return accuracy, precision, sensitivity, specificity, f1, kappa

def compute_specificity(y_true, y_pred, num_classes):
    """ Compute specificity for multi-class classification """
    cm = confusion_matrix(y_true, y_pred)
    specificity_per_class = []
    
    for i in range(num_classes):
        tn = np.sum(cm) - np.sum(cm[i, :]) - np.sum(cm[:, i]) + cm[i, i]
        fp = np.sum(cm[:, i]) - cm[i, i]
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        specificity_per_class.append(specificity)
    
    return np.mean(specificity_per_class)

# Example Usage
# Assuming `model` is the trained model and `test_generator` is the test data generator
accuracy, precision, sensitivity, specificity, f1, kappa = evaluate_model(model, test_generator)
