import os
import json
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    f1_score,
    roc_curve,
    auc,
)
from sklearn.preprocessing import label_binarize


class Tool:
    @staticmethod
    def save_confusion_matrix(y_true, y_score, target_names, filename, normalize=False):
        """
        Saves the confusion matrix to a file.

        Args:
            y_true (array-like): True labels.
            y_score (array-like): Predicted scores.
            target_names (list): Names of the target classes.
            filename (str): Path to save the confusion matrix.
            normalize (bool, optional): Whether to normalize the confusion matrix. Defaults to False.
        Returns:
            cm (numpy.ndarray): The confusion matrix.
        """
        try:
            cm = confusion_matrix(y_true, y_score)
            print("Confusion Matrix:\n", cm)
            if normalize:
                cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
                title = "Normalized Confusion Matrix"
            else:
                title = "Confusion Matrix, Without Normalization"

            plt.figure(figsize=(8, 6))
            sns.heatmap(
                cm,
                annot=True,
                fmt=".2f" if normalize else "d",
                cmap="Blues",
                xticklabels=target_names,
                yticklabels=target_names,
            )
            plt.ylabel("True label")
            plt.xlabel("Predicted label")
            plt.title(title)
            plt.savefig(filename)
            plt.close()
            return cm
        except ValueError as e:
            print(f"Error creating confusion matrix: {e}")
            return None

    @staticmethod
    def save_classification_report(y_true, y_score, filename):
        """
        Saves the classification report to a file.

        Args:
            y_true (array-like): True labels.
            y_score (array-like): Predicted scores.
            filename (str): Path to save the classification report.
        Returns:
            cr (dict): The classification report.
        """
        try:
            cr_nodict = classification_report(y_true, y_score, output_dict=False)
            print("Classification Report:\n", cr)
            cr = classification_report(y_true, y_score, output_dict=True)
            report_df = pd.DataFrame(cr).transpose()
            report_df.drop(
                "support", axis=1, inplace=True
            )  # Bỏ cột support nếu không cần
            report_df.plot(kind="bar", figsize=(10, 6))
            plt.title("Classification Report")
            plt.ylabel("Score")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(filename)
            plt.close()
            return cr_nodict
        except ValueError as e:
            print(f"Error creating DataFrame from classification report: {e}")
            return None

    @staticmethod
    def save_roc_auc_plot(y_true, y_score, n_classes, filename):
        """
        Calculates and saves the ROC AUC plot to a file.

        Args:
            y_true (array-like): True labels.
            y_score (array-like): Predicted scores.
            n_classes (int): Number of classes.
            filename (str): Path to save the plot.
        Returns:
            fpr (dict): False positive rates for each class.
            tpr (dict): True positive rates for each class.
            roc_auc (dict): ROC AUC scores for each class.
        """
        try:
            # Convert y_true and y_score to NumPy arrays if they are lists
            y_true = np.array(y_true)
            y_score = np.array(y_score)

            fpr = dict()
            tpr = dict()
            roc_auc = dict()

            # Binarize the output if more than 2 classes
            if n_classes > 2:
                y_true = label_binarize(y_true, classes=[*range(n_classes)])
                for i in range(n_classes):
                    fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_score[:, i])
                    roc_auc[i] = auc(fpr[i], tpr[i])
            else:
                fpr[1], tpr[1], _ = roc_curve(y_true, y_score[:, 1])
                roc_auc[1] = auc(fpr[1], tpr[1])

            plt.figure(figsize=(8, 6))

            if n_classes == 2:
                plt.plot(
                    fpr[1],
                    tpr[1],
                    lw=2,
                    label="ROC curve (area = {0:0.2f})".format(roc_auc[1]),
                )
            else:
                for i in range(n_classes):
                    plt.plot(
                        fpr[i],
                        tpr[i],
                        lw=2,
                        label="ROC curve of class {0} (area = {1:0.2f})".format(
                            i, roc_auc[i]
                        ),
                    )

            plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("Receiver Operating Characteristic (ROC)")
            plt.legend(loc="lower right")
            plt.savefig(filename)
            plt.close()
            return fpr, tpr, roc_auc
        except ValueError as e:
            print(f"Error creating ROC AUC plot: {e}")
            return None, None, None