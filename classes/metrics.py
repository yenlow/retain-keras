from tensorflow.keras import backend as K
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, roc_curve
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

def specificity(y_true, y_pred):
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos

    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos

    tp = K.sum(y_pos * y_pred_pos)
    tn = K.sum(y_neg * y_pred_neg)

    fp = K.sum(y_neg * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg)
    return (tn / (fp + tn))


def precision_recall(y_true, y_prob, ARGS):
    """Print Precision Recall Statistics and Graph"""
    average_precision = average_precision_score(y_true, y_prob)
    if ARGS.graphs:
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        plt.style.use('ggplot')
        plt.clf()
        plt.plot(recall, precision,
                 label='Precision-Recall Curve  (Area = %0.3f)' % average_precision)
        plt.xlabel('Recall: P(predicted+|true+)')
        plt.ylabel('Precision: P(true+|predicted+)')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.legend(loc="lower left")
        print(f'Precision-Recall Curve saved to {ARGS.out_directory}/pr.png')
        plt.savefig(f'{ARGS.out_directory}/pr.png')
    else:
        print('Average Precision %0.3f' % average_precision)


def probability_calibration(y_true, y_prob, ARGS):
    if ARGS.graphs:
        fig_index = 1
        name = 'My pred'
        n_bins = 20
        fig = plt.figure(fig_index, figsize=(10, 10))
        ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
        ax2 = plt.subplot2grid((3, 1), (2, 0))

        ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")

        fraction_of_positives, mean_predicted_value = \
            calibration_curve(y_true, y_prob, n_bins=n_bins, normalize=True)

        ax1.plot(mean_predicted_value, fraction_of_positives,
                 label=name)

        ax2.hist(y_prob, range=(0, 1), bins=n_bins, label=name,
                 histtype="step", lw=2)

        ax1.set_ylabel("Fraction of Positives")
        ax1.set_ylim([-0.05, 1.05])
        ax1.legend(loc="lower right")
        ax1.set_title('Calibration Plots  (Reliability Curve)')

        ax2.set_xlabel("Mean predicted value")
        ax2.set_ylabel("Count")
        ax2.legend(loc="upper center", ncol=2)
        print(f'Probability Calibration Curves saved to {ARGS.out_directory}/calibration.png')
        plt.tight_layout()
        plt.savefig(f'{ARGS.out_directory}/calibration.png')


def lift(y_true, y_prob, ARGS):
    """Print Precision Recall Statistics and Graph"""
    prevalence = sum(y_true)/len(y_true)
    average_lift = average_precision_score(y_true, y_prob) / prevalence
    if ARGS.graphs:
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        lift_values = precision/prevalence
        plt.style.use('ggplot')
        plt.clf()
        plt.plot(recall, lift_values,
                 label='Lift-Recall Curve  (Area = %0.3f)' % average_lift)
        plt.xlabel('Recall: P(predicted+|true+)')
        plt.ylabel('Lift')
        plt.xlim([0.0, 1.0])
        plt.legend(loc="lower left")
        print(f'Lift-Recall Curve saved to {ARGS.out_directory}/lift.png')
        plt.savefig(f'{ARGS.out_directory}/lift.png')
    else:
        print('Average Lift %0.3f' % average_lift)


def roc(y_true, y_prob, ARGS):
    """Print ROC Statistics and Graph"""
    roc_auc = roc_auc_score(y_true, y_prob)
    if ARGS.graphs:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                 label='ROC curve (Area = %0.3f)'% roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (1 - Specifity)')
        plt.ylabel('True Positive Rate (Sensitivity)')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        print(f'ROC Curve saved to {ARGS.out_directory}/roc.png')
        plt.savefig(f'{ARGS.out_directory}/roc.png')
    else:
        print('ROC-AUC %0.3f' % roc_auc)
