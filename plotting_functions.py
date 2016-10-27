# standard imports
import itertools
import pandas as pd
import numpy as np

# image processing imports
import cv2

# classification metrics imports
from sklearn.metrics import roc_curve, roc_auc_score, auc, confusion_matrix

# plotting imports
import seaborn as sns
import matplotlib
from matplotlib import colors
from matplotlib import pyplot as plt
from matplotlib import cm as cm

def img_plots(fig_h, path_list, plot_title, label_df=None, x_label=None, variable_df=None):
    sns.set_style("white")
    fig, ax = plt.subplots(1,5,figsize=(16,fig_h))

    images_plot = []

    for img in path_list[:5]:
        image = cv2.imread(img)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images_plot.append(image_rgb)

    for i in range(5):
        plt.subplot(1,5,i+1)
        plt.xticks([], [])
        plt.yticks([], [])
        plt.imshow(images_plot[i])
        if not label_df is None:
            label = 'not pizza' if label_df.values[i] == 0 else 'pizza'
            plt.title(label, size=16)
        if not (x_label is None) and not (variable_df is None):
            plt.xlabel(x_label + "\n%.3f" % variable_df.values[i], size=14)
        plt.suptitle(plot_title, size=18)
    plt.show()

def epoch_plot(acc, val_acc, loss, val_loss):
    # A plot of accuracy on the training and validation datasets over training epochs.
    sns.set_style("dark")
    plt.figure(figsize=(20, 6))

    plt.subplot(1,2,1)
    plt.plot(acc)
    plt.plot(val_acc)
    plt.title('Model accuracy', size=18)
    plt.ylabel('Accuracy', size=16)
    plt.xlabel('Epoch', size=16)
    plt.legend(['Train', 'Test'], loc='upper left', fontsize = 14)
    plt.tick_params(axis='both', which='major', labelsize=13)

    # A plot of loss on the training and validation datasets over training epochs.
    plt.subplot(1,2,2)
    plt.plot(loss)
    plt.plot(val_loss)
    plt.title('Model loss', size=18)
    plt.ylabel('Loss', size=16)
    plt.xlabel('Epoch', size=16)
    plt.legend(['Train', 'Test'], loc='upper left', fontsize = 14)
    plt.tick_params(axis='both', which='major', labelsize=13)

    plt.show()

def corr(df, title):
    sns.set_style("dark")
    fig, ax = plt.subplots(figsize=(12,7))
    mask = np.zeros_like(df.corr(), dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    ax = sns.heatmap(df.corr(), mask=mask, annot=True)
    ax.set_xticklabels(ax.xaxis.get_ticklabels(), fontsize=14, rotation=70)
    ax.set_yticklabels(ax.yaxis.get_ticklabels(), fontsize=14, rotation=0)
    plt.suptitle(title, size=18)
    plt.show()


def color_space_plots(space1, color1, space1_label, space2, color2, space2_label, space3, color3, space3_label, title):
    sns.set_style("dark")
    fig, ax = plt.subplots(1,3,figsize=(18,4))
    sns.despine()

    sns.distplot(space1, bins=100, kde=False, hist_kws={"alpha": 1, "color": color1}, ax=ax[0])
    ax[0].set_xlabel(space1_label, size = 16, )

    sns.distplot(space2, bins=100, kde=False, hist_kws={"alpha": 1, "color": color2}, ax=ax[1])
    ax[1].set_xlabel(space2_label, size = 16)

    sns.distplot(space3, bins=100, kde=False, hist_kws={"alpha": 1, "color": color3}, ax=ax[2])
    ax[2].set_xlabel(space3_label, size = 16)

    plt.suptitle(title, size=18)
    plt.show()


def roc(actual, preds):
    sns.set_style("dark")

    fpr_, tpr_, _ = roc_curve(actual, preds)
    auc_ = auc(fpr_, tpr_)
    acc_ = np.abs(0.5 - np.mean(actual)) + 0.5

    fig, axr = plt.subplots(figsize=(8,7))

    axr.plot(fpr_, tpr_, label='ROC (area = %0.2f)' % auc_,
             color='darkred', linewidth=2,
             alpha=0.7)
    axr.plot([0, 1], [0, 1], color='grey', ls='dashed',
             alpha=0.9, linewidth=2, label='baseline accuracy = %0.2f' % acc_)

    axr.set_xlim([-0.05, 1.05])
    axr.set_ylim([0.0, 1.05])
    axr.set_xlabel('False positive rate', fontsize=14)
    axr.set_ylabel('True positive rate', fontsize=14)
    axr.set_title('Pizza vs. not pizza ROC curve\n', fontsize=16)

    axr.legend(loc="lower right", fontsize=12)

    plt.show()


def conf_matrix_plot(actual, predicted, classes, title=None):

    cm = confusion_matrix(actual, predicted)

    plt.figure(figsize=(4, 4))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    if not title is None:
        plt.title(title)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, size=14)
    plt.yticks(tick_marks, classes, size=14)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label', size=14)
    plt.xlabel('Predicted label', size=14)
    plt.show()
