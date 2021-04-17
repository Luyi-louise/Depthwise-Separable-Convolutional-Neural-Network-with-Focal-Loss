from utils import BS,BASE_DIR
# from generator import csv_image_generator
from keras.models import load_model
from keras.metrics import categorical_accuracy
import sklearn
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from utils import BASE_DIR, encoding
import numpy as np
import cv2
import os
from utils import SAMPLE_DIR
import pickle
from model import define_model


encoding = {'Atrial_flutter': 1,
            'Atrial_fibrillation': 2,
            'Supraventricular_tachyarrhythmia': 3,
            'Pre-excitation_(WPW)': 4,
            'Idioventricular_rhythm': 5,
            'Second-degree_heart_block': 6,
            'Ventricular_flutter': 7,
            'Ventricular_tachycardia': 8,
            'Ventricular_trigeminy': 9,
            'Atrial_premature_beat': 10,
            'Pacemaker_rhythm': 11,
            'Premature_ventricular_contraction': 12,
            'Ventricular_bigeminy': 13,
            'FusionIdioventricular_of_ventricular_and_normal_beat': 14,
            'Left_bundle_branch_block_beat': 15,
            'Right_bundle_branch_block_beat': 16,
            'Normal_sinus_rhythm': 17}

model = define_model()
# model.load_weights('models-cnn3/012-0.007-0.999-0.002-0.999.hdf5')
# model.load_weights('models-fl-cnn1/017-0.010-0.988-0.001-0.997.hdf5')
# model.load_weights('models-dsc-cnn1/024-0.006-0.999-0.002-1.000.hdf5')
model.load_weights('models-alpha=0.25-gamma=3/026-0.008-0.988-0.001-0.997.hdf5')

with open('X_data.pkl', 'rb') as f:
    X_test = pickle.load(f)
f.close()
with open('y_label.pkl', 'rb') as f:
    y_test = pickle.load(f)
f.close()
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5)
idx = np.argmax(y_pred, axis=-1)
y_pred = np.zeros(y_pred.shape)
y_pred[np.arange(y_pred.shape[0]), idx] = 1
# acc = categorical_accuracy(y_test, y_pred)
y = []
for i in range(0, len(y_test)):
    # lab = [0, 0, 0, 0, 0, 0, 0, 0]
    lab = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    lab[encoding[y_test[i]] - 1] = 1
    y.append(lab)
acc = accuracy_score(np.array(y), y_pred)
print("Accuracy is: ", acc * 100)
recall = metrics.recall_score(np.array(y), y_pred, average='macro')
print("Recall is: ", recall * 100)
f1 = metrics.f1_score(np.array(y), y_pred, average='weighted')
print("f1 is: ", f1 * 100)
roc = roc_auc_score(np.array(y), y_pred)
print("ROC is: ", roc)
cfm = confusion_matrix(np.array(y).argmax(axis=1), y_pred.argmax(axis=1))
# 下两行为归一化（数值转化为小数点）
con_mat_norm = cfm.astype('float') / cfm.sum(axis=1)[:, np.newaxis]     # 归一化
con_mat_norm = np.around(con_mat_norm, decimals=2)
print("confusion_matrix is :\n", con_mat_norm)

target_names = ['1',  # 324
                '2',  # 5
                '3',  # 14
                '4',  # 3
                '5',  # 13
                '6',  # 1
                '7',  # 895
                '8',  # 909
                '9',  # 2
                '10',  # 28
                '11',  # 4
                '12',  # 102
                '13',  # 1029
                '14',
                '15',
                '16',
                '17']  # 9563
print(classification_report(np.array(y), y_pred, target_names=target_names))


def plot_confusion_matrix(confusion_matrix, target_names, title):
    # cm = sklearn.preprocessing.normalize(cfm, norm='l1', axis=1, copy=True)
    cm = sklearn.preprocessing.normalize(cfm, norm='l1', axis=1, copy=True)
    matplotlib.rcParams['figure.figsize'] = (8, 7)
    plt.pcolor(np.flipud(cm), cmap="Blues")
    cbar = plt.colorbar()
    # cbar.ax.tick_params(labelsize=12)
    plt.title(title)  # 图像标题
    num_local = np.array(range(len(target_names)))
    plt.xticks(num_local + .5, target_names, rotation=90, fontsize=12)
    plt.yticks(num_local + .5, reversed(target_names), fontsize=12)
    plt.clim(0, 1)
    plt.ylabel("True label", fontsize=12)
    plt.xlabel("Predicted lable", fontsize=12)
    plt.title("Confusion Matrix of DSC-FL-CNN", fontsize=16)
    plt.tight_layout()

plot_confusion_matrix(cfm, target_names, "Confusion Matrix")
plt.show()