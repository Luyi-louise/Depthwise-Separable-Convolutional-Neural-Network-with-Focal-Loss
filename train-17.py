from utils import TRAIN_CSV, TEST_CSV, BS, NUM_EPOCHS, BASE_DIR, SAMPLE_DIR#,encoding
from model import define_model
from sklearn.model_selection import StratifiedKFold
import keras
from keras.callbacks import ModelCheckpoint
import os
import cv2
import tensorflow as tf
import numpy as np
import random
from sklearn.metrics import accuracy_score
import statistics
import pickle
import matplotlib.pyplot as plt

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

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()

session = tf.Session(config=config)

def plot(history, filename):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(filename+'_acc')
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show(filename+'_loss')

# 转换训练数据
X = []
y = []
sample_dir = os.path.join(BASE_DIR, 'samples')
list_samples = os.listdir(sample_dir)   #读出路径下每一个文件夹名
print(list_samples)

for sam in list_samples:
    print("Loading images for "+sam)
    im_dir = os.path.join(sample_dir, sam)
    list_imgs = os.listdir(im_dir)
    for i in list_imgs:
        image = cv2.imread(os.path.join(im_dir,i))
        if type(X) == list:
            X.append(image)
        else:
            np.append(X, [image], axis=0)
        y.append(sam)

combined = list(zip(X, y))
random.shuffle(combined)
X[:], y[:] = zip(*combined)
# skf = StratifiedKFold(n_splits=10, shuffle=True)
# skf.get_n_splits(X, y)
X_train = np.array(X)
y_train = np.array(y)

# 转换验证数据
X = []
y = []
sample_dir = os.path.join(BASE_DIR, 'samples-val')
list_samples = os.listdir(sample_dir)   #读出路径下每一个文件夹名
for sam in list_samples:
    print("Loading images for "+sam)
    im_dir = os.path.join(sample_dir, sam)
    list_imgs = os.listdir(im_dir)
    for i in list_imgs:
        image = cv2.imread(os.path.join(im_dir, i))
        if type(X) == list:
            X.append(image)
        else:
            np.append(X, [image], axis=0)
        y.append(sam)

combined = list(zip(X, y))
random.shuffle(combined)
X[:], y[:] = zip(*combined)
# skf = StratifiedKFold(n_splits=10, shuffle=True)
# skf.get_n_splits(X, y)
X_test = np.array(X)
y_test = np.array(y)

accs = []

model = None
model = define_model()   # 8分类网络、新型网络、深度分离网络
model.summary()

with open("X_train"+".pkl", 'wb') as f:
    pickle.dump(X_train, f, protocol=4)
f.close()
with open("X_test"+".pkl", 'wb') as f:
    pickle.dump(X_test, f, protocol=4)
f.close()
with open("y_train"+".pkl", 'wb') as f:
    pickle.dump(y_train, f, protocol=4)
f.close()
with open("y_test"+".pkl", 'wb') as f:
    pickle.dump(y_test, f, protocol=4)
f.close()

def get_filename_for_saving(save_dir):
    return os.path.join(save_dir,
            "{epoch:03d}-{val_loss:.3f}-{val_acc:.3f}-{loss:.3f}-{acc:.3f}.hdf5")

SAVE_DIR = os.path.join(BASE_DIR,'models')
stopping = keras.callbacks.EarlyStopping(patience=6)
reduce_lr = keras.callbacks.ReduceLROnPlateau(
    factor=0.9,
    patience=2,
    min_lr=0.001 * 0.001)
checkpoint = ModelCheckpoint(filepath=get_filename_for_saving(SAVE_DIR), monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint, reduce_lr, stopping]

# Debug message I guess
print("Training new iteration on " + str(X_train.shape[0]) + " training samples, " + str(X_test.shape[0]) + " validation samples, this may be a while...")
y_tr = []
y_te = []
for i in range(0, len(y_train)):
    lab = [0, 0, 0, 0, 0, 0, 0, 0]
    lab[encoding[y_train[i]] - 1] = 1
    y_tr.append(lab)
for i in range(0, len(y_test)):
    lab[encoding[y_test[i]] - 1] = 1
    y_te.append(lab)
print('y_train:', y_train.shape)

y_train = np.array(y_tr)
y_test = np.array(y_te)
X_train = np.array(X_train)
X_test = np.array(X_test)

history = model.fit(X_train, y_train,
                    batch_size=32,
                    validation_data=(X_test, y_test),
                    epochs=50,
                    callbacks=callbacks_list)
eval_model = model.evaluate(X_test, y_test, verbose=1)
print("%s: %.2f%%" % (model.metrics_names[1], eval_model[1]*100))
print("%s: %.2f%%" % (model.metrics_names[0], eval_model[0]))
accs.append(eval_model[1]*100)
# plot(history= history, filename="plot")
# model.save_weights('model'+str()+'.h5')
with open('results.txt', 'a+') as f:
    f.write(" Loss is: "+str(eval_model[0])+" Accuracy is: " + str(eval_model[1] * 100) +'\n')
f.close()

print("List of accuracies: ", accs)
print("Final accuracy: ", str(statistics.mean(accs)))
with open('results.txt', 'a+') as f:
    f.write("Final List of accuracies: " + str(accs) +"Final Mean accuracy: "+str(statistics.mean(accs)) + '\n')
f.close()
