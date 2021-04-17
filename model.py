from keras import Sequential, optimizers
from keras.layers import Conv2D, BatchNormalization, MaxPool2D, Flatten, Dense, Dropout, ELU, SeparableConv2D
# from focal_loss import focal_loss
from focal_loss_cate import catergorical_focal_loss

# def define_model():
#     model = Sequential()
#     model.add(Conv2D(64, (3, 3), strides=(1, 1), input_shape=(128, 128, 3), kernel_initializer='glorot_uniform', data_format="channels_last"))
#     model.add(ELU())
#     model.add(BatchNormalization())
#     model.add(Conv2D(64, (3, 3), strides=(1, 1), kernel_initializer='glorot_uniform'))
#     model.add(ELU())
#     model.add(BatchNormalization())
#     model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
#     model.add(Conv2D(128, (3, 3), strides=(1, 1), kernel_initializer='glorot_uniform'))
#     model.add(ELU())
#     model.add(BatchNormalization())
#     model.add(Conv2D(128, (3, 3), strides=(1, 1), kernel_initializer='glorot_uniform'))
#     model.add(ELU())
#     model.add(BatchNormalization())
#     model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
#     model.add(Conv2D(256, (3, 3), strides=(1, 1), kernel_initializer='glorot_uniform'))
#     model.add(ELU())
#     model.add(BatchNormalization())
#     model.add(Conv2D(256, (3, 3), strides=(1, 1), kernel_initializer='glorot_uniform'))
#     model.add(ELU())
#     model.add(BatchNormalization())
#     model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
#     model.add(Flatten())
#     model.add(Dense(2048))
#     model.add(ELU())
#     model.add(BatchNormalization())
#     model.add(Dropout(0.5))
#     model.add(Dense(17, activation='softmax'))
#     # model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])  # optimizer='adam'
#     # model.compile(optimizer='sgd', loss=[focal_loss(classes_num)], metrics=['accuracy'])
#     model.compile(optimizer='sgd', loss=[catergorical_focal_loss(gamma=2.0, alpha=0.25)], metrics=['accuracy'])
#     return model


def define_model():
    model = Sequential()
    model.add(SeparableConv2D(64, (3, 3), strides=(1, 1), input_shape=(128, 128, 3), kernel_initializer='glorot_uniform', data_format="channels_last"))
    model.add(ELU())
    model.add(BatchNormalization())
    model.add(SeparableConv2D(64, (3, 3), strides=(1, 1), kernel_initializer='glorot_uniform'))
    model.add(ELU())
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(SeparableConv2D(128, (3, 3), strides=(1, 1), kernel_initializer='glorot_uniform'))
    model.add(ELU())
    model.add(BatchNormalization())
    model.add(SeparableConv2D(128, (3, 3), strides=(1, 1), kernel_initializer='glorot_uniform'))
    model.add(ELU())
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(SeparableConv2D(256, (3, 3), strides=(1, 1), kernel_initializer='glorot_uniform'))
    model.add(ELU())
    model.add(BatchNormalization())
    model.add(SeparableConv2D(256, (3, 3), strides=(1, 1), kernel_initializer='glorot_uniform'))
    model.add(ELU())
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Flatten())
    model.add(Dense(2048))
    model.add(ELU())
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(17, activation='softmax'))
    # model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
    model.compile(optimizer='sgd', loss=[catergorical_focal_loss(gamma=3, alpha=0.25)], metrics=['accuracy'])
    return model
