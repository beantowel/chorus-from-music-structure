import keras.backend as KK
import math
from keras import backend as K
from keras.regularizers import l2
from keras.layers.advanced_activations import LeakyReLU, ReLU
from keras.models import model_from_json, Model, Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Dropout,\
    LSTM, Reshape, Bidirectional, TimeDistributed, Input, add



def ResNet_Block(input, block_id, filterNum):
    ''' Create a ResNet block
    Args:
        input: input tensor
        filterNum: number of output filters
    Returns: a keras tensor
    '''

    x = Conv2D(filterNum, (1, 1), name='conv_s'+str(block_id)+'_1x1', padding='same',
               kernel_initializer='he_normal', use_bias=False, kernel_regularizer=l2(1e-5))(input)
    shortcut = BatchNormalization()(x)
    x = LeakyReLU(0.01)(shortcut)

    x = Conv2D(filterNum, (3, 3), name='conv'+str(block_id)+'_1', padding='same',
               kernel_initializer='he_normal', use_bias=False, kernel_regularizer=l2(1e-5))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.01)(x)

    x = Conv2D(filterNum, (3, 3), name='conv'+str(block_id)+'_2', padding='same',
               kernel_initializer='he_normal', use_bias=False, kernel_regularizer=l2(1e-5))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.01)(x)

    x = Conv2D(filterNum, (1, 1), name='conv_f'+str(block_id)+'_1x1', padding='same',
               kernel_initializer='he_normal', use_bias=False, kernel_regularizer=l2(1e-5))(x)
    x = BatchNormalization()(x)

    x = add([x, shortcut])
    x = LeakyReLU(0.01)(x)
    x = MaxPooling2D((1, 4))(x)
    return x




def melody_ResNet():
    num_output = int(55 * 2 ** (math.log(8, 2)) + 2)
    input = Input(shape=(31, 513, 1))

    block_1 = ResNet_Block(input=input, block_id=1,
                             filterNum=64)
    block_2 = ResNet_Block(input=block_1, block_id=2,
                             filterNum=128)
    block_3 = ResNet_Block(input=block_2, block_id=3,
                             filterNum=192)
    block_4 = ResNet_Block(input=block_3, block_id=4,
                             filterNum=256)
    block_4 = Dropout(0.5)(block_4)

    numOutput_P = block_4.shape[2] * block_4.shape[3]
    output = Reshape((31, numOutput_P))(block_4)
    output = Bidirectional(
        LSTM(256, return_sequences=True, recurrent_dropout=0.3, dropout=0.3))(output)
    output = TimeDistributed(Dense(num_output))(output)
    output = TimeDistributed(Activation("softmax", name='softmax'))(output)


    model = Model(inputs=input, outputs=output)
    return model
