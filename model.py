import tensorflow as tf 

from tensorflow.keras.layers import Dense,Dropout, Embedding,LSTM, Bidirectional, SpatialDropout1D

from tensorflow.keras import Model, Input 

def my_model(inp_shape,out_shape):

    inp = Input(shape=(inp_shape))
    i
    dense_1 = Dense(128,activation='relu')(inp)
    dropout_1 = Dropout(rate=0.5)
    dense_2 = Dense(64,activation='relu')(dense_1)
    dropout_2 = Dropout(rate=0.5)
    out = Dense(out_shape,activation='softmax')(dense_2)
    
    model = Model(inputs=inp,outputs=out)

    return model