# Import packages
import pathlib
import numpy as np
from keras.models import Model
from keras.layers import (Dense, GRU, LSTM, Embedding, Input, 
                            concatenate, Dropout, Masking, BatchNormalization)
from keras.regularizers import l2, l1, l1_l2
from keras.optimizers import Adadelta
from utilities import open_pickle, format_data


# Dir to example data:
PATH_EXAMPLE_DATA = pathlib.Path.cwd().joinpath('example_data')

# Dir to helper functions and files:
PATH_UTILITIES = pathlib.Path.cwd().joinpath('utilities')

# Load label_encoder (also event_encoder)
label_encoder = open_pickle(
    path=PATH_UTILITIES, 
    filename='event_encoder.pkl')

# Prepare inputs for the model
# Shape is: (number of patients, 301, input_dim)

# x_event data of shape: (number of patients, 301) 
# will be embedded in the model to (number of patients, 301, embedding_dim)

# Time dimension data (time in days between descending ordered visits) of 
# shape: (number of patients, 301, 1)

# Prepare demographics for each patient that are NOT changing over time of 
# shape: (number of patients, 301, 15)

# Prepare demographics that are changing over time, 
# ensure that they match with the timestemps of the test data (medical_events)
# one-hot-encode the variable demographic features and process to 
# shape: (number of patients, 301, 92)


(x_event, x_time_last_event, 
    x_static_demo, x_variable_demo, y)=format_data(path_example_data=PATH_EXAMPLE_DATA,
                                                    path_utilities=PATH_UTILITIES)
					   

def model_get(embedding_dim=30, rnn_type='GRU', lstm_dim=5, 
    lstm_activation='tanh', lstm_dropout=.1, lstm_recurrent_dropout=.05, 
    embedding_activity_regularizer_l1=.0, lstm_kernel_regularizer_l1=.0001,
    lstm_kernel_regularizer_l2=.0, lstm_activity_regularizer=.00001, 
    output_kernel_regularizer_l1=.0001, label_encoder=label_encoder):
    '''Function to initiate DETECT (many-to-many model) architecture and parameters'''

    
    # Define input data
    max_len_w_fed=301
    event_input=Input(shape=(max_len_w_fed, ), name='input_event')
    embedding_input_dim=len(label_encoder.classes_) + 1

    demographic_variable_input=Input(
        shape=(max_len_w_fed, x_variable_demo.shape[2]), 
        dtype='float32', 
        name='input_var_demo')

    demographic_input=Input(
        shape=(max_len_w_fed, x_static_demo.shape[2]), 
        dtype='float32', 
        name='input_static_demo') #many-to-many

    time_layer=Input(
        shape=(max_len_w_fed, 1), 
        dtype='float32', 
        name='input_time')

    # Takes the encoded sequence of integers representing events and 
    # returns a sequence of dense vectors of size 
	# output_dim that is a Eucledian representation of the event 
    # (two similar events will have similar vectors)
    embed=Embedding(
        input_dim=embedding_input_dim, 
        output_dim=embedding_dim, 
        mask_zero=True,
        activity_regularizer=l1(embedding_activity_regularizer_l1), 
            name='embed')(event_input)
    
    # Add the time since last event as the last element of the embedding vector
    concat_1=concatenate(
        inputs=[embed, time_layer, demographic_variable_input], 
        name='concat_1') 
	# Mask zeros
    mask_1=Masking(
        mask_value=0.0, 
        name='mask_1')(concat_1)
    
    # RNN layer
    if rnn_type=='LSTM':
        rnn=LSTM(lstm_dim, 
            activation=lstm_activation, dropout=lstm_dropout, 
            recurrent_dropout=lstm_recurrent_dropout, return_sequences=True, 
            kernel_regularizer=l1_l2(l1=lstm_kernel_regularizer_l1, 
            l2=lstm_kernel_regularizer_l2), 
            activity_regularizer=l2(lstm_activity_regularizer), name='rnn')(mask_1)
    else:
        rnn=GRU(lstm_dim, activation=lstm_activation, dropout=lstm_dropout, 
            recurrent_dropout=lstm_recurrent_dropout, return_sequences=True, 
            kernel_regularizer =l1_l2(l1=lstm_kernel_regularizer_l1, 
            l2=lstm_kernel_regularizer_l2), 
            activity_regularizer=l2(lstm_activity_regularizer), name='rnn')(mask_1)

    
    # Add demographic info to the output of the RNN
    concat_2=concatenate(
        inputs=[rnn, demographic_input], 
        name='concat_2')
                    
    # Output
    dense_3=Dense(
        1, 
        activation='sigmoid',
        kernel_regularizer=l1(output_kernel_regularizer_l1), 
            name='dense_3')(concat_2)
    
    # Create model
    model=Model(
        [event_input, time_layer, 
        demographic_variable_input, demographic_input], 
        dense_3)
    
    # Optimizer
    opti=Adadelta()
    
    # Compile and fit  
    model.compile(
        loss='binary_crossentropy', 
        optimizer=opti, 
        metrics=['accuracy'],
        sample_weight_mode='temporal') 
				 
    print(model.summary())
    return model


def model_fit(model=None, batch_size=16, epochs=10):
    '''Function for fitting the model (from model_get())''' 

    max_len_w_fed=301    
	# Construct sample weights that only weights the last observation in loss and accuracy
    weights=np.zeros(shape=(x_event.shape[0] ,max_len_w_fed))
    for l in weights:
        l[-1]=1
    
       
    hist=model.fit(
        x=[x_event, x_time_last_event, x_variable_demo, x_static_demo],
        y=y,
        epochs=epochs, 
        verbose=1, 
        batch_size=batch_size,
        sample_weight=weights)

    # Print accuracy and loss on training
    print('Trainings Accuracy: {}'.format(hist.history['acc'][-1]))
    print('Final Train Loss {0}:'.format(hist.history['loss'][-1]))
    
    return model


# Execution of the functions
model_init = model_get()
model_trained = model_fit(
    model=model_init, 
    batch_size=16, 
    epochs=100)