import pickle
import pathlib
import numpy as np
import pandas as pd
from keras.preprocessing import sequence

def format_data(path_example_data=None, path_utilities=None):
    '''Wrapper to format example_data'''
    
    # Events
    dateparser = lambda x: pd.datetime.strptime(x, "%m/%d/%Y")
    event_df = pd.read_csv(
        path_example_data.joinpath('event_data.csv'),
        parse_dates=['event_date'], date_parser=dateparser)

    event_encoder = open_pickle(
        path=path_utilities, 
        filename='event_encoder.pkl')
    

    # Static demographics
    static_demo_df = pd.read_csv(
        path_example_data.joinpath('static_demographic_data.csv'))

    static_demo_encoder = open_pickle(
        path=path_utilities, 
        filename='static_demo_scaler.pkl')

    static_demo_df_cols = list(static_demo_df.columns)
    static_demo_df_cols.remove('patient_id')
    

    # Variable demographics
    variable_demo_df = pd.read_csv(
        path_example_data.joinpath('variable_demographic_data.csv'))

    variable_demo_encoder = open_pickle(
        path=path_utilities, 
        filename='variable_demo_scaler.pkl')

    variable_demo_df_cols = list(variable_demo_df.columns)
    variable_demo_df_cols.remove('patient_id')
    

    # Encode and scale data
    event_df['int_event_encoded']=event_encoder.transform(
        event_df.event_desc.values)
    event_df['int_event_encoded']=[encoding + 1 for 
        encoding in event_df['int_event_encoded']]
    
    static_demo_df_encoded=static_demo_encoder.transform(
        static_demo_df[static_demo_df_cols])

    variable_demo_df_encoded=pd.DataFrame(
        variable_demo_encoder.transform(variable_demo_df[variable_demo_df_cols]), 
        columns=variable_demo_df_cols)

    variable_demo_df_encoded=pd.concat(
        [variable_demo_df['patient_id'], variable_demo_df_encoded], 
        axis=1)
    
    
    # Reshape the encoded data to list of lists (batch_size, timesteps, input_dim)
    x_event=to_seq_by_patient_id(
        event_df, 
        column='int_event_encoded')

    y=to_seq_by_patient_id(
        event_df,
        column='outcome')

    x_time_last_event=to_seq_by_patient_id(
        event_df, 
        column='time_since_last_event')
    
    x_static_demo=static_demo_df_encoded.reshape(
        -1, 1, static_demo_df_encoded.shape[1])

    x_variable_demo=to_3d_from_2d(
        variable_demo_df_encoded,
        columns_to_keep=variable_demo_df_cols,
        patient_id_col='patient_id')


    # Padding and final transformation of the data
    x_event_pad=sequence.pad_sequences(
        x_event, 
        dtype='float32', 
        maxlen=301)
    
    y_pad = sequence.pad_sequences(
        y, 
        dtype='float32', 
        maxlen=301)
    y_pad = np.array(y_pad).reshape(-1, 301, 1)

    x_time_last_event_pad=sequence.pad_sequences(
        x_time_last_event, 
        dtype='float32', 
        maxlen=301)

    x_time_last_event_pad=x_time_last_event_pad.reshape(-1, 301, 1)
    
    x_static_demo_pad=np.broadcast_to(
        x_static_demo, 
        (x_static_demo.shape[0], 301, x_static_demo.shape[2]))

    x_variable_demo_pad=sequence.pad_sequences(
        x_variable_demo, 
        dtype='float32', 
        maxlen=301)

    return (x_event_pad, x_time_last_event_pad, x_static_demo_pad, x_variable_demo_pad, y_pad)
    
    
# Function for loading pickels of encoders and scalers
def open_pickle(path=None, filename=None):
    '''Function for loading pickle files'''

    try:
        path=pathlib.Path(path)
        path_full=path.joinpath(filename)
    except:
        path_full=path + filename

    with open(path.joinpath(filename), 'rb') as f:
        return pickle.load(f)  

    
def to_seq_by_patient_id(df, column, group_on='patient_id'):
    '''
    Function to return list of lists
        
        Input:
        patient id | outcome
        A             1
        A             0
        A             0
        B             1
        
        returns [[1, 0, 0], [1]]
    '''


    if 'event_date' in df.columns and 'time_since_last_event' in df.columns:
        df = df.sort_values(['patient_id', 'event_date', 'time_since_last_event'])
    else:
        df = df.sort_values(group_on)
    
    df = df[['patient_id', column]]
    keys, values = df.values.T
    ukeys, index = np.unique(keys, return_index=True)
    arrays = np.split(values, index[1:])
    return [list(a) for a in arrays]


def to_3d_from_2d(df, columns_to_keep, patient_id_col = 'patient_id'):
    '''
    Function to get for each patient at each timestep (event) 
    the variable demographics into a list of lists
    '''

    formatted_3d = [[list(row[columns_to_keep].values) for index,row in df[df[patient_id_col]==patient].iterrows()] \
                        for patient in df[patient_id_col].unique()]
    return formatted_3d

    
def decoder(patient, pad_value = 0, label_encoder = None):
    '''Inverse encoding function. Takes integer input returns event'''

    
    return label_encoder.inverse_transform(patient[patient != pad_value] - 1)