# Import packages
import pathlib
from keras.models import Model, load_model
from utilities import format_data

# Dir to example data:
PATH_EXAMPLE_DATA = pathlib.Path.cwd().joinpath('example_data')

# Dir to helper functions and files:
PATH_UTILITIES = pathlib.Path.cwd().joinpath('utilities')

# Dir to trained model:
TRAINED_MODEL_FILENAME = 'DETECT_trained.h5'

# Load model
model = load_model(TRAINED_MODEL_FILENAME)
print('Loaded successfully model')
print(model.summary())


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
    x_static_demo, x_variable_demo, _)=format_data(path_example_data=PATH_EXAMPLE_DATA,
                                                    path_utilities=PATH_UTILITIES)

# Use the model for predictions
x_event_predictions = model.predict([
    x_event, 
    x_time_last_event, 
    x_variable_demo,
    x_static_demo])

# prediction probabilities
x_event_predictions = x_event_predictions.flatten()

# find where the padding is
paddings = x_event.flatten() != 0

# remove padding from predictions
x_event_predictions_no_pad = x_event_predictions[paddings]
print(f'Non-padded predictions flattend: {x_event_predictions_no_pad}')