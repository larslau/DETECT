# DETECT
Recurrent neural network model for Dynamic ElecTronic hEalth reCord deTection (DETECT) of individuals at risk of a first-episode of psychosis

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3701836.svg)](https://doi.org/10.5281/zenodo.3701836)


If you use this code or data, please cite

Raket, Lars Lau, Jörn Jaskolowski, Bruce J. Kinon, Jens Christian Brasen, Linus Jönsson, Allan Wehnert, and Paolo Fusar-Poli. "Dynamic ElecTronic hEalth reCord deTection (DETECT) of individuals at risk of a first episode of psychosis: a case-control development and validation study." *The Lancet Digital Health* (2020). [DOI: 10.1016/S2589-7500(20)30024-8](https://doi.org/10.1016/S2589-7500(20)30024-8)


Model structure
--------------------
DETECT a recurrent neural network model built for predicting risk of future psychosis. It was built and trained using de-identified electronic health record data from the [IBM Explorys database](https://www.ibm.com/marketplace/explorys-ehr-data-analysis-tools).

DETECT is a time-dynamic model that uses demographic data combined with medical events in the categories diagnoses, drug prescriptions, procedures, encounters and admissions, observations and laboratory test results. Every time a new medical event is entered in the EHR, the predicted probability is updated. The basic model structure is illustrated below.

<p align="center">
<img src="https://i.imgur.com/7pq5IXG.png" width="1000">
</p>

Model components
--------------------

The first component in the model is an embedding that maps the medical events to a 30-dimensional vector space representation

<p align="center">
<img src="https://i.imgur.com/Rtrlos6.png" width="1000">
</p>

The second component is a memory module that stores an internal representation of the event history of the invidiual patient. This is needed for creating a unified representation of the patient risk that should not depend too much on number of observed events or follow-up time. The module uses the [gated recurrent unit](https://en.wikipedia.org/wiki/Gated_recurrent_unit) architecture.

The third and final component is a prediction module which consists of a fully connected layer that combine the information from the memory module and other demographic data to a single number representing the predicted risk of the individual at the given time point. 

Code structure
-------------------------------
Requirements:
*Create a virtual environment in Python 3.6.7 and install the requirements in* `requirements.txt`

- `example_data` contains artificial patient data to illustrate the data structure
  - `all_medical_events.csv` all eligible medical events that DETECT can take as input
  - `event_data.csv` medical events for artificial patients
  - `static_demographic_data.csv` static demographic data for artificial patients. This file should contain one line per `patient_id`. Note that the variable `std_gender` codes male as `1` and female as `2`
  - `variable_demographic_data.csv` static demographics in addition to demographic data that can change over time for artificial patients. This file should have one line per medical event representing the known demographic variables of the patient at the time of the medical event 
- `utilities` contains utilities for reshaping and scaling data from the format presented in `example_data` to a format suitable for giving as input to DETECT
- `DETECT_architecture.py` code for making and training an RNN model with the same architecture as DETECT 
- `DETECT_prediction_example.py` code for predicting the probability of first-episode psychosis at every event for the artificial patients in example_data using DETECT
- `DETECT_trained.h5` the trained DETECT model object

Code output
-------------------------------
`DETECT_prediction_example.py` will output predicted probabilities for future first-episode psychosis (FEP) for the artificial patient data in  `example_data`. The dynamic predictions of the first artificial patient are illustrated in the patient journey plot below. Note that the predicted FEP probabilities are in the context of a 1:1 matched case-control study where the prior risk for FEP is 0.5. In non-matched studies, the predicted probabilities should be adjusted based on prevalence estimates.

<p align="center">
<img src="https://i.imgur.com/HiVrG2L.png" width="1000">
</p>

License
-------------------------------
Copyright (c) IBM Corporation 2020

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
   
Licenses to standards or ontologies may be governed by separate terms 
depending on how the end user of the code intends to use or modify it.
