# DETECT
Recurrent neural network model for Dynamic ElecTronic hEalth reCord deTection (DETECT) of individuals at risk of a first-episode of psychosis


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
<img src="https://i.imgur.com/e9xG3Vi.png" width="1000">
</p>

The second component is a memory module that stores an internal representation of the event history of the invidiual patient. This is needed for creating a unified representation of the patient risk that should not depend too much on number of observed events or follow-up time. The module uses the [gated recurrent unit](https://en.wikipedia.org/wiki/Gated_recurrent_unit) architecture.

The third and final component is a prediction module which consists of a fully connected layer that combine the information from the memory module and other demographic data to a single number representing the predicted risk of the individual at the given time point. 
