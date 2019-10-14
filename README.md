# DETECT
Recurrent neural network model for Dynamic ElecTronic hEalth reCord deTection (DETECT) of individuals at risk of a first-episode of psychosis


Model structure
--------------------
DETECT a recurrent neural network model built for predicting risk of future psychosis. It was built and trained using de-identified electronic health record data from the [IBM Explorys database](https://www.ibm.com/marketplace/explorys-ehr-data-analysis-tools).

DETECT is a time-dynamic model that uses demographic data combined with medical events in the categories diagnoses, drug prescriptions, procedures, encounters and admissions, observations and laboratory test results. Every time a new medical event is entered in the EHR, the predicted probability is updated. The basic structure of the model is illustrated below.

<p align="center">
<img src="https://i.imgur.com/7pq5IXG.png" width="640">
</p>
