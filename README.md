# Distributed_Learning_for_Automatic_Modulation_Recognition

This repository presents two distributed methods for Automatic Modulation Recognition (AMR). These methods include consensus voting (DAMR-V) and feature sharing (DAMR-F) approaches as detailed in the following article. Please use the following paper for citation:  
[Distributed learning for automatic modulation recognition in bandwidth-limited networks]()

In this study, we utilized the TeMuRAMRD.2023 (Terrain-driven Multi-Receiver Automatic Modulation Recognition Dataset) as a multi-receiver dataset. This dataset has the maximum number of 6 receivers.

To do the training process for different methods, follow these guidelines:   
CentAMR Method:  
-For the CentAMR method, which includes training based on datast from 6 receivers, initiate the training process by running the train file with 'nrx' set to 6.  
Distributed Learning Methods:   
Begin by training the DAMR-V model, which involves training six individual models based on VTCNN2-1D architecture, and then integration of probabilistic predictions.  
For the feature sharing step (DAMR-F8), execute the train file with the following configurations:
Set 'model' to 'DAMR_F8', 'nrx' to 6, and provide the 'checkpoint_path' corresponing to the six trained models from previous step.

