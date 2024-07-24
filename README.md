# Distributed_Learning_for_Automatic_Modulation_Recognition

This repository presents two distributed methods for Automatic Modulation Recognition (AMR). These methods include consensus voting (DAMR-V) and feature sharing (DAMR-F) approaches as detailed in the following article. Please use the following paper for citation:  
[Distributed learning for automatic modulation recognition in bandwidth-limited networks](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/13057/130570X/Distributed-learning-for-automatic-modulation-recognition-in-bandwidth-limited-networks/10.1117/12.3013532.short)

In this study, we utilized the TeMuRAMRD.2023 (Terrain-driven Multi-Receiver Automatic Modulation Recognition Dataset) as a multi-receiver dataset. This dataset has the maximum number of 6 receivers.

To do the training process and replicate the results for different methods, follow these guidelines:   
- For local reeiver, conduct training individually for each receiver by setting the model parameter in the training file to 'VTCNN2-1D' and 'nrx' to 1.
-For the CentAMR method, which includes training based on datast from 6 receivers using the 'VTCNN2-1D' model , initiate the training process by executing the train file with the model argument set to 'VTCNN2-1D' and 'nrx' set to 6.
-For the distributed learning methods, begin with training the DAMR-V model, which involves training six individual models based on VTCNN2-1D architecture, followed by the integration of probabilistic predictions. To proceed, set 'nrx' to 6 and 'model' to 'DAMR-V' in the training file. 
 -Next, for the feature sharing step (DAMR-F8), execute the training file with the following configurations: set 'model' to 'DAMR-F8', 'nrx' to 6, and specify the 'checkpoint_path' corresponding to the six trained models from the previous step.
