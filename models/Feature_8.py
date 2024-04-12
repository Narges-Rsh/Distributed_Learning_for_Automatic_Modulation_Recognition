from models import *
import torch
import os
import numpy as np
import torch.nn as nn
from typing import List
from models import CNN1
from models import Voting
import torchmetrics
from .ModelBase import ModelBase


class Classifier(nn.Module):
    def __init__(
        self,
        input_width: int,
        n_classes: List[str],
        **kwargs
    ):
        super().__init__(**kwargs)

        self.model = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(input_width, 128),
                    nn.ReLU(),
                    nn.Linear(128,64),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(64,n_classes)
                )


    def forward(self, x: torch.Tensor):
        return self.model(x)

class Feature_8 (ModelBase):

    def __init__(self,
                  classes: List[str], 
                  input_samples: int, 
                  input_channels:int,
                  learning_rate:float,
                  feature_width: int = 8,
                  #checkpoint_path="/home/distributed_learning/logs/Voting-SimpleHDF5DataModule/version_0/checkpoints/epoch=2-step=37800.ckpt",
                    **kwargs):
        
        super().__init__(classes=classes, **kwargs)
        map_location='cuda:3'
        #Voting_model=Voting.load_from_checkpoint(checkpoint_path,map_location=map_location)
        Voting_model=Voting.load_from_checkpoint("/home/distributed_learning/logs/Voting-SimpleHDF5DataModule/version_0/checkpoints/epoch=2-step=37800.ckpt",map_location=map_location)


        Voting_model.eval()     

        #print("Voting_model", Voting_model)  

        self.loaded_models = Voting_model.fe_models
        for i in range(input_channels):
            self.loaded_models[i].eval()
            for param in self.loaded_models[i].parameters():
                param.requires_grad = False

        self.lr = learning_rate
        self.nrx = input_channels
        self.loss = nn.CrossEntropyLoss()
        self.feature_width=feature_width

        self.automatic_optimization = False

        self.classifier_model = Classifier(feature_width*input_channels, len(classes))

        self.example=torch.zeros((1,input_channels,input_samples), dtype=torch.float) 


        metrics = {
            f'F1':torchmetrics.classification.MulticlassF1Score(num_classes=len(classes))
        }
        metrics = torchmetrics.MetricCollection(metrics)
        
        self.val_metrics = metrics.clone(f"val/")
        self.test_metrics = metrics.clone(f"test/")

        self.cm_metric = torchmetrics.classification.MulticlassConfusionMatrix(len(classes), normalize='true')


    def forward(self, x: torch.Tensor):
        print("input shape", x.shape)
        f = []
        for i in range(self.nrx):
            
            out, features = self.loaded_models[i](x[:,i:i+1])
            f.append(out)    

        f = torch.stack(f, -1)              
        f = torch.flatten(f, start_dim=1)   
        y = self.classifier_model(f)         

        return y
        

    def configure_optimizers(self):
        opts= torch.optim.AdamW(self.classifier_model.parameters(), lr=self.lr, weight_decay=0.00001)
        
        return opts
    

    def training_step(self, batch, batch_nb):
        data, target,_ =batch
        classifier_opt=self.optimizers()


        with torch.no_grad():   

            f = []
            for i, fe in enumerate(self.loaded_models):  
                
                y, features = fe(data[:,i:i+1])
                f.append(y)   


        #classifier training
        self.toggle_optimizer(classifier_opt)
        f = torch.stack(f, -1)
        f = torch.flatten(f, start_dim=1)
        y = self.classifier_model(f)
             


        # calculate loss for clasifier
        loss = self.loss(y, target)
        if self.global_step!= 0: self.logger.log_metrics({'train/classifier_loss': loss, 'epoch': self.current_epoch}, self.global_step)
        self.log("class_loss", loss, prog_bar=True, logger=False) 


        # Optimize and Backprop
        self.manual_backward(loss)
        classifier_opt.step()
        classifier_opt.zero_grad()
        self.untoggle_optimizer(classifier_opt)


        # Validation step
    def validation_step(self, batch, batch_nb):
        data, target, snr = batch


        # Full classifier pass
        output = self.forward(data)
        self.val_metrics['F1'].update(output, target)
        self.cm_metric.update(output, target)
                                   
        
    # Called each test set step
    def test_step(self, batch, batch_nb):
        data, target, snr = batch
        
        # Full classifier pass
        output = self.forward(data)
        self.test_metrics['F1'].update(output, target)   
        self.cm_metric.update(output, target)    

        
        batch_size = len(snr)    
        batch_idx = batch_nb*batch_size    
        self.outputs_list[batch_idx:batch_idx+batch_size] = output.detach().cpu()    


        snr = snr.squeeze(dim=-1)    
        self.snr_list[batch_idx:batch_idx+batch_size] = snr.detach().cpu()   
        
       
        #extract outputs from each model
        for i, model in enumerate(self.loaded_models ):  
            y, _ = model(data[:,i:i+1])    
            self.test_metrics[f'F1_fe{i}'].update(y, target)  
















    


    



        
    


