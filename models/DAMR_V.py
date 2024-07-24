from typing import List
import torchmetrics
import torch
import torch.nn as nn
from .ModelBase import ModelBase

class VTCNN2_1D(nn.Module):
    
    def __init__(
        self,
        input_samples: int,
        input_channels: int,
        feature_width: int,
        n_classes: int,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.model = nn.Sequential(
            nn.Conv1d(in_channels=2*input_channels, out_channels=256, kernel_size=7, padding=3, bias=False),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Conv1d(in_channels=256, out_channels=80, kernel_size=7, padding=3, bias=True),
            nn.ReLU(),
            nn.BatchNorm1d(80),
            nn.Flatten(),
            nn.Linear(80 * 1 * input_samples, feature_width),
            nn.ReLU(),
            nn.BatchNorm1d(feature_width),

        )

        self.classifier = nn.Linear(feature_width, n_classes)

    
    def forward(self, x: torch.Tensor):
        x = torch.view_as_real(x)
        x = x.transpose(-2,-1).flatten(1,2).contiguous()
        features = self.model(x)
        y = self.classifier(features)   
        return y, features


class DAMR_V(ModelBase):
    
    def __init__(
        self,
        classes: List[str],
        input_samples: int,
        learning_rate: float = 0.001, 
        input_channels: int = 6,
        feature_width: int = 256,
        **kwargs
    ):
        super().__init__(classes=classes, **kwargs)

        
        self.lr = learning_rate
        self.nrx = input_channels
        self.example_input_array = torch.zeros((1,input_channels,input_samples), dtype=torch.cfloat)
        self.loss = nn.CrossEntropyLoss() 
        self.automatic_optimization = False
        self.fe_models = nn.ModuleList()
        
        for i in range(input_channels):
            self.fe_models.append(VTCNN2_1D(input_samples, 1, feature_width, len(classes)))

        
        metrics = {
            f'F1_fe{i}': torchmetrics.classification.MulticlassF1Score(num_classes=len(classes)) for i in range(input_channels)
        }
        metrics['F1'] = torchmetrics.classification.MulticlassF1Score(num_classes=len(classes))
        metrics = torchmetrics.MetricCollection(metrics)
        self.val_metrics = metrics.clone(f"val/")
        self.test_metrics = metrics.clone(f"test/")
        self.cm_metric = torchmetrics.classification.MulticlassConfusionMatrix(len(classes), normalize='true')


    def forward(self, x: torch.Tensor):
        
        y = []
        for i in range(self.nrx):
            out, features = self.fe_models[i](x[:,i:i+1]) 
            y.append(out)

        y = torch.stack(y, -2)
        y = torch.sum(y,dim=1)
        return y

    def configure_optimizers(self):
        opts = []
        for model in self.fe_models:
            opts.append(torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=0.00001))
        return opts


    def training_step(self, batch, batch_nb):
        data, target, _ = batch
        opts = self.optimizers()
        model_opts = opts[:]
        

        # 6 models training
        total_fe_loss = 0
        
        for i, (fe, opt) in enumerate(zip(self.fe_models, model_opts)):
            self.toggle_optimizer(opt)

            
            y, _ = fe(data[:,i:i+1])     
            loss = self.loss(y, target)
            if self.global_step!= 0: self.logger.log_metrics({f'train/fe_{i}_loss': loss}, self.global_step)
            total_fe_loss += loss

            # Optimize and Backprop (manually)
            self.manual_backward(loss)
            opt.step()
            opt.zero_grad()
            self.untoggle_optimizer(opt)
        self.log("total_fe_loss", total_fe_loss, prog_bar=True, logger=False) 



    #Validation step
    def validation_step(self, batch, batch_nb):
        data, target, snr = batch

        # Full Voting pass
        output = self.forward(data)
        self.val_metrics['F1'].update(output, target)
        self.cm_metric.update(output, target)

        # extract outputs from each model
        for i, model in enumerate(self.fe_models):
            y, _ = model(data[:,i:i+1])    
            self.val_metrics[f'F1_fe{i}'].update(y, target)    

    # Called each test set step
    def test_step(self, batch, batch_nb):
        data, target, snr = batch
        
        # Full Voting pass
        output = self.forward(data)
        self.test_metrics['F1'].update(output, target) 
        self.cm_metric.update(output, target)       

        batch_size = len(snr)                  
        batch_idx = batch_nb*batch_size   
        self.outputs_list[batch_idx:batch_idx+batch_size] = output.detach().cpu()    
        snr = snr.squeeze(dim=-1)   
        self.snr_list[batch_idx:batch_idx+batch_size] = snr.detach().cpu()   
       
        # extract outputs from each model
        for i, model in enumerate(self.fe_models):
            y, _ = model(data[:,i:i+1])    
            self.test_metrics[f'F1_fe{i}'].update(y, target)    
