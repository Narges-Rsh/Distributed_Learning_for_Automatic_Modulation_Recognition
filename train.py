import sys
import torch
import os
from argparse import ArgumentParser
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.profilers import PyTorchProfiler
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateFinder, StochasticWeightAveraging

from data.SimpleHDF5DataModule import SimpleHDF5DataModule
from models import *

def train(model, dm, name, epochs=40, precision="32", debug=False):
   
   
    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    version = None if debug or slurm_job_id is None else slurm_job_id
    logger = TensorBoardLogger("./logs", name, version=version, log_graph=False)
    
    callbacks = [
        ModelCheckpoint(monitor='val/F1', mode='max', save_top_k=1, save_last=True),
     
    ]
    profiler = None

    trainer = Trainer(
        fast_dev_run=False, 
        logger=logger, 
        callbacks=callbacks,
        accelerator='auto', 
        devices=[3], 
        num_nodes=1,
        sync_batchnorm=True,
        deterministic='warn',
        precision=precision,
        enable_progress_bar=True, 
        max_epochs=epochs,
        default_root_dir=logger.log_dir,
        profiler=profiler
    )
    trainer.fit(model, datamodule=dm)

    # Test best model on test set
    model = model.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
    trainer.test(model, datamodule=dm, verbose=True)
    
    logger.finalize('success')

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default='Voting') 
 
    #parser.add_argument("--checkpoint_path", type=str,       #Path for loading 6 models for feature sharing step
    #                    default="/home/distributed_learning/logs/Voting-SimpleHDF5DataModule/version_0/checkpoints/epoch=2-step=37800.ckpt")

    parser.add_argument("--epochs", type=int, default=1) 
    parser.add_argument("--bs", type=int, default=256)
    parser.add_argument("--nrx", type=int, default=6) 
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--debug", action='store_true')
    parser.add_argument("--seed", type=float, default=42)
    parser.add_argument("--precision", type=str, default="32", choices=["64", "32", "16", "bf16"])
    args = parser.parse_args()

    ## Initialize torch
    seed_everything(args.seed)
    torch.set_float32_matmul_precision('high')

    dm = SimpleHDF5DataModule("/home/SPIE_Paper/cspb_DisAMR_1a5961a.hdf5", args.bs, n_rx=args.nrx)  

    model_args = {
        # Required
        'input_samples': dm.frame_size, 
        'input_channels': args.nrx, 
        'classes': dm.classes,
        'learning_rate': args.lr,

        # Hyperparameters to log
        'epochs': args.epochs,
        'batch_size': args.bs,
        'precision': args.precision,
       #'checkpoint_path':args.checkpoint_path   #for Step 2 (feature extraction)
    }

    # Create model by finding class matching args.model and initializing with parameters in model_args
    model = getattr(sys.modules['models'], args.model)(**model_args)
    
    session_name = f"{args.model}-{dm.__class__.__name__}"
    train(model, dm, session_name, epochs=args.epochs, precision=args.precision, debug=args.debug)
