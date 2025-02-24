import sys
from pathlib import Path
# Append the directory to your python path using sys
sys.path.append('/home/ubuntu/mimolm')
# Add the project root to sys.path
print(Path.cwd()) 

from argparse import ArgumentParser
from src.config import MODELS_DIR, PROCESSED_DATA_DIR
from pathlib import Path
from os.path  import join
from src.external.hptr.src.data_modules.data_h5_av2 import DataH5av2
from src.mimolm import MimoLM
from lightning.pytorch.tuner import Tuner
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch import Trainer, seed_everything
import torch

def main(args):
    if int(args.devices) > 1:
        BASE_PATH = '/home/ubuntu/'
    else: 
        BASE_PATH = '/home/harshavardhan-patil/Work/Projects/'
        
    seed_everything(args.seed, workers=True)
    data_module = DataH5av2(BASE_PATH + "mimolm/data")
    data_module.setup(stage="fit")
    model = MimoLM(data_size=data_module.tensor_size_train
                , n_rollouts = args.n_rollouts
                , learning_rate = float(args.learning_rate))

    checkpoint_callback = ModelCheckpoint(
        save_top_k=-1,  # Save all checkpoints
        every_n_epochs=1  # Save checkpoint every n epochs
    )
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    trainer = Trainer(precision='16-mixed',
                        callbacks=[checkpoint_callback, lr_monitor],
                        max_epochs=int(args.max_epochs),
                        profiler="simple",
                        devices=int(args.devices),
                        default_root_dir=BASE_PATH + "mimolm/ckpts")
    # tuner = Tuner(trainer)

    # #Run learning rate finder and then train
    # tuner.lr_find(model=model, datamodule=data_module)
    trainer.fit(model=model, datamodule=data_module)#, ckpt_path=BASE_PATH + 'mimolm/ckpts/lightning_logs/version_0/checkpoints/epoch=9-step=2610.ckpt')
    

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--accelerator", default=None)
    parser.add_argument("--devices", default=None)
    parser.add_argument("--strategy", default=None)
    parser.add_argument("--seed", default=43)
    parser.add_argument("--learning_rate", default=6.e-04)
    parser.add_argument("--n_rollouts", default=1)
    parser.add_argument("--max_epochs", default=1)
    args = parser.parse_args()
    main(args)
