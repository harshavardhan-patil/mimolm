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

def main(args):
    seed_everything(args.seed, workers=True)
    data_module = DataH5av2("/home/ubuntu/mimolm/data")
    data_module.setup(stage="fit")
    model = MimoLM(data_size=data_module.tensor_size_train
                , n_rollouts = args.n_rollouts
                , learning_rate = args.learning_rate,)

    checkpoint_callback = ModelCheckpoint(
        save_top_k=-1,  # Save all checkpoints
        every_n_epochs=1  # Save checkpoint every n epochs
    )
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    trainer = Trainer(precision='16-mixed',
                        callbacks=[checkpoint_callback, lr_monitor],
                        max_epochs=args.max_epochs,
                        profiler="simple",
                        devices=args.devices,
                        default_root_dir="/home/ubuntu/mimolm/ckpts")
    tuner = Tuner(trainer)

    #Run learning rate finder and then train
    tuner.lr_find(model=model, datamodule=data_module)
    trainer.fit(model=model, datamodule=data_module) #, ckpt_path='/content/drive/MyDrive/Colab/mimolm/ckpts/lightning_logs/version_1/checkpoints/epoch=4-step=20825.ckpt')
    

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--accelerator", default=None)
    parser.add_argument("--devices", default=None)
    parser.add_argument("--strategy", default=None)
    parser.add_argument("--seed", default=43)
    parser.add_argument("--learning_rate", default=1.e-04)
    parser.add_argument("--n_rollouts", default=1)
    parser.add_argument("--max_epochs", default=1)
    args = parser.parse_args()
    main(args)
