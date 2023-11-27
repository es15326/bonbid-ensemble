"""Examples
# BTCCV:
python train.py --model_cfg config/architectures/btcv_unet.yml \
    --data_cfg dataset/dataset_btcv.json \
    --out_dir temp

# Vessels:
python train.py --model_cfg config/architectures/vessels_unet.yml \
    --data_cfg dataset/vessels/dataset.json \
    --out_dir ckpt_vessels \
    --train_transforms config/transforms/vessels_train_transforms.yml \
    --val_transforms config/transforms/vessels_val_transforms.yml

CUDA_VISIBLE_DEVICES=1 python train.py \
    --model_cfg config/architectures/vessels_swin_unetr_48.yml \
    --data_cfg dataset/vessels/dataset.json \
    --out_dir ckpt_vessels \
    --train_transforms config/transforms/vessels_train_transforms.yml \
    --val_transforms config/transforms/vessels_val_transforms.yml

CUDA_VISIBLE_DEVICES=1 python train.py \
    --model_cfg config/architectures/vessels_swin_unetr_48_pretrained.yml \
    --data_cfg dataset/vessels/dataset.json \
    --out_dir ckpt_vessels \
    --train_transforms config/transforms/vessels_train_transforms.yml \
    --val_transforms config/transforms/vessels_val_transforms.yml

CUDA_VISIBLE_DEVICES=1 python train.py \
    --model_cfg config/architectures/vessels_swin_unetr_48_pretrained.yml \
    --data_cfg dataset/vessels/dataset_single_shot.json \
    --out_dir ckpt_vessels \
    --train_transforms config/transforms/vessels_train_transforms.yml \
    --val_transforms config/transforms/vessels_val_transforms.yml \
    --tag single_shot

CUDA_VISIBLE_DEVICES=1 python train.py \
    --model_cfg config/architectures/vessels_swin_unetr_48_pretrained_scratch.yml \
    --data_cfg dataset/vessels/dataset_single_shot.json \
    --out_dir ckpt_vessels \
    --train_transforms config/transforms/vessels_train_transforms.yml \
    --val_transforms config/transforms/vessels_val_transforms.yml \
    --tag single_shot_scratch

# BONBID - SwinUNETR pretrained
python train.py --model_cfg config/architectures/bonbid_swin_unetr_48.yml \
  --out_dir ckpt_lesions \
  --data_cfg examples/dataset_bonbid.json \
  --train_transforms config/transforms/bonbid_train_transforms.yml \
  --val_transforms config/transforms/bonbid_val_transforms.yml

"""
import argparse
import os

import pytorch_lightning as pl
from monai.config import print_config
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from model import MonaiLightningNet


def main():
    print_config()

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_cfg', type=str,
                        default='dataset/dataset_btcv.json',
                        help='path to the data config file')
    parser.add_argument('--model_cfg', type=str,
                        default='config/architectures/swin_unetr_24.yml',
                        help='path to the model config file')
    parser.add_argument('--out_dir', type=str, default='./ckpt',
                        help='output directory')
    parser.add_argument('--train_transforms', type=str,
                        default='config/transforms/btcv_train_transforms.yml',
                        help='path to the train transforms config file')
    parser.add_argument('--val_transforms', type=str,
                        default='config/transforms/btcv_val_transforms.yml',
                        help='path to the validation transforms config file')
    parser.add_argument('--tag', type=str, required=False,
                        help='tag for the experiment')
    args = parser.parse_args()

    # Setup data directory
    os.makedirs(args.out_dir, exist_ok=True)

    # initialise the LightningModule
    net = MonaiLightningNet(model_config=args.model_cfg,
                            data_config_path=args.data_cfg,
                            train_transforms=args.train_transforms,
                            val_transforms=args.val_transforms)

    # set up checkpoints
    ckpt_name = 'best_' + os.path.basename(args.model_cfg).split('.')[0]
    if args.tag is not None:
        ckpt_name += '_' + args.tag
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.out_dir, filename=ckpt_name,
        mode="max", monitor="val_dice",
        save_top_k=1, verbose=True)

    # set up loggewr
    logdir = os.path.join(args.out_dir, ckpt_name)
    logger = TensorBoardLogger(logdir, version=args.tag)

    # initialise Lightning's trainer.
    trainer = pl.Trainer(
        devices=[0],
        logger=logger,
        max_epochs=net.max_epochs,
        check_val_every_n_epoch=net.check_val,
        callbacks=[checkpoint_callback],
        default_root_dir=args.out_dir,
        log_every_n_steps=4
    )

    # train
    trainer.fit(net)


if __name__ == "__main__":
    main()
