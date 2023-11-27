from typing import Optional, Sequence, Union

import pytorch_lightning as pl
import torch
import yaml
from monai.data import (CacheDataset, DataLoader, decollate_batch,
                        list_data_collate, load_decathlon_datalist)
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.losses import DiceLoss
from monai.losses import FocalLoss
from monai.metrics import DiceMetric
from monai.networks import nets
from monai.transforms import AsDiscrete, Compose, RemoveSmallObjects

from transforms import get_transform_from_config


import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


def get_model_config(model_config_path: str) -> dict:
    with open(model_config_path, 'r') as stream:
        config = yaml.safe_load(stream)
    return config


def edit_keys(d: dict, replace_strs: Sequence[str],
              with_strs: Sequence[str]) -> dict:
    new_d = {}
    for key in d:
        new_key = key
        for replace_str, with_str in zip(replace_strs, with_strs):
            new_key = new_key.replace(replace_str, with_str)
        new_d[new_key] = d[key]
    return new_d


def get_model_from_config(model_config: dict) -> torch.nn.Module:
    model_name = model_config['model']
    model_args = model_config['args']
    model = getattr(nets, model_name)(**model_args)
    if 'weights' in model_config:
        weights = torch.load(model_config['weights']['path'])
        if 'remap_keys' in model_config['weights']:
            replace_str, with_str = model_config['weights']['remap_keys']
            weights['state_dict'] = edit_keys(
                weights['state_dict'], replace_str, with_str)
        model.load_from(weights=weights)
    return model


class MonaiLightningNet(pl.LightningModule):
    def __init__(self, model_config: Union[str, dict],
                 data_config_path: Optional[str] = None,
                 train_transforms: Optional[Union[str, dict]] = None,
                 val_transforms: Optional[Union[str, dict]] = None):
        super().__init__()
        if isinstance(model_config, str):
            self.model_config = get_model_config(model_config)
        self.save_hyperparameters(self.model_config)
        self._model = get_model_from_config(self.model_config)

        num_classes = self.model_config['args']['out_channels']
        #self.loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
        #self.loss_function = DiceLoss(to_onehot_y=True, softmax=True)
        self.loss_function = FocalLoss(to_onehot_y=True)
        self.post_pred = Compose([
            AsDiscrete(argmax=True, to_onehot=num_classes),
            RemoveSmallObjects(min_size=500)
        ])
        self.post_label = AsDiscrete(to_onehot=num_classes)
        self.dice_metric = DiceMetric(include_background=False,
                                      reduction="mean",
                                      ignore_empty=False,
                                      get_not_nans=False)
        self.best_val_dice = 0
        self.best_val_epoch = 0
        self.max_epochs = 5000
        self.check_val = 1
        self.warmup_epochs = 20
        self.metric_values = []
        self.validation_step_outputs = []
        self.data_config_path = data_config_path

        self.train_transforms = get_transform_from_config(train_transforms)
        self.val_transforms = get_transform_from_config(val_transforms)

    def forward(self, x):
        return self._model(x)

    def prepare_data(self):
        assert self.data_config_path is not None, "data_config_path is None"
        # prepare data
        datalist = load_decathlon_datalist(
            self.data_config_path, True, "training")
        val_files = load_decathlon_datalist(
            self.data_config_path, True, "validation")

        self.train_ds = CacheDataset(
            data=datalist,
            transform=self.train_transforms,
            cache_num=24,
            cache_rate=1.0,
            num_workers=8,
        )
        self.val_ds = CacheDataset(
            data=val_files,
            transform=self.val_transforms,
            cache_num=6,
            cache_rate=1.0,
            num_workers=8,
        )

    def train_dataloader(self):
        train_loader = DataLoader(
            self.train_ds,
            batch_size=1,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
            collate_fn=list_data_collate,
        )
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(
            self.val_ds,
            batch_size=1,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        return val_loader

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self._model.parameters(), lr=1e-4, weight_decay=1e-5)
        return optimizer
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     optimizer, factor=0.5, patience=30, min_lr=1e-7, verbose=True)
        # return {
        #     "optimizer": optimizer,
        #     "lr_scheduler": scheduler,
        #     "monitor": "val_dice"
        # }

    def training_step(self, batch, batch_idx):
        images, labels = (batch["image"].cuda(), batch["label"].cuda())
        output = self.forward(images)
        loss = self.loss_function(output, labels)
        tensorboard_logs = {"train_loss": loss.item()}
        self.log_dict(tensorboard_logs, on_step=True, on_epoch=True,
                      prog_bar=True)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]
        roi_size = (96, 96, 96)
        sw_batch_size = 4
        outputs = sliding_window_inference(
            images, roi_size, sw_batch_size, self.forward)
        loss = self.loss_function(outputs, labels)
        outputs = [self.post_pred(i) for i in decollate_batch(outputs)]
        labels = [self.post_label(i) for i in decollate_batch(labels)]
        self.dice_metric(y_pred=outputs, y=labels)
        d = {"val_loss": loss, "val_number": len(outputs)}
        self.validation_step_outputs.append(d)
        return d

    def on_validation_epoch_end(self):
        val_loss, num_items = 0, 0
        for output in self.validation_step_outputs:
            val_loss += output["val_loss"].sum().item()
            num_items += output["val_number"]
        mean_val_dice = self.dice_metric.aggregate().item()
        self.dice_metric.reset()
        mean_val_loss = torch.tensor(val_loss / num_items)
        tensorboard_logs = {
            "val_dice": mean_val_dice,
            "val_loss": mean_val_loss,
            "hp_metric": mean_val_dice,
        }
        if mean_val_dice > self.best_val_dice:
            self.best_val_dice = mean_val_dice
            self.best_val_epoch = self.current_epoch
        print(
            f"current epoch: {self.current_epoch} "
            f"current mean dice: {mean_val_dice:.4f}"
            f"\nbest mean dice: {self.best_val_dice:.4f} "
            f"at epoch: {self.best_val_epoch}"
        )
        self.metric_values.append(mean_val_dice)
        self.validation_step_outputs.clear()  # free memory
        self.log_dict(tensorboard_logs, prog_bar=True)
        return {"log": tensorboard_logs}
