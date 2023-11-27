'''Example
# UNet
python eval.py --model_cfg config/architectures/btcv_unet.yml \
    --ckpt ckpt/unet_best_metric_model.pth \
    --data_cfg dataset/dataset_btcv.json \
    --data_split validation

# UNETR
python eval.py --model_cfg config/architectures/btcv_unetr.yml \
    --ckpt ckpt/unetr_best_metric_model.pth \
    --data_cfg dataset/dataset_btcv.json \
    --data_split validation

# SwinUNETR-24
python eval.py --model_cfg config/architectures/btcv_swin_unetr_24.yml \
    --ckpt ckpt/swin_unetr_best_metric_model.pth \
    --data_cfg dataset/dataset_btcv.json \
    --data_split validation

# SwinUNETR-48
python eval.py --model_cfg config/architectures/btcv_swin_unetr_48.yml \
    --ckpt new_ckpt/unet_best_metric_model.ckpt \
    --data_cfg dataset/dataset_btcv.json \
    --data_split validation

# Vessels - U-Net
python eval.py --model_cfg config/architectures/vessels_unet.yml \
    --ckpt ckpt_vessels/best_vessels_unet.ckpt \
    --data_cfg dataset/vessels/dataset.json \
    --data_split test \
    --transform_cfg config/transforms/vessels_val_transforms.yml

# Vessels - Swin UNETR
python eval.py --model_cfg config/architectures/vessels_swin_unetr_48.yml \
    --ckpt ckpt_vessels/best_vessels_swin_unetr_48.ckpt \
    --data_cfg dataset/vessels/dataset.json \
    --data_split test \
    --transform_cfg config/transforms/vessels_val_transforms.yml

# Vessels - Swin UNETR Pretrained - many-shot
python eval.py --model_cfg config/architectures/vessels_swin_unetr_48.yml \
    --ckpt ckpt_vessels/best_vessels_swin_unetr_48_pretrained.ckpt \
    --data_cfg dataset/vessels/dataset.json \
    --data_split test \
    --transform_cfg config/transforms/vessels_val_transforms.yml

# Vessels - Swin UNETR Pretrained - single-shot
python eval.py --model_cfg config/architectures/vessels_swin_unetr_48.yml \
    --ckpt ckpt_vessels/best_vessels_swin_unetr_48_pretrained-oneshot.ckpt \
    --data_cfg dataset/vessels/dataset.json \
    --data_split test \
    --transform_cfg config/transforms/vessels_val_transforms.yml

# Vessels - Swin UNETR Pretrained - single-shot - from scratch [for ablation]
python eval.py --model_cfg config/architectures/vessels_swin_unetr_48.yml \
    --ckpt ckpt_vessels/best_vessels_swin_unetr_48_pretrained_scratch_single_shot_scratch.ckpt \
    --data_cfg dataset/vessels/dataset.json \
    --data_split test \
    --transform_cfg config/transforms/vessels_val_transforms.yml

'''
import argparse
import os
import time

import torch
from monai.data import Dataset, load_decathlon_datalist
from monai.inferers import sliding_window_inference

from model import MonaiLightningNet
from transforms import get_transform_from_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_cfg', type=str, required=True,
                        help='path to the model config yml file')
    parser.add_argument('--ckpt', type=str, required=True,
                        help='Checkpoint file path')
    parser.add_argument('--data_cfg', type=str,
                        default='dataset/dataset_btcv.json',
                        help='path to the data config json file')
    parser.add_argument('--transform_cfg', type=str,
                        help='path to the transforms config yml file')
    parser.add_argument('--data_split', type=str, default='validation',
                        help='data split to evaluate on')

    args = parser.parse_args()

    # Dataset
    files = load_decathlon_datalist(
        args.data_cfg, True, args.data_split)

    transforms = get_transform_from_config(args.transform_cfg)
    dataset = Dataset(
        data=files,
        transform=transforms,
    )

    # Model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = MonaiLightningNet(model_config=args.model_cfg)
    ckpt = torch.load(args.ckpt)

    # Weights saved by Pytorch Lightning Trainer are wrapped in a dict whereas
    # weights saved by Pytorch are not.
    if "state_dict" in ckpt:
        # Strip "_model" from the keys
        new_ckpt = {}
        for key in ckpt["state_dict"]:
            new_ckpt[key.replace("_model.", "")] = ckpt["state_dict"][key]
        net._model.load_state_dict(new_ckpt)
    else:
        net._model.load_state_dict(ckpt)
    net.eval()
    net.to(device)

    # Inference and evaluation

    with torch.no_grad():
        for case_num in range(len(dataset)):
            begin_time = time.time()
            img_name = os.path.split(
                dataset[case_num]["image"].meta["filename_or_obj"])[1]
            img = dataset[case_num]["image"]
            label = dataset[case_num]["label"]
            val_inputs = torch.unsqueeze(img, 0).cuda()
            val_labels = torch.unsqueeze(label, 0).cuda()
            val_outputs = sliding_window_inference(
                val_inputs, (96, 96, 96), 4, net, overlap=0.25)
            val_outputs = net.post_pred(val_outputs[0])
            val_labels = net.post_label(val_labels[0])
            dice = net.dice_metric(
                [val_outputs], [val_labels])[0].detach().cpu().numpy()
            duration = time.time() - begin_time
            print(img_name, *[f'{d:.4f}' for d in dice], f'{dice.mean():.4f}',
                  f'{duration:.4f}s')
    print(f"Aggregated Dice: {net.dice_metric.aggregate().item():.4f}")


if __name__ == "__main__":
    main()
