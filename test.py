'''Example
# UNet
python test.py --model_cfg config/architectures/btcv_unet.yml \
    --ckpt ckpt/unet_best_metric_model.pth \
    --data_cfg dataset/dataset_btcv.json

# UNETR
python test.py --model_cfg config/architectures/btcv_unetr.yml \
    --ckpt ckpt/unetr_best_metric_model.pth \
    --data_cfg dataset/dataset_btcv.json

# SwinUNETR
python test.py --model_cfg config/architectures/btcv_swin_unetr.yml \
    --ckpt ckpt/swin_unetr_best_metric_model.pth \
    --data_cfg dataset/dataset_btcv.json

# Vessels - UNETR
python test.py --model_cfg config/architectures/vessels_swin_unetr_48.yml \
    --ckpt ckpt_vessels/best_vessels_swin_unetr_48.ckpt \
    --data_cfg dataset/vessels/dataset.json \
    --out_dir dataset/vessels/predsTs \
    --transform_cfg config/transforms/vessels_test_transforms.yml

python test.py --model_cfg config/architectures/vessels_unet.yml \
    --ckpt ckpt_vessels/best_vessels_unet.ckpt \
    --data_cfg dataset/vessels/dataset.json \
    --out_dir dataset/vessels/predsTs \
    --transform_cfg config/transforms/vessels_test_transforms.yml \
    --sliding_window 128,128,32
'''
import argparse
import os

import nibabel as nii
import numpy as np
import torch
from monai.data import CacheDataset, load_decathlon_datalist
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
    parser.add_argument('--split', type=str, default='all',
                        help='Data split.')
    parser.add_argument('--out_dir', type=str, default='dataset/predsTs',
                        help='output directory')
    parser.add_argument('--transform_cfg', type=str,
                        default='config/transforms/bonbid_val_transforms.yml',
                        help='path to the transforms config yml file')
    parser.add_argument('--sliding_window', type=str, default='96,96,96',
                        help='Sliding window size.')
    args = parser.parse_args()

    # Model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = MonaiLightningNet(model_config=args.model_cfg)
    ckpt = torch.load(args.ckpt, map_location=torch.device('cpu'))
    net.load_state_dict(ckpt['state_dict'])
    net.eval()
    net.to(device)


    temp = ckpt['state_dict']

    for k, v in temp.items():
        print(k)
    exit()
    # Dataset
    files = load_decathlon_datalist(args.data_cfg, True, args.split)

    transforms = get_transform_from_config(args.transform_cfg)

    dataset = CacheDataset(
        data=files,
        transform=transforms,
        cache_num=1,
        cache_rate=1.0,
        num_workers=8,
    )

    output_dir = os.path.join(
        args.out_dir,
        os.path.basename(args.model_cfg).split('.')[0])
    os.makedirs(output_dir, exist_ok=True)

    window_size = tuple(map(int, args.sliding_window.split(',')))
    # Inference and evaluation
    for case_num in range(len(dataset)):
        with torch.no_grad():
            img_name = os.path.basename(
                dataset[case_num]["image_meta_dict"]["filename_or_obj"])
            print('Processing', img_name)
            img = dataset[case_num]["image"].cpu()
            #val_inputs = torch.unsqueeze(img, 0).cuda()
            val_inputs = torch.unsqueeze(img, 0).cpu()
            fake_data = torch.tensor(np.zeros(val_inputs.shape), dtype=torch.float32).to(device)
            val_outputs = sliding_window_inference(
                fake_data, window_size, 4, net, overlap=0.5)
            print(val_outputs.shape)
            val_outputs = net.post_pred(val_outputs[0])
            val_outputs = torch.argmax(
                val_outputs, dim=0).squeeze().cpu().numpy()
            #print(val_outputs.shape)
            nii.save(nii.Nifti1Image(val_outputs, None, dtype=np.int16),
                     os.path.join(output_dir, img_name))
    print('Done')


if __name__ == "__main__":
    main()
