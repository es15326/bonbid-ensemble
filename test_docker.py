import argparse
import os

import nibabel as nii
import numpy as np
import torch
from monai.data import CacheDataset, load_decathlon_datalist
from monai.inferers import sliding_window_inference

from model import MonaiLightningNet
from transforms import get_transform_from_config


'''val_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image"],
            pixdim=(1.0, 1.0, 1.0),
            mode=("bilinear", "nearest"),
        ),
        ScaleIntensityRanged(keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image"], source_key="image"),
    ]
)'''



def main():

    transform_cfg = 'config/transforms/bonbid_val_transforms.yml'
    ckpt = '/usr/mvl2/esdft/3d-segmentation-monai-main_main/ckpt_lesions_all_training/best_bonbid_unet-v5.ckpt'
    out_dir = 'dataset/predsTs'

    # Model

    model_cfg = 'config/architectures/bonbid_unet.yml'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = MonaiLightningNet(model_config=model_cfg)
    ckpt = torch.load(ckpt)
    net.load_state_dict(ckpt['state_dict'])
    net.eval()
    net.to(device)

    # Dataset

    data_cfg = '/usr/mvl2/esdft/bonbid_dataset_all_training_data.json'
    split = 'all'
    files = load_decathlon_datalist(data_cfg, True, split)

    transforms = get_transform_from_config(transform_cfg)

    dataset = CacheDataset(
        data=files,
        transform=transforms,
        cache_num=1,
        cache_rate=1.0,
        num_workers=8,
    )


    sliding_window = '96,96,96'

    output_dir = os.path.join(
        out_dir,
        os.path.basename(model_cfg).split('.')[0])
    os.makedirs(output_dir, exist_ok=True)

    window_size = tuple(map(int, sliding_window.split(',')))
    # Inference and evaluation
    for case_num in range(len(dataset)):
        with torch.no_grad():
            img_name = os.path.basename(
                dataset[case_num]["image_meta_dict"]["filename_or_obj"])
            #print('Processing', img_name)
            img_path = dataset[case_num]["image_meta_dict"]["filename_or_obj"]
            nifti_img = nii.load(img_path)
            data_array = nifti_img.get_fdata()
            #transformed_tensor = transforms(torch.from_numpy(data_array))
            #img = dataset[case_num]["image"]
            #val_inputs = torch.unsqueeze(img, 0).cuda()
            #print(val_inputs.shape)
            img = torch.tensor(data_array, dtype=torch.float32)
            val_inputs = torch.unsqueeze(img, 0).cuda()
            val_inputs = torch.unsqueeze(val_inputs, 0).cuda()
            val_outputs = sliding_window_inference(
                val_inputs, window_size, 4, net, overlap=0.5)
            #val_outputs = net.post_pred(val_outputs[0])
            print(torch.tensor(data_array).shape, img.shape, val_inputs.shape, val_outputs.shape)
            val_outputs = torch.argmax(
                val_outputs, dim=0).squeeze().cpu().numpy()
            #print(val_outputs.shape)
            nii.save(nii.Nifti1Image(val_outputs, None, dtype=np.int16),
                     os.path.join(output_dir, img_name))
    print('Done')


if __name__ == "__main__":
    main()
