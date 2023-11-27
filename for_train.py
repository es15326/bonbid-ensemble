import os
import shutil
import tempfile

import numpy as np
import nibabel
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.metrics import f1_score

from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.transforms import (
    AsDiscrete,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    RandRotate90d,
    SpatialPadd,
)

from monai.config import print_config
from monai.metrics import DiceMetric
from monai.networks.nets import UNETR, SwinUNETR

from monai.data import (
    DataLoader,
    CacheDataset,
    load_decathlon_datalist,
    decollate_batch,
)


import torch

print_config()


train_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.0, 1.0, 1.0),
            mode=("bilinear", "nearest"),
        ),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-175,
            a_max=250,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        SpatialPadd(keys=["image", "label"], spatial_size=(256, 256, 32)),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=(96, 96, 32),
            pos=1,
            neg=1,
            num_samples=4,
            image_key="image",
            image_threshold=0,
        ),
        RandFlipd(
            keys=["image", "label"],
            spatial_axis=[0],
            prob=0.10,
        ),
        RandFlipd(
            keys=["image", "label"],
            spatial_axis=[1],
            prob=0.10,
        ),
        RandFlipd(
            keys=["image", "label"],
            spatial_axis=[2],
            prob=0.10,
        ),
        RandRotate90d(
            keys=["image", "label"],
            prob=0.10,
            max_k=3,
        ),
        RandShiftIntensityd(
            keys=["image"],
            offsets=0.10,
            prob=0.50,
        ),
    ]
)
val_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        ScaleIntensityRanged(keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image", "label"], source_key="image"),
    ]
)






#data_dir = "/usr/mvl2/esdft/3d-segmentation-monai-main_main/examples/"
#split_json = "dataset_bonbid.json"

data_dir = "/usr/mvl2/esdft/"
split_json = "bonbid_dataset_ADC.json"


datasets = data_dir + split_json
#datalist = load_decathlon_datalist(datasets, True, "training")
datalist_all = load_decathlon_datalist(datasets, True, "all")
#val_files = load_decathlon_datalist(datasets, True, "validation")
'''train_ds = CacheDataset(
    data=datalist,
    transform=train_transforms,
    cache_num=24,
    cache_rate=1.0,
    num_workers=0,
)'''
#train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=0, pin_memory=True)
#val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_num=6, cache_rate=1.0, num_workers=0)
all_ds = CacheDataset(data=datalist_all, transform=val_transforms, cache_num=6, cache_rate=1.0, num_workers=0)
all_loader = DataLoader(all_ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
#val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)







'''slice_map = {
    "Zmap_MGHNICU_329-VISIT_01-ADC_smooth2mm_clipped10.nii.gz": 170,
    "Zmap_MGHNICU_014-VISIT_01-ADC_smooth2mm_clipped10.nii.gz": 230,
    "Zmap_MGHNICU_010-VISIT_01-ADC_smooth2mm_clipped10.nii.gz": 204,
    "Zmap_MGHNICU_074-VISIT_01-ADC_smooth2mm_clipped10.nii.gz": 204,
    "Zmap_MGHNICU_077-VISIT_01-ADC_smooth2mm_clipped10.nii.gz": 204,
    "Zmap_MGHNICU_447-VISIT_01-ADC_smooth2mm_clipped10.nii.gz": 74,
}
case_num = 0
img_name = os.path.split(val_ds[case_num]["image"].meta["filename_or_obj"])[1]
img = val_ds[case_num]["image"]
label = val_ds[case_num]["label"]
img_shape = img.shape
label_shape = label.shape
print(f"image shape: {img_shape}, label shape: {label_shape}")

plt.figure("image", (18, 6))
plt.subplot(1, 2, 1)
plt.title("image")
plt.imshow(img[0, :, :, slice_map[img_name]].detach().cpu(), cmap="gray")
plt.subplot(1, 2, 2)
plt.title("label")
plt.imshow(label[0, :, :, slice_map[img_name]].detach().cpu())
plt.show()'''






os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SwinUNETR(
    in_channels=1,
    out_channels=2,
    img_size=(96, 96, 96),
    feature_size=12,
).to(device)

'''loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
torch.backends.cudnn.benchmark = True
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)'''





def validation(epoch_iterator_val):
    model.eval()
    dice_scores = []
    with torch.no_grad():
        for batch in epoch_iterator_val:
            val_inputs, val_labels = (batch["image"].cuda(), batch["label"].cuda())
            val_outputs = sliding_window_inference(val_inputs, (96, 96, 96), 4, model, overlap=0.1)
            val_labels_list = decollate_batch(val_labels)
            val_labels_convert = [post_label(val_label_tensor) for val_label_tensor in val_labels_list]
            val_outputs_list = decollate_batch(val_outputs)
            val_output_convert = [post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]
            dice_metric(y_pred=val_output_convert, y=val_labels_convert)
            #print(val_outputs.shape)
            #epoch_iterator_val.set_description("Validate (%d / %d Steps)" % (global_step, 10.0))

            
            temp_pred = val_output_convert[0].as_tensor().cpu().numpy()
            temp_true = val_labels_convert[0].as_tensor().cpu().numpy()
            
            pred = temp_pred.astype(bool)
            gt = temp_true.astype(bool)
            dice = f1_score(gt.flatten(), pred.flatten(), zero_division=1)
            dice_scores.append(dice)

        mean_dice_val = dice_metric.aggregate().item()
        print(dice_scores)
        dice_metric.reset()
    return mean_dice_val



root_dir = f'{os.getcwd()}/checkpoints/'
output_dir = f'{os.getcwd()}/new_results_ADC'
model.load_state_dict(torch.load(os.path.join(root_dir, "best_metric_model.pth")))



post_label = AsDiscrete(to_onehot=2)
post_pred = AsDiscrete(argmax=True, to_onehot=2)
dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
global_step = 0

epoch_iterator_all = tqdm(all_loader, desc="Validate (X / X Steps) (dice=X.X)", dynamic_ncols=True)
#epoch_iterator = tqdm(train_loader, desc="Train (X / X Steps) (dice=X.X)", dynamic_ncols=True)
#####dice_val = validation(epoch_iterator_all)


#####print(dice_val)


for case_num in range(len(all_ds)):

    model.eval()

    with torch.no_grad():

        img_name = os.path.split(all_ds[case_num]["image"].meta["filename_or_obj"])[1]
        img = all_ds[case_num]["image"]
        label = all_ds[case_num]["label"]
        val_inputs = torch.unsqueeze(img, 1).cuda()
        val_labels = torch.unsqueeze(label, 1).cuda()
        val_outputs = sliding_window_inference(val_inputs, (96, 96, 96), 4, model, overlap=0.1)

        pred = torch.argmax(val_outputs, dim=1).detach().squeeze().cpu().numpy()
        gt = val_inputs.as_tensor().squeeze().cpu().numpy()

        #print(img_name, img.shape, label.shape)
        print(pred.shape)
        #pred = pred.astype(bool)
        #gt = gt.astype(bool)

        #dice = f1_score(gt.flatten(), gt.flatten(), zero_division=1)
        #print(dice)

        nibabel.save(nibabel.Nifti1Image(pred, None, dtype=np.int16),os.path.join(output_dir, img_name))




