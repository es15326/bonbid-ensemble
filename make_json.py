import json
import os

'''data = {
    "description": "btcv yucheng",
    "labels": {
        "0": "background",
        "1": "lesion",
    },

    "training": [
        {"image": "imagesTr/img0001.nii.gz", "label": "labelsTr/label0001.nii.gz"},
        {"image": "imagesTr/img0002.nii.gz", "label": "labelsTr/label0002.nii.gz"},
        {"image": "imagesTr/img0003.nii.gz", "label": "labelsTr/label0003.nii.gz"},
    ],
    "validation": [
        {"image": "imagesTr/img0035.nii.gz", "label": "labelsTr/label0035.nii.gz"},
        {"image": "imagesTr/img0036.nii.gz", "label": "labelsTr/label0036.nii.gz"},
    ]
}


with open("btcv_dataset.json", "w") as json_file:
    json.dump(data, json_file, indent=4)

print("JSON file created.")'''


import os


#image_folder = "/usr/mvl2/esdft/trainingset_nii/1ADC_ss"
#label_folder = "/usr/mvl2/esdft/trainingset_nii/3LABEL"

#image_folder = "/usr/mvl5/Images2/BIO2/BONBID2023/trainset_niigz/1ADC_ss"

image_folder = "/usr/mvl5/Images2/BIO2/BONBID2023/trainset_niigz/2Z_ADC"
label_folder = "/usr/mvl5/Images2/BIO2/BONBID2023/trainset_niigz/3LABEL"

all_data = []

for filename in os.listdir(image_folder):
    if filename.endswith(".nii.gz"):
        image_path = os.path.join(image_folder, filename)
        #label_filename = filename.replace('-ADC_ss', '_lesion')
        label_filename = filename.replace('-ADC_smooth2mm_clipped10', '_lesion')
        label_filename = label_filename.replace('Zmap_', '')
        label_path = os.path.join(label_folder, label_filename)
        entry = {"image": image_path, "label": label_path}
        all_data.append(entry)

print("Generated training data dictionary:")


#training_data = all_data[:-10]

training_data = all_data
validation_data = all_data[-10:]

print(len(validation_data))
print(len(training_data))

print(len(all_data))




data = {
    "description": "btcv yucheng",
    "labels": {
        "0": "background",
        "1": "lesion",
    },

    "training": training_data,
    "validation": validation_data,
    "all": all_data
}


with open("bonbid_dataset_all_training_data.json", "w") as json_file:
    json.dump(data, json_file, indent=4)

print("JSON file created.")


