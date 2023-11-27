import SimpleITK as sitk
import glob

out = '/usr/mvl2/esdft/3d-segmentation-monai-main_main/trainset_niig/'

files = glob.glob(f'{out}/*.mha')


for i in range(len(files)):
    img = sitk.ReadImage(files[i]) 
    sitk.WriteImage(img,  files[i].replace('.mha','.nii'))



