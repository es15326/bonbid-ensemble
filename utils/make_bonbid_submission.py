import argparse
import glob
import os

import nibabel as nib
import SimpleITK
from evalutils.io import SimpleITKLoader


def save_mha(image, path):
  """Save numpy array as mha file.

  Args:
      image (np.ndarray): Input image array of shape (H, W, D)
      path (str): Path in which to save the mha file
  """
  SimpleITK.WriteImage(SimpleITK.GetImageFromArray(image), path)


def load_mha(path):
  """Load the mha file as numpy array.

  Args:
      path (str): .mha file path

  Returns:
      np.ndarray: image as a numpy array of shape (H, W, D)
  """
  loader = SimpleITKLoader()
  im = loader.load_image(path)
  im = SimpleITK.GetArrayFromImage(im)
  return im


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '-i', '--in_nii_dir', type=str,
      help='Directory containing nii files generated using test.py')
  parser.add_argument(
      '-o', '--out_mha_dir', type=str,
      help='Directory to save mha files')

  args = parser.parse_args()
  os.makedirs(args.out_mha_dir, exist_ok=True)
  labels = glob.glob(os.path.join(args.in_nii_dir, '*.nii.gz'))
  for label in labels:
    new_label = label.replace('.nii.gz', '.mha')
    new_label = new_label.replace(args.in_nii_dir, args.out_mha_dir)
    new_label = new_label.replace('Zmap_', '')
    new_label = new_label.replace('-ADC_smooth2mm_clipped10', '_lesion')
    im = nib.load(label).get_data().astype('uint8')
    save_mha(im, new_label)


if __name__ == '__main__':
  main()
