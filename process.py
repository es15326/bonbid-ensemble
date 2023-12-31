# Copyright 2023 Radboud University Medical Center
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# codes written by Rina Bao (rina.bao@childrens.harvard.edu) for BONBID-HIE
# MICCAI Challenge 2023 (https://bonbid-hie2023.grand-challenge.org/).
# Demo of Algorithm docker

import json
from dataclasses import dataclass, make_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
import SimpleITK
import argparse
import os
import yaml

import nibabel as nii
import numpy as np
import torch
from monai.data import CacheDataset, load_decathlon_datalist
from monai.inferers import sliding_window_inference

from model import edit_keys, get_model_from_config
#from transforms import get_transform_from_config
#from skimage.util.shape import view_as_windows
from model import MonaiLightningNet


INPUT_PREFIX = Path('/input')
OUTPUT_PREFIX = Path('/output')


class IOKind(str, Enum):
    JSON = 'JSON'
    IMAGE = 'Image'
    FILE = 'File'


class InterfaceKind(str, Enum):
  # TODO: taken from
  # https://github.com/comic/grand-challenge.org/blob/ffbae21af534caed9595d9bc48708c5f753b075c/app/grandchallenge/components/models.py#L69
  # would be better to get this directly from the schema

  def __new__(cls, value, annotation, io_kind):
    member = str.__new__(cls, value)
    member._value_ = value
    member.annotation = annotation
    member.io_kind = io_kind
    return member

  STRING = 'String', str, IOKind.JSON
  INTEGER = 'Integer', int, IOKind.JSON
  FLOAT = 'Float', float, IOKind.JSON
  BOOL = 'Bool', bool, IOKind.JSON
  ANY = 'Anything', Any, IOKind.JSON
  CHART = 'Chart', Dict[str, Any], IOKind.JSON

  # Annotation Types
  TWO_D_BOUNDING_BOX = '2D bounding box', Dict[str, Any], IOKind.JSON
  MULTIPLE_TWO_D_BOUNDING_BOXES = (
      'Multiple 2D bounding boxes', Dict[str, Any], IOKind.JSON)
  DISTANCE_MEASUREMENT = 'Distance measurement', Dict[str, Any], IOKind.JSON
  MULTIPLE_DISTANCE_MEASUREMENTS = (
      'Multiple distance measurements', Dict[str, Any], IOKind.JSON)
  POINT = 'Point', Dict[str, Any], IOKind.JSON
  MULTIPLE_POINTS = 'Multiple points', Dict[str, Any], IOKind.JSON
  POLYGON = 'Polygon', Dict[str, Any], IOKind.JSON
  MULTIPLE_POLYGONS = 'Multiple polygons', Dict[str, Any], IOKind.JSON
  LINE = 'Line', Dict[str, Any], IOKind.JSON
  MULTIPLE_LINES = 'Multiple lines', Dict[str, Any], IOKind.JSON
  ANGLE = 'Angle', Dict[str, Any], IOKind.JSON
  MULTIPLE_ANGLES = 'Multiple angles', Dict[str, Any], IOKind.JSON
  ELLIPSE = 'Ellipse', Dict[str, Any], IOKind.JSON
  MULTIPLE_ELLIPSES = 'Multiple ellipses', Dict[str, Any], IOKind.JSON

  # Choice Types
  CHOICE = 'Choice', int, IOKind.JSON
  MULTIPLE_CHOICE = 'Multiple choice', int, IOKind.JSON

  # Image types
  IMAGE = 'Image', bytes, IOKind.IMAGE
  SEGMENTATION = 'Segmentation', bytes, IOKind.IMAGE
  HEAT_MAP = 'Heat Map', bytes, IOKind.IMAGE

  # File types
  PDF = 'PDF file', bytes, IOKind.FILE
  SQREG = 'SQREG file', bytes, IOKind.FILE
  THUMBNAIL_JPG = 'Thumbnail jpg', bytes, IOKind.FILE
  THUMBNAIL_PNG = 'Thumbnail png', bytes, IOKind.FILE
  OBJ = 'OBJ file', bytes, IOKind.FILE
  MP4 = 'MP4 file', bytes, IOKind.FILE

  # Legacy support
  CSV = 'CSV file', str, IOKind.FILE
  ZIP = 'ZIP file', bytes, IOKind.FILE


@dataclass
class Interface:
  slug: str
  relative_path: str
  kind: InterfaceKind

  @property
  def kwarg(self):
      return self.slug.replace('-', '_').lower()

  def load(self):
    if self.kind.io_kind == IOKind.JSON:
      return self._load_json()
    elif self.kind.io_kind == IOKind.IMAGE:
      return self._load_image()
    elif self.kind.io_kind == IOKind.FILE:
      return self._load_file()
    else:
      raise AttributeError(
          f'Unknown io kind {self.kind.io_kind!r} for {self.kind!r}')

  def save(self, *, data):
    if self.kind.io_kind == IOKind.JSON:
      return self._save_json(data=data)
    elif self.kind.io_kind == IOKind.IMAGE:
      return self._save_image(data=data)
    elif self.kind.io_kind == IOKind.FILE:
      return self._save_file(data=data)
    else:
      raise AttributeError(
          f'Unknown io kind {self.kind.io_kind!r} for {self.kind!r}')

  def _load_json(self):
    with open(INPUT_PREFIX / self.relative_path, 'r') as f:
      return json.loads(f.read())

  def _save_json(self, *, data):
    with open(OUTPUT_PREFIX / self.relative_path, 'w') as f:
      f.write(json.dumps(data))

  def _load_image(self):
    input_directory = INPUT_PREFIX / self.relative_path
    mha_files = {f for f in input_directory.glob("*.mha") if f.is_file()}

    if len(mha_files) == 1:
      mha_file = mha_files.pop()
      return SimpleITK.ReadImage(mha_file)
    elif len(mha_files) > 1:
      raise RuntimeError(
          f'More than one mha file was found in {input_directory!r}'
      )
    else:
      raise NotImplementedError

  def _save_image(self, *, data):
    output_directory = OUTPUT_PREFIX / self.relative_path
    output_directory.mkdir(exist_ok=True, parents=True)
    file_save_name = output_directory / 'overlay.mha'
    SimpleITK.WriteImage(data, file_save_name)

  @property
  def _file_mode_suffix(self):
    if self.kind.annotation == str:
      return ''
    elif self.kind.annotation == bytes:
      return 'b'
    else:
      raise AttributeError(
          f'Unknown annotation {self.kind.annotation!r} for {self.kind!r}')

  def _load_file(self):
    i_file = INPUT_PREFIX / self.relative_path
    mode = 'r' + self._file_mode_suffix
    with open(i_file, mode) as f:
      return f.read()

  def _save_file(self, *, data):
    o_file = OUTPUT_PREFIX / self.relative_path
    mode = 'w' + self._file_mode_suffix
    with open(o_file, mode) as f:
      f.write(data)


INPUT_INTERFACES = [
    Interface(
        slug="z-score-apparent-diffusion-coefficient-map",
        relative_path="images/z-score-adc",
        kind=InterfaceKind.IMAGE),
    Interface(
        slug="skull-stripped-adc",
        relative_path="images/skull-stripped-adc-brain-mri",
        kind=InterfaceKind.IMAGE),
]

OUTPUT_INTERFACES = [
    Interface(
        slug='hypoxic-ischemic-encephalopathy-lesion-segmentation',
        relative_path='images/hie-lesion-segmentation',
        kind=InterfaceKind.SEGMENTATION),
]

Inputs = make_dataclass(
    cls_name='Inputs',
    fields=[(inpt.kwarg, inpt.kind.annotation) for inpt in INPUT_INTERFACES])

Outputs = make_dataclass(
    cls_name='Outputs',
    fields=[(output.kwarg, output.kind.annotation)
            for output in OUTPUT_INTERFACES])


def load() -> Inputs:
  return Inputs(
      **{interface.kwarg: interface.load() for interface in INPUT_INTERFACES}
  )


class BaseNet(object):
  '''def __init__(self, save_model_path):
    with open(save_model_path, 'rb') as f:
      self.model = joblib.load(f)'''

  def __init__(self):
    #self.ckpt = 'ckpt_lesions/best_bonbid_unet-v5.ckpt'
    self.ckpt = 'ckpt_lesions/best_bonbid_swin_unetr_48.ckpt'
    self.model_cfg = 'config/bonbid_swin_unetr_48.yml'
    #self.model_cfg = 'config/bonbid_unet.yml'
    self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

  def predict(
      self,
      zadc: np.ndarray,
      adc: Optional[np.ndarray] = None,):

    '''size = zadc.shape
    out = np.zeros((size[0], size[1], size[2]))
    return out.astype('uint8')'''


    net = MonaiLightningNet(model_config=self.model_cfg)
    ckpt = torch.load(self.ckpt, map_location=self.device)
    net.load_state_dict(ckpt['state_dict'])

    net.eval()
    net.to(self.device)

    sliding_window = '96,96,96'
    window_size = tuple(map(int, sliding_window.split(',')))

    img = torch.tensor(zadc, dtype=torch.float32)
    val_inputs = torch.unsqueeze(img, 0)
    val_inputs = torch.unsqueeze(val_inputs, 0).to(self.device)
    
    val_outputs = sliding_window_inference(val_inputs, window_size, 4, net, overlap=0.5)

    val_outputs = torch.argmax(val_outputs.squeeze(), dim=0).squeeze().cpu().numpy()

    return val_outputs.astype('uint8')


def predict(*, inputs: Inputs) -> Outputs:
  z_adc = inputs.z_score_apparent_diffusion_coefficient_map
  adc_ss = inputs.skull_stripped_adc
  z_adc = SimpleITK.GetArrayFromImage(z_adc)
  adc_ss = SimpleITK.GetArrayFromImage(adc_ss)

  #model_cfg = 'config/bonbid_swin_unetr_12.yml'
  #ckpt = 'ckpt_lesions/best_bonbid_swin_unetr_12.ckpt'

  model = BaseNet()
  out = model.predict(z_adc, adc_ss)

  hie_segmentation = SimpleITK.GetImageFromArray(out)

  outputs = Outputs(
      hypoxic_ischemic_encephalopathy_lesion_segmentation=hie_segmentation
  )
  return outputs


def save(*, outputs: Outputs) -> None:
  for interface in OUTPUT_INTERFACES:
    interface.save(data=getattr(outputs, interface.kwarg))


def main() -> int:
  inputs = load()
  outputs = predict(inputs=inputs)
  save(outputs=outputs)
  return 0


if __name__ == '__main__':
  raise SystemExit(main())

