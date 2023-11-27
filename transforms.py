from typing import Optional, Union

import monai.transforms
import yaml


def get_transform_from_config(
    transform_config: Optional[Union[dict, str]]
) -> Optional[monai.transforms.Compose]:
    if transform_config is None:
      return None
    if isinstance(transform_config, str):
        with open(transform_config, "r") as stream:
            transform_config = yaml.safe_load(stream)
    transform_list = []
    for transform_name, transform_args in transform_config.items():
        transform = getattr(monai.transforms, transform_name)(**transform_args)
        transform_list.append(transform)
    return monai.transforms.Compose(transform_list)
