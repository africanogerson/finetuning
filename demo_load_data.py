from __future__ import print_function, division
from copy import deepcopy
import yaml
import argparse

from src.dataloader import create_mamodataset_from_config
from src.utils import join_yaml_fn, show_case

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/config.yaml", help='config file to be read')
    parser.add_argument("--set", type=str, default="train", help="set to be loaded: 'train' or 'val'")
    parser.add_argument("--num_samples", type=int, default=3, help='number of samples to show')
    parser.add_argument("--show_contours", help="show contours", action="store_true")
    args = parser.parse_args()

    # Read settings
    with open(args.config, 'r') as f:
        yaml.add_constructor('!join', join_yaml_fn)
        config = yaml.load(f, Loader=yaml.Loader)

    dataset_info = deepcopy(config['dataset'])
    dataset_info['transformations']['to_tensor'] = False
    dataset_info['transformations']['contours'] = args.show_contours
    cropping_info = deepcopy(config['cropping'])

    # Load dataset
    mammo_dataset = create_mamodataset_from_config(dataset_info, cropping_info)

    # Show samples
    for n, sample in enumerate(mammo_dataset[args.set]):
        show_case(sample, pause=True, contour=args.show_contours)
        print(n)
        if n >= args.num_samples - 1:
            break
