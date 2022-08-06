from __future__ import print_function, division
import os
import pandas as pd
import argparse
import breast_cancer_classifier.utilities.pickling as pickling

from breast_cancer_classifier.cropping.crop_mammogram import crop_mammogram
from src.constants import VIEWS


def create_dict_paths(data_csv, pkl_filename):
    df = pd.read_csv(data_csv)
    keys = VIEWS.LIST
    datum = []
    for idx in range(len(df)):
        paths = df[['L_CC_path', 'R_CC_path', 'L_MLO_path', 'R_MLO_path']].iloc[idx].tolist()
        paths = [[path] for path in paths]
        data_dict = dict(zip(keys, paths))
        data_dict['horizontal_flip'] = 'NO'
        datum.append(data_dict)
    pickling.pickle_to_file(pkl_filename, datum)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Remove background of image and save cropped files')
    parser.add_argument('--cases-data', required=True)
    parser.add_argument('--controls-data', required=True)
    parser.add_argument('--output-data-folder', required=True)
    parser.add_argument('--cases-exam-list-path', required=True)
    parser.add_argument('--controls-exam-list-path', required=True)
    parser.add_argument('--cropped-exam-list-file', required=True)
    parser.add_argument('--num-processes', default=10, type=int)
    parser.add_argument('--num-iterations', default=100, type=int)
    parser.add_argument('--buffer-size', default=50, type=int)
    args = parser.parse_args()

    create_dict_paths(args.cases_data, args.cases_exam_list_path)
    crop_mammogram(
        exam_list_path=args.cases_exam_list_path,
        cropped_exam_list_path=os.path.join(args.output_data_folder, 'cases', args.cropped_exam_list_file),
        output_data_folder=os.path.join(args.output_data_folder, 'cases'),
        num_processes=args.num_processes,
        num_iterations=args.num_iterations,
        buffer_size=args.buffer_size,
    )

    create_dict_paths(args.controls_data, args.controls_exam_list_path)
    crop_mammogram(
        exam_list_path=args.controls_exam_list_path,
        cropped_exam_list_path=os.path.join(args.output_data_folder, 'controls', args.cropped_exam_list_file),
        output_data_folder=os.path.join(args.output_data_folder, 'controls'),
        num_processes=args.num_processes,
        num_iterations=args.num_iterations,
        buffer_size=args.buffer_size,
    )



