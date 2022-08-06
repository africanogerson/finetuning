import os
import cv2
import numpy as np
import pandas as pd
import argparse


IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']
KEYS_VIEW = ['L_CC_path', 'R_CC_path', 'L_MLO_path', 'R_MLO_path']


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def get_sizes(cropped_folder, dataframe):
    # Extract image size of cropped images in folder
    sizes_info = {}
    for view in KEYS_VIEW:
        sizes_info[view] = []
        for file_path in dataframe[view].tolist():
            filename = os.path.basename(file_path) + '.png'
            if is_image_file(os.path.join(cropped_folder, filename)):
                im = cv2.imread(os.path.join(cropped_folder, filename))
                sizes_info[view].append(im.shape)
    return sizes_info


if __name__ == '__main__':
    # For parsing commandline arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases-data", type=str, required=True, help='path to csv file with data of cases')
    parser.add_argument("--controls-data", type=str, required=True, help='path to csv file with data of controls')
    parser.add_argument("--cropped-image-folder", type=str, required=True, help='path to folder with cropped images')
    parser.add_argument("--output-file", type=str, required=True, help='path to npy file with sizes of mean crops')
    args = parser.parse_args()

    # Read csv files for cases and controls
    df_cases, df_controls = pd.read_csv(args.cases_data), pd.read_csv(args.controls_data)

    # Path to cropped images
    cases_folder = os.path.join(args.cropped_image_folder, 'cases')
    controls_folder = os.path.join(args.cropped_image_folder, 'controls')

    # List of sizes
    sizes_cases = get_sizes(cases_folder, df_cases)
    sizes_controls = get_sizes(controls_folder, df_controls)

    # Compute average crops
    cc_crop = np.mean(np.asarray(
            sizes_cases['L_CC_path'] + sizes_cases['R_CC_path'] +
            sizes_controls['L_CC_path'] + sizes_controls['R_CC_path']),
            axis=0
        )
    mlo_crop = np.mean(np.asarray(
            sizes_cases['L_MLO_path'] + sizes_cases['R_MLO_path'] +
            sizes_controls['L_MLO_path'] + sizes_controls['R_MLO_path']),
            axis=0
        )
    avg_crops = {
        'CC': (int(cc_crop[0]), int(cc_crop[1])),
        'MLO': (int(mlo_crop[0]), int(mlo_crop[1]))
    }
    print('Average crop sizes:')
    print(avg_crops)

    # Save to output file
    np.save(args.output_file, avg_crops)
