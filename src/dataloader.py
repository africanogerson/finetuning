from __future__ import print_function, division
import numpy as np
import pandas as pd
import os
from torch.utils.data import Dataset
from src.ffdm import FFDM
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from copy import deepcopy

from breast_cancer_classifier.utilities.pickling import unpickle_from_file, pickle_to_file
from src.transformations import MammoTransformation
import breast_cancer_classifier.data_loading.loading as loading

class MammoDataset(Dataset):
    """Class definition of Breast Cancer Risk assessment dataset"""

    def __init__(self, df: pd.DataFrame, info: list, parameters: dict, transforms=None):
        #self.opts = opts
        self.transform = transforms
        self._views = ['L_CC_path', 'R_CC_path', 'L_MLO_path', 'R_MLO_path']
        assert all(a in list(df.columns) for a in self._views)
        self.df = df
        self.info = info
        self.hm = parameters["use_heatmaps"]
        self.parameters = parameters

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        view_names = ['L-CC', 'R-CC', 'L-MLO', 'R-MLO']
        sample = dict()
        for c, p in zip(view_names, self._views):
            #added


            short_file_path = self.df[p].iloc[idx]
            case_flag = self.df['isCase'].iloc[idx]


            if case_flag == 1:
                image_folder_path = self.parameters["image_path_cases"]
                heatmap_folder_path = self.parameters["heatmaps_path_cases"]
            else:
                image_folder_path = self.parameters["image_path_controls"]
                heatmap_folder_path = self.parameters["heatmaps_path_controls"]

            loaded_image = loading.load_image(
                image_path=os.path.join(image_folder_path, short_file_path + '.png'),
                view=c,
                horizontal_flip=self.info[idx]["horizontal_flip"],
            )
            if self.hm:
                loaded_heatmaps = loading.load_heatmaps(
                    benign_heatmap_path=os.path.join(heatmap_folder_path, "heatmap_benign",
                                                     short_file_path + ".hdf5"),
                    malignant_heatmap_path=os.path.join(heatmap_folder_path, "heatmap_malignant",
                                                        short_file_path + ".hdf5"),
                    view=c,
                    horizontal_flip=self.info[idx]["horizontal_flip"],
                )
            else:
                loaded_heatmaps = None

            sample[c] = self.transform(loaded_image, loaded_heatmaps, self.info[idx], c)
            #added finished


        sample['output'] = self.df.isCase[idx]
        sample['info'] = self.info[idx]
        sample['output'] = np.asarray(sample['output']).astype(np.float32)
        sample['idnumber'] = self.df.number[idx]
        # sample['p-info'] = self.df[['id', 'round', 'syear', 'sday', 'byear', 'age', 'system']].loc[[idx]].to_dict('list')

        return sample


def create_mamodataset_from_config(dataset_info, cropping_info, parameters, train_idx=None, val_idx=None,):

    df_cases = pd.read_csv(dataset_info['cases'])
    df_controls = pd.read_csv(dataset_info['controls'])
    info_dict_cases = unpickle_from_file(cropping_info['cases'])
    info_dict_controls = unpickle_from_file(cropping_info['controls'])

    # Transformations
    dataset_info['transformations']['seed'] = 0
    dataset_info['transformations']['random_number'] = np.random.RandomState(0)

    parameters_train = deepcopy(dataset_info['transformations'])
    parameters_train['augmentation'] = True

    parameters_val = deepcopy(dataset_info['transformations'])
    parameters_val['augmentation'] = False

    parameters_test = deepcopy(dataset_info['transformations'])
    parameters_test['augmentation'] = False
    trans = {
        phase: MammoTransformation(params) for phase, params in
        zip(
            ['train', 'val', 'test'],
            [parameters_train, parameters_val, parameters_test]
        )
    }
    # added
    skf = StratifiedKFold(n_splits=5, random_state=1, shuffle=False)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=0)
    train_i = []
    test_i = []
    val_i = []
    for train_idx, test_idx in skf.split(X=np.ones(df_cases.shape[0]),
                                         y=df_cases[['isGE']]):
        test_i.append(test_idx)
        t_data = df_cases.loc[train_idx]
        train_aux, val_aux = next(sss.split(X=np.ones(t_data.shape[0]), y=t_data[['isGE']]))
        train_i.append(train_idx[train_aux])
        val_i.append(train_idx[val_aux])

    folds = list((None for i in range(5)))

    for i in range(5):
        df_train = pd.DataFrame(columns=df_cases.columns)
        df_train = df_train.append(df_cases.loc[train_i[i]], ignore_index=True)
        df_train = df_train.append(df_controls.loc[train_i[i]], ignore_index=True)
        df_val = pd.DataFrame(columns=df_cases.columns)
        df_val = df_val.append(df_cases.loc[val_i[i]], ignore_index=True)
        df_val = df_val.append(df_controls.loc[val_i[i]], ignore_index=True)

        df_test = pd.DataFrame(columns=df_cases.columns)
        df_test = df_test.append(df_cases.loc[test_i[i]], ignore_index=True)
        df_test = df_test.append(df_controls.loc[test_i[i]], ignore_index=True)
        ## debug about no intersection
        aux1 = len(set(df_train.number).intersection(df_val.number,df_test.number)) + len(set(df_test.number).intersection(df_val.number))
        assert aux1 == 0
        test_label = 'results/test_id' + str(i) + '.csv'
        df_test.to_csv(test_label, index=False)
        # Spliting cropping info for training and validation
        info_train = [info_dict_cases[i] for i in train_i[i]]
        info_train += [info_dict_controls[i] for i in train_i[i]]
        info_val = [info_dict_cases[i] for i in val_i[i]]
        info_val += [info_dict_controls[i] for i in val_i[i]]

        info_test = [info_dict_cases[i] for i in test_i[i]]
        info_test += [info_dict_controls[i] for i in test_i[i]]

        #

        #
        folds[i] = {phase: MammoDataset(df, info, parameters, transforms=trans[phase]) for phase, df, info in
                    zip(['train', 'val', 'test'],
                        [df_train, df_val, df_test],
                        [info_train, info_val, info_test]
                        )
                    }
    pickle_to_file('CrossVal.pkl', folds)
    return folds


