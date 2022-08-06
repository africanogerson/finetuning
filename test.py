from __future__ import absolute_import, division, print_function
import torch
from torch.utils.data import DataLoader
import copy
import pyprind
import argparse
import yaml
import os
import numpy as np
import pandas as pd

from src.model_handler import BaseRiskModel
from src.utils import join_yaml_fn, cross_test_auc, cross_train_auc, cross_val_auc, cross_train_auc_optimization
from src.constants import VIEWS
from src.dataloader import create_mamodataset_from_config
from sklearn.metrics import roc_auc_score
import scipy.io as sio


class Tester(BaseRiskModel):
    def __init__(self, _config):
        super(Tester, self).__init__(options=_config, mode='test')

    def test(self, _dataset):
        dataloader = DataLoader(_dataset,
                                batch_size=1,
                                shuffle=False,
                                num_workers=4,
                                )
        self.model.eval()  # Set model to evaluate mode

        if self.opts['verbose']:
            prog_bar = pyprind.ProgBar(len(dataloader), monitor=True, title='Test')

        self.t = np.zeros(len(_dataset), dtype=bool)
        self.y = np.zeros(len(_dataset), dtype=np.float32)

        for i, sample in enumerate(dataloader):
            inputs = {view: sample[view].to(self.device) for view in VIEWS.LIST}
            targets = sample['output'].to(self.device)

            # forward
            with torch.no_grad():
                outputs = self.model(inputs)
                self.t[i] = targets.detach().cpu().numpy().astype(bool)
                self.y[i] = outputs.detach().cpu().numpy()

                # update progress bar
                if self.opts['verbose']:
                    prog_bar.update()

        self._save_scores()
        auc = roc_auc_score(self.t, self.y)

        if self.opts['verbose']:
            print('AUC: {:.4f}'.format(auc))

    def _save_scores(self):
        if self.opts['csv_path'] is not None:
            d ={
                'targets': self.t,
                'scores': self.y
            }
            df = pd.DataFrame(data=d)
            df.to_csv(self.opts['csv_path'], index=False)


def _complete_config_from_parser(_config, _args):
    _config['verbose'] = True if _args.verbose else False
    _config['use_gpu'] = True if _args.use_gpu else False
    _config['csv_path'] = os.path.join(_args.scores_folder, _config['experiment']) + '.csv' if _args.scores_folder is not '' else None
    if not os.path.exists(_args.scores_folder) and _config['csv_path'] is not None:
        os.makedirs(_args.scores_folder)
    return _config


if __name__ == '__main__':
    # For parsing commandline arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/config.yaml", help='config file to be read')
    parser.add_argument("--scores_folder", type=str, default="results", help='path to store csv file with scores')
    parser.add_argument("--indices", type=str, default="", help='indices for training and testing')
    parser.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")
    parser.add_argument("--use_gpu", help="use GPU", action="store_true")
    args = parser.parse_args()

    # Read settings
    config_file = args.config
    yaml.add_constructor('!join', join_yaml_fn)
    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.Loader)

    config = _complete_config_from_parser(config, args)

    dataset_info = copy.deepcopy(config['dataset'])
    cropping_info = copy.deepcopy(config['cropping'])
    parameters = copy.deepcopy(config['parameters'])
    parameters['use_heatmaps'] = config['model']['use_heatmaps']
    prefix_path = config['ckpt_path']
    exp_name = config['experiment']
    raizt = config['raiz']


    # Load data
    if args.indices == '':
        mammo_dataset = create_mamodataset_from_config(
            dataset_info,
            cropping_info,
            parameters,
        )
    else:
        indices = sio.loadmat(args.indices)
        mammo_dataset = create_mamodataset_from_config(
            dataset_info,
            cropping_info,
            train_idx=indices['train_idx'][0],
            val_idx=indices['val_idx'][0]
        )

    # Inference loop
    for i, fold in enumerate(mammo_dataset):
        print('fold number', i)
        config['ckpt_path'] = prefix_path+str(i)+'.pt'
        config['csv_path'] = 'results/'+exp_name+'train_fold'+str(i)+'.csv'

        tester = Tester(config)
        tester.test(fold['train'])

    auc_train=cross_train_auc(raizt, exp_name)

    for i, fold in enumerate(mammo_dataset):
        print('fold number', i)
        config['ckpt_path'] = prefix_path+str(i)+'.pt'
        config['csv_path'] = 'results/'+exp_name+'test_fold'+str(i)+'.csv'

        tester = Tester(config)
        tester.test(fold['test'])

    auc_test = cross_test_auc(raizt, exp_name)
    auc_val = cross_val_auc(raizt, exp_name)
    auc_train_opt = cross_train_auc_optimization(raizt, exp_name)
    print('AUC test:', auc_test)
    print('AUC train:', auc_train)
    print('AUC val:', auc_val)
    print('AUC train optmization:', auc_train_opt)
    # tester.test(mammo_dataset['train'])
    # print('train')
    #tester.test(mammo_dataset['test'])
    # print('test')
