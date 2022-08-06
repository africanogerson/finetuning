from __future__ import absolute_import, division, print_function
import torch
from torch.utils.data import DataLoader
import time
import copy
import pyprind
import argparse
import yaml
import numpy as np
import pandas as pd

from breast_cancer_classifier.utilities.pickling import unpickle_from_file, pickle_to_file
from src.model_handler import BaseRiskModel
from src.utils import elapsed_time, join_yaml_fn, get_lr, hyper_par, cross_val_auc, cross_train_auc_optimization
from src.constants import VIEWS, METRICS
from src.dataloader import create_mamodataset_from_config
from sklearn.metrics import roc_auc_score
import scipy.io as sio


class Trainer(BaseRiskModel):
    def __init__(self, _config):
        super(Trainer, self).__init__(options=_config, mode='train')

    def train(self, datasets):
        assert all(i in ['train', 'val', 'test'] for i in datasets.keys())
        assert self.opts['metric'] in METRICS.LIST
        unfreeze_flag = False
        num_epochs = self.opts['train']['num_epochs']
        best_metric = 0.0

        since = time.time()

        for epoch in range(num_epochs):
            # Dataloader
            batch_sizes = {x: self.opts.get(x)['batch_size'] for x in ['train', 'val']}
            print(batch_sizes['train'])
            self.dataloaders = {
                x: DataLoader(datasets[x],
                              batch_size=batch_sizes[x],
                              shuffle=True, num_workers=4, drop_last=False
                              )
                for x in ['train', 'val']}
            self.dataset_sizes = {x: len(datasets[x]) for x in ['train', 'val']}

            if self.opts['verbose']:
                print('-' * 15)
                print('Epoch {}/{}'.format(epoch, num_epochs - 1))

            if epoch >= self.opts['train']['unfreeze'] and not unfreeze_flag:
                self.model.unfreeze()
                unfreeze_flag = True
                print('unfreezing')

            # Training step

            self.model.train()  # Set model to training mode
            print(epoch)
            self._run_epoch('train', epoch)
            self._send_scalars('train', epoch)

            # Validation step
            self.model.eval()  # Set model to evaluate mode
            self._run_epoch('val', epoch)
            self._send_scalars('val', epoch)
            self.scheduler.step(epoch=epoch)

            # deep copy the model when improving in validation
            if self.epoch_metric > best_metric:
                #best_metric = self.epoch_metric if epoch > 5 else 0.5
                best_metric = self.epoch_metric
                best_model_wts = copy.deepcopy(self.model.state_dict())
                self.ckpt_mgr.save_best(best_model_wts)
                ###sve the validation scores
                d = {
                    'targets': self.t,
                    'scores': self.y
                }
                d_train = {
                    'targets': self.t_train,
                    'scores': self.y_train
                }
                df = pd.DataFrame(data=d)
                df_train = pd.DataFrame(data=d_train)

                df_train.to_csv(self.opts['train_name'], index=False)
                df.to_csv(self.opts['val_name'], index=False)

        if self.opts['verbose']:
            print('\nTraining completed in {}'.format(elapsed_time(since, time.time())))
            print('Best val {}: {:4f}'.format(self.opts['metric'], best_metric))

    def _run_epoch(self, phase, epoch):
        running_loss = 0
        running_metric = 0
        N = 0
        if self.opts['metric'] is not METRICS.ACC:
            t = np.zeros(self.dataset_sizes[phase], dtype=bool)
            y = np.zeros(self.dataset_sizes[phase], dtype=np.float32)

        # Progres bar
        if self.opts['verbose']:
            prog_bar = pyprind.ProgBar(len(self.dataloaders[phase]), monitor=True, title=phase)
        # loss_sum = 0
        for aux_index, sample in enumerate(self.dataloaders[phase]):
            inputs = {view: sample[view].to(self.device) for view in VIEWS.LIST}
            targets = sample['output'].to(self.device)

            # zero the parameter gradients
            self.optimizer.zero_grad()

            # forward
            with torch.set_grad_enabled(phase == 'train'):
                outputs = self.model(inputs)
                loss = self.loss(outputs, targets)

                if self.opts['metric'] is METRICS.ACC:
                    corrects = self._cal_corrects(outputs, targets)
                N += len(targets)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    self.optimizer.step()

                # cumulative = int((epoch + 1) / 70) + 1
                # if (phase == 'train') and (((aux_index + 1) % cumulative) == 0):
                #     if phase == 'val':
                #         print('something wrong')
                #     loss_sum = (1 / cumulative) * (loss_sum + loss)
                #     loss_sum.backward()
                #     self.optimizer.step()
                #     loss_sum = 0
                # elif phase == 'train':
                #     loss_sum += loss

            with torch.no_grad():
                # statistics
                running_loss += loss.item()
                if self.opts['metric'] is METRICS.ACC:
                    running_metric += corrects.item()
                else:
                    t[N - len(targets):N] = targets.detach().cpu().numpy().astype(bool)
                    y[N - len(targets):N] = outputs.detach().cpu().numpy()

                # update progress bar
                if self.opts['verbose']:
                    prog_bar.update()
                torch.cuda.empty_cache()

        with torch.no_grad():
            self.epoch_loss = running_loss / N
            if self.opts['metric'] is METRICS.ACC:
                self.epoch_metric = running_metric / N
            else:
                self.epoch_metric = roc_auc_score(t, y)
                if phase == 'train':
                    self.t_train = t
                    self.y_train = y

                else:
                    self.t = t
                    self.y = y

            if self.opts['verbose']:
                print('{} Loss: {:.4f} {}: {:.4f}'.format(
                    phase, self.epoch_loss, self.opts['metric'], self.epoch_metric))

    def _send_scalars(self, phase, epoch):
        if self.opts['logger']['state']:
            # Send epoch loss to logger
            self.logger.add_scalar('{}/loss'.format(phase), self.epoch_loss, epoch)
            # Send epoch metric to logger
            self.logger.add_scalar('{}/{}'.format(phase, self.opts['metric']), self.epoch_metric, epoch)
            # Send lr
            if phase is 'train':
                self.logger.add_scalar('lr', get_lr(self.optimizer), epoch)

    @staticmethod
    def _cal_corrects(pred, target):
        return target.eq((pred > 0.5).type(torch.float)).sum()

    @staticmethod
    def _normalize(img):
        return (img - img.min()) / (img.max() - img.min())


def _complete_config_from_parser(_config, _args):
    _config['verbose'] = True if _args.verbose else False
    _config['use_gpu'] = True if _args.use_gpu else False
    return _config


if __name__ == '__main__':
    # For parsing commandline arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/config.yaml", help='config file to be read')
    parser.add_argument("--indices", type=str, default="", help='indices for training and testing')
    parser.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")
    parser.add_argument("--use_gpu", help="use GPU", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(0)
    np.random.seed(0)
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

    # Load dataset
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
            parameters,
            train_idx=indices['train_idx'][0],
            val_idx=indices['val_idx'][0]
        )

    # #hyperparameter to condiser
    # h_batch = [4, 2]
    # h_lr = np.linspace(5, 6, 8)
    #
    # hyper_list = hyper_par(h_lr, h_batch)
    # if config['outage']==0:
    #     k_0 = 0
    #     i_0 = 0
    #     pickle_to_file(exp_name+'index.pkl', [k_0, i_0])
    #     ls = list((None for i in range(len(hyper_list))))
    #     csv_record = pd.DataFrame(list(zip(ls, ls, ls, ls, ls, ls, ls)),
    #                       columns=['AUC_Train','AUC_Val', 'lr', 'batch', 'finetuning', 'flayer', 'extended_model'])
    #     csv_record.to_csv(exp_name + 'val_record.csv', index=False)
    # else:
    #     k_0, i_0 = unpickle_from_file(exp_name+'index.pkl')
    #     csv_record = pd.read_csv(exp_name+'val_record.csv')
    #     print(k_0,i_0)
    #
    #
    # for k in range(k_0, len(hyper_list)):  #here should go length of hiper_list len(hyper_list)
    #     h1, h2 = hyper_list[k]
    #     config['optimizer']['lr'] = h1
    #     config['train']['batch_size'] = h2
    #     config['val']['batch_size'] = h2
    #     # Training
    #     for i in range(i_0, 5):
    #         print('fold number', i)
    #         pickle_to_file(exp_name+'index.pkl', [k, i])
    #         config['ckpt_path'] = prefix_path+str(i)+'.pt'
    #         config['val_name'] = 'results/'+exp_name+'val_fold'+str(i)+'.csv'
    #         config['train_name'] = 'results/train_folds/' + exp_name + 'train_fold' + str(i) + '.csv'
    #         torch.manual_seed(0)  # this is important for reproducible issues
    #         np.random.seed(0)
    #         trainer = Trainer(config)
    #         trainer.train(mammo_dataset[i])
    #     i_0 = 0
    #     auc_val = cross_val_auc(raizt, exp_name)
    #     auc_train = cross_train_auc_optimization(raizt, exp_name)
    #     csv_record.iloc[k, 0:7] = [auc_train, auc_val, h1, h2, config['model']['finetuning'],
    #                                config['model']['flayer'], config['model']['extended_model']]
    #     csv_record.to_csv(exp_name+'val_record.csv', index=False)

    # when the optimization is done
    h1 = 10**(-np.linspace(5, 6, 8))
    # # h2=4
    k = 0
    i_0 = 0
    config['optimizer']['lr'] = h1[3]
    # config['train']['batch_size'] = 2
    # # config['val']['batch_size'] = h2

    if config['outage'] == 0:
        k_0 = 0
        i_0 = 0
        pickle_to_file(exp_name + 'index.pkl', [k_0, i_0])

    else:
        k_0, i_0 = unpickle_from_file(exp_name + 'index.pkl')
        print('outage k,i:', k_0, i_0)

    for i in range(i_0, 5):  # here should go 'for i in range(i_0, 5):'
        print('fold number', i)
        pickle_to_file(exp_name + 'index.pkl', [k, i])
        config['ckpt_path'] = prefix_path + str(i) + '.pt'
        config['val_name'] = 'results/' + exp_name + 'val_fold' + str(i) + '.csv'
        config['train_name'] = 'results/train_folds/' + exp_name + 'train_fold' + str(i) + '.csv'
        torch.manual_seed(0)
        np.random.seed(0)
        trainer = Trainer(config)
        trainer.train(mammo_dataset[i])
    auc_val = cross_val_auc(raizt, exp_name)
    auc_train_opt = cross_train_auc_optimization(raizt, exp_name)
    print('The experiment was:', exp_name)
    print('AUC_val:', auc_val)
    print('AUC_Train_learning:', auc_train_opt)