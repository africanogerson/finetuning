from src.dataloader import create_mamodataset_from_config
import pandas as pd
import yaml
import copy
from torch.utils.data import DataLoader
from src.utils import join_yaml_fn, show_case
import torch
import matplotlib.pyplot as plt
import torchvision
import os
from breast_cancer_classifier.utilities.pickling import unpickle_from_file, pickle_to_file

# data = unpickle_from_file('CrossVal.pkl')
# for i, fold in enumerate(data):
#     test = fold['test']





config_file='/media/bioimag/DATADRIVE/backup/deeprisk/config/config.yaml'
yaml.add_constructor('!join', join_yaml_fn)
with open(config_file, 'r') as f:
    config = yaml.load(f, Loader=yaml.Loader)
torch.manual_seed(0)
parameters = copy.deepcopy(config['parameters'])
parameters['use_heatmaps'] = config['model']['use_heatmaps']

dataset_info = copy.deepcopy(config['dataset'])
cropping_info = copy.deepcopy(config['cropping'])
parameters = copy.deepcopy(config['parameters'])
parameters['use_heatmaps'] = config['model']['use_heatmaps']
prefix_path = config['ckpt_path']
exp_name = config['experiment']
raizt = config['raiz']

# for i in range(3):
#     print(type(string(i)))
# dataset = dict()
# cropping_info = dict()
# dataset['cases'] = 'added/data/wu_cases_191.csv'
# dataset['controls'] = 'added/data/wu_controls_191.csv'
# cropping_info['cases'] = 'sample_output/data_cases.pkl'
# cropping_info['controls'] = 'sample_output/data_controls.pkl'
# print(config['dataset'])


mammo_dataset = create_mamodataset_from_config(
            config['dataset'],
            config['cropping'],
            parameters,
        )
fold1 = mammo_dataset[0]
print(len(fold1['train']))

# for i, ver in enumerate(mammo_dataset):
#     print(i)
ver = DataLoader(fold1['train'], batch_size=8, shuffle=True, num_workers=1, drop_last=False)

print(len(ver))

print(0/2==int, int(2.1), int(2.99))
# # ver2 = DataLoader(mammo_dataset['train'], batch_size=1, shuffle=False, num_workers=1, drop_last=False)
cumulative = int((19 + 1) / 10) + 1
print(cumulative)
# dataiter = iter(ver)
# sample = dataiter.next()
# print(sample['L-CC'].shape)
# # print(sample['idnumber'],sample['output'])
# # show_case(sample)
# ax = plt.subplot(221)
# aux = sample['L-CC']
# id = sample['idnumber']
# print(id)
# ax.imshow(sample['L-CC'][0,0,:,:], 'gray')
# plt.show()



# print(type(ver))
# a=1
# for sample in ver:
#     #print(a)
#     if a==1:
#         break
# show_case(sample)
# # for sample2 in ver:
# #     if a==1:
# #         break
#
# # for sample2 in ver2:
# #     if a==1:
# #         break
# # bo= sample['L-CC'][0,1,:,:]==sample2['L-CC'][0,1,:,:]

# print(sample['L-CC'].shape,bo.all())