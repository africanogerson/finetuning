import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from breast_cancer_classifier.utilities.pickling import unpickle_from_file, pickle_to_file
from src.utils import cross_t,cross_test_auc,cross_val_auc, cross_train_auc

raiz = '/media/bioimag/DATADRIVE/backup/deeprisk/results/'
name = 'Heat_risk_ext_dp'
auc=cross_test_auc(raiz,name)
print('test:', auc)
auc=cross_val_auc(raiz,name)
print('val:', auc)
auc=cross_train_auc(raiz,name)
print('train:', auc)

# # ls = list((None for i in range(24)))
# # df = pd.DataFrame(list(zip(ls, ls, ls)),
# #                columns =['AUC', 'lr', 'batch'])
# # a='dd'
# # df.iloc[1,0:3]=[1,2,3]
# # df.to_csv('ver.csv', index=False)
# # df.iloc[0,0:3]=[1,2,3]
# # df.to_csv(a+'ver.csv', index=False)
# # print(df)
#
#
# import breast_cancer_classifier.modeling.layers as layers
# from breast_cancer_classifier.modeling.models import SplitBreastModel as CancerClassModel
# from src.constants import VIEWS
# weights_path = '/home/bioimag/Documentos/gafricano/code/own_projects/deeplearning/deeplearning01/data/models/sample_image_model.p'
# cancer_class_model = CancerClassModel(input_channels=1)
# model_r = copy.deepcopy(cancer_class_model)
# # for name, child in cancer_class_model.named_children():
# #     print(name)
# for name, x in model_r.named_children():
#     print(name)
# state_dict = torch.load(weights_path, map_location='cpu')["model"]  # add map_location='cpu' if no gpu
# model_r.load_state_dict(state_dict)
# # print(torch.cuda.is_available())
# # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# # print(torch.cuda.is_available())
# # ver=np.ones((10,1))
# # ver=None
# # print(type(ver))
# # if ver is None:
# #     print('got you')
#
# # for epoch in range(7):
# #     best_metric = 8 if epoch > 5 else 0.5
# #     print(best_metric)
# # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# # print(device)

