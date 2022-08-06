import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

import breast_cancer_classifier.modeling.layers as layers
from breast_cancer_classifier.modeling.models import SplitBreastModel as CancerClassModel
from src.constants import VIEWS

_DEFAULT_WEIGHTS = 'breast_cancer_classifier/models/sample_image_model.p'

_DEFAULT_MODEL_SETTINGS = {
    'batch_norm': False,
    'dropout': {
        'state': False,
        'p': 0.1
    }
}


class ViewWiseRiskModel(nn.Module):
    def __init__(self, device='cpu', pretrained=True, weights_path=_DEFAULT_WEIGHTS, opts=_DEFAULT_MODEL_SETTINGS):
        super().__init__()

        self.opts = opts
        channels = 3 if self.opts['use_heatmaps'] else 1
        print('Loading the heatmap model') if self.opts['use_heatmaps'] else print('loading image only model',channels)
        cancer_class_model = CancerClassModel(input_channels=channels)
        weights_path = self.opts['heatmap_weights_path'] if self.opts['use_heatmaps'] else self.opts['image_weights_path']

        if pretrained:
            # Load weights
            state_dict = torch.load(weights_path, map_location=device)["model"]  # add map_location='cpu' if no gpu
            cancer_class_model.load_state_dict(state_dict)

        # Feature extractor
        self.features = copy.deepcopy(cancer_class_model.four_view_resnet)
        if pretrained:
            self.freeze()
        # del cancer_class_model

        # Batch norm layers
        if opts['batch_norm']:
            self.bn1d_cc = nn.BatchNorm1d(256 * 2)
            self.bn1d_mlo = nn.BatchNorm1d(256 * 2)

        # Dropout layers
        if opts['dropout']['state']:
            self.fc_drop_cc = nn.Dropout(opts['dropout']['p'])
            self.fc_drop_mlo = nn.Dropout(opts['dropout']['p'])
            print('drop_out')

        # AVG-Pooling layer
        self.all_views_avg_pool = layers.AllViewsAvgPool()

        # Fully-connected layers
        if opts['finetuning']:
            self.fc1_cc = copy.deepcopy(cancer_class_model.fc1_cc)
            self.fc1_mlo = copy.deepcopy(cancer_class_model.fc1_mlo)
            print('finetuning')
        else:
            self.fc1_cc = nn.Linear(256 * 2, 256 * 2)
            self.fc1_mlo = nn.Linear(256 * 2, 256 * 2)
            print('extractor')

        if opts['flayer'] & opts['finetuning']:
            for param in self.fc1_cc.parameters():
                param.requires_grad = False
            for paramq in self.fc1_mlo.parameters():
                paramq.requires_grad = False
            print('flayer')
        del cancer_class_model

        #fully connected layer added
        if opts['extended_model']:
            self.fca_cc = nn.Linear(256 * 2, 256 * 2)
            self.fca_mlo = nn.Linear(256 * 2, 256 * 2)
            print('extended')
        # Output layers
        self.fc2_cc = nn.Linear(256 * 2, 1)
        self.fc2_mlo = nn.Linear(256 * 2, 1)

    def freeze(self):
        for view in VIEWS.LIST:
            for param in self.features.model_dict[view].parameters():
                param.requires_grad = False

    def unfreeze(self):
        for param in self.fc1_cc.parameters():
            param.requires_grad = True
        for paramq in self.fc1_mlo.parameters():
            paramq.requires_grad = True
        # for view in VIEWS.LIST:
        #     for param in self.features.model_dict[view].parameters():
        #         param.requires_grad = True

    def forward(self, x):

        # Features
        f_maps = self.features(x)

        # Pooling
        h = self.all_views_avg_pool(f_maps)
        h_cc = torch.cat([h[VIEWS.L_CC], h[VIEWS.R_CC]], dim=1)
        h_mlo = torch.cat([h[VIEWS.L_MLO], h[VIEWS.R_MLO]], dim=1)

        # Fully connected layers, batch normalization and dropout (if included)
        h_cc = self.fc1_cc(h_cc)
        if self.opts['batch_norm']:
            h_cc = self.bn1d_cc(h_cc)
        h_cc = F.relu(h_cc)
        if self.opts['dropout']['state']:
            h_cc = self.fc_drop_cc(h_cc)

        h_mlo = self.fc1_mlo(h_mlo)
        if self.opts['batch_norm']:
            h_mlo = self.bn1d_mlo(h_mlo)
        h_mlo = F.relu(h_mlo)
        if self.opts['dropout']['state']:
            h_mlo = self.fc_drop_mlo(h_mlo)
        #added layers
        if self.opts['extended_model']:
            h_cc = self.fca_cc(h_cc)
            h_cc = F.relu(h_cc)
            h_mlo = self.fca_mlo(h_mlo)
            h_mlo = F.relu(h_mlo)
        if self.opts['dropout']['state']:
            h_mlo = self.fc_drop_mlo(h_mlo)
        if self.opts['dropout']['state']:
            h_cc = self.fc_drop_cc(h_cc)
        # Output layers
        y_cc = torch.sigmoid(self.fc2_cc(h_cc))
        y_mlo = torch.sigmoid(self.fc2_mlo(h_mlo))

        # Final average
        y = torch.cat((y_cc, y_mlo), dim=1).mean(1)
        return y


class ImageWiseRiskModel(nn.Module):
    def __init__(self, pretrained=True, weights_path=_DEFAULT_WEIGHTS, opts=_DEFAULT_MODEL_SETTINGS):
        super().__init__()
        self.opts = opts
        cancer_class_model = CancerClassModel(input_channels=1)
        if pretrained:
            # Load weights
            state_dict = torch.load(weights_path, map_location='cpu')["model"]  # add map_location='cpu' if no gpu
            cancer_class_model.load_state_dict(state_dict)

        # Feature extractor
        self.features = copy.deepcopy(cancer_class_model.four_view_resnet)
        if pretrained:
            self.freeze()
        del cancer_class_model

        # AVG-Pooling layer
        self.all_views_avg_pool = layers.AllViewsAvgPool()

        # Fully-connected layers
        self.fc1_l_cc = nn.Linear(256, 256)
        self.fc1_r_cc = nn.Linear(256, 256)
        self.fc1_l_mlo = nn.Linear(256, 256)
        self.fc1_r_mlo = nn.Linear(256, 256)

        # Batch norm layers
        if opts['batch_norm']:
            self.bn1d_l_cc = nn.BatchNorm1d(256)
            self.bn1d_r_cc = nn.BatchNorm1d(256)
            self.bn1d_l_mlo = nn.BatchNorm1d(256)
            self.bn1d_r_mlo = nn.BatchNorm1d(256)

        # Dropout layers
        if opts['dropout']['state']:
            self.fc_drop_l_cc = nn.Dropout(opts['dropout']['p'])
            self.fc_drop_r_cc = nn.Dropout(opts['dropout']['p'])
            self.fc_drop_l_mlo = nn.Dropout(opts['dropout']['p'])
            self.fc_drop_r_mlo = nn.Dropout(opts['dropout']['p'])

        # Output layers
        self.fc2_l_cc = nn.Linear(256, 1)
        self.fc2_r_cc = nn.Linear(256, 1)
        self.fc2_l_mlo = nn.Linear(256, 1)
        self.fc2_r_mlo = nn.Linear(256, 1)

    def freeze(self):
        for view in VIEWS.LIST:
            for param in self.features.model_dict[view].parameters():
                param.requires_grad = False

    def unfreeze(self):
        for view in VIEWS.LIST:
            for param in self.features.model_dict[view].parameters():
                param.requires_grad = True

    def forward(self, x):
        # Features
        f_maps = self.features(x)

        # Pooling
        h = self.all_views_avg_pool(f_maps)

        # Fully connected layers, batch normalization and dropout (if included)
        h_l_cc = self.fc1_l_cc(h[VIEWS.L_CC])
        if self.opts['batch_norm']:
            h_l_cc = self.bn1d_l_cc(h_l_cc)
        h_l_cc = F.relu(h_l_cc)
        if self.opts['dropout']['state']:
            h_l_cc = self.fc_drop_l_cc(h_l_cc)

        h_r_cc = self.fc1_r_cc(h[VIEWS.R_CC])
        if self.opts['batch_norm']:
            h_r_cc = self.bn1d_r_cc(h_r_cc)
        h_r_cc = F.relu(h_r_cc)
        if self.opts['dropout']['state']:
            h_r_cc = self.fc_drop_r_cc(h_r_cc)

        h_l_mlo = self.fc1_l_mlo(h[VIEWS.L_MLO])
        if self.opts['batch_norm']:
            h_l_mlo = self.bn1d_l_mlo(h_l_mlo)
        h_l_mlo = F.relu(h_l_mlo)
        if self.opts['dropout']['state']:
            h_l_mlo = self.fc_drop_l_mlo(h_l_mlo)

        h_r_mlo = self.fc1_r_mlo(h[VIEWS.R_MLO])
        if self.opts['batch_norm']:
            h_r_mlo = self.bn1d_r_mlo(h_r_mlo)
        h_r_mlo = F.relu(h_r_mlo)
        if self.opts['dropout']['state']:
            h_r_mlo = self.fc_drop_r_mlo(h_r_mlo)

        # Output layers
        y_l_cc = torch.sigmoid(self.fc2_l_cc(h_l_cc))
        y_r_cc = torch.sigmoid(self.fc2_r_cc(h_r_cc))
        y_l_mlo = torch.sigmoid(self.fc2_l_mlo(h_l_mlo))
        y_r_mlo = torch.sigmoid(self.fc2_r_mlo(h_r_mlo))

        # Final average
        y = torch.cat((y_l_cc, y_r_cc, y_l_mlo, y_r_mlo), dim=1).mean(1)
        return y


class BreastWiseRiskModel(nn.Module):
    def __init__(self, pretrained=True, weights_path=_DEFAULT_WEIGHTS, opts=_DEFAULT_MODEL_SETTINGS):
        super().__init__()
        self.opts = opts
        cancer_class_model = CancerClassModel(input_channels=1)
        if pretrained:
            # Load weights
            state_dict = torch.load(weights_path, map_location='cpu')["model"]  # add map_location='cpu' if no gpu
            cancer_class_model.load_state_dict(state_dict)

        # Feature extractor
        self.features = copy.deepcopy(cancer_class_model.four_view_resnet)
        if pretrained:
            self.freeze()
        del cancer_class_model

        # AVG-Pooling layer
        self.all_views_avg_pool = layers.AllViewsAvgPool()

        # Batch norm layers
        if opts['batch_norm']:
            self.bn1d_l = nn.BatchNorm1d(256 * 2)
            self.bn1d_r = nn.BatchNorm1d(256 * 2)

        # Dropout layers
        if opts['dropout']['state']:
            self.fc_drop_l = nn.Dropout(opts['dropout']['p'])
            self.fc_drop_r = nn.Dropout(opts['dropout']['p'])

        # Fully-connected layers
        self.fc1_l = nn.Linear(256 * 2, 256 * 2)
        self.fc1_r = nn.Linear(256 * 2, 256 * 2)

        # Output layers
        self.fc2_l = nn.Linear(256 * 2, 1)
        self.fc2_r = nn.Linear(256 * 2, 1)

    def freeze(self):
        for view in VIEWS.LIST:
            for param in self.features.model_dict[view].parameters():
                param.requires_grad = False

    def unfreeze(self):
        for view in VIEWS.LIST:
            for param in self.features.model_dict[view].parameters():
                param.requires_grad = True

    def forward(self, x):
        # Features
        f_maps = self.features(x)

        # Pooling
        h = self.all_views_avg_pool(f_maps)
        h_l = torch.cat([h[VIEWS.L_CC], h[VIEWS.L_MLO]], dim=1)
        h_r = torch.cat([h[VIEWS.R_CC], h[VIEWS.R_MLO]], dim=1)

        # Fully connected layers, batch normalization and dropout (if included)
        h_l = self.fc1_l(h_l)
        if self.opts['batch_norm']:
            h_l = self.bn1d_l(h_l)
        h_l = F.relu(h_l)
        if self.opts['dropout']['state']:
            h_l = self.fc_drop_l(h_l)

        h_r = self.fc1_r(h_r)
        if self.opts['batch_norm']:
            h_r = self.bn1d_r(h_r)
        h_r = F.relu(h_r)
        if self.opts['dropout']['state']:
            h_r = self.fc_drop_r(h_r)

        # Output layers
        y_l = torch.sigmoid(self.fc2_l(h_l))
        y_r = torch.sigmoid(self.fc2_r(h_r))

        # Final average
        y = torch.cat((y_l, y_r), dim=1).mean(1)
        return y


class JointRiskModel(nn.Module):
    def __init__(self, pretrained=True, weights_path=_DEFAULT_WEIGHTS, opts=_DEFAULT_MODEL_SETTINGS):
        super().__init__()
        self.opts = opts
        cancer_class_model = CancerClassModel(input_channels=1)
        if pretrained:
            # Load weights
            state_dict = torch.load(weights_path, map_location='cpu')["model"]  # add map_location='cpu' if no gpu
            cancer_class_model.load_state_dict(state_dict)

        # Feature extractor
        self.features = copy.deepcopy(cancer_class_model.four_view_resnet)
        if pretrained:
            self.freeze()
        del cancer_class_model

        # AVG-Pooling layer
        self.all_views_avg_pool = layers.AllViewsAvgPool()

        # Fully-connected layers
        self.fc1 = nn.Linear(256 * 4, 256 * 2)

        # Batch norm layers
        if opts['batch_norm']:
            self.bn1d = nn.BatchNorm1d(256 * 2)

        # Dropout layers
        if opts['dropout']['state']:
            self.fc_drop = nn.Dropout(opts['dropout']['p'])

        # Output layers
        self.fc2 = nn.Linear(256 * 2, 1)

    def freeze(self):
        for view in VIEWS.LIST:
            for param in self.features.model_dict[view].parameters():
                param.requires_grad = False

    def unfreeze(self):
        for view in VIEWS.LIST:
            for param in self.features.model_dict[view].parameters():
                param.requires_grad = True

    def forward(self, x):
        # Features
        f_maps = self.features(x)

        # Pooling
        h = self.all_views_avg_pool(f_maps)
        h = torch.cat([h[view] for view in VIEWS.LIST], dim=1)

        # Fully connected layers
        h1 = self.fc1(h)
        if self.opts['batch_norm']:
            h1 = self.bn1d(h1)
        h1 = F.relu(h1)
        if self.opts['dropout']['state']:
            h1 = self.fc_drop(h1)

        # Output layers
        y = torch.sigmoid(self.fc2(h1)).squeeze(dim=1)
        return y