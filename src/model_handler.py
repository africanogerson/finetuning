from __future__ import print_function, division
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.nn.modules.loss import BCELoss
from torch.optim import lr_scheduler
from datetime import datetime
import socket
import os

from src.constants import MODELMODES, OPTIMIZERS
import src.risk_models as models


class BaseRiskModel:
    def __init__(self, options: dict, mode='train'):
        self.opts = options

        # Device checking
        self.device = torch.device(options['model']['GPU_use'] if torch.cuda.is_available() and options['use_gpu'] else "cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Load model
        self.setup_model(self.opts['model']['variant'])

        if mode is 'train':
            # Optimizer
            self.setup_optimizer()
            # Loss function
            self.loss = BCELoss(reduction='sum')
            # Scheduler
            self.setup_lr_scheduler()

        # Setting checkpoint manager
        self.ckpt_mgr = CheckpointSaver(self.opts['ckpt_path'])
        if self.opts['ckpt_path'] is not None:
            self.load_ckpt(mode)

        # Setup logger
        if self.opts['logger']['state'] and mode is 'train':
            self.logger = SummaryWriter(self.get_log_dir(self.opts['logger']['path']))

    def setup_model(self, variant: str):
        """
        Setup network model
        """
        assert variant in MODELMODES.LIST
        if self.opts['verbose']:
            print("Configuring model variant: {}".format(variant))
        model_class = {
            MODELMODES.VIEW: models.ViewWiseRiskModel,
            MODELMODES.IMAGE: models.ImageWiseRiskModel,
            MODELMODES.BREAST: models.BreastWiseRiskModel,
            MODELMODES.JOINT: models.JointRiskModel
        }[variant]
        model = model_class(
            opts=self.opts['model'],
            device=self.device)

        self.model = model.to(self.device)
        print(self.device)

    def setup_optimizer(self):
        """
        Setup optimization algorithm
        """
        optimizer_opts = self.opts['optimizer']
        assert optimizer_opts['variant'] in OPTIMIZERS.LIST
        if self.opts['verbose']:
            print("Configuring {} as optimizer...".format(optimizer_opts['variant']))
        optimizer_class = {
            OPTIMIZERS.ADAM: optim.Adam,
            OPTIMIZERS.SGD: optim.SGD
        }[optimizer_opts['variant']]
        weight_decay = optimizer_opts['weight_decay'] if type(optimizer_opts['weight_decay']) is int \
            else eval(optimizer_opts['weight_decay'])
        self.optimizer= optimizer_class(self.model.parameters(),
                                        lr=optimizer_opts['lr'],
                                        weight_decay=weight_decay
                                        )

    def setup_lr_scheduler(self):
        """
        Setup learning rate scheduler
        """
        if self.opts['verbose']:
            print("Configuring multistep lr scheduler...")

        scheduler_opts = self.opts['scheduler']
        milestones = list(
            range(scheduler_opts['step'],
                  self.opts['train']['num_epochs'],
                  scheduler_opts['step'])
        )
        self.scheduler = lr_scheduler.MultiStepLR(self.optimizer, milestones=milestones,
                                                  gamma=scheduler_opts['gamma'])

    def load_ckpt(self, mode):
        """Load trained model"""
        if mode in ['test', 'inference']:
            if self.opts['verbose']:
                print("Loading trained model...")
            ckpt = self.ckpt_mgr.load_best()
            if ckpt != {}:
                self.model.load_state_dict(ckpt)

    def unfreeze(self):
        self.model.unfreeze()

    @staticmethod
    def get_log_dir(log_path):
        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        log_dir = os.path.join(
            log_path, current_time + '_' + socket.gethostname())
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        return log_dir


class CheckpointSaver(object):
    """Class definition for the checkpoint manager"""
    def __init__(self, ckpt_path: str):

        folder = os.path.dirname(os.path.abspath(ckpt_path))
        if not os.path.exists(folder):
            os.makedirs(folder)

        self._ckpt_path = ckpt_path

    def save_best(self, best_state):
        torch.save(best_state, self._ckpt_path)

    def load_best(self):
        return torch.load(self._ckpt_path, map_location='cpu')
