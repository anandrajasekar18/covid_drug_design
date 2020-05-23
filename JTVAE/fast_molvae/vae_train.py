import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from torch.autograd import Variable

import math, random, sys
import numpy as np
import argparse
from collections import deque
import _pickle as pickle

from fast_jtnn.vocab import Vocab
from fast_jtnn.jtnn_vae import JTNNVAE
from fast_jtnn.datautils import MolTreeFolder
import rdkit


from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import loggers

# lg = rdkit.RDLogger.logger() 
# lg.setLevel(rdkit.RDLogger.CRITICAL)

parser = argparse.ArgumentParser()
parser.add_argument('--train', required=True)
parser.add_argument('--vocab', required=True)
parser.add_argument('--save_dir', required=True)
parser.add_argument('--load_epoch', type=int, default=0)

parser.add_argument('--hidden_size', type=int, default=450)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--latent_size', type=int, default=56)
parser.add_argument('--depthT', type=int, default=20)
parser.add_argument('--depthG', type=int, default=3)

parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--clip_norm', type=float, default=50.0)
parser.add_argument('--opt', type=str, default='adam')
parser.add_argument('--beta', type=float, default=0.0)
parser.add_argument('--step_beta', type=float, default=0.002)
parser.add_argument('--max_beta', type=float, default=1.0)
parser.add_argument('--warmup', type=int, default=40000)

parser.add_argument('--epoch', type=int, default=20)
parser.add_argument('--anneal_rate', type=float, default=0.9)
parser.add_argument('--anneal_iter', type=int, default=40000)
parser.add_argument('--kl_anneal_iter', type=int, default=2000)
parser.add_argument('--print_iter', type=int, default=50)
parser.add_argument('--save_iter', type=int, default=5000)

args = parser.parse_args()
# print(args)

optimizers = {
    'adam': optim.Adam,
    'adamax': optim.Adamax,
    'rmsprop': optim.RMSprop,
    }

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class myJTNNVAE(LightningModule):
    def __init__(self, args=args):
        super(myJTNNVAE, self).__init__()
        self.args = args
        vocab = [x.strip("\r\n ") for x in open(args.vocab)] 
        self.vocab = Vocab(vocab)
        self.model = JTNNVAE(self.vocab, args.hidden_size, args.latent_size, args.depthT, args.depthG)
        self.total_step = args.load_epoch
        self.beta = args.beta
        # self.meters = np.zeros(4)

        for param in self.model.parameters():
            if param.dim() == 1:
                nn.init.constant_(param, 0)
            else:
                nn.init.xavier_normal_(param)

    def forward(self,x, beta):
        return self.model(x, beta)

    def configure_optimizers(self):
        """
            Choose Optimizer
        """
        optimizer = optimizers[self.args.opt](self.parameters(), lr=self.args.lr)
        scheduler = lr_scheduler.ExponentialLR(optimizer, self.args.anneal_rate)
        return [optimizer], [scheduler]

    
    def training_step(self, batch, batch_idx):
        """
            Define one training step
        """
        self.total_step += 1
        loss, kl_div, wacc, tacc, sacc = self(batch, self.beta)

        # self.meters = self.meters + np.array([kl_div, wacc * 100, tacc * 100, sacc * 100])

        if self.total_step % self.args.kl_anneal_iter == 0 and self.total_step >= self.args.warmup:
            self.beta = min(self.args.max_beta, self.beta + self.args.step_beta)
        
        # if self.total_step % self.args.print_iter == 0:
        #     self.meters /= self.args.print_iter
        #     # print("[%d] Beta: %.3f, KL: %.2f, Word: %.2f, Topo: %.2f, Assm: %.2f, PNorm: %.2f, GNorm: %.2f" % (total_step, beta, meters[0], meters[1], meters[2], meters[3], param_norm(model), grad_norm(model)))
        #     self.meters *= 0
        if self.total_step % self.args.save_iter == 0:
            torch.save(self.model.state_dict(), self.args.save_dir + "/model.iter-" + str(self.total_step))

        tensorboard_log = {'trainer_loss':loss,'kl_div':kl_div,'beta':self.beta,}
        return {'loss': loss, 'log': tensorboard_log}

    def validation_step(self, batch, batch_idx):
        """
            Define one training step
        """
        loss, kl_div, wacc, tacc, sacc = self(batch, self.beta)

        # self.meters = self.meters + np.array([kl_div, wacc * 100, tacc * 100, sacc * 100])

        
        tensorboard_log = {'val_loss':loss,'val_kl_div':kl_div,'val_beta':self.beta,}
        return {'loss': loss, 'log': tensorboard_log}

    def param_norm(self, m):
        return math.sqrt(sum([p.norm().item() ** 2 for p in m.parameters()]))

    def grad_norm(self, m):
        return math.sqrt(sum([p.grad.norm().item() ** 2 for p in m.parameters() if p.grad is not None]))

    def train_dataloader(self):
        return MolTreeFolder(self.args.train, self.vocab, self.args.batch_size)

    def val_dataloader(self):
        return MolTreeFolder(self.args.train, self.vocab, self.args.batch_size)

if __name__ == "__main__":
    model = myJTNNVAE()
    tb_logger = loggers.TensorBoardLogger('logs/')

    trainer = Trainer(
                        fast_dev_run=False,                      # make this as True only to check for bugs
                        max_epochs=args.epoch,
                        gradient_clip_val=args.clip_norm,        # gradient clipping
                        logger=tb_logger,                       # tensorboard logger
                        val_percent_check=0.0,
                        nb_sanity_val_steps=0,
                        )
    trainer.fit(model)