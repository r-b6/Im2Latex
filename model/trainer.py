import torch
import numpy as np
import pandas as pd
from score import cal_loss
from torch.nn.utils import clip_grad_norm_
import os


class Trainer():
    def __init__(self, optimizer, model, lr_scheduler, train_loaders, val_loaders, args, num_epochs, start):
        self.optimizer = optimizer
        self.model = model
        self.lr_scheduler = lr_scheduler
        self.train_loaders = train_loaders
        self.val_loaders = val_loaders
        self.args = args
        self.num_epochs = num_epochs
        self.device = torch.device("cuda" if args.cuda else "cpu")
        self.best_val_loss = 1e18
        self.epoch = start
        self.step = 1
        #how often to print results
        self.print_freq = 100
    def get_total_len(self):
        """
        :return: total number of batches in one epoch, used to display progress during training
        """
        length = 0
        for loader in self.train_loaders:
            length += len(loader)
        return length

    def train(self):
        total_len = self.get_total_len()
        mes = "Epoch {}, step:{}/{} {:.2f}%, Loss:{:.4f}, Perplexity:{:.4f}, LR: {:.5f}"
        while self.epoch <= self.num_epochs:
            self.model.train()
            losses = 0
            for loader in self.train_loaders:
                for images, formulas in loader:
                    #targets don't have start token
                    targets = formulas[:, 1:]
                    step_loss, norm = self.train_step(images, formulas, targets)
                    losses += step_loss

                    # log message
                    if self.step % self.print_freq == 0:
                        avg_loss = losses / self.print_freq
                        print(mes.format(
                            self.epoch, self.step, total_len,
                            100 * self.step / total_len,
                            avg_loss,
                            2 ** avg_loss,
                            self.lr_scheduler.get_last_lr()[0]
                        ))
                        losses = 0.0
            val_loss = self.validate()
            self.lr_scheduler.step(val_loss)
            self.step = 1
            self.epoch += 1
            self.save_model('ckpt-{}-{:.4f}'.format(self.epoch, val_loss), self.epoch)

    def train_step(self, images, formulas, targets):
        self.optimizer.zero_grad()
        images = images.to(self.device)
        formulas = formulas.to(self.device)
        targets = targets.to(self.device)
        logits = self.model(images, formulas)
        loss = cal_loss(logits, targets, 0)
        loss.backward()
        norm = clip_grad_norm_(self.model.parameters(), self.args.clip, error_if_nonfinite=True)
        self.optimizer.step()
        self.step += 1
        return loss.item(), norm

    def validate(self):
        self.model.eval()
        val_total_loss = 0
        steps = 0
        mes = "Epoch {}, validation average loss:{:.4f}, Perplexity:{:.4f}"
        with torch.no_grad():
            for loader in self.val_loaders:
                for images, formulas in loader:
                    # targets don't include start token
                    targets = formulas[:, 1:]
                    images = images.to(self.device)
                    formulas = formulas.to(self.device)
                    targets = targets.to(self.device)
                    logits = self.model(images, formulas)
                    loss = cal_loss(logits, targets, 0)
                    val_total_loss += loss
                    steps += 1
            average_loss = val_total_loss / steps
            print(mes.format(
                self.epoch, average_loss, 2 ** average_loss
            ))
        if average_loss < self.best_val_loss:
            self.best_val_loss = average_loss
            self.save_model('best_ckpt', self.epoch)
        return average_loss

    def save_model(self, name, epoch):
        if not os.path.isdir(self.args.save_dir):
            os.makedirs(self.args.save_dir)
        save_path = os.path.join(self.args.save_dir, '{}.pt'.format(name))
        print("Saving checkpoint to {}".format(save_path))
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lr_sche': self.lr_scheduler.state_dict(),
            'args': self.args
        }, save_path)
