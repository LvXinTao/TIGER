import logging

import numpy as np
import torch
from time import time
from torch import optim
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup, get_constant_schedule_with_warmup

from utils import ensure_dir,set_color,get_local_time,delete_file
import os

import heapq
import wandb
class Trainer(object):

    def __init__(self, 
                    lr,
                    epochs,
                    batch_size,
                    num_workers,
                    eval_step,
                    learner,
                    lr_scheduler_type,
                    warmup_epochs,
                    weight_decay,
                    dropout_prob,
                    bn,
                    loss_type,
                    kmeans_init,
                    kmeans_iters,
                    sk_epsilons,
                    sk_iters,
                    device,
                    wandb_logging,
                    num_emb_list,
                    e_dim,
                    quant_loss_weight,
                    beta,
                    layers,
                    save_limit,
                    ckpt_dir,
                    data_path,
                    model, 
                    data_num):
        self.model = model
        self.logger = logging.getLogger()

        self.lr = lr
        self.learner = learner
        self.lr_scheduler_type = lr_scheduler_type

        self.weight_decay = weight_decay
        self.epochs = epochs
        self.warmup_steps = warmup_epochs * data_num
        self.max_steps = epochs * data_num

        self.save_limit = save_limit
        self.wandb_logging = wandb_logging
        self.best_save_heap = []
        self.newest_save_queue = []
        self.eval_step = min(eval_step, self.epochs)
        self.device = device
        self.device = torch.device(self.device)
        self.ckpt_dir = ckpt_dir
        saved_model_dir = "{}".format(get_local_time())
        self.ckpt_dir = os.path.join(self.ckpt_dir,saved_model_dir)
        ensure_dir(self.ckpt_dir)

        self.best_loss = np.inf
        self.best_collision_rate = np.inf
        self.best_loss_ckpt = "best_loss_model.pth"
        self.best_collision_ckpt = "best_collision_model.pth"
        self.optimizer = self._build_optimizer()
        self.scheduler = self._get_scheduler()
        self.model = self.model.to(self.device)

    def _build_optimizer(self):

        params = self.model.parameters()
        learner =  self.learner
        learning_rate = self.lr
        weight_decay = self.weight_decay

        if learner.lower() == "adam":
            optimizer = optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)
        elif learner.lower() == "sgd":
            optimizer = optim.SGD(params, lr=learning_rate, weight_decay=weight_decay)
        elif learner.lower() == "adagrad":
            optimizer = optim.Adagrad(
                params, lr=learning_rate, weight_decay=weight_decay
            )
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(self.device)
        elif learner.lower() == "rmsprop":
            optimizer = optim.RMSprop(
                params, lr=learning_rate, weight_decay=weight_decay
            )
        elif learner.lower() == 'adamw':
            optimizer = optim.AdamW(
                params, lr=learning_rate, weight_decay=weight_decay
            )
        else:
            self.logger.warning(
                "Received unrecognized optimizer, set default Adam optimizer"
            )
            optimizer = optim.Adam(params, lr=learning_rate)
        return optimizer

    def _get_scheduler(self):
        if self.lr_scheduler_type.lower() == "linear":
            lr_scheduler = get_linear_schedule_with_warmup(optimizer=self.optimizer,
                                                           num_warmup_steps=self.warmup_steps,
                                                           num_training_steps=self.max_steps)
        else:
            lr_scheduler = get_constant_schedule_with_warmup(optimizer=self.optimizer,
                                                             num_warmup_steps=self.warmup_steps)

        return lr_scheduler

    def _check_nan(self, loss):
        if torch.isnan(loss):
            raise ValueError("Training loss is nan")


    def _train_epoch(self, train_data, epoch_idx):

        self.model.train()

        total_loss = 0
        total_recon_loss = 0
        total_rq_loss = 0

        for batch_idx, data in enumerate(train_data):
            data = data.to(self.device)
            self.optimizer.zero_grad()
            out, rq_loss, indices = self.model(data)
            loss, loss_recon = self.model.compute_loss(out, rq_loss, xs=data)
            self._check_nan(loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()
            # print(self.scheduler.get_last_lr())
            total_loss += loss.item()
            total_recon_loss += loss_recon.item()
            total_rq_loss += rq_loss.item()

        return total_loss, total_recon_loss, total_rq_loss

    @torch.no_grad()
    def _valid_epoch(self, valid_data):

        self.model.eval()

        indices_set = set()
        num_sample = 0
        for batch_idx, data in enumerate(valid_data):
            num_sample += len(data)
            data = data.to(self.device)
            indices = self.model.get_indices(data)
            indices = indices.view(-1,indices.shape[-1]).cpu().numpy()
            for index in indices:
                code = "-".join([str(int(_)) for _ in index])
                indices_set.add(code)

        collision_rate = (num_sample - len(list(indices_set)))/num_sample

        return collision_rate

    def _save_checkpoint(self, epoch, collision_rate=1, ckpt_file=None):

        ckpt_path = os.path.join(self.ckpt_dir,ckpt_file) if ckpt_file \
            else os.path.join(self.ckpt_dir, 'epoch_%d_collision_%.4f_model.pth' % (epoch, collision_rate))
        state = {
            "epoch": epoch,
            "best_loss": self.best_loss,
            "best_collision_rate": self.best_collision_rate,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        torch.save(state, ckpt_path, pickle_protocol=4)

        return ckpt_path

    def fit(self, data):

        cur_eval_step = 0

        with tqdm(total=self.epochs, desc=set_color("Training", "green")) as pbar:
            for epoch_idx in range(self.epochs):
                total_loss = 0
                total_recon_loss = 0
                total_rq_loss = 0
                # train
                training_start_time = time()
                train_loss, train_recon_loss, train_rq_loss = self._train_epoch(data, epoch_idx)
                training_end_time = time()
                pbar.set_postfix({
                    "train_loss": train_loss,
                    "train_recon_loss": train_recon_loss,
                    "train_rq_loss": train_rq_loss,
                })
                
                
                total_loss += train_loss
                total_recon_loss += train_recon_loss
                total_rq_loss += train_rq_loss

                if self.wandb_logging:
                    train_log = {
                        "train_learning_rate": self.scheduler.get_last_lr()[0],
                        "train_loss": train_loss,
                        "train_recon_loss": train_recon_loss,
                        "train_rq_loss": train_rq_loss,
                    }

                # eval
                eval_log = {}
                if (epoch_idx + 1) % self.eval_step == 0:
                    collision_rate = self._valid_epoch(data)
                    if self.wandb_logging:
                        eval_log = {
                            "eval_collision_rate": collision_rate,
                        }

                    if train_loss < self.best_loss:
                        self.best_loss = train_loss
                        self._save_checkpoint(epoch=epoch_idx, ckpt_file=self.best_loss_ckpt)

                    if collision_rate < self.best_collision_rate:
                        self.best_collision_rate = collision_rate
                        cur_eval_step = 0
                        self._save_checkpoint(epoch_idx, collision_rate=collision_rate,
                                            ckpt_file=self.best_collision_ckpt)
                    else:
                        cur_eval_step += 1

                    ckpt_path = self._save_checkpoint(epoch_idx, collision_rate=collision_rate)
                    now_save = (-collision_rate, ckpt_path)
                    if len(self.newest_save_queue) < self.save_limit:
                        self.newest_save_queue.append(now_save)
                        heapq.heappush(self.best_save_heap, now_save)
                    else:
                        old_save = self.newest_save_queue.pop(0)
                        self.newest_save_queue.append(now_save)
                        if collision_rate < -self.best_save_heap[0][0]:
                            bad_save = heapq.heappop(self.best_save_heap)
                            heapq.heappush(self.best_save_heap, now_save)

                            if bad_save not in self.newest_save_queue:
                                delete_file(bad_save[1])

                        if old_save not in self.best_save_heap:
                            delete_file(old_save[1])
                
                if self.wandb_logging:
                    wandb.log({
                        **train_log,
                        **eval_log,
                    })
                pbar.update(1)

        return self.best_loss, self.best_collision_rate




