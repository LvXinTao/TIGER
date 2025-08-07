import argparse
import random
import torch
import gin
import numpy as np
from time import time
import logging

from torch.utils.data import DataLoader

from datasets import EmbDataset
from models.rqvae import RQVAE
from trainer import  Trainer
from utils import fix_everything

def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, help="Path to gin config file.")
    args = parser.parse_args()
    gin.parse_config_file(args.config_path)


@gin.configurable('train')
def train(
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
    num_emb_list,
    e_dim,
    quant_loss_weight,
    beta,
    layers,
    save_limit,
    ckpt_dir,
    data_path
):
    
    fix_everything()
    
    print("=================================================")
    print(gin.config_str())
    print("=================================================")

    
    """build dataset"""
    data = EmbDataset(data_path)
    model = RQVAE(in_dim=data.dim,
                  num_emb_list=num_emb_list,
                  e_dim=e_dim,
                  layers=layers,
                  dropout_prob=dropout_prob,
                  bn=bn,
                  loss_type=loss_type,
                  quant_loss_weight=quant_loss_weight,
                  beta=beta,
                  kmeans_init=kmeans_init,
                  kmeans_iters=kmeans_iters,
                  sk_epsilons=sk_epsilons,
                  sk_iters=sk_iters,
                  )
    data_loader = DataLoader(data,num_workers=num_workers,
                             batch_size=batch_size, shuffle=True,
                             pin_memory=True)
                            
    trainer = Trainer(  lr,
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
                        num_emb_list,
                        e_dim,
                        quant_loss_weight,
                        beta,
                        layers,
                        save_limit,
                        ckpt_dir,
                        data_path,
                        model=model,
                        data_num=len(data_loader))
    
    best_loss, best_collision_rate = trainer.fit(data_loader)

    print("Best Loss",best_loss)
    print("Best Collision Rate", best_collision_rate)

if __name__ == '__main__':
    parse_config()
    train()

