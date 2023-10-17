from templates import *
from templates_latent import *
import os
import torch
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model_path', '-m', type=str, default="./checkpoints/nature1024/last.ckpt",
                        help='Patch-DM model path')
    parser.add_argument('--semantic_enc', action='store_true',
                        help='use semantic encoder')
    parser.add_argument('--name', '-n', type=str, default="train",
                        help='experiment name')
    args = parser.parse_args()

    model_path = args.model_path

    conf = train_autoenc_latent()
    gpus = [0]
    conf.sample_size = 1
    conf.semantic_enc = args.semantic_enc
    conf.pretrain = PretrainConfig(
        name='72M',
        path=model_path,
    )
    conf.name = args.name
    train(conf, gpus=gpus)
