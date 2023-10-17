import copy
import json
import os
import re
import random
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from numpy.lib.function_base import flip
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import *
from torch import nn
from torch.cuda import amp
from torch.distributions import Categorical
from torch.optim.optimizer import Optimizer
from torch.utils.data.dataset import ConcatDataset, TensorDataset
from torchvision.utils import make_grid, save_image

from config import *
from utils.dataset import *
from utils.dist_utils import *
from utils.lmdb_writer import *
from utils.metrics import *
from utils.renderer import *

from einops import rearrange
from torchvision import transforms


class LitModel(pl.LightningModule):
    def __init__(self, conf: TrainConfig):
        super().__init__()
        assert conf.train_mode != TrainMode.manipulate
        if conf.seed is not None:
            pl.seed_everything(conf.seed)

        self.save_hyperparameters(conf.as_dict_jsonable())

        self.conf = conf
        self.model = conf.make_model_conf().make_model()
        self.ema_model = copy.deepcopy(self.model)
        self.ema_model.requires_grad_(False)
        self.ema_model.eval()

        model_size = 0
        for param in self.model.parameters():
            model_size += param.data.nelement()
        print('Model params: %.2f M' % (model_size / 1024 / 1024))
        self.patch_size = conf.patch_size
        print(f"==Model size is {self.patch_size}==")
        self.sampler = conf.make_diffusion_conf().make_sampler()
        self.eval_sampler = conf.make_eval_diffusion_conf().make_sampler()

        # this is shared for both model and latent
        self.T_sampler = conf.make_T_sampler()

        if conf.train_mode.use_latent_net():
            self.latent_sampler = conf.make_latent_diffusion_conf(
            ).make_sampler()
            self.eval_latent_sampler = conf.make_latent_eval_diffusion_conf(
            ).make_sampler()
        else:
            self.latent_sampler = None
            self.eval_latent_sampler = None
        if conf.pretrain is not None:
            print(f'loading pretrain ... {conf.pretrain.name}')
            state = torch.load(conf.pretrain.path, map_location='cpu')
            print('step:', state['global_step'])
            state_value = dict()
            for key in state['state_dict']:
                if "model.semantic_enc.semantic_enc.weight" in key:
                    state_value["model.encoder.semantic_enc.weight"] = state['state_dict'][key]
                elif "ema_model.semantic_enc.semantic_enc.weight" in key:
                    state_value["ema_model.encoder.semantic_enc.weight"] = state['state_dict'][key]
                else:
                    state_value[key] = state['state_dict'][key]
            self.load_state_dict(state_value, strict=False)

            if conf.train_mode.use_latent_net():
                self.conds = state['state_dict']['model.encoder.semantic_enc.weight'].float()
                conds_mean = self.conds.mean(dim = 0)
                conds_std = self.conds.std(dim = 0)
                self.register_buffer('conds_mean', conds_mean[None, :])
                self.register_buffer('conds_std', conds_std[None, :])
            else:
                self.conds_mean = None
                self.conds_std = None

        else:
            self.conds_mean = None
            self.conds_std = None

    def normalize(self, cond):
        cond = (cond - self.conds_mean.to(self.device)) / self.conds_std.to(
            self.device)
        return cond

    def denormalize(self, cond):
        cond = (cond * self.conds_std.to(self.device)) + self.conds_mean.to(
            self.device)
        return cond

    def sample(self, N, device, T=None, T_latent=None):
        if T is None:
            sampler = self.eval_sampler
            latent_sampler = self.latent_sampler
        else:
            sampler = self.conf._make_diffusion_conf(T).make_sampler()
            latent_sampler = self.conf._make_latent_diffusion_conf(T_latent).make_sampler()

        noise = torch.randn(N,
                            3,
                            self.conf.img_size,
                            self.conf.img_size,
                            device=device)
        pred_img = render_uncondition(
            self.conf,
            self.ema_model,
            noise,
            sampler=sampler,
            latent_sampler=latent_sampler,
            conds_mean=self.conds_mean,
            conds_std=self.conds_std,
        )
        pred_img = (pred_img + 1) / 2
        return pred_img

    def render(self, noise, cond=None, T=None):
        if T is None:
            sampler = self.eval_sampler
        else:
            sampler = self.conf._make_diffusion_conf(T).make_sampler()

        if cond is not None:
            pred_img = render_condition(self.conf,
                                        self.ema_model,
                                        noise,
                                        sampler=sampler,
                                        cond=cond)
        else:
            pred_img = render_uncondition(self.conf,
                                          self.ema_model,
                                          noise,
                                          sampler=sampler,
                                          latent_sampler=None)
        pred_img = (pred_img + 1) / 2
        return pred_img

    def encode(self, x):
        assert self.conf.model_type.has_autoenc()
        cond = self.ema_model.encoder.forward(x)
        return cond

    def encode_stochastic(self, x, cond, T=None):
        if T is None:
            sampler = self.eval_sampler
        else:
            sampler = self.conf._make_diffusion_conf(T).make_sampler()
        out = sampler.ddim_reverse_sample_loop(self.ema_model,
                                               x,
                                               model_kwargs={'cond': cond})
        return out['sample']

    def forward(self, noise=None, x_start=None, ema_model: bool = False):
        with amp.autocast(False):
            if ema_model:
                model = self.ema_model
            else:
                model = self.model
            gen = self.eval_sampler.sample(model=model,
                                           noise=noise,
                                           x_start=x_start)
            return gen

    def setup(self, stage=None) -> None:
        """
        make datasets & seeding each worker separately
        """
        ##############################################
        # NEED TO SET THE SEED SEPARATELY HERE
        if self.conf.seed is not None:
            seed = self.conf.seed * get_world_size() + self.global_rank
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            print('local seed:', seed)
        ##############################################

        self.train_data = self.conf.make_dataset()
        if self.train_data is not None:
            print('train data:', len(self.train_data))
        self.val_data = self.train_data
        if self.val_data is not None:
            print('val data:', len(self.val_data))

    def _train_dataloader(self, drop_last=True):
        """
        really make the dataloader
        """
        # make sure to use the fraction of batch size
        # the batch size is global!
        conf = self.conf.clone()
        conf.batch_size = self.batch_size

        dataloader = conf.make_loader(self.train_data,
                                      shuffle=True,
                                      drop_last=drop_last)
        return dataloader

    def train_dataloader(self):
        """
        return the dataloader, if diffusion mode => return image dataset
        if latent mode => return the inferred latent dataset
        """
        print('on train dataloader start ...')
        if self.conf.train_mode.require_dataset_infer():
            if self.conds is None:
                # usually we load self.conds from a file
                # so we do not need to do this again!
                self.conds = self.infer_whole_dataset()
                self.conds_mean.data = self.conds.float().mean(dim=0,
                                                               keepdim=True)
                self.conds_std.data = self.conds.float().std(dim=0,
                                                             keepdim=True)
            print('mean:', self.conds_mean.mean(), 'std:',
                  self.conds_std.mean())

            # return the dataset with pre-calculated conds
            conf = self.conf.clone()
            conf.batch_size = self.batch_size
            data = TensorDataset(self.conds)
            return conf.make_loader(data, shuffle=True)
        else:
            return self._train_dataloader()

    @property
    def batch_size(self):
        """
        local batch size for each worker
        """
        ws = get_world_size()
        assert self.conf.batch_size % ws == 0
        return self.conf.batch_size // ws

    @property
    def num_samples(self):
        """
        (global) batch size * iterations
        """
        return self.global_step * self.conf.batch_size_effective

    def is_last_accum(self, batch_idx):
        """
        is it the last gradient accumulation loop? 
        used with gradient_accum > 1 and to see if the optimizer will perform "step" in this iteration or not
        """
        return (batch_idx + 1) % self.conf.accum_batches == 0

    def infer_whole_dataset(self,
                            with_render=False,
                            T_render=None,
                            render_save_path=None):
        """
        predicting the latents given images using the encoder

        Args:
            both_flips: include both original and flipped images; no need, it's not an improvement
            with_render: whether to also render the images corresponding to that latent
            render_save_path: lmdb output for the rendered images
        """
        data = self.conf.make_dataset()
        if isinstance(data, CelebAlmdb) and data.crop_d2c:
            # special case where we need the d2c crop
            data.transform = make_transform(self.conf.img_size,
                                            flip_prob=0,
                                            crop_d2c=True)
        else:
            data.transform = make_transform(self.conf.img_size, flip_prob=0)


        loader = self.conf.make_loader(
            data,
            shuffle=False,
            drop_last=False,
            batch_size=self.conf.batch_size_eval,
            parallel=True,
        )
        model = self.ema_model
        model.eval()
        conds = []

        if with_render:
            sampler = self.conf._make_diffusion_conf(
                T=T_render or self.conf.T_eval).make_sampler()

            if self.global_rank == 0:
                writer = LMDBImageWriter(render_save_path,
                                         format='webp',
                                         quality=100)
            else:
                writer = nullcontext()
        else:
            writer = nullcontext()

        with writer:
            for batch in tqdm(loader, total=len(loader), desc='infer'):
                with torch.no_grad():
                    # (n, c)
                    # print('idx:', batch['index'])
                    cond = model.encoder(batch['img'].to(self.device))

                    # used for reordering to match the original dataset
                    idx = batch['index']
                    idx = self.all_gather(idx)
                    if idx.dim() == 2:
                        idx = idx.flatten(0, 1)
                    argsort = idx.argsort()

                    if with_render:
                        noise = torch.randn(len(cond),
                                            3,
                                            self.conf.img_size,
                                            self.conf.img_size,
                                            device=self.device)
                        render = sampler.sample(model, noise=noise, cond=cond)
                        render = (render + 1) / 2
                        render = self.all_gather(render)
                        if render.dim() == 5:
                            render = render.flatten(0, 1)
                        if self.global_rank == 0:
                            writer.put_images(render[argsort])
                    # (k, n, c)
                    cond = self.all_gather(cond)
                    if cond.dim() == 3:
                        # (k*n, c)
                        cond = cond.flatten(0, 1)
                    conds.append(cond[argsort].cpu())
        model.train()
        conds = torch.cat(conds).float()
        return conds

    def training_step(self, batch, batch_idx):
        """
        given an input, calculate the loss function
        no optimization at this stage.
        """
        with amp.autocast(False):
            if self.conf.train_mode.require_dataset_infer():
                cond = batch[0]
                if self.conf.latent_znormalize:
                    cond = (cond - self.conds_mean.to(
                        self.device)) / self.conds_std.to(self.device)
            else:
                imgs_ori, idxs = batch['img'], batch['index']
                patch_size = self.patch_size
                halfp = patch_size // 2
                H,W = imgs_ori.shape[2:]
                assert H%patch_size==0 and W%patch_size== 0, "Image shape should be dividable by patch size"

                imgs_ori_pad = F.pad(imgs_ori, (halfp, halfp, halfp, halfp), 'constant')
                patch_num_x = H // patch_size
                patch_num_y = W // patch_size
                grid_x = torch.linspace(0, patch_num_x, patch_num_x+1, device=self.device)
                grid_y = torch.linspace(0, patch_num_y, patch_num_y+1, device=self.device)
                xx, yy = torch.meshgrid(grid_x, grid_y, indexing='ij')
                pos = torch.stack([xx, yy], dim=-1)
                loss_mask = torch.zeros_like(imgs_ori_pad)
                loss_mask[:, :, halfp:-halfp, halfp:-halfp] = 1.0
                next_loss_mask = None

            if self.conf.train_mode == TrainMode.diffusion:
                """
                main training mode!!!
                """
                t = torch.randint(0, 1000, size=(imgs_ori.shape[0],), device=self.device)
                losses = self.sampler.training_losses(model=self.model,
                                                      x_start = imgs_ori_pad,
                                                      imgs=imgs_ori,
                                                      t=t,
                                                      pos=pos,
                                                      loss_mask=loss_mask,
                                                      next_loss_mask = next_loss_mask,
                                                      idx = idxs,
                                                      patch_size = self.patch_size,)
            elif self.conf.train_mode.is_latent_diffusion():
                """
                training the latent variables!
                """
                # diffusion on the latent
                t, weight = self.T_sampler.sample(len(cond), cond.device)
                latent_losses = self.latent_sampler.training_latent_losses(
                    model=self.model.latent_net, x_start=cond, t=t)
                losses = {
                    'latent': latent_losses['loss'],
                    'loss': latent_losses['loss']
                }
            else:
                raise NotImplementedError()

            loss = losses['loss'].mean()
            # divide by accum batches to make the accumulated gradient exact!
            for key in ['loss', 'vae', 'latent', 'mmd', 'chamfer', 'arg_cnt']:
                if key in losses:
                    losses[key] = self.all_gather(losses[key]).mean()

            if self.global_rank == 0:
                self.logger.experiment.add_scalar('loss', losses['loss'],
                                                  self.num_samples)
                for key in ['vae', 'latent', 'mmd', 'chamfer', 'arg_cnt']:
                    if key in losses:
                        self.logger.experiment.add_scalar(
                            f'loss/{key}', losses[key], self.num_samples)

        return {'loss': loss}

    def on_train_batch_end(self, outputs, batch, batch_idx: int,
                           dataloader_idx: int) -> None:
        """
        after each training step ...
        """
        if self.is_last_accum(batch_idx):
            if self.conf.train_mode == TrainMode.latent_diffusion:
                ema(self.model.latent_net, self.ema_model.latent_net,
                    self.conf.ema_decay)
            else:
                ema(self.model, self.ema_model, self.conf.ema_decay)

            if self.conf.train_mode.require_dataset_infer():
                imgs = None
                idx = None
            else:
                imgs = batch['img']
                idx = batch["index"]
            self.log_sample(x_start = imgs, step = self.global_step, idx = idx)

    def on_before_optimizer_step(self, optimizer: Optimizer,
                                 optimizer_idx: int) -> None:
        # fix the fp16 + clip grad norm problem with pytorch lightinng
        # this is the currently correct way to do it
        if self.conf.grad_clip > 0:
            # from trainer.params_grads import grads_norm, iter_opt_params
            params = [
                p for group in optimizer.param_groups for p in group['params']
            ]
            # print('before:', grads_norm(iter_opt_params(optimizer)))

            torch.nn.utils.clip_grad_norm_(params,
                                           max_norm=self.conf.grad_clip)
            for i in range(len(params)):
                if params[i].size() == (90000, 512):
                    params[i]._grad = params[i]._grad * 0.5

    def log_sample(self, x_start, idx, **kwargs):
        """
        put images to the tensorboard
        """
        if x_start is not None:
            H, W = x_start.shape[2:]
            patch_num_x = H // self.patch_size
            patch_num_y = W // self.patch_size
        def do(model,
               postfix,
               use_xstart,
               save_real=False,
               no_latent_diff=False,
               interpolate=False):
            model.eval()
            with torch.no_grad():
                
                all_x_T = self.split_tensor(torch.randn(self.conf.sample_size * patch_num_x * patch_num_y, 3, self.patch_size, self.patch_size).cuda())
                batch_size = len(all_x_T)
                loader = DataLoader(all_x_T, batch_size=batch_size)
                Gen = []
                for x_T in loader:
                    if use_xstart:
                        _xstart = x_start[:len(x_T)//patch_num_x//patch_num_y]
                        _idx = idx[:len(x_T)//patch_num_x//patch_num_y]
                        
                        b = _xstart.shape[0]
                        all_pos = []            
                        grid_x = torch.linspace(0, patch_num_x, patch_num_x+1, device=self.device)
                        grid_y = torch.linspace(0, patch_num_y, patch_num_y+1, device=self.device)
                        xx, yy = torch.meshgrid(grid_x, grid_y, indexing='ij')
                        pos1 = torch.stack([xx, yy], dim=-1).flatten(0, 1).repeat(b, 1)
                        all_pos = [pos1]
                        imgs = _xstart
                    else:
                        _xstart = None

                    if self.conf.train_mode.is_latent_diffusion(
                    ) and not use_xstart:
                        # diffusion of the latent first
                        gen = render_uncondition(
                            conf=self.conf,
                            model=model,
                            x_T=x_T,
                            sampler=self.eval_sampler,
                            latent_sampler=self.eval_latent_sampler,
                            conds_mean=self.conds_mean,
                            conds_std=self.conds_std)
                    else:
                        if not use_xstart and self.conf.model_type.has_noise_to_cond(
                        ):
                            model: BeatGANsAutoencModel
                            cond = torch.randn(len(x_T),
                                               self.conf.style_ch,
                                               device=self.device)
                            cond = model.noise_to_cond(cond)
                        else:
                            if interpolate:
                                with amp.autocast(self.conf.fp16):
                                    cond = model.encoder(_xstart)
                                    i = torch.randperm(len(cond))
                                    cond = (cond + cond[i]) / 2
                            else:
                                cond = None
                        gen = self.eval_sampler.sample(model=model,
                                                       noise=None,
                                                       cond=cond,
                                                       x_start=_xstart,
                                                       imgs=imgs,
                                                       all_pos=all_pos,
                                                       idx = _idx,
                                                       patch_size = self.patch_size,
                                                       )
                    Gen.append(gen)

                gen = torch.cat(Gen)
                gen = self.all_gather(gen)
                if gen.dim() == 5:
                    gen = gen.flatten(0, 1)

                if save_real and use_xstart:
                    # save the original images to the tensorboard
                    real = self.all_gather(_xstart)
                    if real.dim() == 5:
                        real = real.flatten(0, 1)

                    if self.global_rank == 0:
                        grid_real = (make_grid(real) + 1) / 2
                        self.logger.experiment.add_image(
                            f'sample{postfix}/real', grid_real,
                            self.num_samples)

                if self.global_rank == 0:
                    # save samples to the tensorboard
                    grid = (make_grid(gen) + 1) / 2
                    sample_dir = os.path.join(self.conf.logdir,
                                              f'sample{postfix}')
                    if not os.path.exists(sample_dir):
                        os.makedirs(sample_dir)
                    path = os.path.join(sample_dir,
                                        '%d.png' % self.num_samples)
                    save_image(grid, path)
                    self.logger.experiment.add_image(f'sample{postfix}', grid,
                                                     self.num_samples)
            model.train()

        if (self.conf.sample_every_samples > 0 and is_time(
                self.num_samples, self.conf.sample_every_samples,
                self.conf.batch_size_effective)):
            if self.conf.train_mode.require_dataset_infer():
                do(self.model, '', use_xstart=False)
                do(self.ema_model, '_ema', use_xstart=False)
            else:
                if self.conf.model_type.has_autoenc(
                ) and self.conf.model_type.can_sample():
                    do(self.model, '', use_xstart=False)
                    do(self.ema_model, '_ema', use_xstart=False)
                    # autoencoding mode
                    do(self.model, '_enc', use_xstart=True, save_real=True)
                    do(self.ema_model,
                       '_enc_ema',
                       use_xstart=True,
                       save_real=True)
                elif self.conf.train_mode.use_latent_net():
                    do(self.model, '', use_xstart=False)
                    do(self.ema_model, '_ema', use_xstart=False)
                    # autoencoding mode
                    do(self.model, '_enc', use_xstart=True, save_real=True)
                    do(self.model,
                       '_enc_nodiff',
                       use_xstart=True,
                       save_real=True,
                       no_latent_diff=True)
                    do(self.ema_model,
                       '_enc_ema',
                       use_xstart=True,
                       save_real=True)
                else:
                    do(self.model, '', use_xstart=True, save_real=True)

    def configure_optimizers(self):
        out = {}
        if self.conf.optimizer == OptimizerType.adam:
            optim = torch.optim.Adam(self.model.parameters(),
                                     lr=self.conf.lr,
                                     weight_decay=self.conf.weight_decay)
        elif self.conf.optimizer == OptimizerType.adamw:
            optim = torch.optim.AdamW(self.model.parameters(),
                                      lr=self.conf.lr,
                                      weight_decay=self.conf.weight_decay)
        else:
            raise NotImplementedError()
        out['optimizer'] = optim
        if self.conf.warmup > 0:
            sched = torch.optim.lr_scheduler.LambdaLR(optim,
                                                      lr_lambda=WarmupLR(
                                                          self.conf.warmup))
            out['lr_scheduler'] = {
                'scheduler': sched,
                'interval': 'step',
            }
        return out

    def split_tensor(self, x):
        """
        extract the tensor for a corresponding "worker" in the batch dimension

        Args:
            x: (n, c)

        Returns: x: (n_local, c)
        """
        n = len(x)
        rank = self.global_rank
        world_size = get_world_size()
        # print(f'rank: {rank}/{world_size}')
        per_rank = n // world_size
        return x[rank * per_rank:(rank + 1) * per_rank]

    def test_step(self, batch, *args, **kwargs):
        """
        for the "eval" mode. 
        We first select what to do according to the "conf.eval_programs". 
        test_step will only run for "one iteration" (it's a hack!).
        
        We just want the multi-gpu support. 
        """
        # make sure you seed each worker differently!
        self.setup()
        """
        "infer" = predict the latent variables using the encoder on the whole dataset
        """
        if 'infer' in self.conf.eval_programs:
            if 'infer' in self.conf.eval_programs:
                print('infer ...')
                conds = self.infer_whole_dataset().float()
                # NOTE: always use this path for the latent.pkl files
                save_path = f'checkpoints/{self.conf.name}/latent.pkl'
            else:
                raise NotImplementedError()

            if self.global_rank == 0:
                conds_mean = conds.mean(dim=0)
                conds_std = conds.std(dim=0)
                if not os.path.exists(os.path.dirname(save_path)):
                    os.makedirs(os.path.dirname(save_path))
                torch.save(
                    {
                        'conds': conds,
                        'conds_mean': conds_mean,
                        'conds_std': conds_std,
                    }, save_path)
        """
        "infer+render" = predict the latent variables using the encoder on the whole dataset
        THIS ALSO GENERATE CORRESPONDING IMAGES
        """
        # infer + reconstruction quality of the input
        for each in self.conf.eval_programs:
            if each.startswith('infer+render'):
                m = re.match(r'infer\+render([0-9]+)', each)
                if m is not None:
                    T = int(m[1])
                    self.setup()
                    print(f'infer + reconstruction T{T} ...')
                    conds = self.infer_whole_dataset(
                        with_render=True,
                        T_render=T,
                        render_save_path=
                        f'latent_infer_render{T}/{self.conf.name}.lmdb',
                    )
                    save_path = f'latent_infer_render{T}/{self.conf.name}.pkl'
                    conds_mean = conds.mean(dim=0)
                    conds_std = conds.std(dim=0)
                    if not os.path.exists(os.path.dirname(save_path)):
                        os.makedirs(os.path.dirname(save_path))
                    torch.save(
                        {
                            'conds': conds,
                            'conds_mean': conds_mean,
                            'conds_std': conds_std,
                        }, save_path)

        # generate images
        for each in self.conf.eval_programs:
            if each.startswith('gen'):
                m = re.match(r'gen\(([0-9]+),([0-9]+)\)', each)
                clip_latent_noise = False
                try:
                    T = int(m[1])
                    T_latent = int(m[2])
                except:
                    raise ValueError("Wrong value input!")
                
                # self.train_dataloader()
                sampler = self.conf._make_diffusion_conf(T=T).make_sampler()
                if T_latent is not None:
                    latent_sampler = self.conf._make_latent_diffusion_conf(
                        T=T_latent).make_sampler()
                else:
                    latent_sampler = None

                conf = self.conf.clone()
                conf.eval_num_images = 50_000
                generate(
                    sampler,
                    self.model,
                    conf,
                    device=self.device,
                    train_data=self.train_data,
                    val_data=self.val_data,
                    latent_sampler=latent_sampler,
                    conds_mean=self.conds_mean,
                    conds_std=self.conds_std,
                    remove_cache=False,
                    clip_latent_noise=clip_latent_noise,
                )


def ema(source, target, decay):
    source_dict = source.state_dict()
    target_dict = target.state_dict()
    for key in source_dict.keys():
        target_dict[key].data.copy_(target_dict[key].data * decay +
                                    source_dict[key].data * (1 - decay))


class WarmupLR:
    def __init__(self, warmup) -> None:
        self.warmup = warmup

    def __call__(self, step):
        return min(step, self.warmup) / self.warmup


def is_time(num_samples, every, step_size):
    closest = (num_samples // every) * every
    return num_samples - closest < step_size


def train(conf: TrainConfig, gpus, nodes=1, mode: str = 'train'):
    print('conf:', conf.name)
    model = LitModel(conf)
    callbacks = [LearningRateMonitor()]

    if conf.logdir and not os.path.exists(conf.logdir):
        os.makedirs(conf.logdir)
    if conf.logdir:
        checkpoint = ModelCheckpoint(dirpath=f'{conf.logdir}',
                                 save_last=True,
                                 save_top_k=-1,
                                 every_n_train_steps=conf.save_every_samples //
                                 conf.batch_size_effective)
        callbacks.append(checkpoint)

    if conf.full_model_path:
        checkpoint_path = conf.full_model_path
    else:
        checkpoint_path =  f'{conf.logdir}/last.ckpt'
    print('ckpt path:', checkpoint_path)
    if os.path.exists(checkpoint_path):
        resume = checkpoint_path
        print('resume!')
    else:
        if conf.continue_from is not None:
            # continue from a checkpoint
            resume = conf.continue_from.path
        else:
            resume = None
    if conf.logdir:
        tb_logger = pl_loggers.TensorBoardLogger(save_dir=conf.logdir,
                                             name=None,
                                             version='')
    else:
        tb_logger = None
    plugins = []
    if len(gpus) == 1 and nodes == 1:
        accelerator = None
    else:
        accelerator = 'ddp'
        from pytorch_lightning.plugins import DDPPlugin
        plugins.append(DDPPlugin(find_unused_parameters=False))

    trainer = pl.Trainer(
        max_steps=conf.total_samples // conf.batch_size_effective,
        resume_from_checkpoint=resume,
        gpus=gpus,
        num_nodes=nodes,
        accelerator=accelerator,
        precision=16 if conf.fp16 else 32,
        callbacks=callbacks,
        replace_sampler_ddp=True,
        logger=tb_logger,
        accumulate_grad_batches=conf.accum_batches,
        plugins=plugins,
    )

    if mode == 'train':
        trainer.fit(model)
    elif mode == 'eval':
        dummy = DataLoader(TensorDataset(torch.tensor([0.] * conf.batch_size)),
                           batch_size=conf.batch_size)
        eval_path = conf.eval_path or checkpoint_path
        print('loading from:', eval_path)
        state = torch.load(eval_path, map_location='cpu')
        print('step:', state['global_step'])
        model.load_state_dict(state['state_dict'],strict=False)
        out = trainer.test(model, dataloaders=dummy)
        out = out[0]
        print(out)

        if get_rank() == 0:
            for k, v in out.items():
                tb_logger.experiment.add_scalar(
                    k, v, state['global_step'] * conf.batch_size_effective)

            tgt = f'evals/{conf.name}.txt'
            dirname = os.path.dirname(tgt)
            if not os.path.exists(dirname):
                os.makedirs(dirname)
            with open(tgt, 'a') as f:
                f.write(json.dumps(out) + "\n")
    else:
        raise NotImplementedError()
