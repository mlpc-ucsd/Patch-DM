import os
import shutil
import re
import torch
import torchvision
from pytorch_fid import fid_score
from torch import distributed
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm.autonotebook import tqdm, trange

from utils.renderer import *
from config import *
from diffusion import Sampler
from utils.dist_utils import *


def make_subset_loader(conf: TrainConfig,
                       dataset: Dataset,
                       batch_size: int,
                       shuffle: bool,
                       parallel: bool,
                       drop_last=True):
    dataset = SubsetDataset(dataset, size=conf.eval_num_images)
    if parallel and distributed.is_initialized():
        sampler = DistributedSampler(dataset, shuffle=shuffle)
    else:
        sampler = None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        # with sampler, use the sample instead of this option
        shuffle=False if sampler else shuffle,
        num_workers=conf.num_workers,
        pin_memory=True,
        drop_last=drop_last,
        multiprocessing_context=get_context('fork'),
    )



def psnr(img1, img2):
    """
    Args:
        img1: (n, c, h, w)
    """
    v_max = 1.
    # (n,)
    mse = torch.mean((img1 - img2)**2, dim=[1, 2, 3])
    return 20 * torch.log10(v_max / torch.sqrt(mse))


def generate(
    sampler: Sampler,
    model: Model,
    conf: TrainConfig,
    device,
    train_data: Dataset,
    val_data: Dataset,
    latent_sampler: Sampler = None,
    conds_mean=None,
    conds_std=None,
    remove_cache: bool = True,
    clip_latent_noise: bool = False,
):

    if get_rank() == 0:
        if not os.path.exists(conf.generate_dir):
            os.makedirs(conf.generate_dir)
    barrier()

    world_size = get_world_size()
    rank = get_rank()
    batch_size = chunk_size(conf.batch_size_eval, rank, world_size)

    def filename(idx):
        return world_size * idx + rank

    model.eval()
    with torch.no_grad():
        m = re.match(r'([0-9]+)x([0-9]+)', conf.image_size)
        H, W = int(m[1]), int(m[2])
        patch_num_x = H // conf.patch_size
        patch_num_y = W // conf.patch_size
        grid_x = torch.linspace(0, patch_num_x, patch_num_x+1, device=device)
        grid_y = torch.linspace(0, patch_num_y, patch_num_y+1, device=device)
        xx, yy = torch.meshgrid(grid_x, grid_y, indexing='ij')
        pos1 = torch.stack([xx, yy], dim=-1).flatten(0, 1).repeat(batch_size, 1)
        all_pos = [pos1]
        x_T = torch.randn(
                    (batch_size*patch_num_x*patch_num_y, 3, conf.patch_size, conf.patch_size),
                    device=device)
        
        if conf.model_type.can_sample():
            eval_num_images = chunk_size(conf.eval_num_images, rank,
                                         world_size)
            desc = "generating images"
            for i in trange(0, eval_num_images, batch_size, desc=desc):
                batch_size = min(batch_size, eval_num_images - i)
        
                batch_images = render_uncondition(
                    conf=conf,
                    model=model,
                    x_T=x_T,
                    sampler=sampler,
                    latent_sampler=latent_sampler,
                    conds_mean=conds_mean,
                    conds_std=conds_std,
                    all_pos = all_pos,
                    patch_size=conf.patch_size,
                    img_size = (H,W),
                    ).cpu()

                batch_images = (batch_images + 1) / 2
                # keep the generated images
                for j in range(len(batch_images)):
                    img_name = filename(i + j)
                    torchvision.utils.save_image(
                        batch_images[j],
                        os.path.join(conf.generate_dir, f'{img_name}.png'))
        elif conf.model_type == ModelType.autoencoder:
            if conf.train_mode.is_latent_diffusion():
                # evaluate autoencoder + latent diffusion (doesn't give the images)
                model: BeatGANsAutoencModel
                eval_num_images = chunk_size(conf.eval_num_images, rank,
                                             world_size)
                desc = "generating images"
                for i in trange(0, eval_num_images, batch_size, desc=desc):
                    batch_size = min(batch_size, eval_num_images - i)
                    
                    batch_images = render_uncondition(
                        conf=conf,
                        model=model,
                        x_T=x_T,
                        sampler=sampler,
                        latent_sampler=latent_sampler,
                        conds_mean=conds_mean,
                        conds_std=conds_std,
                        clip_latent_noise=clip_latent_noise,
                        all_pos=all_pos,
                        patch_size=conf.patch_size,
                        img_size = (H,W),
                    ).cpu()
                    batch_images = (batch_images + 1) / 2
                    # keep the generated images
                    for j in range(len(batch_images)):
                        img_name = filename(i + j)
                        torchvision.utils.save_image(
                            batch_images[j],
                            os.path.join(conf.generate_dir, f'{img_name}.png'))
            else:
                train_loader = make_subset_loader(conf,
                                                  dataset=train_data,
                                                  batch_size=batch_size,
                                                  shuffle=True,
                                                  parallel=True)

                i = 0
                for batch in tqdm(train_loader, desc='generating images'):
                    imgs = batch['img'].to(device)
                    
                    batch_images = render_condition(
                        conf=conf,
                        model=model,
                        x_T=x_T,
                        x_start=imgs,
                        cond=None,
                        sampler=sampler,
                        latent_sampler=latent_sampler,).cpu()
                    
                    batch_images = (batch_images + 1) / 2
                    # keep the generated images
                    for j in range(len(batch_images)):
                        img_name = filename(i + j)
                        torchvision.utils.save_image(
                            batch_images[j],
                            os.path.join(conf.generate_dir, f'{img_name}.png'))
                    i += len(imgs)
        else:
            raise NotImplementedError()
    model.train()

    barrier()


def loader_to_path(loader: DataLoader, path: str, denormalize: bool):
    # not process safe!

    if not os.path.exists(path):
        os.makedirs(path)

    # write the loader to files
    i = 0
    for batch in tqdm(loader, desc='copy images'):
        imgs = batch['img']
        if denormalize:
            imgs = (imgs + 1) / 2
        for j in range(len(imgs)):
            torchvision.utils.save_image(imgs[j],
                                         os.path.join(path, f'{i+j}.png'))
        i += len(imgs)
