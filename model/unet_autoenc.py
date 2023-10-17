from enum import Enum
import copy
import torch
from torch import Tensor
from torch.nn.functional import silu
import random
from .latentnet import *
from .unet import *
from utils.choices import *

from einops import rearrange,repeat


@dataclass
class BeatGANsAutoencConfig(BeatGANsUNetConfig):
    # number of style channels
    enc_out_channels: int = 512
    enc_attn_resolutions: Tuple[int] = None
    enc_pool: str = 'depthconv'
    enc_num_res_block: int = 2
    enc_channel_mult: Tuple[int] = None
    enc_grad_checkpoint: bool = False
    latent_net_conf: MLPSkipNetConfig = None

    def make_model(self):
        return BeatGANsAutoencModel(self)


class BeatGANsAutoencModel(BeatGANsUNetModel):
    def __init__(self, conf: BeatGANsAutoencConfig):
        super().__init__(conf)
        
        self.conf = conf
        if conf.semantic_enc:
            self.encoder = BeatGANsEncoderConfig(
                image_size=conf.image_size,
                in_channels=conf.in_channels,
                model_channels=conf.model_channels,
                out_hid_channels=conf.enc_out_channels,
                out_channels=conf.enc_out_channels,
                num_res_blocks=conf.enc_num_res_block,
                attention_resolutions=(conf.enc_attn_resolutions
                                    or conf.attention_resolutions),
                dropout=conf.dropout,
                channel_mult=conf.enc_channel_mult or conf.channel_mult,
                use_time_condition=False,
                conv_resample=conf.conv_resample,
                dims=conf.dims,
                use_checkpoint=conf.use_checkpoint or conf.enc_grad_checkpoint,
                num_heads=conf.num_heads,
                num_head_channels=conf.num_head_channels,
                resblock_updown=conf.resblock_updown,
                use_new_attention_order=conf.use_new_attention_order,
                pool=conf.enc_pool,
            ).make_model()
            style_channels = conf.enc_out_channels
            self.sem_enc = True
        else:
            if conf.semantic_path == "":
                self.encoder = None # Semantic(conf.data_num, initial_pt=None)
            else:
                self.encoder = Semantic(conf.data_num, initial_pt=conf.semantic_path)
            style_channels = conf.enc_out_channels
            self.sem_enc = False
        
        self.time_embed = TimeStyleSeperateEmbed(
            time_channels=conf.model_channels,
            time_out_channels=conf.embed_channels,
            style_channels = style_channels,
        )        

        if conf.latent_net_conf is not None:
            self.latent_net = conf.latent_net_conf.make_model()

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        assert self.conf.is_stochastic
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def sample_z(self, n: int, device):
        assert self.conf.is_stochastic
        return torch.randn(n, self.conf.enc_out_channels, device=device)

    def noise_to_cond(self, noise: Tensor):
        raise NotImplementedError()
        assert self.conf.noise_net_conf is not None
        return self.noise_net.forward(noise)
    
    def encode(self, x):
        cond = self.encoder.forward(x)
        return {'cond': cond}
    
    @property
    def stylespace_sizes(self):
        modules = list(self.input_blocks.modules()) + list(
            self.middle_block.modules()) + list(self.output_blocks.modules())
        sizes = []
        for module in modules:
            if isinstance(module, ResBlock):
                linear = module.cond_emb_layers[-1]
                sizes.append(linear.weight.shape[0])
        return sizes


    def forward(self,
                x,
                t,
                pos=None,
                y=None,
                imgs=None,
                cond=None,
                noise=None,
                t_cond=None,
                idx = None,
                index = None,
                do_train = False,
                patch_size = 64,
                pos_random = None,
                random = None,
                **kwargs):
        """
        Apply the model to an input batch.

        Args:
            x_start: the original image to encode
            cond: output of the encoder
            noise: random noise (to predict the cond)
            random/pose_random: classifier free
        """

        if t_cond is None:
            t_cond = t

        if noise is not None:
            cond = self.noise_to_cond(noise)
     
        if cond is None:
            if self.sem_enc:
                cond = self.encode(imgs)["cond"]
            else:
                cond = self.encoder.forward(idx)
        if random is not None: # semantic code classifier free
            random = random >= 0.1
            cond = cond * random[:,None].to(cond.device)
        cond_tmp = cond.clone()
        cond = cond.repeat_interleave(x.shape[0] // t.shape[0], dim=0)
        H,W = imgs.shape[2:]
        patch_num_x = H // patch_size
        patch_num_y = W // patch_size

        if t is not None:
            t_cur = repeat(t,'h -> (h repeat)',repeat =int(x.shape[0]/t.shape[0]))
            _t_emb = timestep_embedding(t_cur, self.conf.model_channels)
            # _t_cond_emb = timestep_embedding(t_cond, self.conf.model_channels)

            pos_x = timestep_embedding(pos[:, 0], 64)
            pos_y = timestep_embedding(pos[:, 1], 64)
            pos_emb = torch.cat([pos_x, pos_y], dim=1)
        else:
            # this happens when training only autoenc
            _t_emb = None
            _t_cond_emb = None

        if pos_random is not None: # pose embedding classifier free
            pos_random = pos_random >= 0.5
            pos_random_old = pos_random.repeat_interleave(x.shape[0] // t.shape[0], dim=0)
            pos_emb = pos_emb * pos_random_old[:,None].to(pos_emb.device)

        if self.conf.resnet_two_cond:
            res = self.time_embed.forward(
                time_emb=_t_emb,
                cond=cond,
                pos_emb=pos_emb
            )
        else:
            raise NotImplementedError()

        if self.conf.resnet_two_cond:
            # two cond: first = time emb, second = cond_emb
            emb = res.time_emb
            cond_emb = res.emb
        else:
            # one cond = combined of both time and cond
            emb = res.emb
            cond_emb = None

        # override the style if given
        # style = style or res.style

        assert (y is not None) == (
            self.conf.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"

        if self.conf.num_classes is not None:
            raise NotImplementedError()

        # where in the model to supply time conditions
        enc_time_emb = emb
        mid_time_emb = emb
        dec_time_emb = emb
        # where in the model to supply style conditions
        enc_cond_emb = cond_emb
        mid_cond_emb = cond_emb
        dec_cond_emb = cond_emb


        '''====NOTE: Additional embedding===='''
        if not do_train: # rendering, need to revise first
            grid_x = torch.linspace(0.5, patch_num_x-0.5, patch_num_x, device=emb.device)
            grid_y = torch.linspace(0.5, patch_num_y-0.5, patch_num_y, device=emb.device)
            xx, yy = torch.meshgrid(grid_x, grid_y, indexing='ij')
            pos_new = torch.stack([xx, yy], dim=-1).flatten(0, 1).repeat(t.shape[0], 1)
        else:
            pos_new = torch.stack([index[0]+0.5, index[1]+0.5], dim = -1).unsqueeze(0).repeat(t.shape[0], 1)
        pos_x_new = timestep_embedding(pos_new[:, 0], 64)
        pos_y_new = timestep_embedding(pos_new[:, 1], 64)
        pos_emb_new = torch.cat([pos_x_new, pos_y_new], dim=1).to(pos_emb.device)

        if pos_random is not None: # pose embedding classifier free
            pos_random_new = pos_random.repeat_interleave(pos_new.shape[0] // t.shape[0], dim=0)
            pos_emb_new = pos_emb_new * pos_random_new[:,None].to(pos_emb_new.device)
        # change of time embedding
        t_cur_new = repeat(t,'h -> (h repeat)',repeat =int(pos_new.shape[0]/t.shape[0]))
        _t_emb_new = timestep_embedding(t_cur_new, self.conf.model_channels)
        # change of cond embedding
        cond_new = cond_tmp.repeat_interleave(pos_new.shape[0] // t.shape[0], dim=0)
        res_new = self.time_embed.forward(
                time_emb=_t_emb_new,
                cond=cond_new,
                pos_emb=pos_emb_new
                # time_cond_emb=_t_cond_emb
            )

        hs = [[] for _ in range(len(self.conf.channel_mult))]
        hs_train = [[] for _ in range(len(self.conf.channel_mult))]

        if x is not None:
            h = x.type(self.dtype)
            # input blocks
            k = 0
            for i in range(len(self.input_num_blocks)):
                for j in range(self.input_num_blocks[i]):
                    h = self.input_blocks[k](h,
                                             emb=enc_time_emb,
                                             cond=enc_cond_emb)

                    hs[i].append(h)
                    hs_train[i].append(h.clone())
                    k += 1
                # print(h.size())
            assert k == len(self.input_blocks)

            # middle blocks
            h = self.middle_block(h, emb=mid_time_emb, cond=mid_cond_emb)
            h_train = h.clone()
        else:
            # no lateral connections
            # happens when training only the autonecoder
            h = None
            hs = [[] for _ in range(len(self.conf.channel_mult))]

        ''' For training mode '''
        if do_train:
            prob = 1
            p1 = 2
            p2 = 2
        else:
            prob = 1
            p1 = patch_num_x+1
            p2 = patch_num_y+1

        # output blocks
        if do_train: # Change latent space
            # First runturn
            k = 0
            for i in range(len(self.output_num_blocks)):

                '''NOTE: change the input dimension and lateral shortcut dimension before every output num blocks'''
                batch_size, c, height, width = h.shape
                if i == 0:
                    half_p = int(height//2)
                    h_ori = rearrange(h, '(b p1 p2) c h w -> b c (p1 h) (p2 w)', p1 = p1, p2 = p2)
                    h_ori_crop = h_ori[:, :, half_p:-half_p, half_p:-half_p]
                    h_shift = rearrange(h_ori_crop, 'b c (p1 h) (p2 w) -> (b p1 p2) c h w', h = height, w = width)
                    h = h_shift

                for j in range(self.output_num_blocks[i]):
                    try:
                        lateral = hs[-i - 1].pop()
                        '''NOTE: change the size of lateral to add shorcut'''
                        lateral_batch_size = lateral.size(0)
                        if lateral_batch_size != h.size(0):
                            half_p = int(height//2)
                            lateral_ori = rearrange(lateral, '(b p1 p2) c h w -> b c (p1 h) (p2 w)', p1 = p1, p2 = p2)
                            lateral_ori_crop = lateral_ori[:, :, half_p:-half_p, half_p:-half_p]
                            lateral_shift = rearrange(lateral_ori_crop, 'b c (p1 h) (p2 w) -> (b p1 p2) c h w', h = height, w = width)
                            lateral  = lateral_shift
                    except IndexError:
                        lateral = None

                    if dec_time_emb.size(0) != h.size(0): # for change of dimension
                        # change of position embedding
                        assert h.size(0) == res_new.time_emb.size(0) == lateral.size(0) == res_new.emb.size(0)
                        h = self.output_blocks[k](h,
                                            emb=res_new.time_emb,
                                            cond=res_new.emb, #res_new.emb,
                                            lateral=lateral)
                    else:
                        assert h.size(0) == dec_time_emb.size(0) == lateral.size(0) == dec_cond_emb.size(0)
                        h = self.output_blocks[k](h,
                                            emb=dec_time_emb,
                                            cond=dec_cond_emb, #dec_cond_emb,
                                            lateral=lateral)
                    k += 1
            pred1 = self.out(h)

            # Second Runturn
            k = 0
            for i in range(len(self.output_num_blocks)):
                for j in range(self.output_num_blocks[i]):
                    try:
                        lateral = hs_train[-i - 1].pop()
                    except IndexError:
                        lateral = None
                    h_train = self.output_blocks[k](h_train,
                                            emb=dec_time_emb,
                                            cond=dec_cond_emb,
                                            lateral=lateral)
                    k += 1

            pred2 = self.out(h_train)

        else:
            k = 0
            for i in range(len(self.output_num_blocks)):
                '''NOTE: change the input dimension and lateral shortcut dimension before every output num blocks'''
                batch_size, c, height, width = h.shape
                if i == 0:
                    half_p = int(height//2)
                    h_ori = rearrange(h, '(b p1 p2) c h w -> b c (p1 h) (p2 w)', p1 = p1, p2 = p2)
                    h_ori_crop = h_ori[:, :, half_p:-half_p, half_p:-half_p]
                    h_shift = rearrange(h_ori_crop, 'b c (p1 h) (p2 w) -> (b p1 p2) c h w', h = height, w = width)
                    h = h_shift

                for j in range(self.output_num_blocks[i]):
                    try:
                        lateral = hs[-i - 1].pop()
                        '''NOTE: change the size of lateral to add shorcut'''
                        lateral_batch_size = lateral.size(0)
                        if lateral_batch_size != h.size(0):
                            half_p = int(height//2)
                            lateral_ori = rearrange(lateral, '(b p1 p2) c h w -> b c (p1 h) (p2 w)', p1 = p1, p2 = p2)
                            lateral_ori_crop = lateral_ori[:, :, half_p:-half_p, half_p:-half_p]
                            lateral_shift = rearrange(lateral_ori_crop, 'b c (p1 h) (p2 w) -> (b p1 p2) c h w', h = height, w = width)
                            lateral  = lateral_shift
                    except IndexError:
                        lateral = None

                    if dec_time_emb.size(0) != h.size(0): # for change of dimension
                        # change of position embedding
                        assert h.size(0) == res_new.time_emb.size(0) == lateral.size(0) == res_new.emb.size(0)
                        h = self.output_blocks[k](h,
                                            emb=res_new.time_emb,
                                            cond=res_new.emb, #res_new.emb,
                                            lateral=lateral)
                    else:
                        assert h.size(0) == dec_time_emb.size(0) == lateral.size(0) == dec_cond_emb.size(0)
                        h = self.output_blocks[k](h,
                                            emb=dec_time_emb,
                                            cond=dec_cond_emb, #dec_cond_emb,
                                            lateral=lateral)
                    k += 1
            pred1 = self.out(h)
            pred2 = pred1


        return AutoencReturn(pred=pred1, pred2 = pred2, cond=cond)


class AutoencReturn(NamedTuple):
    pred: Tensor
    pred2: Tensor
    cond: Tensor = None


class EmbedReturn(NamedTuple):
    # style and time
    emb: Tensor = None
    # time only
    time_emb: Tensor = None
    # style only (but could depend on time)
    style: Tensor = None


class TimeStyleSeperateEmbed(nn.Module):
    # embed only style
    def __init__(self, time_channels, time_out_channels, style_channels):
        super().__init__()
        self.time_embed = nn.Sequential(
            linear(time_channels, time_out_channels // 2),
            nn.SiLU(),
            linear(time_out_channels // 2, time_out_channels // 2),
        )

        self.pos_embed = nn.Sequential(
            linear(128, time_out_channels // 2),
            nn.SiLU(),
            linear(time_out_channels // 2, time_out_channels // 2),
        )
        
        # self.style = nn.Sequential(
        #     linear(style_channels, time_out_channels),
        #     nn.SiLU(),
        #     linear(time_out_channels, time_out_channels),
        # )

        if style_channels == time_out_channels:
            self.style = nn.Identity()
        else:
            self.style = nn.Sequential(
            linear(style_channels, time_out_channels),
            nn.SiLU(),
            linear(time_out_channels, time_out_channels),
        )

    def forward(self, time_emb=None, cond=None, pos_emb=None, **kwargs):
        if time_emb is None:
            # happens with autoenc training mode
            time_emb = None
        else:
            time_emb = self.time_embed(time_emb)

        pos_emb = self.pos_embed(pos_emb)

        if cond is None:
            style = None
        else:
            style = self.style(cond)
        return EmbedReturn(emb=style, time_emb=torch.cat([time_emb, pos_emb], dim=1), style=style)
