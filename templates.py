from experiment import *


def ddpm():
    """
    base configuration for all DDIM-based models.
    """
    conf = TrainConfig()
    conf.batch_size = 32
    
    # change_code_note default:ddim
    conf.beatgans_gen_type = GenerativeType.ddpm
    # conf.beatgans_gen_type = GenerativeType.ddim

    conf.beta_scheduler = 'linear'
    conf.data_name = 'ffhq'
    conf.diffusion_type = 'beatgans'
    conf.eval_ema_every_samples = 200_000
    conf.eval_every_samples = 200_000
    conf.fp16 = True
    conf.lr = 1e-4
    conf.model_name = ModelName.beatgans_ddpm
    conf.net_attn = (16, )
    conf.net_beatgans_attn_head = 1
    conf.net_beatgans_embed_channels = 512
    conf.net_ch_mult = (1, 2, 4, 8)
    conf.net_ch = 64
    conf.sample_size = 32
    conf.T_eval = 1000 # 20 change_code_note
    conf.T = 1000
    conf.make_model_conf()
    return conf


def autoenc_base():
    """
    base configuration for all Diff-AE models.
    """
    conf = TrainConfig()
    conf.batch_size = 32
    
    # change_code_note default:ddim
    conf.beatgans_gen_type = GenerativeType.ddpm
    # conf.beatgans_gen_type = GenerativeType.ddim

    conf.beta_scheduler = 'linear'
    conf.data_name = 'ffhq'
    conf.diffusion_type = 'beatgans'
    conf.eval_ema_every_samples = 200_000
    conf.eval_every_samples = 200_000
    conf.fp16 = True
    conf.lr = 1e-4
    conf.model_name = ModelName.beatgans_autoenc
    conf.net_attn = (16, )
    conf.net_beatgans_attn_head = 1
    conf.net_beatgans_embed_channels = 512
    conf.net_beatgans_resnet_two_cond = True
    conf.net_ch_mult = (1, 2, 4, 8)
    conf.net_ch = 64
    conf.net_enc_channel_mult = (1, 2, 4, 8, 8)
    conf.net_enc_pool = 'adaptivenonzero'
    conf.sample_size = 32
    conf.T_eval = 1000 # 20 change_code_note
    conf.T = 1000
    conf.make_model_conf()
    return conf


def ffhq64_ddpm():
    conf = ddpm()
    conf.data_name = 'ffhqlmdb256'
    conf.warmup = 0
    conf.total_samples = 72_000_000
    conf.scale_up_gpus(4)
    return conf


def ffhq64_autoenc():
    conf = autoenc_base()
    conf.data_name = 'ffhqlmdb256'
    conf.warmup = 0
    conf.total_samples = 72_000_000
    conf.net_ch_mult = (1, 2, 4, 8)
    conf.net_enc_channel_mult = (1, 2, 4, 8, 8)
    conf.eval_every_samples = 1_000_000
    conf.eval_ema_every_samples = 1_000_000
    conf.scale_up_gpus(4)
    conf.make_model_conf()
    return conf

def ffhq128_ddpm():
    conf = ddpm()
    conf.data_name = 'ffhqlmdb256'
    conf.warmup = 0
    conf.total_samples = 48_000_000
    conf.img_size = 128
    conf.net_ch = 128
    # channels:
    # 3 => 128 * 1 => 128 * 1 => 128 * 2 => 128 * 3 => 128 * 4
    # sizes:
    # 128 => 128 => 64 => 32 => 16 => 8
    conf.net_ch_mult = (1, 1, 2, 3, 4)
    conf.eval_every_samples = 1_000_000
    conf.eval_ema_every_samples = 1_000_000
    conf.scale_up_gpus(4)
    conf.eval_ema_every_samples = 10_000_000
    conf.eval_every_samples = 10_000_000
    conf.make_model_conf()
    return conf

def ffhq128_autoenc_base():
    conf = autoenc_base()
    conf.data_name = 'ffhqlmdb256'
    conf.scale_up_gpus(4)
    conf.img_size = 128
    conf.net_ch = 128
    # final resolution = 8x8
    conf.net_ch_mult = (1, 1, 2, 3, 4)
    # final resolution = 4x4
    conf.net_enc_channel_mult = (1, 1, 2, 3, 4, 4)
    conf.eval_ema_every_samples = 10_000_000
    conf.eval_every_samples = 10_000_000
    conf.make_model_conf()
    return conf

def ffhq128_ddpm_130M():
    conf = ffhq128_ddpm()
    conf.total_samples = 130_000_000
    conf.eval_ema_every_samples = 10_000_000
    conf.eval_every_samples = 10_000_000
    conf.name = 'ffhq128_ddpm_130M'
    return conf


def ffhq128_autoenc_130M():
    conf = ffhq128_autoenc_base()
    conf.total_samples = 130_000_000
    conf.eval_ema_every_samples = 10_000_000
    conf.eval_every_samples = 10_000_000
    conf.name = 'ffhq128_autoenc_130M'
    return conf

def ffhq256_autoenc_eco():
    conf = ffhq128_autoenc_base()
    conf.img_size = 256
    conf.net_ch = 128
    conf.net_ch_mult = (1, 1, 2, 2, 4, 4)
    conf.net_enc_channel_mult = (1, 1, 2, 2, 4, 4, 4)
    conf.eval_every_samples = 10_000_000
    conf.eval_ema_every_samples = 10_000_000
    conf.total_samples = 200_000_000
    conf.batch_size = 64
    conf.make_model_conf()
    conf.name = 'ffhq256_autoenc_eco'
    return conf



def ffhq256_autoenc():
    conf = ffhq64_autoenc()
    conf.data_name = 'ffhqlmdb256'
    conf.data_num = 70001
    conf.eval_every_samples = 72_000_000
    conf.eval_ema_every_samples = 72_000_000
    conf.total_samples = 72_000_000
    conf.name = 'ffhq256_autoenc' # override
    conf.semantic_enc = True
    conf.img_size = 256
    conf.cfg = True
    return conf

def nature2560_autoenc():
    conf = ffhq64_autoenc()
    conf.data_name = 'nature2560'
    conf.data_num = 21443
    conf.eval_every_samples = 2_000_000_000
    conf.eval_ema_every_samples = 2_000_000_000
    conf.total_samples = 2_000_000_000
    conf.name = 'nature2560_autoenc' # override
    conf.img_size = (1024,2560)
    conf.semantic_enc = False
    conf.enc_img_size = 256
    conf.cfg = False
    return conf

def nature1024_autoenc():
    conf = ffhq64_autoenc()
    conf.data_name = 'nature1024'
    conf.data_num = 21443
    conf.eval_every_samples = 2_000_000_000
    conf.eval_ema_every_samples = 2_000_000_000
    conf.total_samples = 2_000_000_000
    conf.name = 'nature1024_autoenc' # override
    conf.img_size = (512,1280)
    conf.semantic_enc = False
    conf.enc_img_size = 256
    conf.cfg = False
    return conf

def lhq1024_autoenc():
    conf = ffhq64_autoenc()
    conf.data_name = 'lhq1024'
    conf.data_num = 90000
    conf.eval_every_samples = 2_000_000_000
    conf.eval_ema_every_samples = 2_000_000_000
    conf.total_samples = 2_000_000_000
    conf.name = 'lhq1024_autoenc' # override
    conf.img_size = 1024
    conf.semantic_enc = False
    conf.enc_img_size = 256
    conf.patch_size = 64
    conf.cfg = False
    return conf

def ffhq1024_autoenc():
    conf = ffhq64_autoenc()
    conf.data_name = 'ffhqlmdb1024'
    conf.data_num = 70000
    conf.eval_every_samples = 2_000_000_000
    conf.eval_ema_every_samples = 2_000_000_000
    conf.total_samples = 2_000_000_000
    conf.name = 'ffhq1024_autoenc'
    conf.img_size = 1024
    conf.semantic_enc = False
    conf.enc_img_size = 256
    conf.patch_size = 64
    conf.cfg = False
    return conf

def church256_autoenc():
    conf = ffhq64_autoenc()
    conf.data_name = 'church256'
    conf.data_num = 126227
    conf.eval_every_samples = 120_000_000
    conf.eval_ema_every_samples = 120_000_000
    conf.total_samples = 120_000_000
    conf.name = 'church256_autoenc' # override
    conf.img_size = 256
    conf.semantic_enc = True
    conf.cfg = True
    return conf

def bedroom256_autoenc():
    conf = ffhq64_autoenc()
    conf.data_name = 'bedroom256'
    conf.data_num = 3033042
    conf.eval_every_samples = 120_000_000
    conf.eval_ema_every_samples = 120_000_000
    conf.total_samples = 120_000_000
    conf.name = 'bedroom256_autoenc' # override
    conf.img_size = 256
    conf.semantic_enc = True
    conf.cfg = True
    return conf

def train_autoenc():
    conf = ffhq64_autoenc()
    conf.eval_every_samples = 2_000_000_000
    conf.eval_ema_every_samples = 2_000_000_000
    conf.total_samples = 2_000_000_000
    conf.name = 'train' # override
    conf.semantic_enc = False
    conf.cfg = False
    return conf

def pretrain_ffhq128_autoenc130M():
    conf = ffhq128_autoenc_base()
    conf.pretrain = PretrainConfig(
        name='130M',
        path=f'checkpoints/{ffhq128_autoenc_130M().name}/last.ckpt',
    )
    conf.latent_infer_path = f'checkpoints/{ffhq128_autoenc_130M().name}/latent.pkl'
    return conf

def pretrain_ffhq256_autoenc():
    conf = ffhq256_autoenc()
    conf.pretrain = PretrainConfig(
        name='72M',
        path=f'checkpoints/{ffhq256_autoenc().name}/last.ckpt',
    )
    conf.latent_infer_path = f'checkpoints/{ffhq256_autoenc().name}/latent.pkl'
    return conf

def pretrain_church256_autoenc():
    conf = church256_autoenc()
    conf.pretrain = PretrainConfig(
        name='72M',
        path=f'checkpoints/{church256_autoenc().name}/last.ckpt',
    )
    conf.latent_infer_path = f'checkpoints/{church256_autoenc().name}/latent.pkl'
    return conf

def pretrain_bedroom256_autoenc():
    conf = bedroom256_autoenc()
    conf.pretrain = PretrainConfig(
        name='72M',
        path=f'checkpoints/{bedroom256_autoenc().name}/last.ckpt',
    )
    conf.latent_infer_path = f'checkpoints/{bedroom256_autoenc().name}/latent.pkl'
    return conf

def pretrain_nature1024_autoenc():
    conf = nature1024_autoenc()
    conf.pretrain = PretrainConfig(
        name='72M',
        path=f'checkpoints/{nature1024_autoenc().name}/last.ckpt',
    )
    conf.latent_infer_path = f'checkpoints/{nature1024_autoenc().name}/latent.pkl'
    return conf

def pretrain_lhq1024_autoenc():
    conf = lhq1024_autoenc()
    conf.pretrain = PretrainConfig(
        name='72M',
        path=f'checkpoints/{lhq1024_autoenc().name}/last.ckpt',
    )
    conf.latent_infer_path = f'checkpoints/{lhq1024_autoenc().name}/latent.pkl'
    return conf

def pretrain_ffhq1024_autoenc():
    conf = ffhq1024_autoenc()
    conf.pretrain = PretrainConfig(
        name='72M',
        path=f'checkpoints/{ffhq1024_autoenc().name}/last.ckpt',
    )
    conf.latent_infer_path = f'checkpoints/{ffhq1024_autoenc().name}/latent.pkl'
    return conf

def pretrain_autoenc():
    conf = train_autoenc()
    conf.pretrain = None
    conf.latent_infer_path = None
    return conf