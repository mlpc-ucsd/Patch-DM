import argparse
import multiprocessing
from functools import partial
from io import BytesIO
from pathlib import Path

import lmdb
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as trans_fn
from tqdm import tqdm
import os


def resize_and_convert(img, size, resample, quality=100):
    img = trans_fn.resize(img, size, resample) 
    img = trans_fn.center_crop(img, size)
    buffer = BytesIO()
    img.save(buffer, format="jpeg", quality=quality)
    val = buffer.getvalue()

    return val


def resize_multiple(img,
                    sizes=(128, 256, 512, 1024),
                    resample=Image.LANCZOS,
                    quality=100):
    imgs = []

    for size in sizes:
        imgs.append(resize_and_convert(img, size, resample, quality))

    return imgs


def resize_worker(img_file, resample):
    i, (file, idx) = img_file
    img = Image.open(file)
    img = img.convert("RGB")
    buffer = BytesIO()
    img.save(buffer, format="jpeg", quality=100)
    val = buffer.getvalue()

    return i, idx, [val]


def prepare(env,
            paths,
            n_worker,
            sizes=(128, 256, 512, 1024),
            resample=Image.LANCZOS):
    resize_fn = partial(resize_worker, resample=resample)

    indexs = []
    for each in paths:
        file = os.path.basename(each)
        name, ext = file.split('.')
        idx = int(name)
        indexs.append(idx)

    files = sorted(zip(paths, indexs), key=lambda x: x[1])
    files = list(enumerate(files))
    total = 0

    with multiprocessing.Pool(n_worker) as pool:
        for i, idx, imgs in tqdm(pool.imap_unordered(resize_fn, files)):
            for img in imgs:
                key = f"{str(idx).zfill(7)}".encode("utf-8")
                with env.begin(write=True) as txn:
                    txn.put(key, img)
            total += 1
        with env.begin(write=True) as txn:
            txn.put("length".encode("utf-8"), str(total).encode("utf-8"))


class ImageFolder(Dataset):
    def __init__(self, folder, exts=['jpg']):
        super().__init__()
        self.paths = [
            p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')
        ]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = os.path.join(self.folder, self.paths[index])
        img = Image.open(path)
        return img


if __name__ == "__main__":
    """
    converting ffhq images to lmdb
    """
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--input', '-i', type=str, default="",
                        help='input image folder path')
    parser.add_argument('--output', '-o', type=str, default="",
                        help='output image folder path')
    parser.add_argument('--num_workers', '-n', type=int, default=12,
                        help='number of workers')
    args = parser.parse_args()

    num_workers = args.num_workers
    in_path = args.input
    out_path = args.output
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    resample_map = {"lanczos": Image.LANCZOS, "bilinear": Image.BILINEAR}
    resample = resample_map['lanczos']


    print(f"Make dataset of image")
    exts = ['jpg','png']
    paths = [p for ext in exts for p in Path(f'{in_path}').glob(f'**/*.{ext}')]
    _file = os.path.basename(paths[0])
    _name, _ext = _file.split('.')
    try:
        _index = int(_name)
    except:
        for i in range(len(paths)):
            path_list = paths[i]._parts
            old_name = '/'.join(path_list)
            path_list[-1] = f"{i}".zfill(7)+'.jpg'
            new_name = '/'.join(path_list)
            os.rename(old_name,new_name)

    paths = [p for ext in exts for p in Path(f'{in_path}').glob(f'**/*.{ext}')]
    with lmdb.open(out_path, map_size=1024**4, readahead=False) as env:
        prepare(env, paths, num_workers, resample=resample)
