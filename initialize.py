import torch
import argparse

from templates import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--data_path', '-d', type=str, default="./dataset",
                        help='dataset name')
    parser.add_argument('--out_path', '-o', type=str, default="semantic",
                        help='output file path')
    args = parser.parse_args()
    conf = train_autoenc()
    conf.data_path = args.data_path

    env = lmdb.open(
            args.data_path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
    if not env:
        raise IOError('Cannot open lmdb dataset', args.data_path)
    with env.begin(write=False) as txn:
        length = int(txn.get('length'.encode('utf-8')).decode('utf-8'))
    model = Semantic(length, device='cuda', conf=conf, initial_clip=True)

    torch.save(
        {
            'model_state_dict':model.state_dict()
        },
        args.out_path
    )