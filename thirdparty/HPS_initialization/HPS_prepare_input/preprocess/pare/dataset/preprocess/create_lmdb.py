import os
import sys
import cv2
import lmdb
import numpy as np
from tqdm import tqdm

from pare.core.config import DATASET_FILES, DATASET_FOLDERS, DATASET_LMDB_PATH

LMDB_MAP_SIZE = 1 << 40  # MODIFY

def convert_to_lmdb(dataset, is_train):
    img_dir = DATASET_FOLDERS[dataset]
    data = np.load(DATASET_FILES[is_train][dataset])
    imgname = list(data['imgname'])

    split = 'train' if is_train else 'test'
    env = lmdb.open(os.path.join(DATASET_LMDB_PATH, f'{dataset}_{split}'), map_size=LMDB_MAP_SIZE)

    print(dataset, split)
    print(f'Number of images: {len(imgname)}')
    with env.begin(write=True) as txn:

        for img_fn in tqdm(imgname):
            f = os.path.join(img_dir, img_fn)
            with open(f, "rb") as file:
                data = file.read()
                # cv_img = cv2.imread(f)[:, :, ::-1].copy().astype(np.float32)
                txn.put(f'{img_fn}'.encode(), data)

    print('=========== END ===========')


if __name__ == '__main__':
    convert_to_lmdb(dataset=sys.argv[1], is_train=int(sys.argv[2]))