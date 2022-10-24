import os
import shutil
import tarfile
from multiprocessing import Process
from time import sleep
from urllib import request

from joblib import Parallel, delayed
from tqdm import tqdm

WORKER = 5
IMAGENET_VAL_PATH = "/root/autodl-pub/ImageNet/ILSVRC2012/ILSVRC2012_img_val.tar"
IMAGENET_TRAIN_PATH = "/root/autodl-pub/ImageNet/ILSVRC2012/ILSVRC2012_img_train.tar"
DATA_ROOT = "/root/autodl-tmp/imagenet"

TRAIN_DEST = os.path.join(DATA_ROOT, "train")
VAL_DEST = os.path.join(DATA_ROOT, "val")


if not os.path.isdir(DATA_ROOT):
    os.mkdir(DATA_ROOT)
    os.mkdir(TRAIN_DEST)
    os.mkdir(VAL_DEST)
else:
    raise Exception("Data already exists")

print("Downloading and unzipping train set...")
with tarfile.open(IMAGENET_TRAIN_PATH, mode="r:") as tar:
    for item in tqdm(tar):
        cls_name = item.name.strip(".tar")
        a = tar.extractfile(item)
        e_path = f"{TRAIN_DEST}/{cls_name}/"
        os.mkdir(e_path)
        with tarfile.open(fileobj=a, mode="r:") as b:
            b.extractall(e_path)

print("Downloading and unzipping validation set...")
with tarfile.open(IMAGENET_VAL_PATH, mode="r:") as tar:
    Process(target=tar.extractall, args=(VAL_DEST,)).start()
    counter = 0
    with tqdm(total=50000) as pbar:
        while counter < 50000:
            pbar.n = counter
            pbar.refresh()
            sleep(1)
            counter = len(os.listdir(VAL_DEST))

print("Validation set unzipped, fetching and decoding processing script...")
lines = []
with request.urlopen(
    "https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh"
) as response:
    for i in tqdm(range(55000)):
        lines.append(response.readline().decode("utf-8"))

print("Script parsed, executing...")
os.chdir(VAL_DEST)
Parallel(n_jobs=WORKER, verbose=3)(
    delayed(os.mkdir)(lines[i][9:18]) for i in range(1000)
)
Parallel(n_jobs=WORKER, verbose=3)(
    delayed(lambda line: shutil.move(line[3:31], line[32:41]))(lines[i])
    for i in range(1000, 51000)
)
