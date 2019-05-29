import sys
from pathlib import Path

import imageio
import pandas as pd

import torch
from torch.utils.data import Dataset, Subset
from torchvision import transforms

from utils import logger

#STANFORD_CXR_BASE = Path("/mnt/hdd/cxr/Stanford/full_resolution_version").resolve()
STANFORD_CXR_BASE = Path("/mnt/hdd/cxr/Stanford/downsampled_version").resolve()

MODES = ["per_image", "per_study"]


def _group_study(images):
    prev_study = None
    num_images_per_entry = []
    entries = []
    cur_entry = None

    for (f, l) in sorted(images, key=lambda e: e[0]):
        study = f.parent
        if cur_entry is None:
            cur_entry = ([f], l)
        if study == prev_study:
            cur_entry[0].append(f)
            assert cur_entry[1] == l
        else:
            num_images_per_entry.append(len(cur_entry[0]))
            entries.append(cur_entry)
            cur_entry = ([f], l)
        prev_study = study

    assert max(num_images_per_entry) <= 3
    return entries


def _load_manifest(file_path, mode="per_image"):
    assert mode in MODES
    if not file_path.exists():
        logger.error(f"manifest file {file_path} not found.")
        sys.exit(1)

    logger.debug(f"loading dataset manifest {file_path} ...")
    df = pd.read_csv(str(file_path)).fillna(0)
    #df = df.loc[df['AP/PA'] == 'PA']
    LABELS = df.columns.values.tolist()[5:]
    paths = df.iloc[:, 0].tolist()
    labels = df.replace(-1, 0).iloc[:, 5:].values.tolist()
    images = [(STANFORD_CXR_BASE.joinpath(p), l) for p, l in zip(paths, labels)]

    if mode == "per_image":
        entries = images
    elif mode == "per_study":
        entries = _group_study(images)
    else:
        raise RuntimeError

    logger.debug(f"{len(entries)} entries are loaded.")
    return entries, LABELS


cxr_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomRotation((-10, 10)),
    transforms.RandomCrop((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    #transforms.Normalize((0.1307,), (0.3081,))
])


def get_image(img_path):
    image = imageio.imread(img_path)
    image_tensor = cxr_transforms(image)
    return image_tensor


def get_study(img_paths):
    image_tensor = torch.zeros(3, 256, 256)
    for i, img_path in enumerate(img_paths):
        image = imageio.imread(img_path)
        image_tensor[i, :, :] = cxr_transforms(image)
    image_tensor = image_tensor[torch.randperm(3), :, :]
    return image_tensor


class StanfordDataset(Dataset):

    def __init__(self, manifest_file, mode="per_image", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.manifest_file = STANFORD_CXR_BASE.joinpath(manifest_file).resolve()
        self.entries, self.labels = _load_manifest(self.manifest_file, mode)
        self.mode = mode

    def __getitem__(self, index):
        if self.mode == "per_image":
            img_path, label = self.entries[index]
            image_tensor = get_image(img_path)
            target_tensor = torch.FloatTensor(label)
        elif self.mode == "per_study":
            img_paths, label = self.entries[index]
            image_tensor = get_study(img_paths)
            target_tensor = torch.FloatTensor(label)
        else:
            raise RuntimeError

        return image_tensor, target_tensor

    def __len__(self):
        return len(self.entries)


def copy_stanford_dataset(src_path, tar_path):
    from tqdm import tqdm
    from PIL import Image
    for m in [src_path.joinpath("train.csv"), src_path.joinpath("valid.csv")]:
        print(f">>> processing {m}...")
        df = pd.read_csv(str(m))
        for i in tqdm(range(len(df)), total=len(df)):
            f = df.iloc[i]["Path"].split('/', 1)[1]
            ff = src_path.joinpath(f).resolve()
            img = Image.open(ff)
            w, h = img.size
            rs = (512, int(h/w*512)) if w < h else (int(w/h*512), 512)
            resized = img.resize(rs, Image.LANCZOS)
            r = ff.relative_to(src_path)
            t = tar_path.joinpath(r).resolve()
            #print(f"{ff} -> {t}")
            Path.mkdir(t.parent, parents=True, exist_ok=True)
            resized.save(t, "JPEG")
            df.at[i, "Path"] = f
        r = m.relative_to(src_path).name
        t = tar_path.joinpath(r).resolve()
        df.to_csv(t, float_format="%.0f", index=False)


def copy_mimic_dataset(src_path, tar_path):
    from tqdm import tqdm
    from PIL import Image
    for m in [src_path.joinpath("train.csv"), src_path.joinpath("valid.csv")]:
        print(f">>> processing {m}...")
        df = pd.read_csv(str(m))
        for i in tqdm(range(len(df)), total=len(df)):
            f = df.iloc[i]["path"]
            ff = src_path.joinpath(f).resolve()
            img = Image.open(ff)
            w, h = img.size
            rs = (512, int(h/w*512)) if w < h else (int(w/h*512), 512)
            resized = img.resize(rs, Image.LANCZOS)
            r = ff.relative_to(src_path)
            t = tar_path.joinpath(r).resolve()
            #print(f"{ff} -> {t}")
            Path.mkdir(t.parent, parents=True, exist_ok=True)
            resized.save(t, "JPEG")
        r = m.relative_to(src_path).name
        t = tar_path.joinpath(r).resolve()
        df.to_csv(t, float_format="%.0f", index=False)


if __name__ == "__main__":
    """
    manifest_file = "CheXpert-v1.0-small/train.csv"
    file_path = STANFORD_CXR_BASE.joinpath(manifest_file).resolve()
    entries, labels = _load_manifest(file_path, mode="per_study")
    #from pprint import pprint
    #pprint(entries[:20])
    """
    # resize & copy stanford dataset
    src_path = Path("/media/nfs/CXR/Stanford/full_resolution_version/CheXpert-v1.0").resolve()
    tar_path = Path("/mnt/hdd/cxr/stanford/v1").resolve()
    copy_stanford_dataset(src_path, tar_path)

    # resize & copy mimic dataset
    src_path = Path("/media/mycloud/MIMIC_CXR").resolve()
    tar_path = Path("/mnt/hdd/cxr/mimic/v1").resolve()
    copy_mimic_dataset(src_path, tar_path)
