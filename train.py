from pathlib import Path
import logging

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from prettytable import PrettyTable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import ConcatDataset, Subset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

import torchvision
import torchnet as tnt

from predict import PredictEnvironment, Predictor
from dataset import STANFORD_CXR_BASE, MIMIC_CXR_BASE, StanfordDataset
from danet import Network
from utils import logger, print_versions, get_devices
from adamw import AdamW


class TrainEnvironment(PredictEnvironment):
    """
    This environment object inherits PredictEnvironment from predict.py,
    with adding the datasets and their corresponding data loaders,
    optimizer, lr scheduler, and the loss function using in training.
    """

    def __init__(self, devices, local_rank=None):
        self.devices = devices
        self.local_rank = local_rank
        if local_rank is None:
            self.device = self.devices[0]
            self.distributed = False
        else:
            self.device = self.devices[local_rank]
            self.distributed = True

        stanford_train_set = StanfordDataset(STANFORD_CXR_BASE, "train.csv", mode="per_study")
        stanford_test_set = StanfordDataset(STANFORD_CXR_BASE, "valid.csv", mode="per_study")
        mimic_train_set = StanfordDataset(MIMIC_CXR_BASE, "train.csv", mode="per_study")
        mimic_test_set = StanfordDataset(MIMIC_CXR_BASE, "valid.csv", mode="per_study")
        concat_set = ConcatDataset([stanford_train_set, stanford_test_set, mimic_train_set, mimic_test_set])

        num_test = 20000
        datasets = random_split(concat_set, [len(concat_set) - num_test, num_test])
        #subset = Subset(concat_set, range(0, 36))
        #datasets = random_split(subset, [len(subset) - 12, 12])

        train_set_percent = len(datasets[0]) / len(concat_set) * 100.
        test_set_percent = len(datasets[1]) / len(concat_set) * 100.
        logger.info(f"using {len(datasets[0])}/{len(concat_set)} ({train_set_percent:.1f}%) entries for training dataset")
        logger.info(f"using {len(datasets[1])}/{len(concat_set)} ({test_set_percent:.1f}%) entries for testing dataset")

        self.train_loader = DataLoader(datasets[0], batch_size=20, num_workers=20, shuffle=True, pin_memory=True)
        self.test_loader = DataLoader(datasets[1], batch_size=20, num_workers=20, shuffle=False, pin_memory=True)

        self.out_dim = len(stanford_train_set.labels)
        self.labels = stanford_train_set.labels

        #img, tar = datasets[0]
        #plt.imshow(img.squeeze(), cmap='gray')

        super().__init__(out_dim=self.out_dim, device=self.device)

        self.optimizer = AdamW(self.model.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-5)
        #self.scheduler = ReduceLROnPlateau(self.optimizer, factor=0.1, patience=5, mode='min')
        self.loss = nn.BCEWithLogitsLoss()

        if self.distributed:
            self.to_distributed()

    def to_distributed(self):
        if self.device != torch.device("cpu"):
            torch.cuda.set_device(self.device)
        dist.init_process_group(backend="nccl", init_method="env://")

        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        logger.info(f"initialized on {self.device} as rank {self.rank} of {self.world_size}")

        self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[self.device], output_device=self.device, find_unused_parameters=True)
        train_set = self.train_loader.dataset
        batch_size = self.train_loader.batch_size
        num_workers = self.train_loader.num_workers
        self.train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=num_workers,
                                       sampler=DistributedSampler(train_set), shuffle=False, pin_memory=True)

    def load_data_indices(self, runtime_path):
        train_idx = runtime_path.joinpath("train.idx")
        logger.debug(f"loading the train dataset indices from {train_idx}")
        self.train_loader.dataset.indices = np.loadtxt(train_idx, dtype=np.int).tolist()
        valid_idx = runtime_path.joinpath("valid.idx")
        logger.debug(f"loading the valid dataset indices from {valid_idx}")
        self.test_loader.dataset.indices = np.loadtxt(valid_idx, dtype=np.int).tolist()

    def save_data_indices(self, runtime_path):
        if self.distributed and self.local_rank > 0:
            return
        runtime_path.mkdir(mode=0o755, parents=True, exist_ok=True)
        train_idx = runtime_path.joinpath("train.idx")
        logger.debug(f"saving the train dataset indices to {train_idx}")
        np.savetxt(train_idx, self.train_loader.dataset.indices, fmt="%d")
        valid_idx = runtime_path.joinpath("valid.idx")
        logger.debug(f"saving the valid dataset indices to {valid_idx}")
        np.savetxt(valid_idx, self.test_loader.dataset.indices, fmt="%d")

    def save_model(self, filename):
        if self.distributed and self.local_rank > 0:
            return
        filedir = Path(filename).parent.resolve()
        filedir.mkdir(mode=0o755, parents=True, exist_ok=True)

        filepath = Path(filename).resolve()
        logger.debug(f"saving the model to {filepath}")
        state = self.model.state_dict()
        torch.save(state, filename)


class Trainer:

    def __init__(self, env, runtime_path="train", tensorboard=False):
        self.env = env
        self.runtime_path = runtime_path
        self.tensorboard = tensorboard
        if tensorboard:
            tblog_path = runtime_path.joinpath("tensorboard").resolve()
            tblog_path.mkdir(mode=0o755, parents=True, exist_ok=True)
            self.writer = SummaryWriter(log_dir=str(tblog_path))

    def train(self, num_epoch, start_epoch=1):
        if start_epoch > 1:
            model_path = runtime_path.joinpath(f"model_epoch_{(start_epoch - 1):03d}.pth.tar")
            self.env.load_model(model_path)
            self.env.load_data_indices(self.runtime_path)
            if self.env.distributed:
                self.env.train_loader.sampler.set_epoch(start_epoch - 1)
        else:
            self.env.save_data_indices(self.runtime_path)

        for epoch in range(start_epoch, num_epoch + 1):
            self.train_epoch(epoch)
            self.test(epoch)

    def train_epoch(self, epoch):
        train_loader = self.env.train_loader
        train_set = train_loader.dataset

        self.env.model.train()
        progress = 0

        ave_len = len(train_loader) // 100 + 1
        ave_loss = tnt.meter.MovingAverageValueMeter(ave_len)

        ckpt_step = 0.1
        ckpts = iter(len(train_set) * np.arange(ckpt_step, 1 + ckpt_step, ckpt_step))
        ckpt = next(ckpts)

        if self.env.distributed:
            tqdm_desc = f"training{self.env.local_rank}"
            tqdm_pos = self.env.local_rank
        else:
            tqdm_desc = "training"
            tqdm_pos = 0

        t = tqdm(enumerate(train_loader), total=len(train_loader), desc=tqdm_desc,
                 dynamic_ncols=True, position=tqdm_pos)

        for batch_idx, (data, target) in t:
            data, target = data.to(self.env.device), target.to(self.env.device)

            self.env.optimizer.zero_grad()
            output = self.env.model(data)
            loss = self.env.loss(output, target)
            loss.backward()
            self.env.optimizer.step()

            #self.env.scheduler.step(loss.item())

            t.set_description(f"{tqdm_desc} (loss: {loss.item():.4f})")
            t.refresh()

            ave_loss.add(loss.item())
            progress += len(data)

            if progress > ckpt and self.tensorboard:
                #logger.info(f"train epoch {epoch:03d}:  "
                #        f"progress/total {progress:06d}/{len(train_loader.dataset):06d} "
                #            f"({100. * batch_idx / len(train_loader):6.2f}%)  "
                #            f"loss {loss.item():.6f}")

                x = (epoch - 1) + progress / len(train_set)
                global_step = int(x / ckpt_step)
                self.writer.add_scalar("loss", loss.item(), global_step=global_step)
                ckpt = next(ckpts)

            del loss

        logger.info(f"train epoch {epoch:03d}:  "
                    f"loss {ave_loss.value()[0]:.6f}")

        self.env.save_model(self.runtime_path.joinpath(f"model_epoch_{epoch:03d}.pth.tar"))

    def test(self, epoch):
        test_loader = self.env.test_loader
        test_set = test_loader.dataset
        out_dim = self.env.out_dim
        labels = self.env.labels

        aucs = [tnt.meter.AUCMeter() for i in range(out_dim)]

        self.env.model.eval()
        with torch.no_grad():
            test_loss = 0
            correct = 0

            if self.env.distributed:
                tqdm_desc = f"testing{self.env.local_rank}"
                tqdm_pos = self.env.local_rank
            else:
                tqdm_desc = "testing"
                tqdm_pos = 0

            t = tqdm(enumerate(test_loader), total=len(test_loader), desc=tqdm_desc,
                     dynamic_ncols=True, position=tqdm_pos)

            ones = torch.ones((out_dim)).int().to(self.env.device)
            zeros = torch.zeros((out_dim)).int().to(self.env.device)

            for batch_idx, (data, target) in t:
                data, target = data.to(self.env.device), target.to(self.env.device)
                output = self.env.model(data)
                for i in range(out_dim):
                    aucs[i].add(output[:, i], target[:, i])
                pred = torch.where(output > 0., ones, zeros)
                correct += pred.eq(target.int()).sum().item()

            #correct /= out_dim
            total = len(test_set) * out_dim
            percent = 100. * correct / total
            if self.tensorboard:
                self.writer.add_scalar("accuracy", percent, global_step=epoch)

            logger.info(f"val epoch {epoch:03d}:  "
                        f"accuracy {correct}/{total} "
                        f"({percent:.2f}%)")

            p = PrettyTable()
            p.field_names = ["findings", "score"]
            for i in range(out_dim):
                p.add_row([labels[i], f"{aucs[i].value()[0]:.6f}"])
            ave_auc = np.mean([k.value()[0] for k in aucs])
            tbl_str = p.get_string(title=f"AUC scores (average {ave_auc:.6f})")
            logger.info(f"\n{tbl_str}")


# We want to visualize the output of the spatial transformers layer
# after the training, we visualize a batch of input images and
# the corresponding transformed batch using STN.

def convert_image_np(inp):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    return inp

def visualize_stn():
    with torch.no_grad():
        # Get a batch of training data
        data = next(iter(test_loader))[0].to(device)
        input_tensor = data.cpu()
        transformed_input_tensor = model.stn(data).cpu()
        in_grid = convert_image_np(
            torchvision.utils.make_grid(input_tensor))
        out_grid = convert_image_np(
            torchvision.utils.make_grid(transformed_input_tensor))
        # Plot the results side-by-side
        f, axarr = plt.subplots(1, 2)
        axarr[0].imshow(in_grid)
        axarr[0].set_title('Dataset Images')
        axarr[1].imshow(out_grid)
        axarr[1].set_title('Transformed Images')


if __name__ == "__main__":
    import os
    import argparse

    parser = argparse.ArgumentParser(description="CXR Training")
    # for testing
    parser.add_argument('--cuda', default=None, type=str, help="use GPUs with its device ids, separated by commas")
    parser.add_argument('--epoch', default=100, type=int, help="max number of epochs")
    parser.add_argument('--start-epoch', default=1, type=int, help="start epoch, especially need to continue from a stored model")
    parser.add_argument('--runtime-dir', default='./runtime', type=str, help="runtime directory to store log, pretrained models, and tensorboard metadata")
    parser.add_argument('--tensorboard', default=False, action='store_true', help="true if logging to tensorboard")
    parser.add_argument('--slack', default=False, action='store_true', help="true if logging to slack")
    parser.add_argument('--local_rank', type=int, help="this is for the use of torch.distributed.launch utility")
    args = parser.parse_args()

    runtime_path = Path(args.runtime_dir).resolve()
    #runtime_path = Path("train_20190527_per_study_256").resolve()

    # set logger
    if args.local_rank is None:
        log_file = "train.log"
        logger.set_log_to_stream()
    else:
        log_file = f"train.{args.local_rank}.log"
        logger.set_log_to_stream(level=logging.INFO)
    logger.set_log_to_file(runtime_path.joinpath(log_file))
    if args.slack:
        set_log_to_slack(Path(__file__).parent.joinpath(".slack"), runtime_path.name)

    print_versions()
    logger.info(f"runtime_path: {runtime_path}")

    # start training
    devices = get_devices(args.cuda)
    env = TrainEnvironment(devices, args.local_rank)
    t = Trainer(env, runtime_path=runtime_path, tensorboard=args.tensorboard)
    t.train(args.epoch, start_epoch=args.start_epoch)

    # Visualize the STN transformation on some input batch
    #visualize_stn()
    #plt.show()
