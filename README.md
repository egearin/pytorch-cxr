# pytorch-cxr

Chest X-ray multi-labels binary classifications

## Prerequisites

- Python 3.6+
- PyTorch 1.1.0+

## Datasets

- [Stanford CheXpert dataset](https://stanfordmlgroup.github.io/competitions/chexpert/)
- [MIT MIMIC-CXR dataset](https://physionet.org/physiobank/database/mimiccxr/)

You need to resize and copy from the extracted original data.
Modify `{src,tar}_path` at the end of `dataset.py` properly for your environment.

```
$ python dataset.py
```

## Training

If no `--cuda <comma-separated gpu-ids>` given, cpus will be used for training.
If no distributed environment, the first cuda id is used for the training and the others are ignored.

```
$ python train.py --cuda <comma-separated gpu-ids> --runtime-dir <your-runtime-dir> [--tensorboard] [--slack]
```

If you want to run in distributed mode, please refer to [here](https://pytorch.org/docs/stable/distributed.html#launch-utility).

```
$ python -m torch.distributed.launch --nproc_per_node=<NUM_GPUS_YOU_HAVE> train.py --cuda <gpu-ids> --runtime-dir <runtime-dir> [any optinal switches]
```
or using the `train.sh` with proper modification of `node id` etc.

if you want to log to slack, add `.slack` file in yaml format:

```
token: <your bot-slack-app token>
recipients: <list of log message recipients username>
```

## Prediction

```
$ python predict.py <study-dir-to-be-predicted>
```

