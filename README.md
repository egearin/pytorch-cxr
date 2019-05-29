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

```
$ python train.py --cuda --runtime-dir <your-runtime-dir> [--tensorboard] [--slack]
```

if you want to log to slack, add `.slack` file in yaml format:
```
token: <your bot-slack-app token>
recipients: <list of log message recipients username>
```

## Prediction

```
$ python predict.py <study-dir-to-be-predicted>
```

