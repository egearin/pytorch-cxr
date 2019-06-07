import os
import sys
import logging
from pathlib import Path
import yaml

import torch
import torchvision


formatter = logging.Formatter('%(asctime)s [%(levelname)-5s] %(message)s')

# stream handler
chdr = logging.StreamHandler(sys.stdout)
chdr.setLevel(logging.DEBUG)
chdr.setFormatter(formatter)

logger = logging.getLogger("pytorch-stn")
logger.setLevel(logging.DEBUG)
logger.addHandler(chdr)


def set_log_to_file(log_file):
    log_path = Path(log_file).resolve()
    Path.mkdir(log_path.parent, parents=True, exist_ok=True)

    fhdr = logging.FileHandler(log_path)
    fhdr.setLevel(logging.DEBUG)
    fhdr.setFormatter(formatter)

    logger.addHandler(fhdr)


class SlackClientHandler(logging.Handler):

    def __init__(self, credential_file, ch_name):
        super().__init__()
        with open(credential_file, 'r') as f:
            tmp = yaml.safe_load(f)
        self.slack_token = tmp['token']
        self.slack_recipients = tmp['recipients']
        #self.slack_token = os.getenv("SLACK_API_TOKEN")
        #self.slack_user = os.getenv("SLACK_API_USER")
        if self.slack_token is None or self.slack_recipients is None:
            raise KeyError

        from slack import WebClient
        self.client = WebClient(self.slack_token)

        # getting user id
        ans = self.client.users_list()
        users = [u['id'] for u in ans['members'] if u['name'] in self.slack_recipients]
        # open DM channel to the users
        ans = self.client.conversations_open(users=','.join(users))
        self.channel = ans['channel']['id']
        ans = self.client.chat_postMessage(channel=self.channel, text=f"*{ch_name}*")
        self.thread = ans['ts']

    def emit(self, record):
        try:
            msg = self.format(record)
            self.client.chat_postMessage(channel=self.channel, thread_ts=self.thread, text=f"```{msg}```")
        except:
            self.handleError(record)


def set_log_to_slack(credential_file, ch_name):
    try:
        credential_path = Path(credential_file).resolve()
        shdr = SlackClientHandler(credential_path, ch_name)
        shdr.setLevel(logging.INFO)
        shdr.setFormatter(formatter)
        logger.addHandler(shdr)
    except:
        raise RuntimeError


def print_versions():
    logger.info(f"pytorch version: {torch.__version__}")
    logger.info(f"torchvision version: {torchvision.__version__}")


def get_devices(cuda=None):
    if cuda is None:
        logger.info(f"use CPUs")
        return [torch.device("cpu")]
    else:
        assert torch.cuda.is_available()
        avail_devices = list(range(torch.cuda.device_count()))
        use_devices = [int(i) for i in cuda.split(",")]
        assert max(use_devices) in avail_devices
        logger.info(f"use cuda on GPU {use_devices}")
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(i) for i in use_devices])
        return [torch.device(f"cuda:{k}") for k in use_devices]

