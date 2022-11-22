#!/bin/env/python
import subprocess as sp
from pathlib import Path

base_dir = Path("runtime").resolve()

title = "cxr_single"

#20190924_noniid_max_dist_per_image_no_postive_weight
#20190927_noniid_max_dist_per_study_no_postive_weight

plots = {
    "dist_per_image_nopw": "20190924_noniid_max_dist_per_image_no_postive_weight/tensorboard.0/",
    "dist_per_study_densenet169": "20190929_noniid_max_dist_per_study_densenet169/tensorboard.0/",
}

logdir = ",".join([f"{k}:{base_dir.joinpath(v)}" for k, v in plots.items()])

cmd = ["tensorboard", "--host", "0.0.0.0", "--window_title", title, "--logdir", logdir]
print(" ".join(cmd))
sp.run(cmd)
