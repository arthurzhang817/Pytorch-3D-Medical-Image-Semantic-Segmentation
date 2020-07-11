# Adapted from https://github.com/meetshah1995/pytorch-semseg/blob/master/ptsemseg/loss/__init__.py

import logging
import functools
import numpy as np

from .loss import (
    cross_entropy3d
)

key2loss = {
    "cross_entropy3d": cross_entropy3d
}

def get_loss_function(cfg):

    if cfg.TRAINING.LOSS_FUNC is None:
        return cross_entropy3d
    else:
        loss_dict = cfg.TRAINING.LOSS_FUNC
        loss_name = loss_dict.name
        loss_params = {k: v for k, v in loss_dict.items() if k != "name"}

        if loss_name not in key2loss:
            raise NotImplementedError("Loss {} not implemented".format(loss_name))
        return functools.partial(key2loss[loss_name], **loss_params)