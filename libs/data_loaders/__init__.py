# Adapted from https://github.com/meetshah1995/pytorch-semseg/blob/master/ptsemseg/loader/__init__.py

import json

from torch.utils import data as torch_data

from .prostate_ct_volume_loader import Prostate_CT_Volume_Loader


def get_loader(name):

    return {
        "prostateCT_vol": Prostate_CT_Volume_Loader
    }[name]


def build_data_loader(config, write, logger):

    # setup data_loader
    data_loader = get_loader(config.DATASET.NAME)
    data_path = config.DATASET.PATH

    t_loader = data_loader(
                root_dir=data_path,
                split=config.DATASET.TRAIN_SPLIT
            )
    v_loader = data_loader(
        root_dir=data_path,
        split=config.DATASET.VAL_SPLIT
    )

    train_loader = torch_data.DataLoader(
        t_loader,
        batch_size=config.TRAINING.BATCH_SIZE,
        num_workers=config.TRAINING.WORKERS,
        shuffle=True
    )
    val_loader = torch_data.DataLoader(
        v_loader, 
        batch_size=config.TRAINING.BATCH_SIZE,
        num_workers=config.TRAINING.WORKERS,
        shuffle=False
    )
    
    logger.info("train_loader, val_loader ready.")

    return t_loader, v_loader, train_loader, val_loader