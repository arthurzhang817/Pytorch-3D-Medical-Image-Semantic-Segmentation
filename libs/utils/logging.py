# Adapted from https://github.com/meetshah1995/pytorch-semseg/blob/master/ptsemseg/utils.py

import os
import datetime
import logging


def get_logger(logdir, job):
    logger = logging.getLogger(job)
    ts = str(datetime.datetime.now()).split(".")[0].replace(" ", "_")
    ts = ts.replace(":", "_").replace("-", "_")
    file_path = os.path.join(logdir, "run_{}.log".format(ts))
    hdlr = logging.FileHandler(file_path)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.setLevel(logging.INFO)
    return logger