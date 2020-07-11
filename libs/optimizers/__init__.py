# Adapted from https://github.com/meetshah1995/pytorch-semseg/blob/master/ptsemseg/optimizers/__init__.py

import logging

from torch.optim import SGD, Adam, ASGD, Adamax, Adadelta, Adagrad, RMSprop

logger = logging.getLogger("ptsemseg")

key2opt = {
    "sgd": SGD,
    "adam": Adam,
    "asgd": ASGD,
    "adamax": Adamax,
    "adadelta": Adadelta,
    "adagrad": Adagrad,
    "rmsprop": RMSprop,
}


def get_optimizer(CONFIG):


    if CONFIG.TRAINING.OPTIM.name is None:
        logger.info("Using SGD optimizer")
        return SGD

    else:
        opt_name = CONFIG.TRAINING.OPTIM.name
        if opt_name not in key2opt:
            raise NotImplementedError("Optimizer {} not implemented".format(opt_name))

        logger.info("Using {} optimizer".format(opt_name))
        return key2opt[opt_name]