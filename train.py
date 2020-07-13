import os
import yaml
import time
import shutil
import numpy as np
from tqdm import tqdm
from addict import Dict

import torch
import torch.nn as nn
from tensorboardX import SummaryWriter

from libs.models import Deeper_ResUnet_3D, ResUnet_3D
from libs.optimizers import get_optimizer
from libs.schedulers import get_scheduler
from libs.utils.logging import get_logger
from libs.loss_funcs import get_loss_function
from libs.data_loaders import build_data_loader
from libs.metrics import running_seg_score, averageMeter
from libs.utils.device import get_device, memory_usage_report, dict_conversion


def train(config_file):
    
    # Configuration
    with open(config_file) as fp:
        CONFIG = Dict(yaml.load(fp))
    
    # Device
    device = get_device(CONFIG.CUDA)
    torch.backends.cudnn.benchmark = True

    # Setup logger&writer and run dir
    run_id = time.strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join("runs", os.path.basename(config_file)[:-4], run_id)
    writer = SummaryWriter(log_dir=log_dir)

    print("RUN DIR: {}".format(log_dir))
    shutil.copy(config_file, log_dir)

    logger = get_logger(log_dir, CONFIG.JOB)
    logger.info("Starting the program...")
    logger.propagate = False

    # Dataset
    logger.info("Using dataset: {}".format(CONFIG.DATASET.NAME))

    # Dataloader
    _, _, train_loader, val_loader = build_data_loader(CONFIG, writer, logger)
    logger.info("Dataloader Ready.")

    if not CONFIG.MODEL.ADV:
        train_base(CONFIG, writer, logger, train_loader, val_loader, device)
    else:
        pass

    print('-'*60)
    memory_usage_report(device=torch.device("cuda:0"), logger=logger)
    print('-'*40)
    memory_usage_report(device=torch.device("cuda:1"), logger=logger)
    print('-'*60)


def train_base(CONFIG, writer, logger, train_loader, val_loader, device):

    # Setup Model
    logger.info("Building: {}".format(CONFIG.MODEL.NAME))
    model = eval(CONFIG.MODEL.NAME)(n_classes=CONFIG.DATASET.N_CLASSES, 
        base_filters=CONFIG.MODEL.BASE_FILTERS, channel_in=CONFIG.MODEL.CHANNEL_IN)
    
    if CONFIG.MODEL.INIT_MODEL is not None:

        # original saved file with DataParallel
        state_dict = torch.load(CONFIG.MODEL.INIT_MODEL)["model_state"]
        
        # create new OrderedDict that does not contain `module.`
        new_state_dict = dict_conversion(state_dict)

        # load parameters
        for m in model.state_dict().keys():
            if m not in new_state_dict.keys():
                print("    Skip init:", m)
        model.load_state_dict(new_state_dict, strict=False)
        print("Pre-trained weights loaded.")

    if CONFIG.PARALLEL:
        model = nn.DataParallel(model)
    model.to(device)
    print("Model is ready.")

    # Setup optimizer, lr_scheduler and loss function
    optimizer_cls = get_optimizer(CONFIG)
    optimizer_params = {k.lower(): v for k, v in CONFIG.TRAINING.OPTIM.items() if k != "name"}
    optim = optimizer_cls(model.parameters(), **optimizer_params)
    logger.info("Using optimizer{}".format(optim))

    scheduler = get_scheduler(optim, CONFIG.TRAINING.LR_SCHEDULER)
    logger.info("Using lr_scheculer {}".format(scheduler))

    loss_func = get_loss_function(CONFIG)
    logger.info("Using loss {}".format(loss_func))

    # setup metrics
    running_metrics = running_seg_score(CONFIG.DATASET.N_CLASSES)

    # meters
    val_loss_meter = averageMeter()
    time_meter = averageMeter()

    best_dsc = -100
    start_iter = -1
    flag = True

    i = start_iter
    while i <= CONFIG.TRAINING.ITER_MAX and flag:
        for _, volumes, labels, _ in train_loader:
            
            i += 1

            ###############################Training###############################
            
            start_time = time.time()
            model.train()

            volumes = volumes.cuda()
            labels = labels.cuda()

            optim.zero_grad()

            outputs = model(volumes)
            loss = loss_func(input=outputs, target=labels)

            loss.backward()
            optim.step()
            scheduler.step()

            time_meter.update(time.time()-start_time)

            if (i + 1) % CONFIG.TRAINING.PRINT_INTERVAL== 0:
                fmt_str = "Iter [{:d}/{:d}]  Loss: {:.4f}  Time/Image: {:.4f}"
                print_str = fmt_str.format(
                    i + 1,
                    CONFIG.TRAINING.ITER_MAX,
                    loss.item(),
                    time_meter.avg / CONFIG.TRAINING.BATCH_SIZE,
                )

                print(print_str)
                logger.info(print_str)
                writer.add_scalar("loss/train_loss", loss.item(), i + 1)
                time_meter.reset()
            ######################################################################

            #############################Validation###############################
            if (i + 1) % CONFIG.TRAINING.VAL_INTERVAL== 0:
                validation_step(CONFIG, writer, logger, optim, scheduler, i,
                    model, val_loader, loss_func, running_metrics,val_loss_meter, best_dsc,
                    device)
            ######################################################################

            ################################End###################################
            if (i + 1) == CONFIG.TRAINING.ITER_MAX:
                flag = False
                break
            ######################################################################


def validation_step(CONFIG, writer, logger, optim, scheduler, i,
    model, val_loader, loss_func, running_metrics, val_loss_meter, best_dsc,
    device):
    
    model.eval()
    with torch.no_grad():
        for _, imgs_val, labels_val, _ in tqdm(val_loader):

            imgs_val = imgs_val.cuda()
            labels_val = labels_val.cuda()
            
            outputs = model(imgs_val)

            val_loss = loss_func(input=outputs, target=labels_val)

            pred = outputs.data.max(1)[1].cpu().numpy()
            gt = labels_val.data.cpu().numpy()

            running_metrics.update(gt, pred)
            val_loss_meter.update(val_loss.item())

    writer.add_scalar("loss/val_loss", val_loss_meter.avg, i + 1)
    logger.info("Iter %d Loss: %.4f" % (i + 1, val_loss_meter.avg))

    # scoring
    score, class_dsc = running_metrics.get_scores()
    for k, v in score.items():
        print(k, v)
        logger.info("{}: {}".format(k, v))
        writer.add_scalar("val_metrics/{}".format(k), v, i + 1)
    
    for k, v in class_dsc.items():
        logger.info("{}: {}".format(k, v))
        writer.add_scalar("val_metrics/cls_{}".format(k), v, i + 1)
    
    print('-' * 40)

    # reset
    val_loss_meter.reset()
    running_metrics.reset()

    if score["Mean Dice Coefficient: \t"] >= best_dsc:
        best_dsc = score["Mean Dice Coefficient: \t"]
        state = {
            "epoch": i + 1,
            "model_state": model.state_dict(),
            "optimizer_state": optim.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_dsc": best_dsc
        }

        writer.add_scalar("best_model/dsc", best_dsc, i+1)

        save_path = os.path.join(
            writer.file_writer.get_logdir(),
            "{}_{}_best_model.pkl".format(CONFIG.MODEL.NAME, CONFIG.DATASET.NAME),
        )
        torch.save(state, save_path)