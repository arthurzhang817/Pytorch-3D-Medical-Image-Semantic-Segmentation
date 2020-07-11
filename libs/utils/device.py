import torch
from collections import OrderedDict


def get_device(cuda):
    cuda = cuda and torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    if cuda:
        print("Device:")
        for i in range(torch.cuda.device_count()):
            print("    {}:".format(i), torch.cuda.get_device_name(i))
    else:
        print("Device: CPU")
    return device

def memory_usage_report(device, logger=None):
    max_memory_allocated = float(torch.cuda.max_memory_allocated(device=device))/(10**9)
    max_memory_cached = float(torch.cuda.max_memory_cached(device=device))/(10**9)
    print("Max memory allocated on {}: ".format(device) + str(max_memory_allocated) + "GB.")
    print("Max memory cached on {}: ".format(device)+ str(max_memory_cached) + "GB.")

    if logger is not None:
        logger.info("Max memory allocated on {}:".format(device) + str(max_memory_allocated) + "GB.")
        logger.info("Max memory cached on {}: ".format(device)+ str(max_memory_cached) + "GB.")


def dict_conversion(d):
    new_state_dict = OrderedDict()
    for k, v in d.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    return new_state_dict