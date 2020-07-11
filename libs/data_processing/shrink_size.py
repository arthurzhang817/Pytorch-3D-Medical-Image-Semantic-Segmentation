import os
import torch
import glob
import numpy as np

def shrink_ct_size(vol):
    s, h, w = vol.shape # (64, 448, 448)
    
    new_h = int(h/7)
    new_w = int(w/7)
    print((s, h, w), "to", (s, new_h, new_w))
    new_vol = np.zeros((s, new_h, new_w), dtype=np.int8)
    for z in range(0, s):
        for r in range(0, new_h):
            for c in range(0, new_w):
                new_vol[z][r][c] = vol[z][2*r][2*c]
    assert new_vol.shape == (64, 64, 64)
    return new_vol


def shrink_label_size(vol):
    s, h, w = vol.shape # (64, 448, 448)
    new_h = int(h/7)
    new_w = int(w/7)
    print((s, h, w), "to", (s, new_h, new_w))
    new_vol = np.zeros((s, new_h, new_w), dtype=np.int8)
    for z in range(0, s):
        for r in range(0, new_h):
            for c in range(0, new_w):
                new_vol[z][r][c] = vol[z][7*r][7*c]
    assert new_vol.shape == (64, 64, 64)
    return new_vol

def shrink_onehot_size(vol):
    vol = torch.from_numpy(vol).permute(1, 2, 3, 0).data.numpy()
    s, h, w, n = vol.shape
    new_h = int(h/7)
    new_w = int(w/7)

    new_vol = np.zeros((s, new_h, new_w, n), dtype=np.int8)
    for z in range(0, s):
        for r in range(0, new_h):
            for c in range(0, new_w):
                new_vol[z][r][c] = vol[z][7*r][7*c]
    new_vol = torch.from_numpy(new_vol).permute(3, 0, 1, 2).data.numpy()
    assert new_vol.shape == (6, 64, 64, 64)
    return new_vol


if __name__ == "__main__":

    ct_vol_dir = "C:\\Users\\zhzhang\\Desktop\\data\\prostate_CT_vol_64_448_448\\CT"    
    label_vol_dir = "C:\\Users\\zhzhang\\Desktop\\data\\prostate_CT_vol_64_448_448\\SEG"
    onehot_vol_dir = "C:\\Users\\zhzhang\\Desktop\\data\\prostate_CT_vol_64_448_448\\SEG\\onehot_encoded"

    ct_vol_list = glob.glob(ct_vol_dir+"\\*.npy")
    label_vol_list = glob.glob(label_vol_dir+"\\*.npy")
    onehot_vol_list = glob.glob(onehot_vol_dir+"\\*.npy")

    for ct_vol in ct_vol_list:
        file_name = ct_vol.split("\\")[-1]
        print("Working on {}.".format(file_name))
        vol = np.load(ct_vol)
        vol = shrink_ct_size(vol)
        np.save(os.path.join("C:\\Users\\zhzhang\\Desktop\\data\\prostate_CT_vol_64_64_64\\CT", file_name), vol)
        print("{} ct volume done.".format(file_name))
        print("-"*30)


    for label_vol in label_vol_list:
        file_name = label_vol.split("\\")[-1]
        print("Working on {}.".format(file_name))
        vol = np.load(label_vol)
        vol = shrink_label_size(vol)
        np.save(os.path.join("C:\\Users\\zhzhang\\Desktop\\data\\prostate_CT_vol_64_64_64\\SEG", file_name), vol)
        print("{} label volume done.".format(file_name))
        print("-"*30)


    for onehot_vol in onehot_vol_list:
        file_name = onehot_vol.split("\\")[-1]
        print("Working on {}.".format(file_name))
        vol = np.load(onehot_vol)
        vol = shrink_onehot_size(vol)
        np.save(os.path.join("C:\\Users\\zhzhang\\Desktop\\data\\prostate_CT_vol_64_64_64\\SEG\\onehot_encoded", file_name), vol)
        print("{} onehot label volume done.".format(file_name))
        print("-"*30)