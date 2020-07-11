import os
import numpy as np

import torch
from torch.utils import data as torch_data

class Prostate_CT_Volume_Loader(torch_data.Dataset):
    
    def __init__(
        self,
        root_dir,
        split,
        normalize=True
    ):

        self.normalize = normalize
        self.n_classes = 6

        self.split_dir = os.path.join(root_dir, split)
        self.split_dir_ct = os.path.join(self.split_dir, "CT")
        self.split_dir_seg = os.path.join(self.split_dir, "SEG")
        self.split_dir_onehot = os.path.join(self.split_dir, "SEG_onehot")

        self.volume_id_list = [x for x in os.listdir(self.split_dir_ct)]
        self.volume_list = [os.path.join(self.split_dir_ct, x) for x in os.listdir(self.split_dir_ct)]
        self.label_list = [os.path.join(self.split_dir_seg, x) for x in os.listdir(self.split_dir_seg)]
        self.onehot_list = [os.path.join(self.split_dir_onehot, x) for x in os.listdir(self.split_dir_onehot)]

    def __len__(self):
        return len(self.volume_id_list)

    def __getitem__(self, index):

        patient_id = self.volume_id_list[index]
        ct_vol_path = self.volume_list[index]
        label_path = self.label_list[index]
        onehot_path = self.onehot_list[index]

        ct_vol = np.load(ct_vol_path)
        norm = np.linalg.norm(ct_vol)
        normalized_ct_vol = ct_vol/norm
        normalized_ct_vol = np.expand_dims(normalized_ct_vol, axis=0)
        ct = torch.from_numpy(normalized_ct_vol).float()

        label = np.load(label_path)
        label = torch.from_numpy(label).long()

        onehot = np.load(onehot_path)
        onehot = torch.from_numpy(onehot).long()

        return patient_id, ct, label, onehot


