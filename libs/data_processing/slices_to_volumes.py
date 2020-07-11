import os
import glob
import torch
import numpy as np
from PIL import Image


def trim(imgs, masks):
    assert len(imgs) == len(masks)

    n = len(imgs)
    if n > 64:
        start_i = int((n-64)/2)-2

        output_imgs = []
        output_masks = []

        for i in range(0,64):
            output_imgs.append(imgs[start_i+i])
            output_masks.append(masks[start_i+i])
        
        return output_imgs, output_masks
    else:
        img_padding = np.zeros((imgs[0].shape), dtype=np.int8)
        mask_padding = np.zeros((masks[0].shape), dtype=np.int8)
        for _ in range(0, 64-n):
            imgs.append(img_padding)
            masks.append(mask_padding)
        
        return imgs, masks

    
    


def list_to_patient_volumes(img_list, color_mask_list):

    '''
    input: img_list, color_mask_list under a dir
    output: [[patient0_img_volume, patient0_label_volume], ...]
    '''

    patients = []

    patient_id = int(img_list[0].split('\\')[-1].split('_')[1])
    patient_imgs = []
    patient_masks = []

    for img_id in range(0, len(img_list)):
        img_path = img_list[img_id]

        img_array = np.asarray(Image.open(img_path).convert('L').crop((32, 32, 480, 480)))
        color_array = np.asarray(Image.open(color_mask_list[img_id]).convert('RGB').crop((32, 32, 480, 480)))

        cur_id = int(img_path.split('\\')[-1].split('_')[1])
        
        if img_id != len(img_list)-1:
            if cur_id == patient_id:
                patient_imgs.append(img_array)
                patient_masks.append(color_array)
            else:
                patient_imgs, patient_masks = trim(patient_imgs, patient_masks)

                imgs = []
                labels = []

                for slice_id in range(0, 64):
                    img = patient_imgs[slice_id]
                    mask = patient_masks[slice_id]
                    label = color_to_label(mask, [[229, 255, 204],[0, 255, 255],[204, 0, 102],[255, 0, 0],[0, 255, 0]])

                    imgs.append(img)
                    labels.append(label)
                    print("Slice {} done for patient {}".format(slice_id, patient_id))

                patients.append([imgs, labels])
                print("Patient {} done.".format(patient_id))
                np.save("C:\\Users\\zhzhang\\Desktop\\data\\prostate_CT_vol_64_448_448\\CT\\patient_{}_volume.npy".format(patient_id),imgs)
                np.save("C:\\Users\\zhzhang\\Desktop\\data\\prostate_CT_vol_64_448_448\\seg\\seg_patient_{}_volume.npy".format(patient_id),labels)
                print('-'*25)

                patient_id = cur_id
                patient_imgs = [img_array]
                patient_masks = [color_array]
        else:
            patient_imgs.append(img_array)
            patient_masks.append(color_array)

            patient_imgs, patient_masks = trim(patient_imgs, patient_masks)

            imgs = []
            labels = []

            for slice_id in range(0, 64):
                img = patient_imgs[slice_id]
                mask = patient_masks[slice_id]
                label = color_to_label(mask, [[229, 255, 204],[0, 255, 255],[204, 0, 102],[255, 0, 0],[0, 255, 0]])

                imgs.append(img)
                labels.append(label)
                print("Slice {} done for patient {}".format(slice_id, patient_id))

            patients.append([imgs, labels])
            print("Patient {} done.".format(patient_id))
            np.save("C:\\Users\\zhzhang\\Desktop\\data\\prostate_CT_vol_64_448_448\\CT\\patient_{}_volume.npy".format(patient_id),imgs)
            np.save("C:\\Users\\zhzhang\\Desktop\\data\\prostate_CT_vol_64_448_448\\seg\\seg_patient_{}_volume.npy".format(patient_id),labels)
            print('-'*25)


    return patients


def color_to_label(color_mask, color_map):

    '''
    input: color_mask in rgb, color_map
    output: label_mask with class_id
    '''
    
    h, w, c = color_mask.shape
    output = np.zeros((h,w), dtype=np.int8)

    for r in range(0,h):
        for c in range(0,w):
            for i in range(0, len(color_map)):
                if (color_mask[r][c] == color_map[i]).all():
                    output[r][c] = i+1
                    break

    return output

def pre_encode_label_vol(label_vol):

    '''
    input: label_vol with class_id
    output: one-hot encoded label vol
    '''
    z, h, w = label_vol.shape
    onehot_vol = np.zeros((z, h, w, 6), dtype=np.int8)

    for s in range(0,z):
        for r in range(0, h):
            for c in range(0, w):
                onehot = np.zeros(6, dtype=np.int8)
                onehot[label_vol[s][r][c]] = 1
                onehot_vol[s][r][c] = onehot
    onehot_vol = torch.from_numpy(onehot_vol).permute(3, 0, 1, 2).data.numpy()
    assert onehot_vol.shape == (6, 64, 448, 448)

    return onehot_vol


if __name__ == "__main__":
    

    # # Slices to Volumes
    # img_dir = os.path.expanduser("C:\\Users\\zhzhang\\Desktop\\data\\prostate_CT_2d\\orig_CT\\*.png")
    # color_mask_dir = os.path.expanduser("C:\\Users\\zhzhang\\Desktop\\data\\prostate_CT_2d\\seg_mask\\*.png")

    # img_list = glob.glob(img_dir)
    # color_mask_list = glob.glob(color_mask_dir)
    # assert len(img_list) == len(color_mask_list)

    # patients = list_to_patient_volumes(img_list, color_mask_list)

    # Test Block
    test_vol = np.load("C:\\Users\\zhzhang\\Desktop\\data\\prostate_CT_vol_64_224_224\\CT\\patient_1_volume.npy")
    test_seg_vol = np.load("C:\\Users\\zhzhang\\Desktop\\data\\prostate_CT_vol_64_224_224\\SEG\\seg_patient_1_volume.npy")

    print(test_vol.shape, test_seg_vol.shape)
    

    # # Onehot-encode Mask Volumes
    # label_vol_dir = "C:\\Users\\zhzhang\\Desktop\\data\\prostate_CT_vol_64_448_448\\seg\\*.npy"
    # onehot_vol_dir = "C:\\Users\\zhzhang\\Desktop\\data\\prostate_CT_vol_64_448_448\\seg\\onehot_encoded"
    # label_vol_list = glob.glob(label_vol_dir)

    # for label_vol in label_vol_list:
    #     patient_id = label_vol.split("\\")[-1].split('_')[2]
    #     vol = np.load(label_vol)
    #     onehot_vol = pre_encode_label_vol(vol)
    #     np.save(os.path.join(onehot_vol_dir, "encoded_seg_patient_{}_vol.npy".format(patient_id)), onehot_vol)
    #     print("Patient {} done.".format(patient_id))
    