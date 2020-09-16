import pickle
import argparse
import roboverse
import os
import numpy as np
import os.path as osp

# TODO(avi): Clean this up
NFS_PATH = '/nfs/kun1/users/avi/imitation_datasets/'

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--data-save-path", type=str)
args = parser.parse_args()

all_files = []
for root, dirs, files in os.walk(args.data_save_path):
    for f in files:
        f_path = os.path.join(root, f)
        print(f_path)
        data = np.load(f_path, allow_pickle=True)
        all_files.append(data)

all_data = np.concatenate(all_files, axis=0)
if osp.exists(NFS_PATH):
    parent_dir = osp.join(NFS_PATH)
else:
    parent_dir = osp.join(__file__, "../..", "data")

save_all_path = os.path.join(parent_dir, args.data_save_path,
                             "{}_{}.npy".format(args.data_save_path, len(all_data)))
np.save(save_all_path, all_data)
