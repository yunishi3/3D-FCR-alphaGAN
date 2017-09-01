import numpy as np
import os
import random

from config import cfg

class DataProcess():
    def __init__(self, data_paths, batch_size, repeat=True):
        self.data_paths = data_paths
        self.num_data = len(data_paths)
        self.repeat = repeat

        self.batch_size = batch_size
        self.shuffle_db_inds()
        self.n_vox = cfg.CONST.N_VOX

    def shuffle_db_inds(self):
        # Randomly permute the training roidb
        if self.repeat:
            self.perm = np.random.permutation(np.arange(self.num_data))
        else:
            self.perm = np.arange(self.num_data)
        self.cur = 0

    def get_next_minibatch(self):
        flag = True
        if (self.cur + self.batch_size) >= self.num_data and self.repeat:
            self.shuffle_db_inds()
            flag=False

        db_inds = self.perm[self.cur:min(self.cur + self.batch_size, self.num_data)]
        self.cur += self.batch_size
        return db_inds, flag

    def get_voxel(self, db_inds):
        batch_voxel = np.zeros(
                    (self.batch_size, self.n_vox[0], self.n_vox[1], self.n_vox[2]), dtype=np.float32)
    
        for batch_id, db_ind in enumerate(db_inds):
            sceneId, model_id = self.data_paths[db_ind]

            voxel_fn = cfg.DIR.VOXEL_PATH % (sceneId, model_id)
            voxel_data = np.load(voxel_fn)

            batch_voxel[batch_id, :, :, :] = voxel_data
        return batch_voxel


def scene_model_id_pair(dataset_portion=[]):
    '''
    Load sceneId, model names from a suncg dataset.
    '''

    scene_name_pair = []  # full path of the objs files
    sceneIds = os.listdir(cfg.DIR.SCENE_ID_PATH)
    
    for k, sceneId in enumerate(sceneIds):  # load by sceneIds
        model_path = os.path.join(cfg.DIR.SCENE_ID_PATH, sceneId)
        models = os.listdir(model_path)

        scene_name_pair.extend([(sceneId, model_id) for model_id in models])

    num_models = len(scene_name_pair)
    portioned_scene_name_pair = scene_name_pair[int(num_models * dataset_portion[0]):int(num_models * dataset_portion[1])]

    return portioned_scene_name_pair

def scene_model_id_pair_test(dataset_portion=[]):

    amount_of_test_sample = 200

    scene_name_pair = []  # full path of the objs files
    sceneIds = os.listdir(cfg.DIR.SCENE_ID_PATH)

    for k, sceneId in enumerate(sceneIds):  # load by sceneIds
        model_path = os.path.join(cfg.DIR.SCENE_ID_PATH, sceneId)
        models = os.listdir(model_path)

        scene_name_pair.extend([(sceneId, model_id) for model_id in models])

    num_models = len(scene_name_pair)
    data_paths_test = scene_name_pair[int(num_models * dataset_portion[1])+1:]
    random.shuffle(data_paths_test)
    #data_paths = scene_name_pair[int(num_models * dataset_portion[1])+1:int(num_models * dataset_portion[1])+amount_of_test_sample+1]
    data_paths = data_paths_test[:amount_of_test_sample]

    num_models = len(data_paths)
    print '---amount of test data:' + str(num_models)

    n_vox = cfg.CONST.N_VOX

    batch_voxel = np.zeros(
                    (num_models, n_vox[0], n_vox[1], n_vox[2]), dtype=np.float32)

    for i in np.arange(num_models):
        sceneId, model_id = data_paths[i]

        voxel_fn = cfg.DIR.VOXEL_PATH % (sceneId, model_id)
        voxel_data = np.load(voxel_fn)

        batch_voxel[i, :, :, :] = voxel_data

    return batch_voxel, num_models


def onehot(voxel, class_num):
    onehot_voxels = np.zeros((voxel.shape[0], voxel.shape[1], voxel.shape[2], voxel.shape[3], class_num))
    for i in np.arange(class_num):
        onehot_voxel = np.zeros(voxel.shape)
        onehot_voxel[np.where(voxel == i)] = 1
        onehot_voxels[:,:,:,:,i]=onehot_voxel[:,:,:,:]
    return onehot_voxels

