import os
import sys
import numpy as np
import h5py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

# Download dataset for point cloud classification
DATA_DIR = os.path.join(BASE_DIR, 'data')
if not os.path.exists(DATA_DIR):
    os.mkdir(DATA_DIR)
if not os.path.exists(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048')):
    www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
    zipfile = os.path.basename(www)
    os.system('wget %s; unzip %s' % (www, zipfile))
    os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
    os.system('rm %s' % (zipfile))

def add_moments(data_xyz):
    data = np.zeros(shape=(np.shape(data_xyz)[0], np.shape(data_xyz)[1], 9))
    data[:, :, 0:3] = data_xyz
    
                    
                    
    data[:, :, 6] = data_xyz[:, :, 0] * data_xyz[:, :, 1]
    #data[:, :, 3] = preprocessing.normalize(data[:, :, 3], axis=1, norm='l1')
    #data[:, :, 3] = data[:, :, 3] / np.sum(data[:, :, 3], axis=1)
    data[:, :, 7] = data_xyz[:, :, 0] * data_xyz[:, :, 2]
    #data[:, :, 4] = preprocessing.normalize(data[:, :, 4], axis=1, norm='l1')
    data[:, :, 8] = data_xyz[:, :, 1] * data_xyz[:, :, 2]
    #data[:, :, 5] = preprocessing.normalize(data[:, :, 5], axis=1, norm='l1')
    data[:, :, 3] = data_xyz[:, :, 0] * data_xyz[:, :, 0]
    #data[:, :, 6] = preprocessing.normalize(data[:, :, 6], axis=1, norm='l1')
    data[:, :, 4] = data_xyz[:, :, 1] * data_xyz[:, :, 1]
    #data[:, :, 7] = preprocessing.normalize(data[:, :, 7], axis=1, norm='l1')
    data[:, :, 5] = data_xyz[:, :, 2] * data_xyz[:, :, 2]
    #data[:, :, 8] = preprocessing.normalize(data[:, :, 8], axis=1, norm='l1')
    
    '''
    data[:, :, 9] = data_xyz[:, :, 0] / data_xyz[:, :, 2]
    data[:, :, 10] = data_xyz[:, :, 2] / data_xyz[:, :, 0]
    data[:, :, 11] = data_xyz[:, :, 1] / data_xyz[:, :, 0]
    data[:, :, 12] = data_xyz[:, :, 0] / data_xyz[:, :, 1]
    data[:, :, 13] = data_xyz[:, :, 1] / data_xyz[:, :, 2]
    data[:, :, 14] = data_xyz[:, :, 2] / data_xyz[:, :, 1]
    data[:, :, 15] = data[:, :, 3] / data[:, :, 6]
    data[:, :, 16] = data[:, :, 3] / data[:, :, 8]
    data[:, :, 17] = data[:, :, 6] / data[:, :, 8]
    
    data[:, :, 9] = data_xyz[:, :, 0] * data_xyz[:, :, 0]*data_xyz[:, :, 0]
    data[:, :, 9] = normalize(data[:, :, 9], axis=1, norm='l1')
    data[:, :, 10] = data_xyz[:, :, 1] * data_xyz[:, :, 1]*data_xyz[:, :, 1]
    data[:, :, 10] = normalize(data[:, :, 10], axis=1, norm='l1')
    data[:, :, 11] = data_xyz[:, :, 2] * data_xyz[:, :, 2]*data_xyz[:, :, 2]
    data[:, :, 11] = normalize(data[:, :, 11], axis=1, norm='l1')
    data[:, :, 12] = data_xyz[:, :, 0] * data_xyz[:, :, 0] * data_xyz[:, :, 0]* data_xyz[:, :, 0]
    data[:, :, 12] = normalize(data[:, :, 12], axis=1, norm='l1')
    data[:, :, 13] = data_xyz[:, :, 1] * data_xyz[:, :, 1] * data_xyz[:, :, 1]* data_xyz[:, :, 1]
    data[:, :, 13] = normalize(data[:, :, 13], axis=1, norm='l1')
    data[:, :, 14] = data_xyz[:, :, 2] * data_xyz[:, :, 2] * data_xyz[:, :, 2]* data_xyz[:, :, 2]
    data[:, :, 14] = normalize(data[:, :, 14], axis=1, norm='l1')
    '''
    '''
    data[:, :, 9] = np.sqrt(data_xyz[:, :, 0] * data_xyz[:, :, 0]+data_xyz[:, :, 1] * data_xyz[:, :, 1]+data_xyz[:, :, 2] * data_xyz[:, :, 2])
    data[:, :, 10] = np.arccos(data_xyz[:, :, 2]/data[:, :, 9])
    data[:, :, 11] = np.arctan(data_xyz[:, :, 1]/data_xyz[:, :, 0])
    '''
    return data
def shuffle_data(data, labels):
    """ Shuffle data and labels.
        Input:
          data: B,N,... numpy array
          label: B,... numpy array
        Return:
          shuffled data, label and shuffle indices
    """
    idx = np.arange(len(labels))
    np.random.shuffle(idx)
    return data[idx, ...], labels[idx], idx


def rotate_point_cloud(batch_data):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    batch_data = batch_data[:,:,0:3]
    rotated_data_x = np.zeros(batch_data.shape, dtype=np.float32)
    rotated_data_y = np.zeros(batch_data.shape, dtype=np.float32)
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        rotation_angle_x = np.random.uniform() * 2 * np.pi
        cosval_x = np.cos(rotation_angle_x)
        sinval_x = np.sin(rotation_angle_x)

        #rotation_angle_y = np.random.uniform() * 2 * np.pi
        #cosval_y = np.cos(rotation_angle_y)
        #sinval_y = np.sin(rotation_angle_y)
        
        #rotation_angle_z = np.random.uniform() * 2 * np.pi
        #cosval_z = np.cos(rotation_angle_z)
        #sinval_z = np.sin(rotation_angle_z)
        
        rotation_matrix_x = np.array([[1, 0, 0],
                                    [0, cosval_x, -sinval_x],
                                    [0, sinval_x, cosval_x]])
        #rotation_matrix_y = np.array([[cosval_y, 0, sinval_y],
        #                                [0, 1, 0],
        #                                [-sinval_y, 0, cosval_y]])
        #rotation_matrix_z = np.array([[cosval_z, -sinval_z, 0],
        #                                [sinval_z, cosval_z, 0],
        #                                [0, 0, 1]])
        #rot_xy = np.dot(rotation_matrix_y,rotation_matrix_x)
        #rotation_matrix = np.dot(rot_xy,rotation_matrix_z)
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix_x)
    rotated_data_=np.zeros(shape=(np.shape(rotated_data)[0],np.shape(rotated_data)[1],9))
    rotated_data_ = add_moments(rotated_data)
    return rotated_data_
    #return rotated_data


def rotate_perturbation_point_cloud(batch_data, angle_sigma=0.06, angle_clip=0.18):
  """ Randomly perturb the point clouds by small rotations
    Input:
      BxNx3 array, original batch of point clouds
    Return:
      BxNx3 array, rotated batch of point clouds
  """
  rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
  for k in range(batch_data.shape[0]):
    angles = np.clip(angle_sigma*np.random.randn(3), -angle_clip, angle_clip)
    Rx = np.array([[1,0,0],
             [0,np.cos(angles[0]),-np.sin(angles[0])],
             [0,np.sin(angles[0]),np.cos(angles[0])]])
    Ry = np.array([[np.cos(angles[1]),0,np.sin(angles[1])],
             [0,1,0],
             [-np.sin(angles[1]),0,np.cos(angles[1])]])
    Rz = np.array([[np.cos(angles[2]),-np.sin(angles[2]),0],
             [np.sin(angles[2]),np.cos(angles[2]),0],
             [0,0,1]])
    R = np.dot(Rz, np.dot(Ry,Rx))
    shape_pc = batch_data[k, ...]
    rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), R)
  return rotated_data

  
def rotate_point_cloud_by_angle(batch_data, rotation_angle):
    """ Rotate the point cloud along up direction with certain angle.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    batch_data = batch_data[:,:,0:3]
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        #rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    rotated_data_=np.zeros(shape=(np.shape(rotated_data)[0],np.shape(rotated_data)[1],9))
    rotated_data_ = add_moments(rotated_data)
    return rotated_data_
    #return rotated_data


def jitter_point_cloud(batch_data, sigma=0.01, clip=0.05):
    """ Randomly jitter points. jittering is per point.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, jittered batch of point clouds
    """
    jittered_data = np.zeros(shape=(np.shape(batch_data)[0],np.shape(batch_data)[1],9))
    batch_data_xyz = batch_data[:,:,0:3]
    batch_data_moments = batch_data[:,:,3:]
    B, N, C = batch_data_xyz.shape
    '''B, N, C = batch_data.shape'''
    assert(clip > 0)
    jittered_data[:,:,0:3] = np.clip(sigma * np.random.randn(B, N, C), -1*clip, clip)
    jittered_data[:,:,0:3] += batch_data_xyz
    '''jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1*clip, clip)
    jittered_data += batch_data_xyz'''
    jittered_data[:, :, 3:] = np.clip(sigma**2 * np.random.randn(B, N, 6), -1 * clip, clip)
    jittered_data[:, :, 3:] += batch_data_moments
    return jittered_data
def shift_point_cloud(batch_data, shift_range=0.1):
  """ Randomly shift point cloud. Shift is per point cloud.
    Input:
      BxNx3 array, original batch of point clouds
    Return:
      BxNx3 array, shifted batch of point clouds
  """
  B, N, C = batch_data.shape
  shifts = np.random.uniform(-shift_range, shift_range, (B,3))
  for batch_index in range(B):
    batch_data[batch_index,:,:] += shifts[batch_index,:]
  return batch_data


def random_scale_point_cloud(batch_data, scale_low=0.8, scale_high=1.25):
  """ Randomly scale the point cloud. Scale is per point cloud.
    Input:
      BxNx3 array, original batch of point clouds
    Return:
      BxNx3 array, scaled batch of point clouds
  """
  B, N, C = batch_data.shape
  scales = np.random.uniform(scale_low, scale_high, B)
  for batch_index in range(B):
    batch_data[batch_index,:,:] *= scales[batch_index]
  return batch_data

def getDataFiles(list_filename):
    return [line.rstrip() for line in open(list_filename)]

def load_h5(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    data =data[:,:,0:3]
    data_xyz = data
    #data_xyz = pca(data_xyz)
    data=add_moments(data_xyz)
    #data=data_xyz
    #data_sum =  np.sum(data, axis=1)
    #data_sum =np.expand_dims(data_sum, axis=1)
    #data = np.concatenate((data_sum,data), axis=1)
    label = f['label'][:]
    return (data, label)

def loadDataFile(filename):
    return load_h5(filename)

def load_h5_data_label_seg(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    seg = f['pid'][:]
    return (data, label, seg)


def loadDataFile_with_seg(filename):
    return load_h5_data_label_seg(filename)
