import cPickle as pickle
import json
import numpy as np
import os
import os.path as osp
import h5py
import scipy
import scipy.io

def read_input_file(fname, dtype=np.float32):
  if fname.split('.')[-1] == 'csv':
      datalines = open(fname).read().splitlines()
      return np.array([map(int,dl.split(',')) for dl in datalines], dtype=dtype)
  elif fname.split('.')[-1] == 'mat':
      data_struct = scipy.io.loadmat(fname)
      keys = [ky for ky in data_struct.keys() if ky[0] != '_']
      return np.array(data_struct[keys[0]]).astype(dtype)

class DataProvider:
  def __init__(self, params):
    # Write the initilization code to load the preprocessed data and labels
    self.data = {}
    for splt in ['train','val', 'test']:
      if splt+'data' in params:
        self.data[splt] = {}
        self.data[splt]['feat'] = read_input_file(osp.join('data',params[splt+'data']))
        # the reshape is to account for i and q channels 
        n_samp = self.data[splt]['feat'].shape[0]
        self.data[splt]['feat'] = self.data[splt]['feat'].reshape([n_samp,-1,2])

        # Read the labels and convert to one-hot
        self.data[splt]['lab'] = read_input_file(osp.join('data', params[splt+'lbl']), dtype = np.int32)
        tempZ = np.zeros([n_samp, params['num_classes']],dtype=np.int8)
        tempZ[np.arange(n_samp),self.data[splt]['lab'].flatten()] = 1
        self.data[splt]['lab'] = tempZ
    self.feat_size = self.data[self.data.keys()[0]]['feat'].shape[1:]
  
  def get_data_array(self, model, splits, cntxt=-1, shufdata=1, idx = -1):
    output = []

    for spt in splits:
        feats = self.data[spt]['feat'] if idx == -1 else [self.data[spt]['feat'][idx]]
        labs = self.data[spt]['lab'] if idx == -1 else [self.data[spt]['lab'][idx]]
        shfidx = np.random.permutation(feats.shape[0]) if shufdata == 1 else np.arange(feats.shape[0])
        feats = feats[shfidx,...]
        labs = labs[shfidx,:]
        output.extend([feats,labs])
    return output

  def getSplitSize(self, split='train'):
    return self.data[split]['feat'].shape[0] 
  
  def getSplitSize(self, split='train'):
    return self.data[split]['feat'].shape[0] 
