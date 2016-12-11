# -*- coding: utf-8 -*-
import numpy as np
import theano
import argparse
import json
import os
import os.path as osp
from utils.dataprovider import DataProvider
from core.cnnClassifier import CnnClassifier

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(cm,class_names, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def getModelObj(params):
  if params['model_type'] == 'MLP':
    raise ValueError('MLPs is not yet supported')
  elif params['model_type'] == 'CNN':
    mdl = CnnClassifier(params)
  elif params['model_type'] == 'RNN':  
    raise ValueError('RNNs is not yet supported') 
  else:
    raise ValueError('ERROR: %s --> This model type is not yet supported'%(params['model_type']))
  return mdl

def main(params):
  
  # main training and validation loop goes here
  # This code should be independent of which model we use
  
  cv = json.load(open(params['saved_model'],'r'))
  cv_params = cv['params']
  print 'Evaluating using model %s'%(params['saved_model'])
  cv_params['testdata'] = params['testdata']
  cv_params['testlbl'] = params['testlbl']
  cv_params.pop('traindata'); cv_params.pop('valdata')

  # fetch the data provider object
  dp = DataProvider(cv_params)
  ## Add the model intiailization code here

  
  modelObj = getModelObj(cv_params)

  # Build the model Architecture
  f_train = modelObj.build_model(cv_params)
  #Load the saved weights
  modelObj.model.load_weights(cv['weights_file'])
  
  test_x, test_y = dp.get_data_array(cv_params['model_type'], ['test'])
  
  predOut = modelObj.model.predict_classes(test_x, batch_size=100)
  accuracy =  100.0*np.sum(predOut == test_y.nonzero()[1]) / predOut.shape[0]
  print('Accuracy of %s the test set is %0.2f \n Saving now...'%(params['saved_model'], accuracy))
  
  np.save(osp.join(params['out_dir'],'predict_out_'+cv_params['out_file_append']+'.npy'), predOut)

  # plotting confusion matrix
  if params['plot_confmat'] != 0:
      cm = confusion_matrix(test_y.nonzero()[1], predOut) 
      cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
      plt.figure()
      plot_confusion_matrix(cm, np.arange(cv_params['num_classes']))
      plt.show()

  return modelObj


if __name__ == "__main__":
  
  parser = argparse.ArgumentParser()

  # IO specs
  parser.add_argument('-t','--testdata', dest='testdata', type=str, default='dummy', help='Target class for the test set')
  parser.add_argument('-tL','--testlbl', dest='testlbl', type=str, default='dummy', help='Target class for the test set')
  parser.add_argument('-m','--saved_model', dest='saved_model', type=str, default='', help='input the saved model json file to evluate on')
  parser.add_argument('--plot_confmat', dest='plot_confmat', type=int, default=0, help='Should we plot the confusion matrix')
  parser.add_argument('--out_dir', dest='out_dir', type=str, default='preds/', help='String to append to the filename of the trained model')

  # Learning related parmeters
  args = parser.parse_args()
  params = vars(args) # convert to ordinary dict

  model = main(params)
