# -*- coding: utf-8 -*-
import numpy as np
import theano
import argparse
import json
import os
from utils.dataprovider import DataProvider
from core.cnnClassifier import CnnClassifier

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
  print('Accuracy of %s the test set is %0.2f'%(params['saved_model'], accuracy))

  ## Now let's build a gradient computation graph and rmsprop update mechanism
  ##grads = tensor.grad(cost, wrt=model.values())
  ##lr = tensor.scalar(name='lr',dtype=config.floatX)

  #num_frames_total = dp.getSplitSize('train')
  #num_iters_one_epoch = num_frames_total/ batch_size
  #max_iters = max_epochs * num_iters_one_epoch
  ##
  #for it in xrange(max_iters):
  #  batch = dp.getBatch(batch_size)
  #  cost = f_train(*batch)
    
    #cost = f_grad_shared(inp_list)
    #f_update(params['learning_rate'])

    #Save model periodically
  return modelObj


if __name__ == "__main__":
  
  parser = argparse.ArgumentParser()

  # IO specs
  parser.add_argument('-t','--testdata', dest='testdata', type=str, default='dummy', help='Target class for the test set')
  parser.add_argument('-tL','--testlbl', dest='testlbl', type=str, default='dummy', help='Target class for the test set')
  parser.add_argument('-m','--saved_model', dest='saved_model', type=str, default='', help='input the saved model json file to evluate on')

  # Learning related parmeters
  args = parser.parse_args()
  params = vars(args) # convert to ordinary dict

  model = main(params)
