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
  batch_size = params['batch_size']
  max_epochs = params['max_epochs']
  
  
  # fetch the data provider object
  dp = DataProvider(params)
  params['feat_size'] = dp.feat_size
  ## Add the model intiailization code here
  
  modelObj = getModelObj(params)

  # Build the model Architecture
  f_train = modelObj.build_model(params)
  
  if params['saved_model'] !=None: 
    cv = json.load(open(params['saved_model'],'r'))
    modelObj.model.load_weights(cv['weights_file'])
    print 'Conitnuing training from model %s'%(params['saved_model'])
  
  train_x, train_y, val_x, val_y = dp.get_data_array(params['model_type'], ['train', 'val'])
  fname, best_val_loss = modelObj.train_model(train_x, train_y, val_x, val_y, params)

  checkpoint = {}
    
  checkpoint['params'] = params
  checkpoint['weights_file'] = fname.format(val_loss=best_val_loss)
  filename = 'model_%s_%s_%s_%.2f.json' % (params['dataset'], params['model_type'], params['out_file_append'], best_val_loss)
  filename = os.path.join(params['out_dir'],filename)
  print 'Saving to File %s'%(filename)
  json.dump(checkpoint, open(filename,'w'))

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
  parser.add_argument('-d','--traindata', dest='traindata', type=str, default='dummy', help='Input features for the training set')
  parser.add_argument('-dL','--trainlbl', dest='trainlbl', type=str, default='dummy', help='Target class for the training set')
  parser.add_argument('-v','--valdata', dest='valdata', type=str, default='dummy', help='Target class for the validation set')
  parser.add_argument('-vL','--vallbl', dest='vallbl', type=str, default='dummy', help='Target class for the validation set')
  parser.add_argument('-t','--testdata', dest='testdata', type=str, default='dummy', help='Target class for the test set')
  parser.add_argument('-tL','--testlbl', dest='testlbl', type=str, default='dummy', help='Target class for the test set')

  parser.add_argument('--num_classes', dest='num_classes', type=int, default=20, help='number of classes in the output')
  parser.add_argument('--output_file_append', dest='out_file_append', type=str, default='dummyModel', help='String to append to the filename of the trained model')

  parser.add_argument('--out_dir', dest='out_dir', type=str, default='cv/', help='String to append to the filename of the trained model')
  
  # Learning related parmeters
  parser.add_argument('-m', '--max_epochs', dest='max_epochs', type=int, default=20, help='number of epochs to train for')
  parser.add_argument('-l', '--learning_rate', dest='lr', type=float, default=1e-1, help='solver learning rate')
  parser.add_argument('-b', '--batch_size', dest='batch_size', type=int, default=100, help='batch size')
  
  # Solver related parameters
  parser.add_argument('--solver', dest='solver', type=str, default='adam', help='solver types supported: rmsprop')
  
  # Model architecture related parameters
  parser.add_argument('--model_type', dest='model_type', type=str, default='CNN', help='Can take values MLP, RNN or CNN')
  parser.add_argument('--drop_prob', dest='drop_prob', type=float, default=0.0, help='what dropout to apply right after the encoder to an RNN/LSTM')
  parser.add_argument('--num_layers', dest='num_layers', nargs='+',type=int, default=[64, 32, 16], help='the number of filters in different layers of the CNN')
  parser.add_argument('--kernel_size', dest='kernel_size', nargs='+',type=int, default=5, help='the kernel size in the CNN')
  parser.add_argument('--max_pool_sz', dest='max_pool_sz',type=int, default=3, help='the kernel size in the CNN')
  parser.add_argument('--max_pool_stride', dest='max_pool_stride',type=int, default=3, help='the kernel size in the CNN')
  
  # RNN Model architecture related parameters
  parser.add_argument('--continue_training', dest='saved_model', type=str, default=None, help='input the saved model json file to evluate on')

  args = parser.parse_args()
  params = vars(args) # convert to ordinary dict

  model = main(params)
