import numpy as np
import theano
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, SpatialDropout1D, Flatten
from keras.layers.convolutional import Convolution1D
from keras.layers.pooling import MaxPooling1D
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import SGD, RMSprop, Adam
from os import path

def getSolver(params):
    if params['solver'] == 'sgd':
      return SGD(lr=params['lr'], decay=1-params['decay_rate'], momentum=0.9, nesterov=True)
    elif params['solver'] == 'rmsprop':  
      return RMSprop(lr=params['lr'])
    elif params['solver'] == 'adam':  
      return Adam(lr=params['lr'])
    raise ValueError('ERROR in RNN: %s --> This solver type is not yet supported '%(params['solver']))
      
class CnnClassifier:
  def __init__(self, params):
    self.model = Sequential()
    print('----------Using CNN model with the below configuration----------') 
    print('nLayers:%d'%(len(params['num_layers'])))
    print('Layer sizes: [%s]'%(' '.join(map(str,params['num_layers']))))
    print('Dropout Prob: %.2f '%(params['drop_prob']))

  def build_model(self, params):
    num_layers = params['num_layers']
    input_dim = params['feat_size']
    output_dim = params['num_classes']
    drop_prob = params['drop_prob']
    k_sz = params['kernel_size']
    self.nLayers = len(num_layers)

    # First Layer takes the input directly.
    self.model.add(Convolution1D(num_layers[0], k_sz, activation='relu', border_mode='same', 
        input_shape=input_dim))
    self.model.add(MaxPooling1D(params['max_pool_sz'], stride = params['max_pool_stride'])) 
    #Further layers don't need input specification 
    for i in xrange(len(num_layers)-1):
        self.model.add(Convolution1D(num_layers[i+1], k_sz, activation='relu', border_mode='same')) 
        # Then we add dense projection layer to map the RNN outputs to Vocab size 
        self.model.add(MaxPooling1D(params['max_pool_sz'], stride = params['max_pool_stride'])) 
        self.model.add(SpatialDropout1D(drop_prob))

    self.model.add(Flatten())
    self.model.add(Dense(output_dim, init='uniform', activation='softmax'))
  
    self.solver = getSolver(params)
    self.model.compile(loss='categorical_crossentropy', optimizer=self.solver)
    self.f_train = self.model.train_on_batch
    print('\nCNN model cofiguration done:')
    self.model.summary()

    return self.f_train

  def train_model(self, train_x, train_y, val_x, val_y,params):
    epoch= params['max_epochs']
    batch_size=params['batch_size']
    out_dir=params['out_dir']
    fname = path.join(out_dir, 'CNN_weights_'+params['out_file_append'] +'_{val_loss:.2f}.hdf5')
    checkpointer = ModelCheckpoint(filepath=fname, verbose=1, save_best_only=True)
    earlystopper= EarlyStopping(monitor='val_loss', patience=params.get('patience',5), verbose=1)
    self.model.fit(train_x, train_y,validation_data=(val_x, val_y), nb_epoch=epoch, batch_size=batch_size, callbacks=[checkpointer, earlystopper])
    return fname, checkpointer.best
      

