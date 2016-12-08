This contains code to apply deep CNN networks for LTE signal classification

Basic Info:

    train_model.py is the main file for training the model.
    
    core/ contains the code to introduce CNN layers
    All data needs to be stored in folder data/
    
    Before running create a folder to save checkpoints in and provide it as input

Sample Command:

    python train_model.py -d lte_tdwaveform.csv -dL train_lbl.csv -v lte_tdwaveform_val.csv -t lte_tdwaveform_test.csv -tL test_lbl.csv -vL val_lbl.csv

Dependencies:
    1. Python 2.7 distribution with numpy
       - Ideally use anaconda2
    
    2. Theano
       - gcc
       - blas
       - CUDA (optional)
       - Follow instruction here for simple cpu install on windows
            http://stackoverflow.com/questions/33687103/how-to-install-theano-on-anaconda-python-2-7-x64-on-windows
       - Instructions here for adding blas and gpu support
            https://github.com/Lasagne/Lasagne/wiki/From-Zero-to-Lasagne
            http://computerscienceunveiled.blogspot.in/2015/08/installing-openblas-for-theano-on.html

    3. Keras
       - simple 'pip install keras'
