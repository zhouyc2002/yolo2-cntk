# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 13:03:05 2017

@author: Administrator
"""

import cntk as C
import _cntk_py
import cntk.layers
import cntk.initializer
import cntk.losses
import cntk.metrics
import cntk.logging
import os
import cntk.io.transforms as xforms
import cntk.io
import cntk.train


# default Paths relative to current python file.
abs_path   = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(abs_path, "Models")

# model dimensions
image_height = 224
image_width  = 224
num_channels = 3  # RGB
num_classes  = 259

model_name   = "DarkNet.model"

# Create a minibatch source.
def create_image_mb_source(map_file, is_training, total_number_of_samples):
    if not os.path.exists(map_file):
        raise RuntimeError("File '%s' does not exist." %map_file)

    # transformation pipeline for the features has jitter/crop only when training
    transforms = []
    if is_training:
        transforms += [
            xforms.crop(crop_type='randomside', side_ratio=0.88671875, jitter_type='uniratio') # train uses jitter
        ]
    else:
        transforms += [
            xforms.crop(crop_type='center', side_ratio=0.88671875) # test has no jitter
        ]

    transforms += [
        xforms.scale(width=image_width, height=image_height, channels=num_channels, interpolations='linear'),
    ]

    # deserializer
    return cntk.io.MinibatchSource(
        cntk.io.ImageDeserializer(map_file, cntk.io.StreamDefs(
            features=cntk.io.StreamDef(field='image', transforms=transforms), # first column in map file is referred to as 'image'
            labels=cntk.io.StreamDef(field='label', shape=num_classes))),   # and second as 'label'
        randomize=is_training,
        max_samples=total_number_of_samples,
        multithreaded_deserializer=True)

def create_darknet():
    # Input variables denoting the features and label data
    feature_var = C.input_variable((num_channels, image_height, image_width))
    label_var = C.input_variable((num_classes))

    # apply model to input
    # remove mean value 
    mean_removed_features = C.ops.minus(feature_var, C.ops.constant(114), name='mean_removed_input')
    
    with C.default_options(activation=None, pad=True, bias=True):
        z = cntk.layers.Sequential([
            # we separate Convolution and ReLU to name the output for feature extraction (usually before ReLU) 
            cntk.layers.layers.Convolution2D((3,3), 32, 
                                             init=cntk.initializer.he_normal(),
                                             activation=cntk.ops.leaky_relu,
                                             name='conv1'),
            cntk.layers.BatchNormalization(map_rank=1, name='bn1'),

            cntk.layers.MaxPooling((2,2), (2,2), name='pool1'),

            cntk.layers.layers.Convolution2D((3,3), 64, 
                                             init=cntk.initializer.he_normal(),
                                             activation=cntk.ops.leaky_relu,
                                             name='conv2'),
            cntk.layers.BatchNormalization(map_rank=1, name='bn2'),
                                          
            cntk.layers.MaxPooling((2,2), (2,2), name='pool2'),

            cntk.layers.layers.Convolution2D((3,3), 128, 
                                             init=cntk.initializer.he_normal(),
                                             activation=cntk.ops.leaky_relu,
                                             name='conv3a'),
            cntk.layers.BatchNormalization(map_rank=1, name='bn3a'),
            cntk.layers.layers.Convolution2D((1,1), 64, 
                                             init=cntk.initializer.he_normal(),
                                             activation=cntk.ops.leaky_relu,
                                             name='conv3b'),
            cntk.layers.BatchNormalization(map_rank=1, name='bn3b'),
            cntk.layers.layers.Convolution2D((3,3), 128, 
                                             init=cntk.initializer.he_normal(),
                                             activation=cntk.ops.leaky_relu,
                                             name='conv3c'),
            cntk.layers.BatchNormalization(map_rank=1, name='bn3c'),
                                          
            cntk.layers.MaxPooling((2,2), (2,2), name='pool3'),

            cntk.layers.layers.Convolution2D((3,3), 256, 
                                             init=cntk.initializer.he_normal(),
                                             activation=cntk.ops.leaky_relu,
                                             name='conv4a'),
            cntk.layers.BatchNormalization(map_rank=1, name='bn4a'),                                             
            cntk.layers.layers.Convolution2D((1,1), 128, 
                                             init=cntk.initializer.he_normal(),
                                             activation=cntk.ops.leaky_relu,
                                             name='conv4b'),
            cntk.layers.BatchNormalization(map_rank=1, name='bn4b'),
            cntk.layers.layers.Convolution2D((3,3), 256, 
                                             init=cntk.initializer.he_normal(),
                                             activation=cntk.ops.leaky_relu,
                                             name='conv4c'),
            cntk.layers.BatchNormalization(map_rank=1, name='bn4c'),
                                             
            cntk.layers.MaxPooling((2,2), (2,2), name='pool4'),

            cntk.layers.layers.Convolution2D((3,3), 512, 
                                             init=cntk.initializer.he_normal(),
                                             activation=cntk.ops.leaky_relu,
                                             name='conv5a'),
            cntk.layers.BatchNormalization(map_rank=1, name='bn5a'),                                    
            cntk.layers.layers.Convolution2D((1,1), 256, 
                                             init=cntk.initializer.he_normal(),
                                             activation=cntk.ops.leaky_relu,
                                             name='conv5b'),
            cntk.layers.BatchNormalization(map_rank=1, name='bn5b'),                                             
            cntk.layers.layers.Convolution2D((3,3), 512, 
                                             init=cntk.initializer.he_normal(),
                                             activation=cntk.ops.leaky_relu,
                                             name='conv5c'),
            cntk.layers.BatchNormalization(map_rank=1, name='bn5c'),
            cntk.layers.layers.Convolution2D((1,1), 256, 
                                             init=cntk.initializer.he_normal(),
                                             activation=cntk.ops.leaky_relu,
                                             name='conv5d'),
            cntk.layers.BatchNormalization(map_rank=1, name='bn5d'),                                             
            cntk.layers.layers.Convolution2D((3,3), 512, 
                                             init=cntk.initializer.he_normal(),
                                             activation=cntk.ops.leaky_relu,
                                             name='conv5e'),
            cntk.layers.BatchNormalization(map_rank=1, name='bn5e'),
                                          
            cntk.layers.MaxPooling((2,2), (2,2), name='pool5'),
                                  
            cntk.layers.layers.Convolution2D((3,3), 1024, 
                                             init=cntk.initializer.he_normal(),
                                             activation=cntk.ops.leaky_relu,
                                             name='conv6a'),
            cntk.layers.BatchNormalization(map_rank=1, name='bn6a'),
            cntk.layers.layers.Convolution2D((1,1), 512, 
                                             init=cntk.initializer.he_normal(),
                                             activation=cntk.ops.leaky_relu,
                                             name='conv6b'),
            cntk.layers.BatchNormalization(map_rank=1, name='bn6b'),                                             
            cntk.layers.layers.Convolution2D((3,3), 1024, 
                                             init=cntk.initializer.he_normal(),
                                             activation=cntk.ops.leaky_relu,
                                             name='conv6c'),
            cntk.layers.BatchNormalization(map_rank=1, name='bn6c'),
            cntk.layers.layers.Convolution2D((1,1), 512, 
                                             init=cntk.initializer.he_normal(),
                                             activation=cntk.ops.leaky_relu,
                                             name='conv6d'),
            cntk.layers.BatchNormalization(map_rank=1, name='bn6d'),                                             
            cntk.layers.layers.Convolution2D((3,3), 1024, 
                                             init=cntk.initializer.he_normal(),
                                             activation=cntk.ops.leaky_relu,
                                             name='conv6e'),
            cntk.layers.BatchNormalization(map_rank=1, name='bn6e'),
                                          
            cntk.layers.layers.Convolution2D((1,1), num_classes, 
                                             init=cntk.initializer.he_normal(),
                                             name='conv_class'),
            cntk.layers.layers.AveragePooling((7,7), strides=7, name='avg_pool6'),
            ])(mean_removed_features)
    
    net = cntk.ops.reshape(z, (num_classes), name='output')
    # loss and metric
    ce  = cntk.losses.cross_entropy_with_softmax(net, label_var)
    pe  = cntk.metrics.classification_error(net, label_var)

    cntk.logging.log_number_of_parameters(net) ; print()

    return {
        'feature': feature_var,
        'label': label_var,
        'ce' : ce,
        'pe' : pe,
        'output': net
    }
    
# Create trainer
def create_trainer(network, epoch_size, num_quantization_bits, printer, block_size, warm_up):
    # Set learning parameters
    lr_per_mb         = [0.01]*25 + [0.001]*25 + [0.0001]*25 + [0.00001]*25 + [0.000001]
    lr_schedule       = C.learning_rate_schedule(lr_per_mb, unit=C.learners.UnitType.minibatch, epoch_size=epoch_size)
    mm_schedule       = C.learners.momentum_schedule(0.9)
    l2_reg_weight     = 0.0005 # CNTK L2 regularization is per sample, thus same as Caffe

    if block_size != None and num_quantization_bits != 32:
        raise RuntimeError("Block momentum cannot be used with quantization, please remove quantized_bits option.")

    # Create learner
    local_learner = C.learners.momentum_sgd(network['output'].parameters, lr_schedule, mm_schedule, unit_gain=False, l2_regularization_weight=l2_reg_weight)
    # Since we reuse parameter settings (learning rate, momentum) from Caffe, we set unit_gain to False to ensure consistency

    # Create trainer
    if block_size != None:
        parameter_learner = cntk.train.distributed.block_momentum_distributed_learner(local_learner, block_size=block_size)
    else:
        parameter_learner = cntk.train.distributed.data_parallel_distributed_learner(local_learner, num_quantization_bits=num_quantization_bits, distributed_after=warm_up)

    return C.Trainer(network['output'], (network['ce'], network['pe']), parameter_learner, printer)

# Train and test
def train_and_test(network, trainer, train_source, test_source, minibatch_size, epoch_size, restore):

    # define mapping from intput streams to network inputs
    input_map = {
        network['feature']: train_source.streams.features,
        network['label']: train_source.streams.labels
    }

    # Train all minibatches 
    cntk.train.training_session(
        trainer=trainer, mb_source = train_source,
        model_inputs_to_streams = input_map,
        mb_size = minibatch_size,
        progress_frequency=epoch_size,
        checkpoint_config = C.CheckpointConfig(filename=os.path.join(model_path, model_name), frequency=epoch_size, restore=restore, preserve_all=True),
        test_config= C.TestConfig(test_source, minibatch_size=minibatch_size)
    ).train()


# Train and evaluate the network.
def alexnet_train_and_eval(train_data, test_data, 
                           num_quantization_bits=32, 
                           block_size=3200, warm_up=0, 
                           minibatch_size=96, 
                           epoch_size = 1281167, 
                           max_epochs=112,
                           restore=True, 
                           log_to_file=None, 
                           num_mbs_per_log=None, 
                           gen_heartbeat=True):
    _cntk_py.set_computation_network_trace_level(0)

    log_printer = cntk.logging.progress_print.ProgressPrinter(
        freq=1,
        tag='Training',
        log_to_file = os.path.join(model_path, log_to_file),
        num_epochs=max_epochs)
    
    progress_printer = cntk.logging.progress_print.ProgressPrinter(freq=1, tag='Training', 
                                                  num_epochs=max_epochs,test_freq=1)    


    network = create_darknet()
    trainer = create_trainer(network, epoch_size, num_quantization_bits, [progress_printer, log_printer], block_size, warm_up)
    train_source = create_image_mb_source(train_data, True, total_number_of_samples=max_epochs * epoch_size)
    test_source = create_image_mb_source(test_data, False, total_number_of_samples=cntk.io.FULL_DATA_SWEEP)
    train_and_test(network, trainer, train_source, test_source, minibatch_size, epoch_size, restore)

#
# get train sample size evaluate sample size
#
def get_sample_counts(train_file, test_file):
    counts = [0, 0]
    if os.path.exists(train_file):
        ff = open(train_file)
        counts[0] = len(ff.readlines())
        ff.close()
    if os.path.exists(test_file):
        ff = open(test_file)
        counts[1] = len(ff.readlines())
        ff.close();
    return counts

if __name__=='__main__':
    train_data = 'train.txt'
    test_data = 'test.txt'
    sample_size = get_sample_counts(train_data, test_data)
    
    alexnet_train_and_eval(train_data, test_data,
                           epoch_size=sample_size[0],
                           block_size = None,
                           max_epochs = 300,
                           log_to_file = 'DarkNet.log')
    # Must call MPI finalize when process exit without exceptions
    cntk.train.distributed.Communicator.finalize()
    