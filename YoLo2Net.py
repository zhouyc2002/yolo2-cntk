# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 13:03:05 2017

@author: ZHOU Yuncheng
"""

import cntk as C
import _cntk_py
import cntk.layers
import cntk.initializer
import cntk.losses
import cntk.metrics
import cntk.logging
import cntk.io.transforms as xforms
import cntk.io
import cntk.train
import os
import numpy as np
import yolo2
import CloneModel

# default Paths relative to current python file.
abs_path   = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(abs_path, "Models")

# model dimensions
image_height = 416
image_width  = 416
num_channels = 3  # RGB

num_truth_boxes  = 14
box_dim = 5             # centerX, centerY, Width, Height, class_type

num_classes = 3         # object type count. i.e. tomato, flower, stem, et, al.
num_anchors = 5

model_name   = "Yolo2Net.model"

# Create a minibatch source.
def create_image_mb_source(image_file, rois_file, is_training, total_number_of_samples):
    if not os.path.exists(image_file):
        raise RuntimeError("File '%s' does not exist." %image_file)
    if not os.path.exists(rois_file):
        raise RuntimeError("File '%s' does not exist." %rois_file)

    # transformation pipeline for the features has jitter/crop only when training
    transforms = [xforms.scale(width=image_width, height=image_height, 
                               channels=num_channels, interpolations='linear')]    
    if is_training:
        transforms += [
            xforms.color(brightness_radius=0.2, contrast_radius=0.2, saturation_radius=0.2)
        ]
        
    # deserializer        
    imageReader = cntk.io.ImageDeserializer(image_file, 
        cntk.io.StreamDefs(
            features=cntk.io.StreamDef(field='image', transforms=transforms),
            ignored=cntk.io.StreamDef(field='label', shape=1)))
    txtReader = cntk.io.CTFDeserializer(rois_file, 
        cntk.io.StreamDefs(
            rois=cntk.io.StreamDef(field='rois',shape=num_truth_boxes*box_dim)))

    return cntk.io.MinibatchSource([imageReader, txtReader],
        randomize=is_training,
        max_samples=total_number_of_samples,
        multithreaded_deserializer=True)

# Create the network.
def create_yolo2net(anchor_dims = None):
    # Input variables denoting the features and label data
    feature_var = C.input_variable((num_channels, image_height, image_width))
    label_var = C.input_variable((num_truth_boxes, box_dim))

    net = CloneModel.CloneModel('Models/DarkNet.model', 'mean_removed_input', 'bn6e', 
                                cntk.ops.functions.CloneMethod.clone, feature_var)
    
    det1 = cntk.layers.layers.Convolution2D((3,3), 1024, 
        init=cntk.initializer.he_normal(), pad=True,
        activation=cntk.ops.leaky_relu,
        name='det1')(net)
    detbn1 = cntk.layers.BatchNormalization(map_rank=1, name='detbn1')(det1)
    det2 = cntk.layers.layers.Convolution2D((3,3), 1024, 
        init=cntk.initializer.he_normal(), pad=True,
        activation=cntk.ops.leaky_relu,
        name='det2')(detbn1)
    detbn2 = cntk.layers.BatchNormalization(map_rank=1, name='detbn2')(det2)    
    det3 = cntk.layers.layers.Convolution2D((3,3), 1024, 
        init=cntk.initializer.he_normal(), pad = True,
        activation=cntk.ops.leaky_relu,
        name='det3')(detbn2)
    detbn3 = cntk.layers.BatchNormalization(map_rank=1, name='detbn3')(det3)    
    z = cntk.layers.layers.Convolution2D((1,1), (5+num_classes) * num_anchors, 
                                     init=cntk.initializer.normal(0.01), pad = True,
                                     name='output')(detbn3)
    # loss and metric
    ce = C.user_function(yolo2.Yolo2Error(z, label_var, class_size = num_classes, priors = anchor_dims))
    pe = C.user_function(yolo2.Yolo2Metric(z, label_var, class_size = num_classes, priors = anchor_dims, 
                           metricMethod = yolo2.Yolo2MetricMethod.Avg_iou))
    
    cntk.logging.log_number_of_parameters(z) ; print()

    return {
        'feature': feature_var,
        'label': label_var,
        'ce' : ce,
        'pe' : pe,
        'output': z
    }

# Create trainer
def create_trainer(network, epoch_size, num_quantization_bits, printer, block_size, warm_up):
    # Set learning parameters
    lr_per_mb         = [0.001]*25 + [0.0001]*25 + [0.00001]*25 + [0.000001]*25 + [0.0000001]
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
        network['label']: train_source.streams.rois
    }
    
    # Train all minibatches
    cntk.train.training_session(
        trainer=trainer, mb_source = train_source,
        model_inputs_to_streams = input_map,
        mb_size = minibatch_size,
        progress_frequency=epoch_size,
        checkpoint_config = C.CheckpointConfig(filename=os.path.join(model_path, model_name), restore=restore),
        test_config= C.TestConfig(test_source, minibatch_size=minibatch_size)
    ).train()

# Train and evaluate the network.
def net_train_and_eval(train_data, train_rois, test_data, test_rois,
                           priors = None,
                           num_quantization_bits=32, 
                           block_size=3200, warm_up=0, 
                           minibatch_size=1, 
                           epoch_size = 1281167, 
                           max_epochs=1,
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

    network = create_yolo2net(priors)
    
    trainer = create_trainer(network, epoch_size, num_quantization_bits, 
                             [progress_printer, log_printer], block_size, warm_up)
                             
    train_source = create_image_mb_source(train_data, train_rois, True, 
        total_number_of_samples=max_epochs * epoch_size)
    
    train_source
    
    test_source = create_image_mb_source(test_data, train_rois, False, 
        total_number_of_samples=cntk.io.FULL_DATA_SWEEP)

    train_and_test(network, 
                   trainer, 
                   train_source, 
                   test_source, 
                   minibatch_size, 
                   epoch_size, 
                   restore)

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
        ff.close()
    return counts

def open_anchor_file(anchor_file):
    anchors = []
    file = open(anchor_file)
    lines = file.readlines()
    for line in lines:
        if len(line.strip()) > 0:
            dims = line.strip().split("\t")
            anchors.append([float(dims[0]), float(dims[1])])
    file.close()
    return np.array(anchors).astype(np.float32)

if __name__=='__main__':
    
    anchor_data = 'anchor.txt'
    if not os.path.exists(anchor_data):
        raise RuntimeError("File '%s' does not exist." %anchor_data)
    anchors = open_anchor_file(anchor_data)
    if anchors.shape[0] < num_anchors:
        raise RuntimeError("Anchor dimension is less than %s" %num_anchors)
    
#    network = create_yolo2net(anchors)
#    cntk.logging.graph.plot(network['output'], 'yolo2.png')
    train_data = 'train.txt'
    train_rois = 'train.rois.txt'
    test_data = 'train.txt'
    test_rois = 'train.rois.txt'
    
    sample_size = get_sample_counts(train_data, test_data)
    
    net_train_and_eval(train_data, train_rois, test_data, test_rois,
                           priors = anchors,
                           epoch_size=sample_size[0],
                           block_size = None,
                           minibatch_size = 32,
                           max_epochs = 130,
                           log_to_file = 'Yolo2Net.log')
    # Must call MPI finalize when process exit without exceptions
    cntk.train.distributed.Communicator.finalize()
    