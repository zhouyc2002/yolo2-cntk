# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 18:34:25 2017

@author: zhouyc
"""
import math
import numpy as np
import cntk
import cntk.ops
import cntk.logging.graph
import cntk.ops.functions

# np.set_printoptions(threshold=np.nan)
class PureLine(cntk.ops.functions.UserFunction):
    def __init__(self, arg, name='PureLine'):
        super(PureLine, self).__init__([arg], name=name)
    
    def forward(self, argument, device=None, outputs_to_retain=None):
#        print('PureLine Forward', argument.shape)
#        print(argument[0][1])
        return None, argument
    
    def backward(self, state, root_gradients):
#        print('PureLine Backward', root_gradients.shape)
#        print(root_gradients[0][1])
        return root_gradients
    
    def infer_outputs(self):
        return [cntk.output_variable(self.inputs[0].shape,
                                     self.inputs[0].dtype,
                                     self.inputs[0].dynamic_axes)]
    @staticmethod
    def deserialize(inputs, name, state):
        return PureLine(inputs[0], name)
    
class Box(object):
    def __init__(self, cx = 0., cy = 0., width = 0., height = 0.):
        super(Box, self).__init__()
        self.cx = cx
        self.cy = cy
        self.width = width
        self.height  =height
        
    def empty(self):
        if self.width * self.height < 1e-6:
            return True
        else:
            return False
        
    def iou(self, other):
        ax1 = self.cx - self.width / 2.
        ay1 = self.cy - self.height / 2.
        ax2 = ax1 + self.width
        ay2 = ay1 + self.height
        
        bx1 = other.cx - other.width / 2.
        by1 = other.cy - other.height / 2.
        bx2 = bx1 + other.width
        by2 = by1 + other.height
        
        maxLeft = max(ax1, bx1)
        maxTop = max(ay1, by1)
        minRight = min(ax2, bx2)
        minBottom = min(ay2, by2)
        
        if maxLeft > minRight or maxTop > minBottom:
            return 0.
        
        ix = maxLeft
        iy = maxTop
        iw = minRight - ix
        ih = minBottom - iy
        
        ux = min(ax1, bx1)
        uy = min(ay1, by1)
        uw = max(ax2, bx2) - ux
        uh = max(ay2, by2) - uy
        
        return (iw * ih) / (uw * uh)

class FeatureMapParser(object):
    def __init__(self, mapLayers, priorDimentions, classesSize):
        super(FeatureMapParser, self).__init__()
        self.map_layers = mapLayers
        self.loss_layers = np.zeros(mapLayers.shape, dtype=mapLayers.dtype)
        self.prior_dims = priorDimentions
        self.classes_size = classesSize
        shape = list(mapLayers.shape)
        self.sequence_count = shape[0]
        self.layers_count = shape[1]
        self.cell_rows = shape[2]
        self.cell_cols = shape[3]
        self.bbox_count = int(shape[1] / (5 + classesSize))
        
    def get_box_pred(self, sequence_index, cell_row_index, cell_col_index, bbox_index):
        stride = self.classes_size + 5
        tx = self.map_layers[sequence_index][bbox_index*stride][cell_row_index][cell_col_index]
        ty = self.map_layers[sequence_index][bbox_index*stride+1][cell_row_index][cell_col_index]
        tw = self.map_layers[sequence_index][bbox_index*stride+2][cell_row_index][cell_col_index]
        th = self.map_layers[sequence_index][bbox_index*stride+3][cell_row_index][cell_col_index]
        
        return tx, ty, tw, th
    
    def set_box_loss(self, tx_loss, ty_loss, tw_loss, th_loss,
                     sequence_index, cell_row_index, cell_col_index, bbox_index):
        stride = self.classes_size + 5
        self.loss_layers[sequence_index][bbox_index*stride][cell_row_index][cell_col_index] = tx_loss
        self.loss_layers[sequence_index][bbox_index*stride+1][cell_row_index][cell_col_index] = ty_loss
        self.loss_layers[sequence_index][bbox_index*stride+2][cell_row_index][cell_col_index] = tw_loss
        self.loss_layers[sequence_index][bbox_index*stride+3][cell_row_index][cell_col_index] = th_loss
    
    def get_box(self, sequence_index, cell_row_index, cell_col_index, bbox_index):
        tx, ty, tw, th = self.get_box_pred(sequence_index, 
                                           cell_row_index, 
                                           cell_col_index, 
                                           bbox_index)
        bx = tx + cell_col_index
        by = ty + cell_row_index
        pw = self.prior_dims[bbox_index][0] * self.cell_cols
        ph = self.prior_dims[bbox_index][1] * self.cell_rows
        bw = pw * math.exp(tw)
        bh = ph * math.exp(th)
        
        return bx, by, bw, bh

    def get_norm_box(self, sequence_index, cell_row_index, cell_col_index, bbox_index):
        bx, by, bw, bh = self.get_box(sequence_index,
                                      cell_row_index, 
                                      cell_col_index, 
                                      bbox_index)
        nx = bx / self.cell_cols
        ny = by / self.cell_rows
        nw = bw / self.cell_cols
        nh = bh / self.cell_rows
        return nx, ny, nw, nh
    
    def get_norm_bbox(self, sequence_index, cell_row_index, cell_col_index, bbox_index):
        nx,ny,nw,nh = self.get_norm_box(sequence_index, 
                                        cell_row_index, 
                                        cell_col_index, 
                                        bbox_index)
        return Box(nx, ny, nw, nh)
    
    def get_confidence(self, sequence_index, cell_row_index, cell_col_index, bbox_index):
        return self.map_layers[sequence_index][bbox_index*(self.classes_size + 5)+4][cell_row_index][cell_col_index]
    
    def set_confidence_loss(self, loss, sequence_index, cell_row_index, cell_col_index, bbox_index):
        self.loss_layers[sequence_index][bbox_index*(self.classes_size + 5)+4][cell_row_index][cell_col_index] = loss
    
    def get_classes(self, sequence_index, cell_row_index, cell_col_index, bbox_index):
        stride = self.classes_size + 5
        cls = np.zeros(self.classes_size, dtype=np.float32)
        for i in range(self.classes_size):
            cls[i] = self.map_layers[sequence_index][bbox_index*stride+5+i][cell_row_index][cell_col_index]
        return cls

    def set_classes_loss(self, cls_loss, sequence_index, cell_row_index, cell_col_index, bbox_index):
        stride = self.classes_size + 5
        for i in range(self.classes_size):
            self.loss_layers[sequence_index][bbox_index*stride+5+i][cell_row_index][cell_col_index] = cls_loss[i]
    
    def calc_gradient(self):
        stride = self.classes_size + 5
        for seq_index in range(self.sequence_count):
            for box_index in range(self.bbox_count):
                self.loss_layers[seq_index][box_index*stride] *= (self.map_layers[seq_index][box_index*stride] * (1-self.map_layers[seq_index][box_index*stride]))
                self.loss_layers[seq_index][box_index*stride+1] *= (self.map_layers[seq_index][box_index*stride+1] * (1-self.map_layers[seq_index][box_index*stride+1]))
                self.loss_layers[seq_index][box_index*stride+4] *= (self.map_layers[seq_index][box_index*stride+4] * (1-self.map_layers[seq_index][box_index*stride+4]))                
            
class TruthParser(object):
    def __init__(self, ground_truths):
        super(TruthParser, self).__init__()
        self.ground_truths = ground_truths
        shape = list(ground_truths.shape)
        self.sequence_count = shape[0]
        self.truth_count = shape[1]
    
    def get_norm_truth(self, sequence_index, truth_index):
        x = self.ground_truths[sequence_index][truth_index][0]
        y = self.ground_truths[sequence_index][truth_index][1]
        w = self.ground_truths[sequence_index][truth_index][2]
        h = self.ground_truths[sequence_index][truth_index][3]
        cls_type = int(self.ground_truths[sequence_index][truth_index][4])
        return x, y, w, h, cls_type
    
    def get_norm_truth_box(self, sequence_index, truth_index):
        x,y,w,h,ct=self.get_norm_truth(sequence_index, truth_index)
        return Box(x, y, w, h)
    
    def get_truth_class_type(self, sequence_index, truth_index):
        return int(self.ground_truths[sequence_index][truth_index][4])

def yolo2_output(input_shape, anchor_nums, class_size):
    tmp_var = cntk.input_variable(input_shape, name='yolo2_output_net_input_var')
    per_size = class_size + 5
    per_route = []
    for i in range(anchor_nums):
        per_route.append(cntk.ops.sigmoid(tmp_var[per_size * i : per_size * i + 2]))
        per_route.append(tmp_var[per_size * i + 2 : per_size * i + 4])
        per_route.append(cntk.ops.sigmoid(tmp_var[per_size * i + 4 : per_size * i + 5]))
        per_route.append(cntk.ops.softmax(tmp_var[per_size * i + 5 : per_size * (i+1)], axis=0))
    out_tuple = tuple(per_route)
    z = cntk.ops.splice(*out_tuple, axis=0, name='output')
    return {'output' : z, 'input_var' : tmp_var}
    
class Yolo2Error(cntk.ops.functions.UserFunction):
    def __init__(self, arg1, arg2, class_size =0, priors = None, 
                 objectScale = 5.0, noobjectScale = 1.0, classScale = 1.0,
                 coordScale = 1.0, nocoordScale = 0.01,
                 iouThresh = 0.6,
                 biasMatch = False, rescore = True,
                 name='Yolo2Error'):
        super(Yolo2Error, self).__init__([arg1,arg2], name=name)
        self.class_size = class_size
        self.priors = priors
        self.object_scale = objectScale
        self.noobject_scale = noobjectScale
        self.class_scale = classScale
        self.coord_scale = coordScale
        self.nocoord_scale = nocoordScale
        self.iou_thresh = iouThresh
        self.bias_match = biasMatch
        self.rescore = rescore
        anchor_num = int(arg1.shape[0] / (5 + class_size))
        self.output_network = yolo2_output(arg1.shape, anchor_num, class_size)
        
    def forward(self, arguments, device=None, outputs_to_retain=None):
        mapLayers = self.output_network['output'].eval({self.output_network['input_var'] : arguments[0]})
        # mapLayers = arguments[0]
        truthes = arguments[1]
#        
#        print('Yolo2Error Forward:', mapLayers.shape)
#        print(mapLayers[0][1])
#        
        map_parser = FeatureMapParser(mapLayers, self.priors, self.class_size)
        truth_parser = TruthParser(truthes)
        
        sequence_count = map_parser.sequence_count
        
        cell_rows = map_parser.cell_rows
        cell_cols = map_parser.cell_cols
        bbox_count = map_parser.bbox_count
        truth_count = truth_parser.truth_count

        for seq_index in range(sequence_count):
            for row_index in range(cell_rows):
                for col_index in range(cell_cols):
                    for box_index in range(bbox_count):
                        pred = map_parser.get_norm_bbox(seq_index, 
                                                        row_index, col_index, 
                                                        box_index)
                        best_iou = 0.
                        for t in range(truth_count):
                            truth = truth_parser.get_norm_truth_box(seq_index, t)
                            if truth.empty() == True:
                                break;
                            iou = pred.iou(truth)
                            if iou > best_iou:
                                best_iou = iou
                        confidence = map_parser.get_confidence(seq_index, 
                                                               row_index, col_index, 
                                                               box_index)
                        conf_loss = self.noobject_scale * (confidence - 0.0)
                        map_parser.set_confidence_loss(conf_loss, 
                                                       seq_index, 
                                                       row_index, col_index, 
                                                       box_index)
                        if best_iou > self.iou_thresh:
                            map_parser.set_confidence_loss(0, seq_index, 
                                                           row_index, col_index, 
                                                           box_index)
                        tx, ty, tw, th = map_parser.get_box_pred(seq_index, 
                                                                 row_index, col_index, 
                                                                 box_index)
                        map_parser.set_box_loss(self.nocoord_scale*(tx - 0.5), 
                                                self.nocoord_scale*(ty - 0.5), 
                                                self.nocoord_scale*(tw - 0.0), 
                                                self.nocoord_scale*(th - 0.0),
                                                seq_index, row_index, col_index, box_index)
                        
            for truth_index in range(truth_count):
                truth = truth_parser.get_norm_truth_box(seq_index, truth_index)
                
                if truth.empty() == True:
                    break
                
                best_iou = 0.
                best_n = 0
                
                row_index = int(truth.cy * cell_rows)
                col_index = int(truth.cx * cell_cols)
                
                truth_shift = Box(0., 0., 0., 0.)
                truth_shift.width = truth.width
                truth_shift.height = truth.height
                
                # 找寻与当前ground truth最匹配的max(iou)的anchor box
                for box_index in range(bbox_count):
                    pred = map_parser.get_norm_bbox(seq_index, row_index,
                                                    col_index, box_index)
                    if self.bias_match == True:
                        pred.width = self.priors[box_index][0]
                        pred.height = self.priors[box_index][1]
                    
                    pred.cx = 0.
                    pred.cy = 0.
                    
                    iou = pred.iou(truth_shift)
                    if iou > best_iou:
                        best_iou = iou
                        best_n = box_index
                    
                pred = map_parser.get_norm_bbox(seq_index, row_index,
                                                col_index, best_n)
                iou = pred.iou(truth)
                
                tx = truth.cx * cell_cols - col_index
                ty = truth.cy * cell_rows - row_index
                tw = np.log(truth.width / self.priors[best_n][0])
                th = np.log(truth.height / self.priors[best_n][1])
                
                scale = self.coord_scale * (2 - truth.width * truth.height)
                
                ptx,pty,ptw,pth=map_parser.get_box_pred(seq_index, 
                                                        row_index, col_index, best_n)
                tx_loss = scale * (ptx - tx)
                ty_loss = scale * (pty - ty)
                tw_loss = scale * (ptw - tw)
                th_loss = scale * (pth - th)
                
                map_parser.set_box_loss(tx_loss, ty_loss, tw_loss, th_loss,
                                        seq_index, row_index, col_index, best_n)
                
                confidence = map_parser.get_confidence(seq_index, row_index,
                                                       col_index, best_n)
                conf_loss = self.object_scale * (confidence - 1.0)
                if self.rescore == True:
                    conf_loss = self.object_scale * (confidence - iou)
                map_parser.set_confidence_loss(conf_loss, seq_index,
                                               row_index, col_index,
                                               best_n)
                
                truth_type = truth_parser.get_truth_class_type(seq_index, 
                                                               truth_index)
                box_classes = map_parser.get_classes(seq_index, row_index,
                                                     col_index, best_n)
                truth_classes = np.zeros(len(box_classes),dtype=box_classes.dtype)
                truth_classes[truth_type-1]=1.0
                classes_loss = self.class_scale * (box_classes - truth_classes)
                map_parser.set_classes_loss(classes_loss, seq_index, row_index,
                                            col_index, best_n)
                
        #loss_layers = map_parser.loss_layers
        lost = np.zeros((sequence_count,1), dtype=self.inputs[0].dtype)
        for seq_index in range(sequence_count):
            lost[seq_index][0] = np.sum(np.power(map_parser.loss_layers[seq_index], 2.))
            
        map_parser.calc_gradient()
        
        return map_parser.loss_layers, lost
    
    def backward(self, state, root_gradients, variables):
#        print('Yolo2Error Backward:', state.shape)
#        print(state[0][1])
        for var in variables:
            variables[var] = state
    
    def infer_outputs(self):
        return [cntk.output_variable(1,
                                     self.inputs[0].dtype,
                                     self.inputs[0].dynamic_axes)]
    
    def serialize(self):
        return {'prior_dimens': self.priors, 
                'class_size' : self.class_size,
                'object_scale' : self.object_scale,
                'noobject_scale' : self.noobject_scale,
                'class_scale' : self.class_scale,
                'coord_scale' : self.coord_scale,
                'nocoord_scale' : self.nocoord_scale,
                'iou_thresh' : self.iou_thresh,
                'bias_match' : self.bias_match,
                'rescore' : self.rescore}
    
    @staticmethod
    def deserialize(inputs, name, state):
        f = Yolo2Error(inputs[0], inputs[1], 
                       class_size=state['class_size'], 
                       priors = state['prior_dimens'], 
                       objectScale = state['object_scale'],
                       noobjectScale = state['noobject_scale'],
                       classScale = state['class_scale'],
                       coordScale = state['coord_scale'],
                       nocoordScale = state['nocoord_scale'],
                       iouThresh = state['iou_thresh'],
                       biasMatch = state['bias_match'],
                       rescore = state['rescore'],
                       name = name)
        return f
    
#
# Yolo2 Metric
#  
class Yolo2MetricMethod(object):
    Avg_iou = 1
    Avg_recall = 2
    Avg_confidence = 3
    Avg_classes = 4
    
class Yolo2Metric(cntk.ops.functions.UserFunction):
    def __init__(self, arg1, arg2, class_size =0, priors = None, 
                 metricMethod = Yolo2MetricMethod.Avg_iou,
                 name='Yolo2Metric'):
        super(Yolo2Metric, self).__init__([arg1,arg2], name=name)
        self.class_size = class_size
        self.priors = priors
        self.metric_method = metricMethod
        
        anchor_num = int(arg1.shape[0] / (5 + class_size))
        self.output_network = yolo2_output(arg1.shape, anchor_num, class_size)
        
    def forward(self, arguments, device=None, outputs_to_retain=None):
        mapLayers = self.output_network['output'].eval({self.output_network['input_var'] : arguments[0]})
        # mapLayers = arguments[0]
        truthes = arguments[1]
        
        map_parser = FeatureMapParser(mapLayers, self.priors, self.class_size)
        truth_parser = TruthParser(truthes)
        
        sequence_count = map_parser.sequence_count
        
        cell_rows = map_parser.cell_rows
        cell_cols = map_parser.cell_cols
        bbox_count = map_parser.bbox_count
        truth_count = truth_parser.truth_count
        
        avg_iou = 0.0
        avg_classes = 0.0
        avg_object = 0.0
        avg_recall= 0.0
        total_count = 0
        
        for seq_index in range(sequence_count):                        
            for truth_index in range(truth_count):
                truth = truth_parser.get_norm_truth_box(seq_index, truth_index)
                
                if truth.empty() == True:
                    break
                
                best_iou = 0.
                best_n = 0
                
                row_index = int(truth.cy * cell_rows)
                col_index = int(truth.cx * cell_cols)
                
                # 找寻与当前ground truth最匹配的max(iou)的anchor box
                for box_index in range(bbox_count):
                    pred = map_parser.get_norm_bbox(seq_index, row_index, col_index, box_index)
                    iou = pred.iou(truth)
                    if iou > best_iou:
                        best_iou = iou
                        best_n = box_index
                #
                avg_iou += best_iou
                if best_iou > 0.5:
                    avg_recall += 1.
                #
                
                confidence = map_parser.get_confidence(seq_index, row_index, col_index, best_n)
                
                truth_type = truth_parser.get_truth_class_type(seq_index, truth_index)
                box_classes = map_parser.get_classes(seq_index, row_index, col_index, best_n)
                truth_classes = np.zeros(len(box_classes),dtype=box_classes.dtype)
                truth_classes[truth_type-1]=1.0
                             
                classes_loss = np.sum(np.power(confidence * box_classes - truth_classes, 2))
                #
                avg_classes += classes_loss
                avg_object += confidence
                total_count += 1
                #
                
        #metrics
        avg_classes /= total_count
        avg_object /= total_count
        avg_iou /= total_count
        avg_recall /= total_count
        avg = 0
        if self.metric_method == Yolo2MetricMethod.Avg_classes:
            avg = avg_classes
        elif self.metric_method == Yolo2MetricMethod.Avg_confidence:
            avg = avg_object
        elif self.metric_method == Yolo2MetricMethod.Avg_iou:
            avg = avg_iou
        else:
            avg = avg_recall
        
        metric = np.array([avg]*sequence_count, dtype=self.inputs[0].dtype).reshape((sequence_count,1))
        
        return None, metric
    
    def backward(self, state, root_gradients, variables):
        print('Yolo2Metric.........')
        for var in variables:
            variables[var] = None
    
    def infer_outputs(self):
        return [cntk.output_variable(1,
                                     self.inputs[0].dtype,
                                     self.inputs[0].dynamic_axes)]
    
    def serialize(self):
        return {'prior_dimens': self.priors, 
                'class_size' : self.class_size,
                'metric_method' : self.metric_method}
    
    @staticmethod
    def deserialize(inputs, name, state):
        f = Yolo2Metric(inputs[0], inputs[1], 
                       class_size=state['class_size'], 
                       priors = state['prior_dimens'], 
                       metricMethod = state['metric_method'],
                       name = name)
        return f
