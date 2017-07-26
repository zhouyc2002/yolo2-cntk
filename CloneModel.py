# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 12:14:51 2017

@author: Zhou Yuncheng
"""
import cntk
import cntk.ops
import cntk.layers
import cntk.logging.graph
import cntk.io.transforms
import os
import numpy as np

def Visitor(node):
    if isinstance(node, cntk.functions.Function) == True:
        return True
    else:
        return False
    
def CloneModel(modelFile, beginNodeName, endNodeName, cloneMethod, newInput):
    if not os.path.exists(modelFile):
        raise RuntimeError("Model file '%s' does not exist." %modelFile)
        
    net = cntk.functions.load_model(modelFile)
    net_p = net.find_by_name(endNodeName)
    net_b = net.find_by_name(beginNodeName)
    
    if net_b is None:
        raise RuntimeError("Node '%s' does not exist in model." %beginNodeName)
    if net_p is None:
        raise RuntimeError("Node '%s' does not exist in model." %endNodeName)
    
    net_c = cntk.ops.as_composite(net_p)
    funcs = cntk.logging.graph.depth_first_search(net_c, Visitor)
    funcs.reverse()
    
    parts = []
    clone = False
    for i in funcs:
        if i.name == beginNodeName:
            clone = True
        if i.name == endNodeName:
            parts.append(i)
            clone = False
        if clone == True:
            parts.append(i)
    
    fin = net.arguments[0]
    for func in parts:
        fin = func(fin)
    
    return fin.clone(cloneMethod, {net.arguments[0]:newInput})

#
# start route
#
if __name__ == "__main__":
    input_var = cntk.input_variable((3,112,112),name='feature')
    net = CloneModel('Models/DarkNet.model', 'mean_removed_input', 'bn6e', 
                     cntk.functions.CloneMethod.clone,
                     input_var)
    data = np.random.rand(2,3,112,112).astype(np.float32)
    result = net.eval({input_var : data})
    print(result)
    cntk.logging.graph.plot(net, 'net.png')