"""
Created on Wed Jul 8 14:37:11 2020

@author: ARawat4
"""

import numpy as np

from .utils import cut_rois, resize_input
from .ie_module import Module

color_label = ["White", "Gray", "Yellow", "Red", "Green", "Blue", "Black"]
type_label = ["Car", "Bus", "Truck", "Van"]

class VehicleAttributeDetector(Module):
    POINTS_NUMBER = 5

    class Result:
        def __init__(self, color_output, type_output):
            self.color = color_label[np.argmax(color_output)]
            self.type = type_label[np.argmax(type_output)]
            self.color_confidence = round(np.max(color_output) * 100, 2)
            self.type_confidence = round(np.max(type_output) * 100, 2)
            
    def __init__(self, model):
        super(VehicleAttributeDetector, self).__init__(model)

        assert len(model.inputs) == 1, "Expected 1 input blob"
        assert len(model.outputs) == 2, "Expected 2 output blob"
        self.input_blob = next(iter(model.inputs))
        self.color_type_iter = iter(model.outputs)
        self.color_out_blob = next(self.color_type_iter)
        self.type_out_blob = next(self.color_type_iter)
        self.input_shape = model.inputs[self.input_blob].shape

        assert np.array_equal([1, 7, 1, 1],
                              model.outputs[self.color_out_blob].shape), \
            "Expected model color output shape %s, but got %s" % \
            ([1, 7, 1, 1],
             model.outputs[self.color_out_blob].shape)
        
        assert np.array_equal([1, 4, 1, 1],
                              model.outputs[self.type_out_blob].shape), \
            "Expected model type output shape %s, but got %s" % \
            ([1, 4, 1, 1],
             model.outputs[self.type_out_blob].shape)

    def preprocess(self, frame, rois):
        assert len(frame.shape) == 4, "Frame shape should be [1, c, h, w]"
        inputs = cut_rois(frame, rois, "min_max")
        inputs = [resize_input(input, self.input_shape) for input in inputs]
        return inputs

    def enqueue(self, input):
        return super(VehicleAttributeDetector, self).enqueue({self.input_blob: input})

    def start_async(self, frame, rois):
        inputs = self.preprocess(frame, rois)
        for input in inputs:
            self.enqueue(input)

    def get_attributes(self):
        outputs = self.get_outputs()
        results = [VehicleAttributeDetector.Result(out[self.color_out_blob].reshape(7), out[self.type_out_blob].reshape(4)) \
                      for out in outputs]
        return results
