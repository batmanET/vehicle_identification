"""
Created on Wed Jul 8 13:01:27 2020

@author: ARawat4
"""

import numpy as np
from numpy import clip
from .ie_module import Module
from .utils import resize_input

class VehicleDetector(Module):
    class Result:
        OUTPUT_SIZE = 7

        def __init__(self, output):
            self.image_id = output[0]
            self.label = int(output[1])
            self.confidence = output[2]
            self.position_min = np.array((output[3], output[4])) # (x, y)
            self.position_max = np.array((output[5], output[6])) # (x, y)
            self.size = np.array((self.position_max[0] - self.position_min[0], self.position_max[1] - self.position_min[1]))

        def rescale_roi(self, roi_scale_factor=1.0):
            self.position_min -= self.size * 0.5 * (roi_scale_factor - 1.0)
            self.position_max += self.size * 0.5 * (roi_scale_factor - 1.0)
            self.size *= roi_scale_factor

        def resize_roi(self, frame_width, frame_height):
            self.position_min[0] *= frame_width
            self.position_min[1] *= frame_height
            self.position_max[0] *= frame_width
            self.position_max[1] *= frame_height
            self.size = np.array((self.position_max[0] - self.position_min[0], self.position_max[1] - self.position_min[1]))

        def clip(self, width, height):
            min = [0, 0]
            max = [width, height]
            self.position_min[:] = clip(self.position_min, min, max)
            self.position_max[:] = clip(self.position_max, min, max)
            self.size[:] = clip(self.size, min, max)

    def __init__(self, model, confidence_threshold=0.5, roi_scale_factor=1.15):
        super(VehicleDetector, self).__init__(model)

        assert len(model.inputs) == 1, "Expected 1 input blob"
        assert len(model.outputs) == 1, "Expected 1 output blob"
        self.input_blob = next(iter(model.inputs))
        self.output_blob = next(iter(model.outputs))
        self.input_shape = model.inputs[self.input_blob].shape
        self.output_shape = model.outputs[self.output_blob].shape

        assert len(self.output_shape) == 4 and \
               self.output_shape[3] == self.Result.OUTPUT_SIZE, \
            "Expected model output shape with %s outputs" % \
            (self.Result.OUTPUT_SIZE)

        assert 0.0 <= confidence_threshold and confidence_threshold <= 1.0, \
            "Confidence threshold is expected to be in range [0; 1]"
        self.confidence_threshold = confidence_threshold

        assert 0.0 < roi_scale_factor, "Expected positive ROI scale factor"
        self.roi_scale_factor = roi_scale_factor

    def preprocess(self, frame):
        assert len(frame.shape) == 4, "Frame shape should be [1, c, h, w]"
        assert frame.shape[0] == 1
        assert frame.shape[1] == 3
        input = resize_input(frame, self.input_shape)
        return input

    def start_async(self, frame):
        input = self.preprocess(frame)
        self.enqueue(input)

    def enqueue(self, input):
        return super(VehicleDetector, self).enqueue({self.input_blob: input})

    def get_roi_proposals(self, frame):
        outputs = self.get_outputs()[0][self.output_blob]
        # outputs shape is [N_requests, 1, 1, N_max_faces, 7]

        frame_width = frame.shape[-1]
        frame_height = frame.shape[-2]

        vehicles = []
        for output in outputs[0][0]:
            result = VehicleDetector.Result(output)
            if result.confidence < self.confidence_threshold:
                break # results are sorted by confidence decrease

            result.resize_roi(frame_width, frame_height)
            result.rescale_roi(self.roi_scale_factor)
            result.clip(frame_width, frame_height)
            vehicles.append(result)
        return vehicles
