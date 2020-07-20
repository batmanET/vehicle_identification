"""
Created on Wed Jul 8 14:37:11 2020

@author: ARawat4
"""

import numpy as np

from .utils import cut_rois, resize_input
from .ie_module import Module

license_dict = {"0": "0", "1": "1", "2": "2", "3": "3", "4": "4", "5": "5", "6": "6", "7": "7", "8": "8", "9": "9", 
                "10": "<Anhui>", "11": "<Beijing>", "12": "<Chongqing>", "13": "<Fujian>", "14": "<Gansu>", 
                "15": "<Guangdong>", "16": "<Guangxi>", "17": "<Guizhou>", "18": "<Hainan>", "19": "<Hebei>", 
                "20": "<Heilongjiang>", "21": "<Henan>", "22": "<HongKong>", "23": "<Hubei>", "24": "<Hunan>", 
                "25": "<InnerMongolia>", "26": "<Jiangsu>", "27": "<Jiangxi>", "28": "<Jilin>", "29": "<Liaoning>", 
                "30": "<Macau>", "31": "<Ningxia>", "32": "<Qinghai>", "33": "<Shaanxi>", "34": "<Shandong>", "35": "<Shanghai>", 
                "36": "<Shanxi>", "37": "<Sichuan>", "38": "<Tianjin>", "39": "<Tibet>", "40": "<Xinjiang>", "41": "<Yunnan>", 
                "42": "<Zhejiang>", "43": "<police>", "44": "A", "45": "B", "46": "C", "47": "D", "48": "E", "49": "F", "50": "G", 
                "51": "H", "52": "I", "53": "J", "54": "K", "55": "L", "56": "M", "57": "N", "58": "O", "59": "P", "60": "Q", "61": "R", 
                "62": "S", "63": "T", "64": "U", "65": "V", "66": "W", "67": "X", "68": "Y", "69": "Z"}

class LicenseNumberExtractor(Module):
    
    class Result:
        def __init__(self, output):
            self.license_number = ""
            self.encoded_result = np.delete(output, np.argwhere(output == -1)).astype(str)
            for encoding in self.encoded_result:
                self.license_number += license_dict[encoding]

    def __init__(self, model):
        super(LicenseNumberExtractor, self).__init__(model)

        assert len(model.inputs) == 2, "Expected 1 input blob"
        assert len(model.outputs) == 1, "Expected 1 output blob"
        self.input_iter = iter(model.inputs)
        self.input_img_blob = next(self.input_iter)
        self.input_seq_blob = next(self.input_iter)
        self.output_blob = next(iter(model.outputs))
        self.input_shape = model.inputs[self.input_img_blob].shape

        assert np.array_equal([1, 88, 1, 1],
                              model.outputs[self.output_blob].shape), \
            "Expected model output shape %s, but got %s" % \
            ([1, 88, 1, 1],
             model.outputs[self.output_blob].shape)

    def preprocess(self, frame, rois):
        assert len(frame.shape) == 4, "Frame shape should be [1, c, h, w]"
        inputs = cut_rois(frame, rois, "min_max")
        inputs = [resize_input(input, self.input_shape) for input in inputs]
        return inputs

    def enqueue(self, input, seq):
        return super(LicenseNumberExtractor, self).enqueue({self.input_img_blob: input, self.input_seq_blob: seq})

    def start_async(self, frame, rois):
        inputs = self.preprocess(frame, rois)
        seq = np.array([0] + [1] * 87)[:, np.newaxis]
        for input in inputs:
            self.enqueue(input, seq)

    def get_license_numbers(self):
        outputs = self.get_outputs()
        results = [LicenseNumberExtractor.Result(out[self.output_blob].reshape(88).astype(int)) \
                      for out in outputs]
        return results
