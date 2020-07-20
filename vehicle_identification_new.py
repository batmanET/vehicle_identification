#!/usr/bin/env python

"""
Created on Wed Jul 8 13:01:10 2020

@author: ARawat4
"""

import logging as log
import os.path as osp
import sys
import time
from argparse import ArgumentParser

import cv2
import numpy as np

from openvino.inference_engine import IENetwork
from src.ie_module import InferenceContext
from src.vehicle_licence_detection import VehicleLicenceDetector
from src.vehicle_detection import VehicleDetector
from src.license_number_extraction import LicenseNumberExtractor
from src.vehicle_attribute import VehicleAttributeDetector

DEVICE_KINDS = ['CPU', 'GPU', 'FPGA', 'MYRIAD', 'HETERO', 'HDDL']
MATCH_ALGO = ['HUNGARIAN', 'MIN_DIST']


def build_argparser():
    parser = ArgumentParser()

    general = parser.add_argument_group('General')
    general.add_argument('-i', '--input', metavar="PATH", default='0',
                         help="(optional) Path to the input video " \
                         "('0' for the camera, default)")
    general.add_argument('-o', '--output', metavar="PATH", default="",
                         help="(optional) Path to save the output video to")
    general.add_argument('--no_show', action='store_true',
                         help="(optional) Do not display output")
    general.add_argument('-tl', '--timelapse', action='store_true',
                         help="(optional) Auto-pause after each frame")
    general.add_argument('-cw', '--crop_width', default=0, type=int,
                         help="(optional) Crop the input stream to this width " \
                         "(default: no crop). Both -cw and -ch parameters " \
                         "should be specified to use crop.")
    general.add_argument('-ch', '--crop_height', default=0, type=int,
                         help="(optional) Crop the input stream to this height " \
                         "(default: no crop). Both -cw and -ch parameters " \
                         "should be specified to use crop.")
    general.add_argument('--match_algo', default='HUNGARIAN', choices=MATCH_ALGO,
                         help="(optional)algorithm for face matching(default: %(default)s)")

    gallery = parser.add_argument_group('Faces database')
    gallery.add_argument('-fg', metavar="PATH", default="face_db",
                         help="Path to the face images directory")
    gallery.add_argument('--run_detector', action='store_true',
                         help="(optional) Use Face Detection model to find faces" \
                         " on the face images, otherwise use full images.")

    models = parser.add_argument_group('Models')
    models.add_argument('-m_vd', metavar="PATH", default="models/vehicle-detection-adas-0002.xml",
                        help="Path to the Vehicle Detection model XML file")
    models.add_argument('-m_ld', metavar="PATH", default="models/vehicle-license-plate-detection-barrier-0106.xml",
                        help="Path to the Licence Plate Detection model XML file")
    models.add_argument('-m_le', metavar="PATH", default="models/license-plate-recognition-barrier-0001.xml",
                        help="Path to the Licence Number Extraction model XML file")
    models.add_argument('-m_va', metavar="PATH", default="models/vehicle-attributes-recognition-barrier-0039.xml",
                        help="Path to the Vehicle Attributes model XML file")
    models.add_argument('-vl_iw', '--vl_input_width', default=0, type=int,
                         help="(optional) specify the input width of detection model " \
                         "(default: use default input width of model). Both -vl_iw and -vl_ih parameters " \
                         "should be specified for reshape.")
    models.add_argument('-vl_ih', '--vl_input_height', default=0, type=int,
                         help="(optional) specify the input height of detection model " \
                         "(default: use default input height of model). Both -vl_iw and -vl_ih parameters " \
                         "should be specified for reshape.")
    
    infer = parser.add_argument_group('Inference options')
    infer.add_argument('-d_vd', default='CPU', choices=DEVICE_KINDS,
                       help="(optional) Target device for the " \
                       "Vehicle Detection model (default: %(default)s)")
    infer.add_argument('-d_ld', default='CPU', choices=DEVICE_KINDS,
                       help="(optional) Target device for the " \
                       "Licence Plate Detection model (default: %(default)s)")
    infer.add_argument('-d_le', default='CPU', choices=DEVICE_KINDS,
                       help="(optional) Target device for the " \
                       "Licence Number Extraction model (default: %(default)s)")
    infer.add_argument('-d_va', default='CPU', choices=DEVICE_KINDS,
                       help="(optional) Target device for the " \
                       "Vehicle Attributes model (default: %(default)s)")
    infer.add_argument('-l', '--cpu_lib', metavar="PATH", default="",
                       help="(optional) For MKLDNN (CPU)-targeted custom layers, if any. " \
                       "Path to a shared library with custom layers implementations")
    infer.add_argument('-c', '--gpu_lib', metavar="PATH", default="",
                       help="(optional) For clDNN (GPU)-targeted custom layers, if any. " \
                       "Path to the XML file with descriptions of the kernels")
    infer.add_argument('-v', '--verbose', action='store_true',
                       help="(optional) Be more verbose")
    infer.add_argument('-pc', '--perf_stats', action='store_true',
                       help="(optional) Output detailed per-layer performance stats")
    infer.add_argument('-t_vl', metavar='[0..1]', type=float, default=0.4,
                       help="(optional) Probability threshold for detections" \
                       "(default: %(default)s)")
    infer.add_argument('-exp_r_vl', metavar='NUMBER', type=float, default=1.15,
                       help="(optional) Scaling ratio for bboxes passed to recognition " \
                       "(default: %(default)s)")

    return parser



class FrameProcessor:
    QUEUE_SIZE = 16

    def __init__(self, args):
        used_devices = set([args.d_vd, args.d_ld, args.d_le, args.d_va])
        self.context = InferenceContext()
        context = self.context
        context.load_plugins(used_devices, args.cpu_lib, args.gpu_lib)
        for d in used_devices:
            context.get_plugin(d).set_config({
                "PERF_COUNT": "YES" if args.perf_stats else "NO"})

        log.info("Loading models")
        licence_detector_net = self.load_model(args.m_ld)
        vehicle_detector_net = self.load_model(args.m_vd)

        assert (args.vl_input_height and args.vl_input_width) or \
               (args.vl_input_height==0 and args.vl_input_width==0), \
            "Both -vl_iw and -vl_ih parameters should be specified for reshape"
        
        if args.vl_input_height and args.vl_input_width :
            licence_detector_net.reshape({"data": [1, 3, args.vl_input_height,args.vl_input_width]})
            vehicle_detector_net.reshape({"data": [1, 3, args.vl_input_height,args.vl_input_width]})
            

        licence_extractor_net = self.load_model(args.m_le)
        vehicle_attr_detection_net = self.load_model(args.m_va)
        
        self.licence_detector = VehicleLicenceDetector(licence_detector_net,
                                          confidence_threshold=args.t_vl,
                                          roi_scale_factor=args.exp_r_vl)
        
        self.vehicle_detector = VehicleDetector(vehicle_detector_net,
                                          confidence_threshold=args.t_vl,
                                          roi_scale_factor=args.exp_r_vl)

        self.licence_extractor = LicenseNumberExtractor(licence_extractor_net)
        self.vehicle_attr_detection = VehicleAttributeDetector(vehicle_attr_detection_net)
        
        self.licence_detector.deploy(args.d_ld, context)
        self.vehicle_detector.deploy(args.d_vd, context)
        self.licence_extractor.deploy(args.d_le, context,
                                       queue_size=self.QUEUE_SIZE)
        self.vehicle_attr_detection.deploy(args.d_va, context,
                                    queue_size=self.QUEUE_SIZE)
        log.info("Models are loaded")

    def load_model(self, model_path):
        model_path = osp.abspath(model_path)
        model_weights_path = osp.splitext(model_path)[0] + ".bin"
        log.info("Loading the model from '%s'" % (model_path))
        assert osp.isfile(model_path), \
            "Model description is not found at '%s'" % (model_path)
        assert osp.isfile(model_weights_path), \
            "Model weights are not found at '%s'" % (model_weights_path)
        model = IENetwork(model_path, model_weights_path)
        log.info("Model is loaded")
        return model

    def process(self, frame):
        assert len(frame.shape) == 3, \
            "Expected input frame in (H, W, C) format"
        assert frame.shape[2] in [3, 4], \
            "Expected BGR or BGRA input"

        orig_image = frame.copy()
        frame = frame.transpose((2, 0, 1)) # HWC to CHW
        frame = np.expand_dims(frame, axis=0)

        self.licence_detector.clear()
        self.vehicle_detector.clear()
        self.licence_extractor.clear()
        self.vehicle_attr_detection.clear()

        
        self.licence_detector.start_async(frame)
        temp, licence_rois = self.licence_detector.get_roi_proposals(frame)
        self.vehicle_detector.start_async(frame)
        vehicle_rois = self.vehicle_detector.get_roi_proposals(frame)
        if self.QUEUE_SIZE < len(vehicle_rois)+len(licence_rois):
            log.warning("Too many faces for processing." \
                    " Will be processed only %s of %s." % \
                    (self.QUEUE_SIZE, len(rois)))
            rois = rois[:self.QUEUE_SIZE]
        
        self.vehicle_attr_detection.start_async(frame, vehicle_rois)
        vehicle_attributes = self.vehicle_attr_detection.get_attributes()

        self.licence_extractor.start_async(frame, licence_rois)
        licence_numbers = self.licence_extractor.get_license_numbers()

        outputs = [[vehicle_rois, vehicle_attributes], [licence_rois, licence_numbers]]

        return outputs


    def get_performance_stats(self):
        stats = {
            'face_detector': self.face_detector.get_performance_stats(),
            'landmarks': self.landmarks_detector.get_performance_stats(),
            'face_identifier': self.face_identifier.get_performance_stats(),
        }
        return stats


class Visualizer:
    BREAK_KEY_LABELS = "q(Q) or Escape"
    BREAK_KEYS = {ord('q'), ord('Q'), 27}


    def __init__(self, args):
        self.frame_processor = FrameProcessor(args)
        self.display = not args.no_show
        self.print_perf_stats = args.perf_stats

        self.frame_time = 0
        self.frame_start_time = time.perf_counter()
        self.fps = 0
        self.frame_num = 0
        self.frame_count = -1

        self.input_crop = None
        if args.crop_width and args.crop_height:
            self.input_crop = np.array((args.crop_width, args.crop_height))

        self.frame_timeout = 0 if args.timelapse else 1

    def update_fps(self):
        now = time.perf_counter()
        self.frame_time = max(now - self.frame_start_time, sys.float_info.epsilon)
        self.fps = 1.0 / self.frame_time
        self.frame_start_time = now

    def draw_text_with_background(self, frame, text, origin,
                                  font=cv2.FONT_HERSHEY_SIMPLEX, scale=1.0,
                                  color=(0, 0, 0), thickness=1, bgcolor=(255, 255, 255)):
        text_size, baseline = cv2.getTextSize(text, font, scale, thickness)
        cv2.rectangle(frame,
                      tuple((origin + (0, baseline)).astype(int)),
                      tuple((origin + (text_size[0], -text_size[1])).astype(int)),
                      bgcolor, cv2.FILLED)
        cv2.putText(frame, text,
                    tuple(origin.astype(int)),
                    font, scale, color, thickness)
        return text_size, baseline

    def draw_licence_detection_roi(self, frame, roi, license):
        # Draw person ROI border
        cv2.rectangle(frame,
                      tuple(roi.position_min), tuple(roi.position_max),
                      (0, 220, 0), 2)
        
        # Draw identity label
        text_scale = 0.4
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize("H1", font, text_scale, 1)
        line_height = np.array([0, text_size[0][1]])
        text = f"{license.license_number}"
        self.draw_text_with_background(frame, text,
                                       roi.position_min - line_height * 0.4,
                                       font, scale=text_scale)


    def draw_detection_roi(self, frame, roi, attribute):
        # Draw face ROI border
        cv2.rectangle(frame,
                      tuple(roi.position_min), tuple(roi.position_max),
                      (0, 220, 0), 2)

        # Draw identity label
        text_scale = 0.4
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize("H1", font, text_scale, 1)
        line_height = np.array([0, text_size[0][1]])
        text = f"Type: {attribute.type} ({attribute.type_confidence}%), Color: {attribute.color} ({attribute.color_confidence}%)"
        self.draw_text_with_background(frame, text,
                                       roi.position_min - line_height * 0.4,
                                       font, scale=text_scale)

    def draw_detections(self, frame, detections):
        for roi, attribute in zip(*detections[0]):
            self.draw_detection_roi(frame, roi, attribute)
        for roi, license in zip(*detections[1]):
            self.draw_licence_detection_roi(frame, roi, license)

    def draw_status(self, frame, detections):
        origin = np.array([10, 10])
        color = (10, 160, 10)
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_scale = 0.5
        text_size, _ = self.draw_text_with_background(frame,
                                                      "Frame time: %.3fs" % (self.frame_time),
                                                      origin, font, text_scale, color)
        self.draw_text_with_background(frame,
                                       "FPS: %.1f" % (self.fps),
                                       (origin + (0, text_size[1] * 1.5)), font, text_scale, color)

        log.debug('Frame: %s/%s, detections: %s, ' \
                  'frame time: %.3fs, fps: %.1f' % \
                     (self.frame_num, self.frame_count, len(detections[-1]), self.frame_time, self.fps))

        if self.print_perf_stats:
            log.info('Performance stats:')
            log.info(self.frame_processor.get_performance_stats())

    def display_interactive_window(self, frame):
        color = (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_scale = 0.5
        text = "Press '%s' key to exit" % (self.BREAK_KEY_LABELS)
        thickness = 2
        text_size = cv2.getTextSize(text, font, text_scale, thickness)
        origin = np.array([frame.shape[-2] - text_size[0][0] - 10, 10])
        line_height = np.array([0, text_size[0][1]]) * 1.5
        cv2.putText(frame, text,
                    tuple(origin.astype(int)), font, text_scale, color, thickness)
        cv2.imshow('Interactive Face recognition', frame)

    def should_stop_display(self):
        key = cv2.waitKey(self.frame_timeout) & 0xFF
        return key in self.BREAK_KEYS

    def process(self, input_stream, output_stream):
        self.input_stream = input_stream
        self.output_stream = output_stream
        while self.input_stream.isOpened():
            has_frame, frame = self.input_stream.read()
            if not has_frame:
                break

            if self.input_crop is not None:
                frame = Visualizer.center_crop(frame, self.input_crop)
            detections = self.frame_processor.process(frame)

            self.draw_detections(frame, detections)
            # self.draw_status(frame, detections)

            if self.output_stream:
                self.output_stream.write(frame)
            if self.display:
                self.display_interactive_window(frame)
                if self.should_stop_display():
                    break
            
            self.update_fps()
            self.frame_num += 1


    @staticmethod
    def center_crop(frame, crop_size):
        fh, fw, fc = frame.shape
        crop_size[0] = min(fw, crop_size[0])
        crop_size[1] = min(fh, crop_size[1])
        return frame[(fh - crop_size[1]) // 2 : (fh + crop_size[1]) // 2,
                     (fw - crop_size[0]) // 2 : (fw + crop_size[0]) // 2,
                     :]

    def run(self, args):
        input_stream = Visualizer.open_input_stream(args.input)
        if input_stream is None or not input_stream.isOpened():
            log.error("Cannot open input stream: %s" % args.input)
        fps = input_stream.get(cv2.CAP_PROP_FPS)
        frame_size = (int(input_stream.get(cv2.CAP_PROP_FRAME_WIDTH)),
                      int(input_stream.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.frame_count = int(input_stream.get(cv2.CAP_PROP_FRAME_COUNT))
        if args.crop_width and args.crop_height:
            crop_size = (args.crop_width, args.crop_height)
            frame_size = tuple(np.minimum(frame_size, crop_size))
        log.info("Input stream info: %d x %d @ %.2f FPS" % \
            (frame_size[0], frame_size[1], fps))
        if args.output == "":
            args.output = "output/{}.avi".format("cam" if args.input == "0" else args.input.split("/")[-1][:-4])
        output_stream = Visualizer.open_output_stream(args.output, fps, frame_size)

        self.process(input_stream, output_stream)

        # Release resources
        if output_stream:
            output_stream.release()
        if input_stream:
            input_stream.release()

        cv2.destroyAllWindows()

    @staticmethod
    def open_input_stream(path):
        log.info("Reading input data from '%s'" % (path))
        stream = path
        try:
            stream = int(path)
        except ValueError:
            pass
        return cv2.VideoCapture(stream)

    @staticmethod
    def open_output_stream(path, fps, frame_size):
        output_stream = None
        if path != "":
            if not path.endswith('.avi'):
                log.warning("Output file extension is not 'avi'. " \
                        "Some issues with output can occur, check logs.")
            log.info("Writing output to '%s'" % (path))
            output_stream = cv2.VideoWriter(path,
                                            cv2.VideoWriter.fourcc(*'MJPG'), fps, frame_size)
        return output_stream


def main():
    args = build_argparser().parse_args()

    log.basicConfig(format="[ %(levelname)s ] %(asctime)-15s %(message)s",
                    level=log.INFO if not args.verbose else log.DEBUG, stream=sys.stdout)

    log.debug(str(args))

    visualizer = Visualizer(args)
    visualizer.run(args)


if __name__ == '__main__':
    main()
