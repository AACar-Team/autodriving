import os
import pickle
import sys

import cv2
import numpy as np
import torch
from openvino.inference_engine import IECore

from detection.utils.extension import ColorPalette
from detection.utils.models import YOLO, SSD
from detection.utils.pipelines import AsyncPipeline
from utils.logger import Logger
from utils.reader import ParseConfig
from detection.utils.models.utils import Detection


def get_plugin_configs(device, num_streams, num_threads):
    device = str(device)
    config_user_specified = {}

    devices_nstreams = {}
    if num_streams:
        devices_nstreams = {device: num_streams for device in ['cpu', 'gpu'] if device in device} \
            if num_streams.isdigit() \
            else dict(device.split(':', 1) for device in num_streams.split(','))
    if 'cpu' in device:
        if num_threads is not None:
            config_user_specified['CPU_THREADS_NUM'] = str(num_threads)
        if 'cpu' in devices_nstreams:
            config_user_specified['CPU_THROUGHPUT_STREAMS'] = devices_nstreams['CPU'] \
                if int(devices_nstreams['CPU']) > 0 \
                else 'CPU_THROUGHPUT_AUTO'

    if 'gpu' in device:
        if 'gpu' in devices_nstreams:
            config_user_specified['GPU_THROUGHPUT_STREAMS'] = devices_nstreams['GPU'] \
                if int(devices_nstreams['GPU']) > 0 \
                else 'GPU_THROUGHPUT_AUTO'

    return config_user_specified


def calculate_position(bbox: Detection, transform_matrix, warped_size, pix_per_meter):
    pos = np.array((bbox.xmax / 2 + bbox.xmin / 2, bbox.ymax)).reshape(1, 1, -1)
    dst = cv2.perspectiveTransform(pos, transform_matrix).reshape(-1, 1)
    return np.array((warped_size[1] - dst[1]) / pix_per_meter[1])


class ModelDeviceException(Exception):
    '''Basic exception class for Detector'''


class VehicleDetector:
    def __init__(self):
        self.logger = Logger(icon="sun", color="red")
        config = ParseConfig("../config/detection.json", "Vehicle-Detection")
        self.cfg = config.read_config()["vehicle_detection"]
        self.img_size = (self.cfg["width"], self.cfg["height"])
        self.num_streams = self.cfg["num_streams"]
        self.num_threads = self.cfg["num_threads"]
        self.threshold = self.cfg["threshold"]
        self.iou_threshold = self.cfg["iou_threshold"]
        self.camera_id = self.cfg["camera_id"]
        self.label = self.cfg["label_path"]
        self.perspective_transform = list()
        self.pixels_per_meter = tuple()
        self.orig_points = object()
        self.model_path = ""
        self.model_bin = ""
        self.model_xml = ""
        self.UNWARPED_SIZE = tuple()

        self.logger.log_customize("Choosing device to run Detector and Loading model...", icon="success", color="green")
        if self.cfg["use_gpu"] and torch.cuda.is_available():
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')
        if self.cfg["use_torch"]:
            self.build_torch_detector()
        elif not self.cfg["use_gpu"]:
            self.build_openvino_detector()
        else:
            raise ModelDeviceException("Can't loading openvino model with cuda")

        self.logger.log_customize("Initializing camera " + self.camera_id + "...", icon="success", color="green")
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            self.logger.log(('OpenCV: Failed to open capture: ' + str(self.camera_id)))
            sys.exit(1)

        self.perspective_file = self.cfg["distance_camera_parameters"]
        self.logger.log_customize("Loading camera parameter for distance...", icon="success", color="green")
        self.load_camera_parameters("distance")
        self.logger.log_customize("Camera distance parameters ready...", icon="success", color="green")

    def build_torch_detector(self):
        self.model_path = self.cfg["model"]
        self.logger.log_customize("Loading model...", icon="success", color="green")
        self.model = torch.load(self.cfg["model"])
        if self.cfg["FP16"]:
            self.model.half()
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.logger.log_customize("VehicleDetector Ready....", icon="success", color="green")

    def build_openvino_detector(self):
        self.logger.log_customize("Loading model...", icon="success", color="green")
        self.model_xml = self.cfg["model_xml"]
        if not os.path.isfile(self.model_xml):
            self.logger.log(f"{self.model_xml} dose not exist...")
            raise ValueError(f"{self.model_xml} dose not exist...")
        self.model_bin = self.model_xml.split(".")[0] + ".bin"
        self.logger.log_customize("Initializing Inference Engine...", icon="success", color="green")
        ie = IECore()
        # model = YOLO(ie, self.model_xml, threshold=self.threshold, iou_threshold=self.iou_threshold, labels=self.label)
        model = SSD(ie, self.model_xml, labels=self.label)
        self.model = model
        plugin_config = get_plugin_configs(self.device, self.num_streams, self.num_threads)
        self.logger.log_customize("Initializing Detector AsyncPipeline...", icon="success", color="green")

        self.detector_pipeline = AsyncPipeline(ie, model, plugin_config, device=self.device, max_num_requests=1)
        self.palette = ColorPalette(len(model.labels) if model.labels else 100)

        self.logger.log_customize("VehicleDetector Ready....", icon="success", color="green")

    def load_camera_parameters(self, name):
        if name == "distance":
            with open(self.perspective_file, 'rb') as f:
                perspective_data = pickle.load(f)
            self.perspective_transform = perspective_data["perspective_transform"]
            self.pixels_per_meter = perspective_data['pixels_per_meter']
            self.orig_points = perspective_data["orig_points"]
            self.UNWARPED_SIZE = 500, 600

    def preprocess(self, img):
        raw = img.copy()
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.half()  # 半精度
        img /= 255.0  # 图像归一化
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        return raw, img

    def draw_result(self, frame, result):
        labels = self.model.labels
        for detection in result:
            if detection.score > self.threshold:
                distance = calculate_position(bbox=detection,
                                              transform_matrix=self.perspective_transform,
                                              warped_size=self.UNWARPED_SIZE,
                                              pix_per_meter=self.pixels_per_meter)
                xmin = max(int(detection.xmin), 0)
                ymin = max(int(detection.ymin), 0)
                xmax = min(int(detection.xmax), self.img_size[0])
                ymax = min(int(detection.ymax), self.img_size[1])
                class_id = int(detection.id)
                color = self.palette[class_id]
                det_label = labels[class_id - 1] if labels and len(labels) >= class_id else '#{}'.format(class_id)
                print('{:^9} | {:10f} | {:4} | {:4} | {:4} | {:4} | {:2} '
                      .format(det_label, detection.score, xmin, ymin, xmax, ymax, round(distance[0], 2)))
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
                cv2.putText(frame, '{} {:.1%}'.format(det_label, detection.score),
                            (xmin, ymin - 7), cv2.FONT_HERSHEY_PLAIN, 1.5, color, 2)
                cv2.putText(frame, 'dis:{:.2f}'.format(round(distance[0], 2)),
                            (xmin, ymin + (ymax - ymin) + 20), cv2.FONT_HERSHEY_PLAIN, 1.5, color, 2)
        return frame

    def inference(self):
        next_frame_id = 0
        next_frame_id_to_show = 0
        while self.cap.isOpened():
            if self.detector_pipeline.callback_exceptions:
                raise self.detector_pipeline.callback_exceptions[0]
            results = self.detector_pipeline.get_result(next_frame_id_to_show)
            if results:
                objects, frame_meta = results
                frame = frame_meta["frame"]
                frame = self.draw_result(frame, objects)
                cv2.imshow("Detection Result", frame)
                cv2.waitKey(1)
                next_frame_id_to_show += 1

            if self.detector_pipeline.is_ready():
                # Get new image/frame
                ret, frame = self.cap.read()
                if not ret:
                    self.cap.release()
                self.detector_pipeline.submit_data(frame, next_frame_id, {"frame": frame})
                next_frame_id += 1
            else:
                self.detector_pipeline.await_any()
        self.detector_pipeline.await_all()

        # Process completed requests
        while self.detector_pipeline.has_completed_request():
            results = self.detector_pipeline.get_result(next_frame_id_to_show)
            if results:
                objects, frame_meta = results
                frame = frame_meta['frame']
                frame = self.draw_result(frame, objects)
                cv2.imshow('Detection Results', frame)
                cv2.waitKey(0)
                next_frame_id_to_show += 1
            else:
                break

    def detect(self, im):
        pass
