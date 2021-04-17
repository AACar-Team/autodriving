import cv2
import numpy as np
import scipy
import torch
from PIL.Image import Image
from torchvision.transforms import transforms

from detection.lane_detection.model.lane_model import parsingNet
from utils.logger import Logger
from utils.reader import ParseConfig


class LaneDetector:
    def __init__(self):
        config = ParseConfig("../config/detection.json", "lane_detector")
        self.cfg = config.read_config()
        self.logger = Logger(icon="sun", color="red")
        self.w = self.cfg["width"]
        self.h = self.cfg["height"]
        assert self.cfg["backbone"] in ['18', '34', '50', '101', '152', '50next', '101next', '50wide', '101wide']
        self.griding_num = self.cfg["griding_num"]
        self.num_lanes = self.cfg["num_lanes"]
        self.cls_num_per_lane = self.cfg["cls_num_per_lane"]
        self.row_anchor = self.cfg["row_anchor"]
        self.use_aux = self.cfg["use_aux"]
        self.img_transforms = transforms.Compose([
            transforms.Resize((288, 800)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        self.logger.log_customize("Loading Lane Net...", icon="success", color="green")
        self.model = parsingNet(cls_dim=(self.griding_num, self.cls_num_per_lane, 4), use_aux=self.use_aux)
        if self.cfg["use_gpu"]:
            self.model = self.model.cuda()
        state_dict = torch.load(self.cfg["model"], map_location='cpu')['model']
        compatible_state_dict = {}
        for k, v in state_dict.items():
            if 'module.' in k:
                compatible_state_dict[k[7:]] = v
            else:
                compatible_state_dict[k] = v
        self.model.load_state_dict(compatible_state_dict, strict=False)
        self.model.eval()
        self.logger.log_customize("Model ready...", icon="success", color="green")
        col_sample = np.linspace(0, 800 - 1, self.griding_num)
        self.col_sample_w = col_sample[1] - col_sample[0]

    def preprocess(self, frame):
        raw = frame
        raw = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
        raw = Image.fromarray(raw)
        processed = self.img_transforms(raw)
        imgs = processed.unsqueeze(0)
        if self.cfg["use_gpu"]:
            imgs = imgs.cuda()
        return frame, imgs

    def detect(self, image):
        if type(image) == np.ndarray:
            image = Image.fromarray(image)
        else:
            raise Exception(
                f"Unresolved input type: {type(image)}. {type(torch.Tensor)} and {type(np.ndarray)} are allowed.")
        image = torch.reshape(torch.unsqueeze(self.preprocess(image), -1), (1, 3, 288, 800))
        return self.model(image)

    def postprocess(self, out, size_processed=(288, 800)):
        out_j = out[0].data.cpu().numpy()
        out_j = out_j[:, ::-1, :]
        prob = scipy.special.softmax(out_j[:-1, :, :], axis=0)
        idx = np.arange(self.griding_num) + 1
        idx = idx.reshape(-1, 1, 1)
        loc = np.sum(prob * idx, axis=0)
        out_j = np.argmax(out_j, axis=0)
        loc[out_j == self.griding_num] = 0
        out_j = loc

        pos = []
        for i in range(out_j.shape[1]):
            if np.sum(out_j[:, i] != 0) > 2:
                for k in range(out_j.shape[0]):
                    if out_j[k, i] > 0:
                        # cv2.circle(frame, ppp, 5, (0, 255, 0), -1)
                        pos.append((int(out_j[k, i] * self.col_sample_w * self.w / size_processed[1]) - 1,
                                    int(self.h * (self.row_anchor[self.cls_num_per_lane - 1 - k] / size_processed[
                                        0])) - 1))
        return pos
