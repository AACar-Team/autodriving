from detection.utils.thread import DetectionThread
from detection.utils.detector import BaseDetector

class BehaviorDector(BaseDetector):
    def __init__(self, config_path, model_path, video_port, video_size=None, name=None):
        super(BehaviorDector, self).__init__(config_path, model_path, video_port, video_size=None, name=None)

    def init_thread(self):
        self.thread = DetectionThread()

    def preprocess(self):
        pass

    def postprocess(self):
        pass

    def detect(self):
        pass

    def start(self):

        pass

