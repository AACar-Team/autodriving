from threading import Thread

# A interface.
class BaseDetector:
    def __init__(self, config_path, model_path, video_port, video_size=None, name=None):
        self.config_path = config_path
        self.model_path = model_path
        self.name = name
        self.video_port = video_port
        self.video_size = video_size
        self.threshold = 0.5

