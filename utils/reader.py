import json

from utils.logger import Logger, IconColor, Icon


class ParseConfig:
    def __init__(self, cfg_path, name):
        self.cfg_path = cfg_path
        self.name = name
        self.icon_color = IconColor()
        self.icon = Icon()
        self.logger = Logger("success", "green")

    def read_config(self):
        self.logger.log("Loading " + self.name + " configuration...")
        with open(self.cfg_path, 'r') as f:
            content = json.loads(f.read())
        self.logger.log("Loading " + self.name + " configuration successfully...")
        return content


if __name__ == '__main__':
    arg = ParseConfig('../config/radar.json', 'radar')
    arg.read_config()
