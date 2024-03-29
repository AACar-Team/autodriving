import json
import sys

from serial_adapter.meter import Meter
from utils.logger import Logger
from utils.reader import ParseConfig

"""
out:{"wendu": 227, "xinlv": 255, "gaoya": 255, "diya": 255, "jiujing": 16, "data": 248}
"""


class SerialSensor(object):

    def __init__(self):

        config = ParseConfig("../config/meter.json", "meter")
        self.cfg = config.read_config()
        self.serial_port = self.cfg["serial_port"]
        self.baudrate = self.cfg["baudrate"]
        self.byte_size = self.cfg["byte_size"]
        self.timeout = self.cfg["timeout"]
        self.logger = Logger(icon="sun", color="red")
        self.loop = False
        self.iter_data = dict()
        self.meter = Meter(self.serial_port, self.baudrate, self.timeout)

    async def run(self):
        try:
            self.show_hardware_info()
        except:
            self.logger.log("Failed to initialize meter")
            raise ValueError("No such Serial Port to open")
        # self.logger.log_customize("serialnumber-" + self.meter.get_info() + ">Start to initialize radar sensor...",
        #                           "success",
        #                           "green")

        if self.meter.iter_scan() is not None:
            # self.logger.log_customize(
            #     "serialnumber-" + self.meter.get_info() + ">Lidar sensor ready...", "success", "green")
            self.loop = True
            self.logger.log("Enter infinity loop...")
            while self.loop:
                self.iter_data = await self._read()
                print(self.iter_data)

    async def _read(self):
        """
        read data here
        :return: str
        """
        data = self.meter.iter_scan()
        if data is None:
            return None
        return json.loads(data.__next__())

    def show_hardware_info(self):
        info = self.meter.get_info()
        print("=" * 80)
        print("Hardware Name: AACar meter sensor A1")
        print("Firmware Version:", info["firmware"])
        print("Serial Number:", info["serialnumber"])
        print("=" * 80)

    def send_cmd(self, cmd: str):
        ecmd = cmd.encode("utf-8")
        self.meter.send_cmd_internal(ecmd)
        self.meter.flush_output()

    def get_msg(self):
        return "123,123"
