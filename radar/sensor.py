import sys

import pandas as pd
from rplidar import RPLidar

from utils.logger import Logger
from utils.reader import ParseConfig


class LidarSensor(object):

    def __init__(self):
        config = ParseConfig("./config/radar.json", "radar")
        self.cfg = config.read_config()
        self.mode = self.cfg["mode"]
        try:
            self.simulation_data_file = self.cfg["simulation_data_file"]
            self.serial_port = self.cfg["serial_port"]
            self.baudrate = self.cfg["baudrate"]
            self.byte_size = self.cfg["byte_size"]
            self.logger = Logger(icon="sun", color="red")
            self.scan = []
            self.loop = False
            self.iter_list = ()

        except:
            print('ERROR OPENING SENSOR CONNECTION: {}\nWill continue using simulation data...'.format(sys.exc_info()))
            if self.mode == "simulation" and self.simulation_data_file is not None:
                self.simulation_data = pd.read_csv(self.simulation_data_file)

    async def run(self):
        try:
            self.lidar = RPLidar(self.serial_port)
            self.lidar.reset()
            self._data = self.lidar.iter_scans()
            self.show_lidar_info()
        except:
            self.logger.log("Failed to initialize radar...")
            raise ValueError("No such Serial Port to open...")

        self.logger.log_customize(
            "serialnumber-" + self.lidar.get_info()['serialnumber'] + ">Start to initialize radar sensor...", "success",
            "green")
        if self.lidar.iter_scans() is not []:
            self.logger.log_customize(
                "serialnumber-" + self.lidar.get_info()['serialnumber'] + ">Lidar sensor ready...", "success", "green")
            self.loop = True
            self.logger.log("Enter infinity loop...")
            while self.loop:
                self.iter_list = await self._read()
                print(self.iter_list)
        else:
            self.logger.log("Error initialize radar sensor...")
            sys.exit(0)

    def stop(self):
        self.loop = False

    async def _read(self):
        """
        read and process data here
        :return: tuple
        """
        return self._data.__next__()
        # for scan_item in self._data.__next__():
        #     angel_list = []
        #     distance_list = []
        #     for item in scan_item:
        #         _, angel, distance = item
        #         angel_list.append(angel)
        #         distance_list.append(distance)
        #     self.iter_list = (angel_list)

    def show_lidar_info(self):
        info = self.lidar.get_info()
        print("=" * 80)
        firmware = str(info["firmware"]).split("(")[-1].split(")")[0]
        print("Radar Name: Slamtec RPLidar A1M8")
        print("Firmware Version:", firmware)
        print("Serial Number:", info["serialnumber"])
        print("=" * 80)
