from utils.logger import Logger

# Ready all of the preparation
logger = Logger()
logger.show_banner()
logger.print_sys_info()

import asyncio
from radar.sensor import LidarSensor

from utils.reader import ParseConfig

lidar = LidarSensor()

loop = asyncio.get_event_loop()
loop.create_task(lidar.run())
loop.run_forever()


import asyncio
from serial_adapter.serial_sensor import SerialSensor

from utils.reader import ParseConfig

sensor = SerialSensor()

loop = asyncio.get_event_loop()
loop.create_task(sensor.run())
loop.run_forever()

from detection.vehicle_detection.vehicle_detector import VehicleDetector

import os


detector = VehicleDetector()

detector.inference()