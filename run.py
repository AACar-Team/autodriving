from utils.logger import Logger

# Ready all of the preparation
logger = Logger()
logger.show_banner()
logger.print_sys_info()

import asyncio

from radar.sensor import LidarSensor
from multiprocessing import Process, Queue
from serial_adapter.serial_sensor import SerialSensor
from detection.vehicle_detection.vehicle_detector import VehicleDetector

def start_lidar():
    lidar = LidarSensor()
    loop = asyncio.get_event_loop()
    loop.create_task(lidar.run())
    loop.run_forever()

def start_meter():
    sensor = SerialSensor()
    loop = asyncio.get_event_loop()
    loop.create_task(sensor.run())
    loop.run_forever()

def start_detection():
    vdet = VehicleDetector()
    loop = asyncio.get_event_loop()
    loop.create_task(vdet.inference())
    loop.run_forever()

if __name__ == '__main__':

    process0 = Process(target=start_lidar)
    process1 = Process(target=start_meter)
    process2 = Process(target=start_detection)

    process1.start()