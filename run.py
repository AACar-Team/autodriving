from utils.logger import Logger

# Ready all of the preparation
logger = Logger()
logger.show_banner()
logger.print_sys_info()

import asyncio
from radar.sensor import LidarSensor
from serial_adapter.serial_sensor import SerialSensor
from detection.vehicle_detection.vehicle_detector import VehicleDetector

lidar = LidarSensor()

sensor = SerialSensor()

detector = VehicleDetector()

loop = asyncio.get_event_loop()
loop.create_task(sensor.run())
loop.run_forever()

loop = asyncio.get_event_loop()
loop.create_task(lidar.run())
loop.run_forever()

detector.inference()
