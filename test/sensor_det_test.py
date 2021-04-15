import asyncio
from multiprocessing import Process
from multiprocessing.managers import BaseManager

from detection.vehicle_detection.vehicle_detector import VehicleDetector
from serial_adapter.serial_sensor import SerialSensor


def init():
    global sensor
    sensor = SerialSensor()


def start_vdet():
    detector = VehicleDetector()
    detector.inference(sensor)


def start_meter():
    pass


class SensorManager(BaseManager):
    """share object"""



init()

# meter.start()
start_vdet()
loop = asyncio.get_event_loop()
loop.create_task(sensor.run())
loop.run_forever()