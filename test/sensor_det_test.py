import asyncio

from detection.vehicle_detection.vehicle_detector import VehicleDetector
from serial_adapter.serial_sensor import SerialSensor


def init():
    global sensor
    sensor = SerialSensor()


def start_vdet():
    detector = VehicleDetector()
    detector.inference(sensor)


init()
start_vdet()
loop = asyncio.get_event_loop()
loop.create_task(sensor.run())
loop.run_forever()
