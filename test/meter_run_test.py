import asyncio
from serial_adapter.serial_sensor import SerialSensor

from utils.reader import ParseConfig

sensor = SerialSensor()

loop = asyncio.get_event_loop()
loop.create_task(sensor.run())
loop.run_forever()

