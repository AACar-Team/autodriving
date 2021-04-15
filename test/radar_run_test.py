import asyncio
from radar.sensor import LidarSensor

from utils.reader import ParseConfig

lidar = LidarSensor()

loop = asyncio.get_event_loop()
loop.create_task(lidar.run())
loop.run_forever()


