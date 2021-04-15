import time

from serial import Serial
import serial

s = Serial("COM11", 115200, timeout=1)

while True:
    data = s.readline()
    print(s.inWaiting())
    print("1", data, type(data), len(data))
    s.write("1ab".encode("utf-8"))
    # time.sleep(1)
    # s.write("0ab".encode())