import json
import logging

import serial


class MeterException(Exception):
    '''Basic exception class for Meter'''


class Meter(object):
    def __init__(self, port, baudrate=115200, timeout=1, logger=None):
        self._serial = None
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.data_speed = 0.5
        self.hardware_running = None
        if logger is None:
            logger = logging.getLogger("Meter")
        self.logger = logger
        self.connect()

    def connect(self):
        """
        Connect to meter sensor
        :return:
        """
        if self._serial is not None:
            self.disconnection()

        try:
            self._serial = serial.Serial(
                self.port, self.baudrate,
                parity=serial.PARITY_NONE, stopbits=serial.STOPBITS_ONE,
                timeout=self.timeout)
        except serial.SerialException as e:
            raise MeterException("Failed to connect to the meter "
                                 'due to: %s' % e)

    def disconnection(self):
        """
        Disconnects from the serial port
        :return: none
        """
        if self._serial is None:
            return
        self._serial.close()

    def clean_input(self):
        """
        Clean input buffer by reading all available data
        :return: none
        """
        if self.scanning[0]:
            return 'Cleanning not allowed during scanning process active !'
        self._serial.flushInput()

    def reset(self):
        """
        Resets sensor core, reverting it to a similar state as it has
        just been powered up
        :return: none
        """
        self.logger.info('Reseting the sensor')
        self.clean_input()

    def _read_response(self):
        """
        Reads response packet with length of `dsize` bytes
        :param: dsize
        :return: string
        """
        self.logger.debug('Trying to read response...')
        # while self._serial.inWaiting() < dsize:
        #     time.sleep(0.001)
        data = self._serial.readline()
        self.logger.debug("Received data: %s" % data)
        if data is {}:
            return None
        return data

    def iter_scan(self):
        """
        Start to load data
        :return:
        """
        while True:
            data = self._read_response()
            yield data

    def get_info(self):
        if self._serial.inWaiting() > 0:
            return ('Data in buffer, you can\'t have info ! '
                    'Run clean_input() to emptied the buffer.')
        serialnumber = ""
        info = {
            'model': "Self Maker v0.1",
            'firmware': "1",
            'serialnumber': serialnumber
        }
        return info

    def send_json_data(self):
        pass

    def send_cmd_internal(self, cmd):
        if not isinstance(cmd, bytes):
            raise ValueError("Incorrect command type, needed encoding to byte...")
        self._serial.write(cmd)

    def flush_output(self):
        self._serial.reset_output_buffer()
