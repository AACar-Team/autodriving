from threading import Thread


class DetectionSocket:

    def __init__(self):
        # todo: open socket port here
        pass

    def send_data(self, data):
        # todo: send the to frontend
        pass


class RadarSocket:

    def __init__(self):
        # todo: open socket port here

        pass

    def send_data(self, data):
        # todo: send the to frontend
        pass


class SerialSocket:
    def __init__(self):
        # todo: open socket port here
        pass

    def send_data(self, data):
        # todo: send the to frontend
        pass


class SocketThread(Thread):

    def __init__(self):
        super().__init__()

    def run(self) -> None:
        # run send function in thread
        pass

    def connect(self):
        # connect corresponding socket
        pass
