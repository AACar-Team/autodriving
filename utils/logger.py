# 默认日志格式
DEFAULT_LOG_FMT = '%(asctime)s %(filename)s [line:%(lineno)d] %(levelname)s: %(message)s'
# 默认时间格式
DEFUALT_LOG_DATEFMT = '%Y-%m-%d %H:%M:%S.%f'
import platform
from datetime import datetime


class Logger:

    def __init__(self, icon="sun", color="red"):
        self.color = IconColor().font_color[color]
        self.icon = Icon().icon_list[icon]
        self.banner = ""
        self.ret = {}
        self.split_sign = "="
        self.front_sign = ">"
        self.color_font_format = '\033[{};{};{}m'
        self.color_font_without_bg_format = '\033[{};{}m'
        self.color_font_end = '\033[0m'
        self.c = 80

    def print_sys_info(self):
        self.ret["python_version"] = platform.python_version()
        plat_form = platform.platform()
        self.ret["platform"] = plat_form
        self.ret["version"] = platform.version()
        self.ret['version_bit'] = platform.architecture()[0][0:2]
        self.ret["cpu"] = platform.processor()
        print(self.split_sign * self.c)
        print("Current OS:", self.ret["platform"])
        print("Current OS version:", self.ret["version"].split(" ")[0])
        print("Current Python version: ", self.ret["python_version"])
        print("OS CPU info:", self.ret["cpu"])
        print(self.split_sign * self.c)

    def log(self, content):
        if content is not str:
            try:
                content = str(content)
            except TypeError as e:
                print(e.with_traceback())
        now_time = datetime.now()
        formated_time = now_time.strftime(DEFUALT_LOG_DATEFMT)
        print(self.colorize_string_without_background(0, self.icon,
                                                      self.color) + self.front_sign + formated_time + self.front_sign + content)

    def log_customize(self, content, icon, color):
        if content is not str:
            try:
                content = str(content)
            except TypeError as e:
                print(e.with_traceback())
        now_time = datetime.now()
        formated_time = now_time.strftime(DEFUALT_LOG_DATEFMT)
        print(self.colorize_string_without_background(0, Icon().icon_list[icon], IconColor().font_color[color]) + self.front_sign + formated_time + self.front_sign + content)


    def show_banner(self):
        with open("./assets/banner.txt") as f:
            txt = f.readlines()
            for line in txt:
                self.banner += line
        print(self.banner)

    def colorize_string(self, font_style, content, color, background):
        return self.color_font_format.format(font_style, color, background) + content + self.color_font_end

    def colorize_string_without_background(self, font_style, content, color):
        return self.color_font_without_bg_format.format(font_style, color) + content + self.color_font_end


class IconColor:

    def __init__(self):
        self.green = 'green'
        self.red = 'red'
        self.blue = 'blue'
        self.gray = 'gray'
        self.purple = 'purple'
        self.yellow = 'yellow'
        self.font_color = {
            "black": 30,
            "red": 31,
            "green": 32,
            "yellow": 33,
            "blue": 34,
            "pink": 35,
            "cyan": 36,
            "white": 37
        }
        self.background_color = {
            "black": 40,
            "red": 41,
            "green": 42,
            "yellow": 43,
            "blue": 44,
            "pink": 45,
            "cyan": 46,
            "white": 47
        }


class Icon:

    def __init__(self):
        self.icon_list = {
            "sun": '☀',
            "star_hollow": '✩',
            "star": '★',
            "flower": '✿',
            "split": '§',
            "music_1": '♪',
            "music_2": '♬',
            "success": '😄'
        }
