import datetime

class Time():
    def __init__(self):
       self.diff = {}
       self.time_begin = {}

    def begin(self, mark):
        self.time_begin[mark] = self.time_now_ori_self()

    def end(self, mark):
        if mark in self.time_begin:
            diff = self.time_now_ori_self() - self.time_begin[mark]
            diff = diff.total_seconds()
            if mark not in self.diff:
                self.diff[mark] = 0
            self.diff[mark] += diff

    def print_diff(self, mark):
        print("Time:" + mark + "\t" + str(self.diff[mark]) + " sec")

    def print_all(self):
        keys = self.diff.keys()
        keys_s = sorted(keys)
        for k in keys_s:
            self.print_diff(k)

    def time_now_ori_self(self):
        return datetime.datetime.now()

    @staticmethod
    def time_now_ori():
        return datetime.datetime.now()

    @staticmethod
    def time_now():
        return str(Time.time_now_ori().strftime("%Y-%m-%d %H:%M:%S"))
