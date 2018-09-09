from __future__ import division


class Average(object):

    def __init__(self):
        self.reset()

    def update(self, value, number):
        self.sum += value * number
        self.count += number

    def reset(self):
        self.sum = 0
        self.count = 0

    @property
    def average(self):
        return self.sum / self.count
