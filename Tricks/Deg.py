import random

import cv2
import numpy as np
import ffmpeg
import torch.nn as nn
import torch


class Deg_pipeline():
    def __init__(self):

        """ Configs """
        self.compress_ratio = []
        self.sample_ratio = [2, 3, 4]

    def random_deg(self):
        pass

    def type_1(self):
        """ Only different compress """
        pass

    def type_2(self):
        """ Different downsamples and then interploate to origin shape """
        ratio = random.choice(self.sample_ratio)

    def type_3(self):
        """ Bicubic downsample and then SR use various VSR to origin shape """
        pass


if __name__ == '__main__':
    x = random.choice([2,3,4], )
    print(x)