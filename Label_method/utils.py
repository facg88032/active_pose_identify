import numpy as np
import random
class Utils:

    def __init__(self):
        pass

    def combine(self,data1,data2):
        if data1.size == 0:
            data1 = data2
        else:
           data1 = np.vstack((data1, data2))
        return data1

    def DataProcess(self,data, indexs):
        process = np.asarray([])
        for index in indexs:
            process =self.combine(process, data[index])
        return process

    def gcd(self,x, y):
        m = x
        while (m > 0):
            m = x % y
            x = y
            y = m
        return x

    def StraifiedRandomSample(self,num_population, num_sample):
        StraifiedSample = []
        stratums = self.gcd(num_population, num_sample)
        interval = int(num_population / stratums)
        stratum_num_sample = int(num_sample / stratums)
        for stratum in range(stratums):
            PartofSample = random.sample(range(stratum * interval, (stratum + 1) * interval), stratum_num_sample)
            PartofSample.sort()
            StraifiedSample.append(PartofSample)
        return StraifiedSample

