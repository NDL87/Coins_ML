import numpy as np
import cv2
import os
#import tensorflow as tf
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
#tf.autograph.set_verbosity(0)
from functions_wolmar import *
from forex_python.converter import CurrencyRates
c = CurrencyRates()
import datetime
from datetime import datetime

#startdate = datetime(2000,1,1)
#enddate = datetime(2023,1,31)

#rate = c.get_rates('RUB')
#print(rate)
#for i in rate:
#    print(i)
date = datetime(2022, 1, 29)
#print(date)
rate = c.get_rate('USD', 'RUB', date)
print(rate)

