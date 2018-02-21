import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

from sklearn.datasets import fetch_mldata

mnist = fetch_mldata('MNIST original', data_home="./")
mnist_data = mnist.data
mnist_label = mnist.target
print(mnist_data)
print(mnist_label)
