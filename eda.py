import pandas as pd
import numpy as np
from pandas_profiling import ProfileReport

train = pd.read_csv("data/Train.csv")
test = pd.read_csv("data/Test.csv")

train_pr = ProfileReport(train)
test_pr = ProfileReport(test)

train_pr.to_file("eda/train.html")
test_pr.to_file("eda/test.html")