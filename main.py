import arviz as az
import pymc as pm
from pymc.math import switch, ge
import pandas as pd
import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt

data = pd.read_json("./gmail_dataset.jsonl")
print(data.head())