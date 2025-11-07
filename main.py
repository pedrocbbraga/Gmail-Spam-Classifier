import arviz as az
import pymc as pm
from pymc.math import switch, ge
import pandas as pd
import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt

data = pd.read_json("./gmail_dataset.jsonl", orient="records", lines=True)

Y = data[["id", "from", "date", "urls", "body"]].to_numpy()

print(Y[:, 4])