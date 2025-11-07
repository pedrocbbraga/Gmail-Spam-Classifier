import arviz as az
import pymc as pm
from pymc.math import switch, ge
import pandas as pd
import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt

data = pd.read_json("./gmail_dataset.jsonl", orient="records", lines=True)

Y = data[["id", "from", "date", "url count", "spam token count"]].to_numpy()

senders = Y[:, 1]
unique_senders = np.unique(senders)
url_count = Y[:, 3].astype(int)
spam_token_count = Y[:, 4].astype(int)
is_opened = data["opened"].astype(int).to_numpy()

coords = {"id": data.index}

current_sender = unique_senders[0]

with pm.Model(coords=coords) as model:
    url_count_data = pm.Data("url_count_data", url_count, dims="id")


    t = pm.NegativeBinomial("t", mu=5, alpha=1, observed=spam_token_count[0]) # token count likelihood

    u = pm.NegativeBinomial("u", mu=2, alpha=1, observed=url_count[0]) # url count likelihood

    p_o = pm.Beta("p_o", alpha=2, beta=2, observed=is_opened[0]) # opened prior
    o = pm.Bernoulli("o", p=p_o) # opened likelihood

    p_y = pm.Beta("p_y", alpha=2, beta=2) # prior prior
    y = pm.Bernoulli("y", p=p_y, dims="id") # prior

    trace = pm.sample(1000, tune=1000, target_accept=0.9, return_inferencedata=True)
