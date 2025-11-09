import arviz as az
import pymc as pm
from pymc.math import switch, ge
import pandas as pd
import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt
from tqdm import tqdm

data = pd.read_json("./gmail_dataset.jsonl", orient="records", lines=True)

Y = data[["id", "from", "date", "url count", "spam token count"]].to_numpy()

senders = Y[:, 1]
unique_senders = np.unique(senders)
url_counts = Y[:, 3].astype(int)
spam_token_counts = Y[:, 4].astype(int)
is_opened = data["opened"].astype(int).to_numpy()

# Collect data and organize emails per unique sender
# Need a list:
# - sender email
# list of email ids with respective url counts, spam token counts, opened status
sender_email_data = {}
for sender in unique_senders:
    sender_indices = np.where(senders == sender)[0]
    sender_url_counts = url_counts[sender_indices]
    sender_spam_token_counts = spam_token_counts[sender_indices]
    sender_is_opened = is_opened[sender_indices]
    sender_email_data[sender] = {
        "sender": sender,
        "email_ids": Y[sender_indices, 0].tolist(),
        "url_counts": sender_url_counts.tolist(),
        "spam_token_counts": sender_spam_token_counts.tolist(),
        "is_opened": sender_is_opened.tolist()
    }

# create spam filter model for each sender
for sender in tqdm(sender_email_data):
    print(sender)
    grp_email_ids = np.array(sender_email_data[sender]["email_ids"])
    grp_url_counts = np.array(sender_email_data[sender]["url_counts"])
    grp_spam_token_counts = np.array(sender_email_data[sender]["spam_token_counts"])
    grp_is_opened = np.array(sender_email_data[sender]["is_opened"])
    
    coords = {"grp_email_ids": grp_email_ids}

    with pm.Model(coords=coords) as m:
        grp_url_counts_data = pm.Data("grp_url_counts_data", grp_url_counts, dims="grp_email_ids")
        grp_spam_token_counts_data = pm.Data("grp_spam_token_counts_data", grp_spam_token_counts, dims="grp_email_ids")
        grp_is_opened_data = pm.Data("grp_is_opened_data", grp_is_opened, dims="grp_email_ids")




    # with pm.Model() as model:

# with pm.Model(coords=coords) as model:
#     url_count_data = pm.Data("url_count_data", url_count, dims="id")


#     t = pm.NegativeBinomial("t", mu=5, alpha=1, observed=spam_token_count[0]) # token count likelihood

#     u = pm.NegativeBinomial("u", mu=2, alpha=1, observed=url_count[0]) # url count likelihood

#     p_o = pm.Beta("p_o", alpha=2, beta=2, observed=is_opened[0]) # opened prior
#     o = pm.Bernoulli("o", p=p_o) # opened likelihood

#     p_y = pm.Beta("p_y", alpha=2, beta=2) # prior prior
#     y = pm.Bernoulli("y", p=p_y, dims="id") # prior

#     trace = pm.sample(1000, tune=1000, target_accept=0.9, return_inferencedata=True)
