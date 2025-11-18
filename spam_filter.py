import arviz as az
import pymc as pm
import pandas as pd
import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt
from tqdm import tqdm

"""
Inputs: clean email data, spam token library
Purpose: runs naive bayes mixture model to determine spam emails
Output: list of email ids to send to spam


"""

OUT_FILE = "emails_to_move.csv"

data = pd.read_json("./gmail_dataset_500.jsonl", orient="records", lines=True)

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
move_emails_to_spam = []
loo_results_all = {}
for sender in tqdm(sender_email_data):
    print(f"Processing {sender}:")
    grp_email_ids = np.array(sender_email_data[sender]["email_ids"])
    grp_url_counts = np.array(sender_email_data[sender]["url_counts"])
    grp_spam_token_counts = np.array(sender_email_data[sender]["spam_token_counts"])
    grp_is_opened = np.array(sender_email_data[sender]["is_opened"])
    # print(grp_spam_token_counts)
    
    coords = {"grp_email_ids": grp_email_ids}

    if len(grp_email_ids) < 6:
        print(f"Skipping {sender}: not enough emails for mixture model.")
        continue

    with pm.Model(coords=coords) as m:
        url_data = pm.Data("url_data", grp_url_counts, dims="grp_email_ids")
        token_data = pm.Data("token_data", grp_spam_token_counts, dims="grp_email_ids")
        opened_data = pm.Data("opened_data", grp_is_opened, dims="grp_email_ids")

        # Prior on spam
        p_y = pm.Beta("p_y", alpha=1, beta=1)

        # Prior on spam tokens
        lambda_t = pm.Gamma("lambda_t",
                            alpha=2,
                            beta=1,
                            shape=2,
                            transform=pm.distributions.transforms.ordered
                            )

        lambda_u = pm.Gamma("lambda_u", alpha=2, beta=1, shape=2)

        p_o = pm.Beta("p_o", alpha=1, beta=1, shape=2)

        y_i = pm.Bernoulli("y_i", p=p_y, dims="grp_email_ids")

        t_i = pm.Poisson("t_i", mu=lambda_t[y_i], observed=token_data)
        u_i = pm.Poisson("u_i", mu=lambda_u[y_i], observed=url_data)
        o_i = pm.Bernoulli("o_i", p=p_o[y_i], observed=opened_data)

        trace = pm.sample(2000,
                          tune=1500,
                          target_accept=0.95,
                          return_inferencedata=True,
                          initvals={"lambda_t": [0.5,15.0]}
                         )
        pm.compute_log_likelihood(trace)

    log_like_sum = trace.log_likelihood["t_i"] + trace.log_likelihood["u_i"] + trace.log_likelihood["o_i"]
    trace.log_likelihood["obs"] = log_like_sum
    loo_results_all[sender] = az.loo(trace, var_name="obs")
    
    email_spam_probs = trace.posterior["y_i"].mean(dim=["chain", "draw"]).to_numpy()
    
    print("Results for " + sender + ":")
    print("\nSpam Probabilities for each email:")
    for email_id, prob in zip(grp_email_ids, email_spam_probs):
        if prob > 0.8:
            print(f"  [SPAMMMM]   Email {email_id}: {prob:.2f}")
            move_emails_to_spam.append(email_id)
        else:
            print(f"  [HAM]    Email {email_id}: {prob:.2f}")
    


    print()
    # print(az.summary(trace))
    # print("Actual data entry values:")
    # print("grp_spam_token_counts", grp_spam_token_counts)
    # print("grp_url_counts", grp_url_counts)
    # print("grp_is_opened", grp_is_opened)

total_n = 0
total_elpd = 0
for sender, loo in loo_results_all.items():
    n = len(sender_email_data[sender]["email_ids"])
    total_n += n
    total_elpd += loo.elpd_loo

elpd_per_email = total_elpd / total_n
print(f"elpd total: {elpd_per_email}")

print(len(move_emails_to_spam))
with open(OUT_FILE, "w") as f:
    f.write("ids\n")
    for id in move_emails_to_spam:
        f.write(f"{id}\n")
