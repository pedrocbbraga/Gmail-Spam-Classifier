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

email_ids = Y[:, 0]
senders = Y[:, 1]
unique_senders, sender_idx = np.unique(senders, return_inverse=True)
url_counts = Y[:, 3].astype(int)
spam_token_counts = Y[:, 4].astype(int)
is_opened = data["opened"].astype(int).to_numpy()

# Collect data and organize emails per unique sender
# Need a list:
# - sender email
# list of email ids with respective url counts, spam token counts, opened status
# sender_email_data = {}
# for sender in unique_senders:
#     sender_indices = np.where(senders == sender)[0]
#     sender_url_counts = url_counts[sender_indices]
#     sender_spam_token_counts = spam_token_counts[sender_indices]
#     sender_is_opened = is_opened[sender_indices]
#     sender_email_data[sender] = {
#         "sender": sender,
#         "email_ids": Y[sender_indices, 0].tolist(),
#         "url_counts": sender_url_counts.tolist(),
#         "spam_token_counts": sender_spam_token_counts.tolist(),
#         "is_opened": sender_is_opened.tolist()
#     }

log_url = np.log1p(url_counts)
log_tokens = np.log1p(spam_token_counts)

n_email = len(email_ids)
n_sender = len(unique_senders)

coords = {"email": np.arange(n_email), "sender": np.arange(n_sender), "class": ["ham", "spam"] }

move_emails_to_spam = []
with pm.Model(coords=coords) as m_sigmoid:
    sender_idx_data = pm.Data("sender_idx", sender_idx, dims="email")

    log_url_data = pm.Data("log_url_data", log_url, dims="email")
    log_token_data = pm.Data("log_token_data", log_tokens, dims="email")
    opened_data = pm.Data("opened_data", is_opened, dims="email")

    mu_alpha = pm.Normal("mu_alpha", mu=0, sigma=5)
    sigma_alpha = pm.HalfCauchy("sigma_alpha", beta=5)
    alpha_s = pm.Normal("alpha_sender", mu=mu_alpha, sigma=sigma_alpha, dims="sender")

    # Prior on spam
    p_y = pm.Deterministic("spam_sender", pm.math.sigmoid(alpha_s), dims="sender") 

    y = pm.Bernoulli("y", p=p_y[sender_idx_data], dims="email")

    # Token likelihood priors
    mu_t = pm.Normal("mu_tokens",
                            mu=0,
                            sigma=5,
                            # transform=pm.distributions.transforms.ordered,
                            dims="class"
                            )
    sigma_t = pm.HalfCauchy("sigma_t", beta=5, dims="class")

    # URL likelihood priors
    mu_u = pm.Normal("mu_url", mu=0, sigma=5, dims="class")
    sigma_u = pm.HalfCauchy("sigma_u", beta=5, dims="class")

    # Opened likelihood prior
    p_o = pm.Beta("p_o", alpha=1, beta=1, dims="class")

    t = pm.Normal("t", mu=mu_t[y], sigma=sigma_t[y], observed=log_token_data, dims="email")
    u = pm.Normal("u", mu=mu_u[y], sigma=sigma_u[y], observed=log_url_data, dims="email")
    o = pm.Bernoulli("o", p=p_o[y], observed=opened_data, dims="email")

    trace_sigmoid = pm.sample(draws=1500,
                        tune=1500,
                        target_accept=0.95,
                        return_inferencedata=True
                        )

    pm.compute_log_likelihood(trace_sigmoid)
    pm.sample_posterior_predictive(trace_sigmoid, extend_inferencedata=True)

with pm.Model(coords=coords) as m_beta:
    sender_idx_data = pm.Data("sender_idx", sender_idx, dims="email")

    log_url_data = pm.Data("log_url_data", log_url, dims="email")
    log_token_data = pm.Data("log_token_data", log_tokens, dims="email")
    opened_data = pm.Data("opened_data", is_opened, dims="email")

    # Prior on spam
    p_y = pm.Beta("spam_sender", alpha=1, beta=1, dims="sender") 

    y = pm.Bernoulli("y", p=p_y[sender_idx_data], dims="email")

    # Token likelihood priors
    mu_t = pm.Normal("mu_tokens",
                            mu=0,
                            sigma=5,
                            # transform=pm.distributions.transforms.ordered,
                            dims="class"
                            )
    sigma_t = pm.HalfCauchy("sigma_t", beta=5, dims="class")

    # URL likelihood priors
    mu_u = pm.Normal("mu_url", mu=0, sigma=5, dims="class")
    sigma_u = pm.HalfCauchy("sigma_u", beta=5, dims="class")

    # Opened likelihood prior
    p_o = pm.Beta("p_o", alpha=1, beta=1, dims="class")

    t = pm.Normal("t", mu=mu_t[y], sigma=sigma_t[y], observed=log_token_data, dims="email")
    u = pm.Normal("u", mu=mu_u[y], sigma=sigma_u[y], observed=log_url_data, dims="email")
    o = pm.Bernoulli("o", p=p_o[y], observed=opened_data, dims="email")

    trace_beta = pm.sample(draws=1500,
                        tune=1500,
                        target_accept=0.95,
                        return_inferencedata=True
                        )

    pm.compute_log_likelihood(trace_beta)
    pm.sample_posterior_predictive(trace_beta, extend_inferencedata=True)


log_like_sum_1 = trace_sigmoid.log_likelihood["t"] + trace_sigmoid.log_likelihood["u"] + trace_sigmoid.log_likelihood["o"]
trace_sigmoid.log_likelihood["obs"] = log_like_sum_1
loo_results_sigmoid = az.loo(trace_sigmoid, var_name="obs")

log_like_sum_2 = trace_beta.log_likelihood["t"] + trace_beta.log_likelihood["u"] + trace_beta.log_likelihood["o"]
trace_beta.log_likelihood["obs"] = log_like_sum_2
loo_results_beta = az.loo(trace_beta, var_name="obs")

print(f"loo model sigmoid: {loo_results_sigmoid}")
print(f"loo model beta: {loo_results_beta}")

compare_dict = {
    "Sigmoid": loo_results_sigmoid,
    "Beta": loo_results_beta
}

loo_compare = az.compare(compare_dict)

print("\nModel Comparison:")
print(loo_compare)

# az.plot_ppc(trace_sigmoid, var_names=["t", "u", "o"])
# plt.title("Sigmoid Model Trace")
# plt.show()

# az.plot_ppc(trace_sigmoid, var_names=["t", "u", "o"])
# plt.title("Beta Model Trace")
# plt.show()

if loo_results_sigmoid.elpd_loo > loo_results_beta.elpd_loo:
    trace = trace_sigmoid
else:
    trace = trace_beta

spam_probs = trace.posterior["y"].mean(dim=["chain", "draw"]).to_numpy()

print("\nSpam Probabilities for each email:")
for email_id, prob in zip(email_ids, spam_probs):
    if prob > 0.8:
        print(f"  [SPAMMMM]   Email {email_id}: {prob:.2f}")
        move_emails_to_spam.append(email_id)
    else:
        print(f"  [HAM]    Email {email_id}: {prob:.2f}")

with open(OUT_FILE, "w") as f:
    f.write("ids\n")
    for id in move_emails_to_spam:
        f.write(f"{id}\n")