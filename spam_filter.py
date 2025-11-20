import arviz as az
import pymc as pm
import pandas as pd
import numpy as np
import os
from itertools import combinations
import matplotlib.pyplot as plt
from tqdm import tqdm

"""
Inputs: clean email data, spam token library
Purpose: runs naive bayes mixture model to determine spam emails
Output: list of email ids to send to spam


"""

OUT_FILE = "emails_to_move_500.csv"

data = pd.read_json("./gmail_dataset_500.jsonl", orient="records", lines=True)

Y = data[["id", "from", "date", "url count", "spam token count"]].to_numpy()

email_ids = Y[:, 0]
senders = Y[:, 1]
unique_senders, sender_idx = np.unique(senders, return_inverse=True)
url_counts = Y[:, 3].astype(int)
spam_token_counts = Y[:, 4].astype(int)
is_opened = data["opened"].astype(int).to_numpy()

log_url = np.log1p(url_counts)
log_tokens = np.log1p(spam_token_counts)
log_url_std = (log_url - log_url.mean()) / log_url.std()
log_tokens_std = (log_tokens - log_tokens.mean()) / log_tokens.std()

n_email = len(email_ids)
n_sender = len(unique_senders)

coords = {"email": np.arange(n_email), "sender": np.arange(n_sender), "class": ["ham", "spam"] }

move_emails_to_spam = []
with pm.Model(coords=coords) as m_sigmoid:
    sender_idx_data = pm.Data("sender_idx", sender_idx, dims="email")

    log_url_data = pm.Data("log_url_data", log_url_std, dims="email")
    log_token_data = pm.Data("log_token_data", log_tokens_std, dims="email")
    opened_data = pm.Data("opened_data", is_opened, dims="email")

    mu_alpha = pm.Normal("mu_alpha", mu=0, sigma=2)
    sigma_alpha = pm.HalfNormal("sigma_alpha", sigma=2)
    alpha_s = pm.Normal("alpha_s", mu=mu_alpha, sigma=sigma_alpha, dims="sender")
    # alpha_offset = pm.Normal("alpha_offset", mu=0, sigma=1, dims="sender")
    # alpha_s = pm.Deterministic("alpha_s", mu_alpha + sigma_alpha * alpha_offset, dims="sender")

    # Prior on spam
    p_y = pm.Deterministic("spam_sender", pm.math.sigmoid(alpha_s), dims="sender") 

    y = pm.Bernoulli("y", p=p_y[sender_idx_data], dims="email")

    # Token likelihood priors
    mu_t = pm.Normal("mu_tokens",
                            mu=0,
                            sigma=2,
                            transform=pm.distributions.transforms.ordered,
                            dims="class"
                            )
    # sigma_t = pm.HalfNormal("sigma_t", sigma=2, dims="class")
    sigma_t_raw = pm.HalfNormal("sigma_t_raw", sigma=1, dims="class")
    sigma_u_raw = pm.HalfNormal("sigma_u_raw", sigma=1, dims="class")

    # URL likelihood priors
    mu_u = pm.Normal("mu_url", mu=0, sigma=2, dims="class")
    # sigma_u = pm.HalfNormal("sigma_u", sigma=2, dims="class")
    sigma_t = pm.Deterministic("sigma_t", 0.01 + sigma_t_raw, dims="class")
    sigma_u = pm.Deterministic("sigma_u", 0.01 + sigma_u_raw, dims="class")

    # Opened likelihood prior
    p_o = pm.Beta("p_o", alpha=1, beta=1, dims="class")

    t = pm.Normal("t", mu=mu_t[y], sigma=sigma_t[y], observed=log_token_data, dims="email")
    u = pm.Normal("u", mu=mu_u[y], sigma=sigma_u[y], observed=log_url_data, dims="email")
    o = pm.Bernoulli("o", p=p_o[y], observed=opened_data, dims="email")

    trace_sigmoid = pm.sample(draws=1000,
                        tune=1500,
                        target_accept=0.99,
                        init="jitter+adapt_diag",
                        nuts={"max_treedepth": 15},
                        return_inferencedata=True,
                        initvals={"mu_tokens": np.array([-1.0, 1.0])}
                        )

    pm.compute_log_likelihood(trace_sigmoid)
    pm.sample_posterior_predictive(trace_sigmoid, extend_inferencedata=True)

with pm.Model(coords=coords) as m_beta:
    sender_idx_data = pm.Data("sender_idx", sender_idx, dims="email")

    log_url_data = pm.Data("log_url_data", log_url_std, dims="email")
    log_token_data = pm.Data("log_token_data", log_tokens_std, dims="email")
    opened_data = pm.Data("opened_data", is_opened, dims="email")

    # Prior on spam
    p_y = pm.Beta("spam_sender", alpha=1, beta=1, dims="sender") 

    y = pm.Bernoulli("y", p=p_y[sender_idx_data], dims="email")

    # Token likelihood priors
    mu_t = pm.Normal("mu_tokens",
                            mu=0,
                            sigma=2,
                            transform=pm.distributions.transforms.ordered,
                            dims="class"
                            )
    sigma_t_raw = pm.HalfNormal("sigma_t_raw", sigma=1, dims="class")
    sigma_u_raw = pm.HalfNormal("sigma_u_raw", sigma=1, dims="class")

    # URL likelihood priors
    mu_u = pm.Normal("mu_url", mu=0, sigma=2, dims="class")
    # sigma_u = pm.HalfNormal("sigma_u", sigma=2, dims="class")
    sigma_t = pm.Deterministic("sigma_t", 0.01 + sigma_t_raw, dims="class")
    sigma_u = pm.Deterministic("sigma_u", 0.01 + sigma_u_raw, dims="class")

    # Opened likelihood prior
    p_o = pm.Beta("p_o", alpha=1, beta=1, dims="class")

    t = pm.Normal("t", mu=mu_t[y], sigma=sigma_t[y], observed=log_token_data, dims="email")
    u = pm.Normal("u", mu=mu_u[y], sigma=sigma_u[y], observed=log_url_data, dims="email")
    o = pm.Bernoulli("o", p=p_o[y], observed=opened_data, dims="email")

    trace_beta = pm.sample(draws=1000,
                        tune=1500,
                        target_accept=0.99,
                        init="jitter+adapt_diag",
                        nuts={"max_treedepth": 15},
                        return_inferencedata=True,
                        initvals={"mu_tokens": np.array([-1.0, 1.0])}
                        )

    pm.compute_log_likelihood(trace_beta)
    pm.sample_posterior_predictive(trace_beta, extend_inferencedata=True)




""" ---------- HIERARCHICAL MODEL DAGs ---------- """
# Sigmoid model DAG
g_sigmoid = pm.model_to_graphviz(m_sigmoid)
g_sigmoid.render("sigmoid_model", format="png")
g_sigmoid.view()

# Beta model DAG
g_beta = pm.model_to_graphviz(m_beta)
g_beta.render("beta_model", format="png")
g_beta.view()




""" ---------- ARVIZ SUMMARIES ---------- """
# Sigmoid model summary
print(f"SUMMARY sigmoid model:")
print(az.summary(trace_sigmoid))

# Beta model summary
print(f"SUMMARY beta model:")
print(az.summary(trace_beta))




""" ---------- LOO RESULTS ---------- """
# Sigmoid model LOO
log_like_sum_1 = trace_sigmoid.log_likelihood["t"] + trace_sigmoid.log_likelihood["u"] + trace_sigmoid.log_likelihood["o"]
trace_sigmoid.log_likelihood["obs"] = log_like_sum_1
loo_results_sigmoid = az.loo(trace_sigmoid, var_name="obs")
print("LOO sigmoid model:")
print(loo_results_sigmoid)

# Beta model LOO
log_like_sum_2 = trace_beta.log_likelihood["t"] + trace_beta.log_likelihood["u"] + trace_beta.log_likelihood["o"]
trace_beta.log_likelihood["obs"] = log_like_sum_2
loo_results_beta = az.loo(trace_beta, var_name="obs")
print("LOO beta model:")
print(loo_results_beta)

# LOO comparison
compare_dict = {"Sigmoid": loo_results_sigmoid, "Beta": loo_results_beta}
loo_compare = az.compare(compare_dict)
print("\nModel Comparison:")
print(loo_compare)




""" ---------- ARVIZ PLOTS ---------- """
output_folder = "/Users/pedro/Documents/Georgia Tech OMSCS/ISYE 6420 Bayesian Statistics/Project/Gmail-Spam-Classifier/Results"

# Sigmoid model PPC
az.plot_ppc(trace_sigmoid, var_names=["t", "u", "o"])
plt.savefig(os.path.join(output_folder, "Sigmoid Model PPC.png"))
plt.close()

# Beta model PPC
az.plot_ppc(trace_beta, var_names=["t", "u", "o"])
plt.savefig(os.path.join(output_folder, "Beta Model PPC.png"))
plt.close()




""" ---------- OUTPUT SPAM EMAILS ---------- """
# Pick better model
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