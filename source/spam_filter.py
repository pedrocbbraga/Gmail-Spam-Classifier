"""

spam_filter.py: This module runs the naive bayes hierarchical mixture models to classify spam emails.

There are two models:
1. Sigmoid model: partial-pooling model with global shrinkage on sender spamminess
2. Beta mode: no-pooling with a beta prior on sender spamminess

Inputs: clean email data in JSONL format
Output: list of email ids to send to spam

Usage: e.g., from the root Gmail-Spam-Classifier directory, python3 source/spam_filter.py --N 500

"""

import arviz as az
import pymc as pm
import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt
from config import dataset_path, emails_to_move_path, arviz_plots_path, dags_path

parser = argparse.ArgumentParser()
parser.add_argument("--N", type=int, default=500)
args = parser.parse_args()

N = args.N
OUT_FILE = emails_to_move_path(N)
IN_FILE = dataset_path(N)

# Read data
data = pd.read_json(IN_FILE, orient="records", lines=True)
Y = data[["id", "from", "date", "url count", "spam token count"]].to_numpy()

# Extract variables
email_ids = Y[:, 0]
senders = Y[:, 1]
unique_senders, sender_idx = np.unique(senders, return_inverse=True)
url_counts = Y[:, 3].astype(int)
spam_token_counts = Y[:, 4].astype(int)
is_opened = data["opened"].astype(int).to_numpy()

# Convert to logs
log_url = np.log1p(url_counts)
log_tokens = np.log1p(spam_token_counts)

# Standardize
log_url_std = (log_url - log_url.mean()) / log_url.std()
log_tokens_std = (log_tokens - log_tokens.mean()) / log_tokens.std()

n_email = len(email_ids)
n_sender = len(unique_senders)

counts = np.bincount(sender_idx)
# print(counts)
low_sender_mask = counts < 5
low_senders = np.where(low_sender_mask)[0]
# print(low_senders)

coords = {"email": np.arange(n_email), "sender": np.arange(n_sender), "class": ["ham", "spam"] }

with pm.Model(coords=coords) as m_sigmoid:
    sender_idx_data = pm.Data("sender_idx", sender_idx, dims="email")

    log_url_data = pm.Data("log_url_data", log_url_std, dims="email")
    log_token_data = pm.Data("log_token_data", log_tokens_std, dims="email")
    opened_data = pm.Data("opened_data", is_opened, dims="email")

    mu_alpha = pm.Normal("mu_alpha", mu=0, sigma=2)
    sigma_alpha = pm.HalfNormal("sigma_alpha", sigma=2)
    alpha_s = pm.Normal("alpha_s", mu=mu_alpha, sigma=sigma_alpha, dims="sender")

    # Prior on spam
    p_y = pm.Deterministic("spam_sender", pm.math.sigmoid(alpha_s), dims="sender") 

    y = pm.Bernoulli("y", p=p_y[sender_idx_data], dims="email")

    # Token likelihood priors
    mu_t = pm.Normal("mu_t",
                            mu=0,
                            sigma=2,
                            transform=pm.distributions.transforms.ordered,
                            dims="class"
                            )
    sigma_t_raw = pm.HalfNormal("sigma_t_raw", sigma=1, dims="class")
    sigma_t = pm.Deterministic("sigma_t", 0.01 + sigma_t_raw, dims="class")

    # URL likelihood priors
    mu_u = pm.Normal("mu_u", mu=0, sigma=2, dims="class")
    sigma_u_raw = pm.HalfNormal("sigma_u_raw", sigma=1, dims="class")
    sigma_u = pm.Deterministic("sigma_u", 0.01 + sigma_u_raw, dims="class")

    # Opened likelihood prior
    p_o = pm.Beta("p_o", alpha=1, beta=1, dims="class")

    t = pm.Normal("t", mu=mu_t[y], sigma=sigma_t[y], observed=log_token_data, dims="email")
    u = pm.Normal("u", mu=mu_u[y], sigma=sigma_u[y], observed=log_url_data, dims="email")
    o = pm.Bernoulli("o", p=p_o[y], observed=opened_data, dims="email")

    trace_sigmoid = pm.sample(draws=2000,
                        tune=1500,
                        chains=4,
                        target_accept=0.99,
                        init="jitter+adapt_diag",
                        nuts={"max_treedepth": 15},
                        return_inferencedata=True,
                        initvals={"mu_t": np.array([-1.0, 1.0])}
                        )

    pm.compute_log_likelihood(trace_sigmoid)
    pm.sample_posterior_predictive(trace_sigmoid, var_names=["y", "t", "u", "o"], extend_inferencedata=True)

with pm.Model(coords=coords) as m_beta:
    sender_idx_data = pm.Data("sender_idx", sender_idx, dims="email")

    log_url_data = pm.Data("log_url_data", log_url_std, dims="email")
    log_token_data = pm.Data("log_token_data", log_tokens_std, dims="email")
    opened_data = pm.Data("opened_data", is_opened, dims="email")

    # Prior on spam
    p_y = pm.Beta("spam_sender", alpha=1, beta=1, dims="sender") 

    y = pm.Bernoulli("y", p=p_y[sender_idx_data], dims="email")

    # Token likelihood priors
    mu_t = pm.Normal("mu_t",
                            mu=0,
                            sigma=2,
                            transform=pm.distributions.transforms.ordered,
                            dims="class"
                            )
    sigma_t_raw = pm.HalfNormal("sigma_t_raw", sigma=1, dims="class")
    sigma_t = pm.Deterministic("sigma_t", 0.01 + sigma_t_raw, dims="class")

    # URL likelihood priors
    mu_u = pm.Normal("mu_u", mu=0, sigma=2, dims="class")
    sigma_u_raw = pm.HalfNormal("sigma_u_raw", sigma=1, dims="class")
    sigma_u = pm.Deterministic("sigma_u", 0.01 + sigma_u_raw, dims="class")

    # Opened likelihood prior
    p_o = pm.Beta("p_o", alpha=1, beta=1, dims="class")

    t = pm.Normal("t", mu=mu_t[y], sigma=sigma_t[y], observed=log_token_data, dims="email")
    u = pm.Normal("u", mu=mu_u[y], sigma=sigma_u[y], observed=log_url_data, dims="email")
    o = pm.Bernoulli("o", p=p_o[y], observed=opened_data, dims="email")

    trace_beta = pm.sample(draws=2000,
                        tune=1500,
                        chains=4,
                        target_accept=0.99,
                        init="jitter+adapt_diag",
                        nuts={"max_treedepth": 15},
                        return_inferencedata=True,
                        initvals={"mu_t": np.array([-1.0, 1.0])}
                        )

    pm.compute_log_likelihood(trace_beta)
    pm.sample_posterior_predictive(trace_beta, var_names=["y", "t", "u", "o"], extend_inferencedata=True)




""" ---------- HIERARCHICAL MODEL DAGs ---------- """

# Sigmoid model DAG
g_sigmoid = pm.model_to_graphviz(m_sigmoid)
g_sigmoid.render(filename=dags_path("sigmoid_model", N), format="png", cleanup=True)
# g_sigmoid.view()

# Beta model DAG
g_beta = pm.model_to_graphviz(m_beta)
g_beta.render(filename=dags_path("beta_model", N), format="png", cleanup=True)
# g_beta.view()




""" ---------- ARVIZ SUMMARIES ---------- """
sigmoid_posterior = trace_sigmoid.posterior["spam_sender"].values[:, :, low_senders]
beta_posterior = trace_beta.posterior["spam_sender"].values[:, :, low_senders]

# Sigmoid model summary
print(f"SUMMARY sigmoid model:")
print(az.summary(trace_sigmoid, var_names=[
                                            "mu_alpha",
                                            "sigma_alpha",
                                            "mu_t",
                                            "sigma_t_raw",
                                            "mu_u",
                                            "sigma_u_raw",
                                            "p_o"
                                            ]))
print(az.summary(trace_sigmoid, var_names=["spam_sender"], coords={"sender": low_senders}))
# print(az.summary(trace_sigmoid, var_names=[
#                                             "y"
#                                             ]))

# Beta model summary
print(f"SUMMARY beta model:")
print(az.summary(trace_beta, var_names=[
                                        "mu_t",
                                        "sigma_t_raw",
                                        "mu_u",
                                        "sigma_u_raw",
                                        "p_o"
                                        ]))
print(az.summary(trace_beta, var_names=["spam_sender"], coords={"sender": low_senders}))
# print(az.summary(trace_beta, var_names=[
#                                         "y"
#                                         ]))




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




""" ---------- DISAGREEMENT CHECK ---------- """
print("Posterior predictive comparison between models:")
pp_sigmoid = trace_sigmoid.posterior_predictive["y"].values.mean(axis=(0,1))
pp_beta = trace_beta.posterior_predictive["y"].values.mean(axis=(0,1))
# print(np.abs(pp_sigmoid - pp_beta))

class_sigma = (pp_sigmoid > 0.5).astype(int)
class_beta = (pp_beta > 0.5).astype(int)

disagreements = np.where(class_sigma != class_beta)[0]
print(f"Number of disagreements between models: {len(disagreements)}")




""" ---------- ARVIZ PLOTS ---------- """

# Sigmoid model posteriors
az.plot_posterior(trace_sigmoid, var_names=["mu_alpha", "sigma_alpha", "mu_t", "mu_u", "p_o"], hdi_prob=0.95)
plt.tight_layout()
plt.savefig(arviz_plots_path("sigmoid_model_posteriors", N))
plt.close()

# Sigmoid model PPC
az.plot_ppc(trace_sigmoid, var_names=["t", "u", "o"])
plt.savefig(arviz_plots_path("sigmoid_model_ppc", N))
plt.close()

# Beta model posteriors
az.plot_posterior(trace_beta, var_names=["mu_t", "mu_u", "p_o"], hdi_prob=0.95)
plt.tight_layout()
plt.savefig(arviz_plots_path("beta_model_posteriors", N))
plt.close()

# Beta model PPC
az.plot_ppc(trace_beta, var_names=["t", "u", "o"])
plt.savefig(arviz_plots_path("beta_model_ppc", N))
plt.close()




""" ---------- OUTPUT SPAM EMAILS ---------- """
# Pick better model
if loo_results_sigmoid.elpd_loo > loo_results_beta.elpd_loo:
    trace = trace_sigmoid
else:
    trace = trace_beta

spam_probs = trace.posterior["y"].mean(dim=["chain", "draw"]).to_numpy()

print("\nProcessing spam probabilities for each email:")
move_emails_to_spam = []
for email_id, prob in zip(email_ids, spam_probs):
    if prob > 0.9:
        # print(f"  [SPAMMMM]   Email {email_id}: {prob:.2f}")
        move_emails_to_spam.append(email_id)
    # else:
    #     print(f"  [HAM]    Email {email_id}: {prob:.2f}")

with open(OUT_FILE, "w") as f:
    f.write("ids\n")
    for id in move_emails_to_spam:
        f.write(f"{id}\n")