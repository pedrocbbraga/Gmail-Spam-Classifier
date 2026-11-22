Gmail Spam Classifier

* ISYE 6420 project that makes use of bayesian hierarchical modeling techniques to classify Gmail inbox data as spam/ham.

What this project does:
* Builds a spam token library from Apache SpamAssassin's available labeled data
* Extracts email features from Gmail: sender, spam token count, URL count, opened/not-opened
* Fits and compares two Naive Bayes mixture models:
    * Sigmoid model (partially-pooled)
    * Beta model (no-pooling)
* Uses PSIS-LOO and PPC to compare models
* Uses the better model (as per LOO) to flag emails whose spam probability exceeds a threshold


IMPORTANT: To use, place "client_secret.json" (from Gmail API OAuth client) in the project root directory.
Client Secret Tutorial: https://www.youtube.com/watch?v=D8DMj2lQMwo


How to run:
0. Run ./scripts/fetch_data.sh - Downloads the Apache SpamAssassin public corpus into data/raw
1. Run create_spamtoken_dataset.py - Processes and cleans Apache SpamAssassin's spam data
2. Run create_spamtoken_library.py - Processes the above-created dataset and extracts a library of spammy tokens
3. Run create_input_dataset.py - Extracts and cleans Gmail data for model processing
4. Run spam_filter.py - Runs both bayesian models, prints metrics, saves plots+dags, and determines spam
5. Run main.py - Takes the spam emails from the chosen model,
                 sends them to a spam label, and removes them from Gmail's inbox
