# Evaluation Metrics

All metrics explained using the actual results from this pipeline on the Amazon-Google benchmark.

## The Confusion Matrix — Ground Truth

The test set has **2,293 pairs**. The model predicted each as "match" or "non-match":

```
                  Predicted
                  Non-match    Match
Actual Non-match    1922        137    <- 2059 actual non-matches
       Match          70        164    <- 234 actual matches
```

The 4 cells have names:

| Cell | Name | Count | Meaning |
|---|---|---:|---|
| Top-left | True Negatives (TN) | 1,922 | Correctly said "non-match" |
| Top-right | False Positives (FP) | 137 | Said "match" but was wrong (false alarm) |
| Bottom-left | False Negatives (FN) | 70 | Said "non-match" but it was actually a match (missed) |
| Bottom-right | True Positives (TP) | 164 | Correctly said "match" |

## Precision and Recall

These answer two different questions.

### Precision = 0.545 — "When the model says 'match', how often is it right?"

```
Precision = TP / (TP + FP) = 164 / (164 + 137) = 164 / 301 = 0.545
```

The model predicted 301 matches. Only 164 were real. 45.5% of its "match" calls were wrong. If you auto-linked all predicted matches into a database, about half the links would be garbage.

### Recall = 0.701 — "Of all real matches, how many did the model find?"

```
Recall = TP / (TP + FN) = 164 / (164 + 70) = 164 / 234 = 0.701
```

There were 234 true matches. The model caught 164 of them. It missed 70 real matches (30%).

### The Precision-Recall Tradeoff

You can't maximize both. The tradeoff depends on the **decision threshold** — the cutoff probability above which the model says "match."

The model outputs a calibrated probability for each pair (e.g., 0.12, 0.47, 0.83). To make a yes/no decision, you pick a threshold and say: "if probability >= threshold, predict match."

If you **lower the threshold** (say "match" more easily):
- Recall goes up — you catch more true matches
- Precision goes down — you also let in more false alarms

If you **raise the threshold** (only say "match" when very confident):
- Precision goes up — fewer false alarms
- Recall goes down — you miss more real matches

### F1 = 0.613

The harmonic mean of precision and recall — a single number that balances both. The calibration step picked the threshold (0.362) that maximizes F1.

```
F1 = 2 * (Precision * Recall) / (Precision + Recall)
   = 2 * (0.545 * 0.701) / (0.545 + 0.701)
   = 0.613
```

## The PR Curve and AUC-PR

The Precision-Recall curve shows what happens as you sweep the decision threshold from high to low:

- **Threshold = 0.95**: Only predict "match" when extremely confident. Very few predictions, almost all correct. Precision ~1.0, recall ~0.05. (top-left of the curve)
- **Threshold = 0.362**: The chosen operating point. Precision = 0.545, recall = 0.701. (middle of the curve)
- **Threshold = 0.01**: Predict "match" for almost everything. You catch nearly all true matches (recall ~1.0) but drown in false alarms (precision ~0.10). (bottom-right of the curve)

Each point on the PR curve is one threshold value. The curve traces all of them continuously.

**AUC-PR = 0.621** is the area under this curve. It summarizes model quality across *all possible thresholds*, not just the one you chose. Think of it as: "on average, when the model ranks a true match higher than a non-match, how precise is it?"

- A perfect model has AUC-PR = 1.0 (the curve hugs the top-right corner)
- A random model on this dataset would have AUC-PR ~ 234/2293 = 0.102 (the base rate of matches)
- This model at 0.621 is well above random but far from perfect

AUC-PR is the better metric for this problem because the classes are imbalanced (only 10.2% matches). It focuses on how well the model ranks the minority class.

## AUC-ROC = 0.934

AUC-ROC answers a different question: "If I pick one random true match and one random true non-match, what's the probability the model scores the match higher?"

A value of 0.934 means the model ranks a true match above a true non-match 93.4% of the time. This sounds great, but AUC-ROC can be misleadingly optimistic with imbalanced data — correctly predicting the 2,059 non-matches is easy and inflates the score. That's why AUC-PR (0.621) tells a more honest story for entity resolution.

## Brier Score = 0.055

Measures how accurate the probability values themselves are (not just the yes/no decisions). It's the mean squared error between the predicted probability and the actual label (0 or 1).

- Brier = 0.0 means perfect calibration
- Brier = 1.0 means worst possible

A Brier score of 0.055 means the predicted probabilities are close to reality: when the model says "0.8 probability of match," roughly 80% of those pairs truly are matches. This is important for operational use — it means you can trust the probabilities to set tiered thresholds (auto-link above 0.95, send to human review between 0.5-0.95, auto-reject below 0.5).

## Summary

| Metric | Value | Question it answers |
|---|---|---|
| **Precision** | 0.545 | "When I say match, am I right?" |
| **Recall** | 0.701 | "Did I find all the real matches?" |
| **F1** | 0.613 | "Balance of precision and recall" |
| **AUC-PR** | 0.621 | "Overall ranking quality (focused on matches)" |
| **AUC-ROC** | 0.934 | "Overall ranking quality (both classes equally)" |
| **Brier** | 0.055 | "How accurate are the probability values themselves?" (lower = better) |

For entity resolution, AUC-PR and F1 are the metrics that matter most. AUC-ROC (0.934) says the model *ranks* well overall, but AUC-PR (0.621) says that when you actually try to *use* those rankings to make decisions on the minority class, there's a meaningful precision/recall tradeoff.
