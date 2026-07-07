# End-to-End Pipeline Flow

The pipeline answers: **"Do these two product records (from Amazon and Google) refer to the same real-world product?"** It's a binary classification problem.

```
tableA.csv (Amazon) --+
                      +-> Normalize -> Blocking -> Feature Engineering -> LightGBM -> Calibration -> Evaluation
tableB.csv (Google) --+
```

## 1. Normalization (`normalize.py`)

Every text field goes through:

1. **Lowercase** — "Adobe Photoshop" becomes "adobe photoshop"
2. **Strip punctuation** — "Norton™ Anti-Virus" becomes "Norton Anti Virus"
3. **Split digit/letter boundaries** — "v5" becomes "v 5", "CS3" becomes "CS 3"
4. **Remove stop/legal tokens** — strips "inc", "ltd", "corp", "the", "a", "and", "of", etc.
5. **Collapse whitespace** — multiple spaces become one, leading/trailing spaces removed

This ensures downstream string comparisons aren't thrown off by casing, punctuation, or filler words. For example, "The Best Software Inc." and "best software" both normalize to "best software".

## 2. Blocking (`blocking.py`)

### The Problem

Table A has ~1,363 products. Table B has ~3,226 products. The full cartesian product A x B is **~4.4 million pairs**. Computing 20 features for every pair would be wasteful — most pairs are obviously not matches.

### The Solution: TF-IDF Cosine Blocking

1. **Vectorize** normalized titles from both tables using character-level TF-IDF (character 3-grams like "ado", "dob", "obe" for "adobe")
2. **L2-normalize** the sparse TF-IDF vectors so each row has unit length
3. **Dot product** between the two normalized matrices gives cosine similarity for all pairs at once (sparse matrix multiplication, so only non-zero entries are computed)
4. **Filter** — keep only pairs with similarity >= 0.1 (the configured threshold)

### Result

- 4.4M pairs reduced to **~225K candidates** (94.9% reduction)
- **Recall ceiling = 1.000** — all 1,167 labeled true matches survived blocking. No true match was dropped

The recall ceiling is measured and reported separately because the blocker imposes a hard limit: no downstream model can recover pairs the blocker drops. A perfect recall ceiling means blocking is not the bottleneck.

## 3. Feature Engineering (`features.py`)

For each candidate pair, **20 features** are computed across 4 families.

### String Features (9 features)

Computed on product titles after normalization.

| Feature | What it measures |
|---|---|
| `jaro_winkler_raw` | Character-level similarity on raw (unnormalized) titles. Rewards matching prefixes |
| `jaro_winkler_norm` | Same metric on normalized titles. Removes noise so "The Best Software Inc" vs "best software" score high |
| `token_sort_ratio` | Sorts tokens alphabetically then computes fuzzy ratio. Catches reorderings like "photoshop cs3 adobe" vs "adobe photoshop cs3" |
| `token_set_ratio` | Compares intersection of tokens vs remainder. Handles subset relationships like "microsoft word" vs "microsoft word 2007 upgrade pc" |
| `partial_ratio` | Best substring match — highest-scoring alignment of the shorter string within the longer one |
| `levenshtein_norm` | Edit distance normalized by max length, flipped to similarity (1.0 = identical) |
| `trigram_jaccard` | Jaccard overlap of character 3-grams. Robust to small typos |
| `numeric_token_overlap` | Jaccard overlap of digit-only tokens. Critical for matching software version/model numbers |
| `length_ratio` | `min(len_a, len_b) / max(len_a, len_b)`. Very different lengths suggest different products |

### Embedding Features (2 features)

Cosine similarity of sentence-transformer vectors (`all-MiniLM-L6-v2`). Embeddings are precomputed once for all records and cached to `artifacts/embeddings_a.npz` and `artifacts/embeddings_b.npz`.

| Feature | What it measures |
|---|---|
| `emb_name_cosine` | Semantic similarity of titles. Catches meaning-level matches that string metrics miss, like abbreviations or synonyms |
| `emb_record_cosine` | Semantic similarity of full serialized records ("title \| manufacturer \| price" concatenated). Incorporates structured context |

### Structured Features (7 features)

Computed from price and manufacturer fields. NaN values are passed through to LightGBM, which handles missingness natively.

| Feature | What it measures |
|---|---|
| `price_abs_diff` | Absolute difference between the two prices |
| `price_log_ratio` | Absolute log ratio of prices. Scale-invariant: $10 vs $20 matters more than $500 vs $510 |
| `price_rel_diff` | Absolute price difference divided by the larger price (0 = same price, 1 = one is free) |
| `manufacturer_exact` | 1.0 if normalized manufacturers match exactly, 0.0 otherwise, NaN if either is missing |
| `manufacturer_jw` | Jaro-Winkler on normalized manufacturer names |
| `has_price_both` | 1.0 if both records have a non-null price. Missingness indicator for LightGBM |
| `has_manufacturer_both` | Same for manufacturer. Table B is 89% missing on manufacturer, so this flag tells the model when manufacturer features are informative vs noise |

### Interaction Features (2 features)

Cross-field signals combining string and structured features.

| Feature | What it measures |
|---|---|
| `high_name_sim_price_mismatch` | 1.0 if titles are very similar (JW > 0.9) BUT prices differ a lot (relative diff > 0.5). Flags "same name, very different price" — often different editions/versions |
| `disagreeing_fields` | Count (0-2) of structured fields that disagree: price relative diff > 0.3 counts as 1, manufacturer mismatch counts as 1 |

## 4. Training (`train.py`)

The model is **LightGBM** (gradient boosted decision trees).

### Configuration

```yaml
n_estimators: 1000        # max boosting rounds
learning_rate: 0.05       # shrinkage per tree
num_leaves: 31            # max leaves per tree
early_stopping_rounds: 50 # stop if no improvement for 50 rounds
```

### How It Works

1. **Objective**: Binary cross-entropy (log loss). The model learns to output log-odds of "match"
2. **Class imbalance**: `scale_pos_weight` is auto-computed as `n_neg / n_pos` (~8.83). This upweights the minority "match" class so the model doesn't just predict "non-match" for everything
3. **Boosting**: Trees are added sequentially. Each new tree fits the residual errors from all previous trees. The first tree uses shrinkage=1.0 to set a reasonable base prediction; subsequent trees use 0.05
4. **Early stopping**: Training monitors validation log loss and stops if it doesn't improve for 50 consecutive rounds. Configured for up to 1000 rounds, training stopped at **22 trees**
5. **Output**: The trained model is saved to `artifacts/model.txt` (see `MODEL_FORMAT.md` for a field-by-field explanation)

### Feature Importance (gain-based)

After training, the top features by LightGBM gain:

| Rank | Feature | Importance |
|---:|---|---:|
| 1 | token_set_ratio | 26,502 |
| 2 | partial_ratio | 21,539 |
| 3 | price_log_ratio | 15,213 |
| 4 | emb_record_cosine | 11,122 |
| 5 | price_abs_diff | 7,627 |

String similarity dominates, but price signals and embedding cosines contribute meaningfully — validating that structured and embedding features add signal beyond string metrics alone.

## 5. Calibration (`calibrate.py`)

Raw LightGBM scores aren't probabilities — they're uncalibrated log-odds. A raw score of 0.73 doesn't mean "73% chance of match."

### Isotonic Regression

Isotonic regression is fitted on the validation set to learn a monotonic mapping from raw scores to calibrated probabilities. After calibration, a probability of 0.73 means "73% of pairs at this score are true matches."

### Threshold Selection

The decision threshold is chosen by sweeping the precision-recall curve on the validation set and picking the threshold that maximizes F1. The selected threshold is **0.362**.

This enables operational tiers:
- Auto-link above 0.95
- Send to human review between 0.5-0.95
- Auto-reject below 0.5

## 6. Evaluation (`evaluate.py`)

On the held-out test set (2,293 pairs, 234 true matches), the pipeline computes:

### Metrics

| Metric | Model | Baseline (token_sort_ratio) |
|---|---:|---:|
| F1 | 0.613 | 0.413 |
| Precision | 0.545 | 0.348 |
| Recall | 0.701 | 0.509 |
| AUC-PR | 0.621 | 0.335 |
| AUC-ROC | 0.934 | 0.754 |

The LightGBM model outperforms the naive baseline (a single string feature with a validation-tuned threshold) by 48% F1. See `EVALUATION_METRICS.md` for a detailed explanation of each metric.

### Plots

4 plots are generated in `results/`:

| Plot | What it shows |
|---|---|
| `pr_curve.png` | Precision-Recall curve across all thresholds, with AUC-PR |
| `confusion_matrix.png` | TP/FP/TN/FN counts at the chosen threshold |
| `calibration_curve.png` | Before vs after isotonic regression, with Brier scores |
| `feature_importance.png` | LightGBM gain-based feature importance |

### Naive Baseline

The baseline uses a single feature (`token_sort_ratio`) as the sole predictor with a validation-tuned threshold. This measures how much value the full 20-feature model + LightGBM adds over the simplest possible approach.
