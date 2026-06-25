# Model File Format (`artifacts/model.txt`)

The trained LightGBM model is saved in LightGBM's native text format. This document explains every field in the file.

## Header (lines 1-10)

| Line | Field | Meaning |
|---:|---|---|
| 1 | `tree` | File format identifier — this is a tree-based model |
| 2 | `version=v4` | LightGBM serialization format version 4 |
| 3 | `num_class=1` | 1 output per iteration (binary classification uses a single scalar, not one-per-class) |
| 4 | `num_tree_per_iteration=1` | Each boosting round produces 1 tree (multiclass would produce `num_class` trees per round) |
| 5 | `label_index=0` | The label is the 0th target column (only relevant for multi-output) |
| 6 | `max_feature_idx=19` | Highest feature index = 19, meaning **20 features** (0-indexed) |
| 7 | `objective=binary sigmoid:1` | Binary classification with sigmoid activation (sigmoid slope = 1). The model outputs log-odds, sigmoid converts to probabilities |
| 8 | `feature_names=Column_0 ... Column_19` | Generic names because features were passed as a numpy array, not a named DataFrame. They map 1:1 to the `FEATURE_NAMES` list in `features.py` (see [Feature Name Mapping](#feature-name-mapping) below) |
| 9 | `feature_infos=[min:max] ...` | The observed **min and max** of each feature in the training data. LightGBM uses these to build histogram bins. For example, Column_11 (`price_abs_diff`) ranges from `[0:156011.41]`, Column_14 (`manufacturer_exact`) is `[0:1]` |
| 10 | `tree_sizes=3310 3360 ...` | Byte size of each serialized tree block (22 values = 22 trees). Used for fast seeking |

### Feature Name Mapping

`Column_N` in the model file maps to the features defined in `features.py`:

| Column | Feature Name | Family |
|---:|---|---|
| 0 | `jaro_winkler_raw` | String |
| 1 | `jaro_winkler_norm` | String |
| 2 | `token_sort_ratio` | String |
| 3 | `token_set_ratio` | String |
| 4 | `partial_ratio` | String |
| 5 | `levenshtein_norm` | String |
| 6 | `trigram_jaccard` | String |
| 7 | `numeric_token_overlap` | String |
| 8 | `length_ratio` | String |
| 9 | `emb_name_cosine` | Embedding |
| 10 | `emb_record_cosine` | Embedding |
| 11 | `price_abs_diff` | Structured |
| 12 | `price_log_ratio` | Structured |
| 13 | `price_rel_diff` | Structured |
| 14 | `manufacturer_exact` | Structured |
| 15 | `manufacturer_jw` | Structured |
| 16 | `has_price_both` | Structured |
| 17 | `has_manufacturer_both` | Structured |
| 18 | `high_name_sim_price_mismatch` | Interaction |
| 19 | `disagreeing_fields` | Interaction |

## Tree Blocks (repeated 22 times)

The model has **22 trees** (Tree=0 through Tree=21). Early stopping kicked in — training was configured for up to 1000 rounds but stopped at 22 because validation loss stopped improving for 50 consecutive rounds.

Each tree block contains the fields described below. Tree=0 is used as the running example.

### Structure

| Field | Meaning |
|---|---|
| `num_leaves=31` | This tree has 31 leaf nodes (configured via `num_leaves: 31` in the YAML). A tree with 31 leaves has **30 internal (split) nodes** |
| `num_cat=0` | No categorical splits — all features are treated as numeric/continuous |

### Split Definitions (30 values each — one per internal node)

| Field | Meaning |
|---|---|
| `split_feature` | Which feature index (0-19) is used at each internal node. E.g., Tree=0's first split is on feature **3** (`token_set_ratio`) |
| `split_gain` | The loss reduction (information gain) achieved by each split. Higher = more informative. Tree=0's root split has gain **12,282** — the single most informative split across all trees |
| `threshold` | The split cutpoint. For Tree=0 node 0: "if `token_set_ratio` <= 0.8166, go left; else go right". For `price_abs_diff` (Column_11), threshold=83.98 means "if price difference <= $83.98" |
| `decision_type` | Encodes the split direction and handling of missing values as a bitmask. Bit 1 (value 2) = default direction for NaN goes left; Bit 3 (value 8) = default direction for NaN goes right. This is how LightGBM handles the `NaN` values from missing prices/manufacturers |

### Tree Topology (30 values each)

| Field | Meaning |
|---|---|
| `left_child` | Index of the left child for each internal node. **Positive** = another internal node index. **Negative** = a leaf (e.g., `-5` means leaf index 4, since `-1` = leaf 0) |
| `right_child` | Same for right child. Node 0's right_child=`1` means "go to internal node 1" |

Example: the root node of Tree=0 says: *"Split on feature 3 (`token_set_ratio`) at threshold 0.8166. If <= 0.8166, go to internal node 2. If > 0.8166, go to internal node 1."*

### Leaf Statistics (31 values each — one per leaf)

| Field | Meaning |
|---|---|
| `leaf_value` | The **raw prediction** (log-odds contribution) for samples landing in this leaf. All values in Tree=0 are negative (around -1.7 to -2.2) because the base rate is heavily negative — there are ~8.8x more non-matches than matches (`scale_pos_weight: 8.83405`). Less negative values (like -1.68) indicate leaves where samples are *more likely* to be matches |
| `leaf_weight` | Sum of Hessians (second derivatives of the loss) for all samples in this leaf. This is the effective "weight" — higher means more samples with higher certainty fell here. It's used by LightGBM to determine split quality |
| `leaf_count` | Number of training samples that land in this leaf. E.g., leaf 0 has 2,873 samples, leaf 5 has 643. These tell you how much data supports each leaf's prediction |

### Internal Node Statistics (30 values each)

| Field | Meaning |
|---|---|
| `internal_value` | The weighted average prediction of all samples passing through each internal node. Useful for understanding the "direction" at each node before further splitting |
| `internal_weight` | Sum of Hessians at each internal node (sum of its children's weights) |
| `internal_count` | Number of training samples at each node. Node 0 (root) has **6,874** = the entire training set |

### Tree Metadata

| Field | Meaning |
|---|---|
| `is_linear=0` | Not using linear trees (no linear model in each leaf — just a constant value) |
| `shrinkage` | **Learning rate applied to this tree.** Tree=0 has `shrinkage=1` — the first tree gets full weight to set a reasonable base prediction. Trees 1-21 all have `shrinkage=0.05` (the configured `learning_rate`), meaning each subsequent tree's contribution is scaled to 5% to prevent overfitting |

## How Inference Works Across Trees

At prediction time, for a given pair:

1. Drop the 20-feature vector through **each of the 22 trees**
2. In each tree, follow the splits (left/right based on thresholds) until reaching a leaf
3. **Sum up** the leaf values: `prediction = leaf_value_tree0 * 1.0 + leaf_value_tree1 * 0.05 + ... + leaf_value_tree21 * 0.05`
4. Apply **sigmoid**: `probability = 1 / (1 + exp(-prediction))`
5. Apply **isotonic calibration** (from `calibrate.py`) to map to a calibrated probability
6. Compare to the threshold to decide match/non-match

## Footer

### Feature Importances (split count)

After `end of trees`, the file lists how many times each feature was chosen for a split across all 22 trees:

```
Column_12=107    -> price_log_ratio: used in 107 splits
Column_10=78     -> emb_record_cosine
Column_3=77      -> token_set_ratio
Column_11=68     -> price_abs_diff
Column_6=56      -> trigram_jaccard
Column_4=52      -> partial_ratio
Column_9=40      -> emb_name_cosine
Column_5=39      -> levenshtein_norm
Column_2=36      -> token_sort_ratio
Column_8=33      -> length_ratio
Column_0=31      -> jaro_winkler_raw
Column_1=21      -> jaro_winkler_norm
Column_7=10      -> numeric_token_overlap
Column_14=7      -> manufacturer_exact
Column_15=5      -> manufacturer_jw
```

This is **split count** importance (how often each feature was selected). It differs from the **gain-based** importance in `FEATURES.md` — `token_set_ratio` has fewer splits but much higher total gain per split. Features not listed (`price_rel_diff`, `has_price_both`, `has_manufacturer_both`, `high_name_sim_price_mismatch`, `disagreeing_fields`) were **never used** in any split — the model found them redundant.

### Parameters

The full LightGBM parameter dump. Key parameters:

| Parameter | Value | Meaning |
|---|---|---|
| `num_iterations` | 1000 | Max boosting rounds configured (stopped early at 22) |
| `learning_rate` | 0.05 | Shrinkage per tree after Tree=0 |
| `num_leaves` | 31 | Max leaves per tree |
| `min_data_in_leaf` | 20 | A leaf must have >= 20 samples (prevents overfitting to tiny groups) |
| `scale_pos_weight` | 8.83405 | Auto-computed: ~8.8x more negatives than positives, so positive samples get 8.8x weight |
| `use_missing` | 1 | LightGBM natively routes NaN values (from missing prices/manufacturers) using `decision_type` |
| `max_bin` | 255 | Each feature is discretized into up to 255 histogram bins for efficient split finding |
| `seed` | 42 | Reproducibility seed |
| `boost_from_average` | 1 | The initial prediction starts from the training set's average log-odds (not zero), which is why Tree=0's leaf values are all around -2.0 |
| `lambda_l1` | 0 | No L1 regularization |
| `lambda_l2` | 0 | No L2 regularization |
| `max_depth` | -1 | No depth limit (tree complexity controlled by `num_leaves` instead) |

### Tail

`pandas_categorical:null` — no pandas categorical features were used.