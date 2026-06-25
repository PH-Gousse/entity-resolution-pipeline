# Feature Catalog

20 features across 4 families, computed for each candidate pair.

## String (9 features)

Computed on product titles after normalization (lowercase, strip punctuation, remove stop/legal tokens, split digit boundaries).

| Feature | What it measures |
|---|---|
| `jaro_winkler_raw` | Character-level similarity on raw (unnormalized) titles. Rewards matching prefixes, good for catching "Adobe Photoshop" vs "Adobe Photoshop CS3". |
| `jaro_winkler_norm` | Same metric on normalized titles. Removes noise so "The Best Software Inc" vs "best software" score high. |
| `token_sort_ratio` | Sorts tokens alphabetically then computes fuzzy ratio. Catches reorderings: "photoshop cs3 adobe" vs "adobe photoshop cs3" score 100. |
| `token_set_ratio` | Compares intersection of tokens vs remainder. Handles subset relationships: "microsoft word" vs "microsoft word 2007 upgrade pc" still scores high. **Top feature by importance.** |
| `partial_ratio` | Best substring match -- highest-scoring alignment of the shorter string within the longer one. Good when one listing buries the product name in a longer description. |
| `levenshtein_norm` | Edit distance normalized by max length, flipped to similarity (1.0 = identical). General-purpose string distance. |
| `trigram_jaccard` | Jaccard overlap of character 3-grams ("ado","dob","obe"...). Robust to small typos and token boundary differences. |
| `numeric_token_overlap` | Jaccard overlap of digit-only tokens. "version 5.0 pro" and "v5.0 standard" share {"5","0"}. Critical for matching software version/model numbers. |
| `length_ratio` | `min(len_a, len_b) / max(len_a, len_b)`. Proxy for whether titles describe things at similar granularity. Very different lengths suggest different products. |

## Embedding (2 features)

Cosine similarity of sentence-transformer vectors (all-MiniLM-L6-v2). Embeddings are precomputed once for all records and cached to `artifacts/`.

| Feature | What it measures |
|---|---|
| `emb_name_cosine` | Semantic similarity of titles. Catches meaning-level matches that string metrics miss, like abbreviations or synonyms. |
| `emb_record_cosine` | Semantic similarity of full serialized records ("title, manufacturer, price" concatenated). Incorporates structured context so two records with matching manufacturer and price are closer than titles alone would suggest. |

## Structured (7 features)

Computed from price and manufacturer fields. NaN values are passed through to LightGBM, which handles missingness natively.

| Feature | What it measures |
|---|---|
| `price_abs_diff` | Absolute difference between the two prices. Large difference = probably different products. |
| `price_log_ratio` | Absolute log ratio of prices. Scale-invariant: a $10 vs $20 difference matters more than $500 vs $510. **3rd most important feature.** |
| `price_rel_diff` | Absolute price difference divided by the larger price. Normalized to 0-1: 0 = same price, 1 = one is free. |
| `manufacturer_exact` | 1.0 if normalized manufacturers match exactly, 0.0 otherwise. NaN if either is missing. |
| `manufacturer_jw` | Jaro-Winkler on normalized manufacturer names. Catches "adobe-education-box" vs "adobe" as a partial match. |
| `has_price_both` | 1.0 if both records have a non-null price. Missingness indicator -- LightGBM can learn that "one price missing" changes how to interpret other features. |
| `has_manufacturer_both` | Same for manufacturer. Table B is 89% missing on manufacturer, so this flag tells the model when manufacturer features are informative vs noise. |

## Interaction (2 features)

Cross-field signals combining string and structured features.

| Feature | What it measures |
|---|---|
| `high_name_sim_price_mismatch` | 1.0 if titles are very similar (JW > 0.9) BUT prices differ a lot (relative diff > 0.5). Flags suspicious pairs: "same name, very different price" often means different editions/versions, not a true match. |
| `disagreeing_fields` | Count (0-2) of structured fields that disagree: price relative diff > 0.3 counts as 1, manufacturer mismatch counts as 1. Summary signal for "how many structured fields say these are different entities." |

## Feature Importance (Amazon-Google)

After training, the top features by LightGBM gain:

| Rank | Feature | Importance |
|---:|---|---:|
| 1 | token_set_ratio | 26,502 |
| 2 | partial_ratio | 21,539 |
| 3 | price_log_ratio | 15,213 |
| 4 | emb_record_cosine | 11,122 |
| 5 | price_abs_diff | 7,627 |
| 6 | trigram_jaccard | 3,433 |
| 7 | emb_name_cosine | 2,173 |
| 8 | token_sort_ratio | 2,086 |
| 9 | levenshtein_norm | 1,837 |
| 10 | jaro_winkler_raw | 1,540 |

String similarity dominates, but price signals and embedding cosines contribute meaningfully -- validating that structured and embedding features add signal beyond string metrics alone.
