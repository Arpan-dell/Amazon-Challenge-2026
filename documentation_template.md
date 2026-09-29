# Business Entity Resolution: Methodology

Amazon ML Challenge 2026. This document describes the pipeline in `code/business_entity_resolution/`
and the exact run that produced `output/matching_results.tsv` and `output/candidate_pairs.tsv`.

---

## 1. Summary

For each Source 1 (S1) business, we find its records in Source 2 and Source 3 (S2/S3) using only
the name and address. The pipeline has five steps:

1. **Normalisation.** Names and addresses are cleaned into canonical token sets. Native-script
   (Indic) text is converted with a dictionary **learned from the training labels**, legal forms are
   separated from the core name, and address components are reordered, expanded and matched to a state.
2. **Blocking.** Six kinds of hash key, built from each record's *rarest* tokens, generate candidate
   pairs. Over-generic buckets are skipped. A cheap fuzzy score then prunes the candidates to about 15
   per S1 entity.
3. **Pair features.** Each candidate pair gets 57 features: fuzzy string scores, IDF-weighted token
   overlap, number agreement, state agreement, name frequency, which keys fired, and how the pair ranks
   against its competitors.
4. **Model.** A LightGBM binary classifier scores each pair. It is validated with GroupKFold by S1
   entity, and the decision threshold is chosen on out-of-fold **macro F0.5** (the challenge metric).
5. **Assignment.** Pairs above the threshold are kept, and each S2/S3 record is given to at most one S1
   entity (the one it scores highest with). S1 entities with no pair above the threshold are predicted
   empty (singletons).

| Result (full training data, out-of-fold) | Value |
|---|---|
| Macro F0.5 | **0.9552** |
| Pair precision / recall | 0.9912 / 0.9005 |
| Singleton accuracy (correctly predicted empty) | 0.9670 |
| Candidate recall after blocking (share of true pairs ever scored) | 0.9352 |

No external data, APIs or pretrained models are used. The only fixed knowledge is hand-written
language dictionaries in `lexicon.py`: legal-form words, street-type abbreviations and state names.

---

## 2. Understanding the data

| | Train | Test |
|---|---|---|
| Source 1 entities | 2.21 M | 1.73 M |
| Source 2 records | 5.03 M | 4.89 M |
| Source 3 records | 5.29 M | 5.08 M |
| Countries | US, India | US, India, **France** (no labels) |

Findings that shaped the design:

- **Matches per S1:** 3.5 on average, up to 11. 5.6% of S1 entities are singletons, which score 1
  only if predicted empty.
- **Records are never shared.** No S2/S3 record belongs to two S1 entities, so one-to-many
  assignment with exclusive ownership is safe.
- **Country always agrees** between matched records. Everything is processed per country, which
  shrinks every join.
- **About 27% of S2/S3 records match no S1 entity.** Many are near-copies of an S1 entity (same name
  with a different house number, or one name word substituted). These are the main source of false
  merges.
- **Noise we observed and handle:**
  - token reordering;
  - legal forms anywhere in the name ("Inc Family Blue Clinic");
  - typos and leetspeak ("8oclaris");
  - domain-style names ("exportstirupati.com");
  - names and states in Devanagari, Gurmukhi, Tamil, Kannada, Gujarati, Bengali and Odia;
  - mojibake;
  - "(ID: 90108)" suffixes;
  - shuffled address components;
  - state names vs abbreviations ("Oregon" / "OR", "Punjab" / "PB" / "ਪੰਜਾਬ");
  - zero-padded house numbers ("0014225");
  - abbreviations ("St", "Rd", "Ngr", "R.");
  - about 3.4% empty addresses.

---

## 3. Methodology

### 3.1 Normalisation (`normalize.py`, `translit.py`, `lexicon.py`)

**Native script → Latin, learned from the labels (`translit.py`).** Generic transliteration
(`unidecode`) turns "प्राइवेट लिमिटेड" into something that no longer matches "Private Limited". So we
learn a dictionary from the training data:

- **Name tokens:** for matched (S1, S2/S3) pairs whose names have the same number of tokens, tokens
  are aligned by position. Each native-script token is mapped to the Latin token it is most often
  aligned with, with at least 2 votes and a majority share.
- **Address components:** a native-script component is mapped to the state of the matching S1 record,
  with at least 50 votes and an 80% share, so that only state names qualify.
- **Fallback:** tokens not in the dictionary fall back to `unidecode`.

On the full training data this learns 1,347 name tokens and 16 state spellings.

**Names**
1. Strip mojibake byte sequences, lowercase, and remove apostrophes.
2. Join dotted abbreviations ("S.A.S." → "sas", "L.L.C." → "llc").
3. Detect domain-style names ("acme.com" → "acme"). Whether a name was a domain is kept as a feature.
4. Fix leetspeak inside tokens that mix letters and digits ("8oclaris" → "boclaris"; ordinals such as
   "4th" are left alone).
5. Take out legal-form words wherever they appear, and record their canonical set (e.g. "ltd pvt")
   separately. Covered: US, Indian and French forms (Inc, Corp, LLC, LLP, PLLC, Pvt, Ltd, SARL, SAS,
   SASU, EURL, SCI, …).
6. Remove stop words. The remaining **core tokens** give three representations:
   - `name_key`: sorted unique tokens, robust to reordering;
   - `name_seq`: tokens in their original order;
   - `compact`: tokens glued together, which matches domain names.

**Addresses**
1. Convert to Latin (as above), lowercase, and split on commas. Components arrive shuffled, so none
   is assumed to be in a fixed position.
2. Detect the state per component using a country-specific map: US state names and codes, and Indian
   states with common abbreviations and aliases. The state is kept as a separate field, not as a token.
3. Canonicalise Indian city names (Bombay → Mumbai, Calcutta → Kolkata, …).
4. Expand street-type abbreviations with a per-country table. Abbreviations collide between countries:
   "st" is "street" in the US but "saint" in France.
5. Strip leading zeros from numbers and pull all digit runs into a `nums` set (house, plot and unit
   numbers).
6. Produce `addr_key` (sorted unique tokens) and `addr_toks` (alphabetic tokens of at least 2
   characters).

Normalisation is vectorised in Polars and runs in chunks of 300k records per source and country,
cached as Parquet.

### 3.2 Candidate generation and blocking (`blocking.py`)

Scoring all pairs within a country (for example 1.3 M × 6.2 M for US train) is impossible, so we
generate candidates by **hash-joining keys**. A pair becomes a candidate if the S1 record and the
S2/S3 record share at least one key.

Keys are built from each record's **rarest tokens**, ranked by document frequency within the
country. The rarest tokens are the most identifying, and they usually survive typos in other tokens,
reordering and legal-suffix changes.

| Key | Built from | Catches |
|---|---|---|
| `name` | all core name tokens, sorted | reordered names, changed legal suffix |
| `compact` | core tokens glued together (≥ 5 characters) | domain names ("fortunebuilderscom") |
| `name_addr` | each of the 3 rarest name tokens × each of the 2 rarest address tokens | a typo in some name tokens |
| `name_pair` | pairs of the 3 rarest name tokens | records with empty addresses |
| `name_state` | each of the 3 rarest name tokens × state | street or house number rewritten |
| `num_addr` | each number (≤ 6 digits) × each of the 2 rarest address tokens | same address, heavily altered name |

- **Bucket cap:** a key bucket is skipped if (S1 records in it) × (S2/S3 records in it) exceeds
  2,000. Such keys are too generic to be informative and would add millions of useless pairs.
- **Pruning:** the union of all keys still gives 70–90 candidates per S1 entity. Each pair gets a
  cheap score, 0.6 × token-set ratio of names + 0.4 × token-set ratio of addresses, and we keep a pair
  only if it is in its S1 entity's **top 25** *and* in its S2/S3 record's **top 5**. For pairs where
  either address is empty, the address term is replaced by a constant 0.2. The second condition
  removes records that clearly belong to a different, better-matching S1 entity.
- **Bookkeeping:** the key bitmask of each pair is kept and becomes model features.

Blocking on the full data (Kaggle run):

| Split / country | S1 | S2+S3 | Candidates before pruning | After pruning | Recall of true pairs (keys / after pruning) |
|---|---|---|---|---|---|
| train / US | 1.32 M | 6.19 M | 113.4 M (85.7 per S1) | 18.5 M (14.0 per S1) | 0.9758 / 0.9559 |
| train / India | 0.88 M | 4.13 M | 64.8 M (73.3 per S1) | 11.6 M (13.1 per S1) | 0.9205 / 0.9040 |
| test / US | 0.66 M | 3.82 M | 61.2 M | 10.5 M (15.8 per S1) | – |
| test / India | 0.81 M | 4.72 M | 67.8 M | 12.0 M (14.9 per S1) | – |
| test / France | 0.26 M | 1.43 M | 18.7 M | 3.9 M (15.1 per S1) | – |

`output/candidate_pairs.tsv` is exactly this pruned test candidate set: 26.4 M pairs, the complete
input to the model. Every final match is a subset of it. Only 502 of 1.73 M test S1 entities have
no candidate.

### 3.3 Feature engineering (`features.py`)

There are 57 features per candidate pair (A = S1 record, B = S2/S3 record), computed in chunks of
500k pairs with RapidFuzz (C++, multi-threaded) and Polars list operations.

| Group | Features |
|---|---|
| Name similarity | token-set ratio (sorted key), ratio, Jaro-Winkler, partial ratio and ratio on the ordered name, ratio and partial ratio on the compact form |
| Name token overlap | Jaccard; **IDF-weighted Jaccard**; IDF of the **rarest shared token**; number of shared tokens; token counts of A and B |
| Address similarity | token-set ratio, ratio, partial ratio |
| Address token overlap | Jaccard, IDF-weighted Jaccard, rarest shared token, number of shared tokens |
| Numbers | Jaccard of number sets, shared count, **conflict flag** (both have numbers, none shared) |
| Exact / categorical | exact name key, exact compact form, legal form equal / different / missing, state equal / different / missing, address empty (A, B), B was a domain name, B comes from Source 3 |
| Name frequency ("chain" signal) | how many records in the country share A's name key, and B's. Common names need stronger address evidence. |
| Blocking keys | which of the 6 keys fired, and how many |
| Competition context | for the cheap score, name token-set ratio and address token-set ratio: this pair's **rank** and **gap to the best** among (i) all candidates of the S1 entity and (ii) all candidate S1 entities of the S2/S3 record; number of candidates on each side |

IDF is computed per country over all records (S1 + S2 + S3) of the split, as log(N / document
frequency).

The most important features, by LightGBM split count, were:
`nm_wjacc, nm_rarest_shared, ad_ratio, name_freq_b, name_freq_a, ad_partial, nm_ratio, ad_tset,
ad_rarest_shared, num_jacc, cp_ratio, ntok_b, ad_n_inter, nm_seq_ratio, nm_jw`.

### 3.4 Model architecture (`pipeline.py`)

- **Classifier:** LightGBM `LGBMClassifier`, binary objective, with 600 trees, learning rate 0.05, 127
  leaves, minimum 50 samples per leaf, 80% row bagging every iteration, 80% feature sampling, and L2
  regularisation of 1.0.
- **Labels:** a candidate pair is positive if it appears in the training ground truth. True pairs lost
  in blocking cannot be learned and count as misses in validation.
- **Training set:** blocking and features use *all* training data, so every token frequency and
  competition feature is computed at full scale. The classifier is fitted on a deterministic 40% of
  training S1 entities (CRC32 hash of the id) to fit Kaggle's 30 GB memory: 12.0 M pairs, 2.86 M
  positives, 881,825 S1 entities.
- **Validation:** 4-fold **GroupKFold grouped by S1 entity**, so all candidates of an entity are in
  the same fold. This mirrors the test setting, where whole entities are unseen.

### 3.5 Decision rule and assignment (`metric.py`)

The macro F0.5 metric averages a per-S1 F0.5, and a singleton scores 1 only if predicted empty. So
the decision rule is tuned directly on the metric rather than on pair accuracy.

- **Policy "all":** keep every pair with probability ≥ threshold.
- **Policy "excl":** keep pairs with probability ≥ threshold, then give each S2/S3 record only to the
  S1 entity it scores highest with. This follows from the finding that no record belongs to two
  entities.

Both policies and thresholds from 0.20 to 0.98 (step 0.02) are evaluated on out-of-fold
probabilities with the official metric. The best is **"excl" at threshold 0.68**. The high threshold
reflects F0.5's preference for precision: a false merge costs about twice as much as a miss.

---

## 4. Validation results

Out-of-fold, full training data (881,825 S1 entities):

| Policy | Best threshold | Macro F0.5 |
|---|---|---|
| all | 0.68 | 0.9551 |
| **excl (chosen)** | **0.68** | **0.9552** |

Breakdown at the chosen setting:
- macro F0.5 0.9552;
- singleton accuracy 0.9670;
- macro F0.5 over matched entities 0.9545;
- pair precision 0.9912 and pair recall 0.9005.

For reference, predicting nothing scores 0.0555.

**Development sample.** A local sample was cut by **whole states**, so that every entity keeps its
same-state look-alike competitors, with separate states for training and testing. On it, the held-out
states scored 0.9655 macro F0.5 (precision 0.986, recall 0.935).

**Test output.** 1,732,544 rows, one per test S1 entity, of which 111,903 are empty and 1,620,641
have matches, with 5,485,059 matched pairs in total. The official validator passes.

| Test country | S1 entities | Predicted empty | Average matches |
|---|---|---|---|
| India | 809,986 | 7.0% | 3.32 |
| US | 663,106 | 6.1% | 3.45 |
| France | 259,452 | 5.7% | 3.42 |

The empty rates are close to the 5.6% singleton rate seen in training, including for France, which
has no labelled data.

---

## 5. Other relevant information

### 5.1 Compute and engineering
- **Hardware:** development ran on an 8 GB laptop using the state-based sample; the full run was a
  Kaggle CPU notebook (4 cores, 30 GB RAM). Everything is CPU-only, with no GPU.
- **Full-run time:** about 2 hours end to end. Train normalisation took 11 min, train blocking and
  features 17 min, model CV and fit 61 min, and test normalisation, features and prediction 35 min.
- **Memory control:** normalisation is chunked per source and country; features are chunked at 500k
  pairs; blocking uses hash joins on 64-bit key hashes with integer record indices.
- **Caching:** every stage writes Parquet caches, so stages re-run independently (e.g. retrain
  without recomputing features).

### 5.2 Error analysis (`errors.py`)
Out-of-fold mistakes were grouped into false merges, missed matches and blocking misses, and printed
with the original text:

- **Most false merges** are S2/S3 records that belong to no S1 entity but look almost identical to
  one. Some are clearly generated look-alikes (house number 219 vs 221, "Nelson Scientific" vs "White
  Scientific" at the same address). Others appear identical to records the ground truth does link.
- **Most missed matches** have empty addresses (only a name to go on), rewritten house numbers, or a
  **completely different brand name at an identical address** (about 2.5% of true pairs).
- **Blocking misses** are concentrated in India at full scale. Indian business names reuse a small
  vocabulary ("Tirupati Exports", "Shree Solutions"), so name buckets exceed the cap country-wide.

### 5.3 Known limitations of this submission
- France has no training labels, and French regions and départements were not mapped to a state
  field. France is scored by the US/India model using name and address similarity alone.
- Records whose name is replaced by an unrelated brand name are found only through `num_addr`, and
  are often pruned by the name-weighted cheap score.
- Name suffixes like "(ID: 90108)" were not stripped in this version.

### 5.4 Reproducing
See `code/business_entity_resolution/README.md`. In short:
```bash
pip install -r code/business_entity_resolution/requirements.txt
python code/business_entity_resolution/run_pipeline.py --data <dataset folder with train/ and test/> \
       --work <work folder> learn normalize-train featurize-train
python code/business_entity_resolution/run_pipeline.py --data <...> --work <...> train --train-fraction 0.4
python code/business_entity_resolution/run_pipeline.py --data <...> --work <...> normalize-test featurize-test predict
```
Outputs are written to `<work folder>/output/matching_results.tsv` and `candidate_pairs.tsv`.
`kaggle_run.ipynb` is the notebook used for the submitted run.
