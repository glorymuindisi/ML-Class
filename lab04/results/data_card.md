# Data Card: UCI Adult (OpenML version 2)

## Source
UCI Adult (Census Income) dataset, created by Ronny Kohavi and Barry Becker from the
1994 US Census database. Loaded through OpenML with
`fetch_openml(name='adult', version=2, as_frame=True)`.

## License
The UCI Machine Learning Repository lists Adult under CC BY 4.0. Check the dataset page
before redistributing.

## Size
- Raw: 48842 rows x 15 columns (14 features + target).
- Processed: 34165 train rows (after dropping 24 exact duplicates) and 14653 test rows.
- Target: income >50K is the positive class (23.9% of rows).

## Processing summary
70/30 stratified split before cleaning; mode/median imputation, 1st/99th percentile
clipping and duplicate removal use the training set only; fitted cleaner saved at
`results/preprocessor.joblib`. Derived features: `capital_net`, `hours_per_age`.
Processed file SHA256: `134c1a721928df219d748df4286646d54173b22bf933a9ee3f824231aede1498`.

## Quality checks
- Schema violations (types and ranges): 0
- Duplicate feature rows with conflicting labels: 5 groups (10 rows)
- Random sample of 50 rows saved for manual review; automated impossible values found: 0

## Baseline results (logistic regression, test set)
- Without derived features: accuracy 0.8540, F1 0.6601
- With derived features: accuracy 0.8578, F1 0.6696

## Known biases
- Data comes from the 1994 US Census, so it does not reflect today's incomes or economy.
- Strong imbalance by sex and race (mostly male and White), and the labels show a
  large income gap between groups. A model can learn and repeat these patterns.
- Income is cut at a single $50K threshold, and that value is not adjusted for inflation.
- `capital-gain` is capped at 99999, a placeholder-like top value.
- `fnlwgt` is a census sampling weight, not a personal attribute.
- Missing values (workclass, occupation, native-country) are not random.

## Privacy concerns
The data is public and has no names, but combinations of age, sex, race, country,
occupation and education can act as quasi-identifiers and may re-identify people when
joined with other data. Do not try to link it to other datasets.

## Reproducibility
Run `python run_pipeline.py` (seed 42, settings in `config.yaml`).
