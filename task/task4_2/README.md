# Task 4.2 — unsupervised protein-sequence clustering

Cluster the protein sequences without using the biological group labels during fitting. The public training FASTA includes labels only for evaluation and interpretation.

## Output contract

- Read public data from `dataset/task4_2/train.fasta` and `dataset/task4_2/test_features.fasta`.
- Write `outputs/task4_2_predictions.csv`.
- Include exactly one row per held-out FASTA record.
- Required columns: `sequence_id` and `cluster`.
- `sequence_id` values must match the held-out FASTA headers, and `cluster` may contain any consistent cluster labels.
- Do not write duplicate IDs or missing values.
