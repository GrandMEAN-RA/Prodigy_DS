# Decision Tree Analysis Report

## Summary

- Dataset: `C:\Users\EBUNOLUWASIMI\Dropbox\Portfolio\Internships\prodigy\task 3\Data\bank additional\bank-additional.csv`

- Rows: 4,119  Columns: 21

## Key Findings (summary of runs)

### With `duration` included

- Baseline accuracy: 0.8883

- Baseline precision: 0.4884

- Baseline recall: 0.4667

- Baseline f1: 0.4773

- Baseline roc_auc: 0.7034

- Pruned accuracy: 0.9138

- Pruned precision: 0.6727

- Pruned recall: 0.4111

- Pruned f1: 0.5103

- Pruned roc_auc: 0.8913

- Best params: {'dt__ccp_alpha': 0.0013332674911750234, 'dt__max_depth': None, 'dt__min_samples_leaf': 10}


### Without `duration` (real-world scenario)

- Baseline accuracy: 0.8265

- Baseline precision: 0.2022

- Baseline recall: 0.2000

- Baseline f1: 0.2011

- Baseline roc_auc: 0.5516

- Pruned accuracy: 0.8847

- Pruned precision: 0.4444

- Pruned recall: 0.2222

- Pruned f1: 0.2963

- Pruned roc_auc: 0.6932

- Best params: {'dt__ccp_alpha': 0.0, 'dt__max_depth': None, 'dt__min_samples_leaf': 5}


## Recommendations

- Do not use `duration` in a production pre-call prediction model; it leaks post-hoc information.

- Use the pruned tree parameters (see metrics) or consider ensemble models like RandomForest/GradientBoosting for production.

- Consider cost-sensitive thresholds (tune for recall/precision trade-offs depending on business cost of false positives/negatives).

