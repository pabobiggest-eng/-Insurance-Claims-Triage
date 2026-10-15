# Insurance Claims Triage — Final Project Report

## 1. Executive summary
This project builds and evaluates a binary classification workflow for triaging insurance claims into the target classes in **`is_inpatient`**. The workflow includes data quality assessment, exploratory data analysis (EDA), skewness and kurtosis statistics, feature inclusion/exclusion decisions, class-imbalance diagnostics, ten classifier attempts, model comparison, and a business-use proposal.

**Best completed model by F1, with PR-AUC as tie-breaker:** `Random_Forest`.

| Metric | Result |
|---|---:|
| Accuracy | 0.9620 |
| Balanced accuracy | 0.9398 |
| Precision | 0.6981 |
| Recall | 0.9134 |
| F1 score | 0.7914 |
| ROC-AUC | 0.9867 |
| PR-AUC / Average precision | 0.9172 |
| Matthews correlation coefficient | 0.7794 |

These are test-set metrics from the files and split described below. They are not evidence of real-world financial savings, fraud detection, or deployment readiness.

## 2. Dataset and split
- Dataset rows across train/test files: **846,242**
- Training rows available: **676,992**
- Test rows evaluated: **169,250**
- Columns in training file: **44**
- Target: **`is_inpatient`**
- Predictors retained for modeling: **28**
- Features/columns excluded from predictors: **16** (including target, identifiers, payment fields, post-outcome or unusable columns)
- Feature arithmetic check: **44 total columns = 28 retained predictors + 16 excluded columns**
- Split method: **existing train.csv/test.csv**

### Retained predictor names (28 total)
`BENE_SEX_IDENT_CD`, `BENE_RACE_CD`, `BENE_HI_CVRAGE_TOT_MONS`, `BENE_SMI_CVRAGE_TOT_MONS`, `BENE_HMO_CVRAGE_TOT_MONS`, `PLAN_CVRG_MOS_NUM`, `SP_ALZHDMTA`, `SP_CHF`, `SP_CHRNKIDN`, `SP_CNCR`, `SP_COPD`, `SP_DEPRESSN`, `SP_DIABETES`, `SP_ISCHMCHT`, `SP_OSTEOPRS`, `SP_RA_OA`, `SP_STRKETIA`, `n_dx`, `n_prcdr`, `n_physicians`, `dx_chapter`, `BENE_ESRD_IND`, `SP_STATE_CODE`, `claim_year`, `claim_month`, `age_at_claim`, `is_deceased`, `chronic_count`

### Excluded columns and reasons
- `DESYNPUF_ID` — Identifier; high-cardinality / memorisation risk
- `CLM_ID` — Identifier; high-cardinality / memorisation risk
- `is_inpatient` — Target label; never a predictor
- `CLM_PMT_AMT` — Claim payment/reimbursement; potential target leakage or post-decision information
- `NCH_PRMRY_PYR_CLM_PD_AMT` — Claim payment/reimbursement; potential target leakage or post-decision information
- `NCH_BENE_BLOOD_DDCTBL_LBLTY_AM` — Claim payment/reimbursement; potential target leakage or post-decision information
- `MEDREIMB_IP` — Claim payment/reimbursement; potential target leakage or post-decision information
- `BENRES_IP` — Claim payment/reimbursement; potential target leakage or post-decision information
- `PPPYMT_IP` — Claim payment/reimbursement; potential target leakage or post-decision information
- `MEDREIMB_OP` — Claim payment/reimbursement; potential target leakage or post-decision information
- `BENRES_OP` — Claim payment/reimbursement; potential target leakage or post-decision information
- `PPPYMT_OP` — Claim payment/reimbursement; potential target leakage or post-decision information
- `MEDREIMB_CAR` — Claim payment/reimbursement; potential target leakage or post-decision information
- `BENRES_CAR` — Claim payment/reimbursement; potential target leakage or post-decision information
- `PPPYMT_CAR` — Claim payment/reimbursement; potential target leakage or post-decision information
- `claim_duration_days` — Potential post-outcome field; excluded to reduce temporal leakage

### Target distribution
- Counts: `{"0": 779537, "1": 66705}`
- Percentages: `{"0": 92.1175, "1": 7.8825}`
- Majority/minority ratio: **11.69:1**
- Imbalance handling recommended by threshold (1.5:1): **True**
- Handling actually used in model training: **True** — class_weight='balanced' for supported models; no resampling

## 3. Data quality and EDA outputs
The pipeline writes the following tables under `reports/final_tables/`:
- `data_inventory.csv`: type, non-null count, missingness, and unique-value count for every column.
- `missingness_report.csv`: missing counts and percentages.
- `numeric_descriptive_statistics_skew_kurtosis.csv`: count, mean, standard deviation, quantiles, skewness, kurtosis, zeros, negatives and infinities.
- `skewness_kurtosis_all_numeric_features.csv`: full numeric-feature skewness/kurtosis ranking.
- `iqr_outlier_audit.csv`: IQR outlier counts. Outliers are flagged, not blindly deleted.
- `numeric_correlation_matrix.csv` and `high_correlation_pairs_abs_0_85.csv`.
- `feature_inclusion_exclusion_audit.csv`: each column's inclusion/exclusion decision and reason.
- `preprocessing_plan.csv`: imputation, scaling and encoding plan.

The EDA plots under `reports/final_plots/` include target balance, missingness, numeric distributions, skewness, correlation heatmap, target-rate plots, metric comparisons, and the best-model confusion matrix. The script does not automatically remove outliers or transform every skewed variable: those actions require domain justification and should be tested against held-out performance.

## 4. Feature selection and leakage prevention
The target is excluded from predictors. Claim/beneficiary identifiers are excluded to reduce memorisation. Payment/reimbursement columns are excluded because they may encode downstream decisions or leak information unavailable at triage time. `claim_duration_days` is excluded by default because it can be unavailable at initial triage and may act as a proxy for inpatient status. Constant and all-missing training columns are also excluded.

**Important:** the inclusion/exclusion list is a conservative starting point, not a substitute for a data dictionary and timestamp review. Before deployment, the insurer must confirm that every retained predictor is genuinely available at the moment a triage decision is made.

## 5. Model comparison
The complete model comparison is in `reports/final_model_comparison.csv`. The pipeline attempts ten models: majority-class dummy baseline, logistic regression, SGD classifier, decision tree, random forest, extra trees, Ridge classifier, AdaBoost, Gaussian Naive Bayes, and linear SVM. Individual model reports and confusion matrices are saved under `reports/final_models/`.

Top completed models by F1:
- **Random_Forest** — F1 0.7914; recall 0.9134; precision 0.6981; PR-AUC 0.9172
- **Extra_Trees** — F1 0.7706; recall 0.8838; precision 0.6831; PR-AUC 0.8899
- **AdaBoost** — F1 0.7602; recall 0.6301; precision 0.9579; PR-AUC 0.8989
- **Logistic_Regression** — F1 0.7494; recall 0.9388; precision 0.6237; PR-AUC 0.9232
- **SGD_Classifier** — F1 0.7458; recall 0.8001; precision 0.6984; PR-AUC 0.6112

F1 is used as the primary ranking metric to balance precision and recall; PR-AUC is a useful complementary metric for imbalanced classification. The best model should not be chosen using accuracy alone.

## 6. Train/test validation and limitations
- The model is evaluated on a held-out test set, which is not used for fitting preprocessing or model parameters.
- Preprocessing is fitted inside each model pipeline on training data only.
- Beneficiary overlap audit: `{"checked": true, "identifier": "DESYNPUF_ID", "unique_train_identifiers": 69396, "unique_test_identifiers": 17309, "overlapping_identifiers": 0, "overlap_percent_of_test_unique_ids": 0.0, "warning": "No beneficiary overlap detected."}`.
- The script caps model-fitting rows at 100,000 for practical memory use when the training file is larger; the full test set is evaluated.
- High scores can still reflect dataset-specific patterns, temporal leakage, target construction, or repeated beneficiaries. Verify the data-generation process and run a beneficiary-grouped and/or time-based holdout before claiming generalisation.
- This is classification/triage support, not fraud detection and not an automatic claim-denial system.

## 7. Practical value to an insurance company
If validated in a controlled pilot, the model could:
1. **Route claims into appropriate queues** (for example, likely inpatient vs other) so staff can prioritise and assign workloads.
2. **Reduce manual sorting** for routine claims and make the process more consistent.
3. **Support staffing and capacity planning** using claim-volume patterns and predicted queue composition.
4. **Help monitor service levels** by comparing predicted and actual categories, error rates, and processing times.
5. **Identify cases for human review** when confidence is low or model outputs conflict with existing rules.

The project does **not** prove that an insurer has already saved money, reduced processing time, or detected fraud. Those benefits require a prospective pilot with measured baselines and outcomes.

## 8. Recommended operating strategy
1. Use the model as a **decision-support and routing aid**, never as the sole basis for denial or reduction of benefits.
2. Keep a human reviewer for uncertain, high-value, unusual, or disputed claims.
3. Log model version, prediction, confidence/score, final human decision, and reason for override.
4. Monitor precision, recall, PR-AUC, false positives/negatives, latency, and drift monthly.
5. Check performance by relevant demographic and geographic groups, and investigate unequal error rates.
6. Protect sensitive beneficiary data with access controls, encryption, retention limits, and approved handling.
7. Compare the model with the current workflow in a pilot; only scale if measurable improvements occur without unacceptable harm.

## 9. Key fundamentals
- **EDA:** understand structure, missing data, distributions, skewness, kurtosis, outliers, correlations and class balance.
- **Cleaning:** standardise target labels and inspect missingness; imputation is learned from training data to avoid leakage.
- **Features:** predictors are columns available at prediction time; identifiers and downstream outcomes are excluded.
- **Train/test:** training fits the model; test estimates performance on unseen rows.
- **Imbalance:** if the majority/minority ratio crosses the configured threshold, class weights are enabled for models that support them. No resampling is applied blindly.
- **Precision:** among predicted positive cases, the fraction that is truly positive.
- **Recall:** among actual positive cases, the fraction identified by the model.
- **F1:** harmonic mean of precision and recall.
- **ROC-AUC / PR-AUC:** ranking metrics; PR-AUC is particularly informative when the positive class is rare.
- **Confusion matrix:** shows correct and incorrect predictions by class.

## 10. Final status
This script creates reproducible analysis outputs and a report. Before operational use, confirm target meaning, predictor availability at decision time, beneficiary/time split validity, privacy approval, subgroup performance, and pilot results.
