# Insurance Claims Triage

Classifies incoming Medicare claims by type/urgency and assigns them to a queue.
Data: CMS 2008-2010 DE-SynPUF (Sample 1) - synthetic Medicare claims.
Download: https://www.cms.gov/data-research/statistics-trends-and-reports/medicare-claims-synthetic-public-use-files/cms-2008-2010-data-entrepreneurs-synthetic-public-use-file-de-synpuf

Raw data is NOT in this repo (size limits). Put the CSVs in `data/` and run `python src/01_merge_data.py data`.

## Progress
- [x] Step 1: Merge Inpatient + Outpatient + Beneficiary 2008
- [ ] Step 2: Cleaning  - [ ] Step 3: EDA  - [ ] Step 4: Features/labels  - [ ] Step 5: Modeling  - [ ] Step 6: Evaluation
