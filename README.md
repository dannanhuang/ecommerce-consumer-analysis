# E-commerce Channel Performance and Customer Segmentation

## Project Overview

This project examines two marketing decisions: which channels attract customers
with high estimated lifetime value relative to acquisition cost, and which
customer groups may deserve closer attention from the retention team.

We compare marketing channels, group customers by purchasing behavior and
engagement, and present the results in a Dash dashboard. GitHub Actions will
automate data processing and report generation as new batches arrive.

## Team Members

- Name 1 — DANNAN HUANG
- Name 2 — ZHUN NI
- Name 3 — LINGJIN SHAN

## Dataset

Dataset: Multi-Channel E-commerce with Behavioral Metrics 

Source: (https://www.kaggle.com/datasets/rifatalam3/multi-channel-e-commerce-with-behavioral-metrics/data)

| File | Description |
| --- | --- |
| `customer.csv` | Original dataset with 50,000 records and 53 columns |
| `data/customer_selected_columns.csv` | Project input with all 50,000 records and 19 selected columns |
| `data/customer_clean.csv` | Planned output of the data examination and cleaning script |

The 19 selected columns cover customer identifiers, purchase and acquisition
dates, marketing and first-purchase channels, acquisition cost, estimated
customer lifetime value, purchase behavior, engagement, satisfaction, loyalty,
and churn status. Additional financial fields are retained for consistency checks.

## Research Questions

1. How do customer numbers, acquisition costs, and estimated customer lifetime
   value relative to acquisition cost vary across marketing channels? Do the
   same patterns hold across first-purchase channels?

2. Can we identify a group of historically frequent buyers who have not
   purchased recently or show low engagement? How does this group compare
   with other segments in estimated lifetime value, satisfaction, and the
   proportion of customers marked as churned?

## Planned Methods

### 1. Data examination and cleaning

Use `data_exam.py` to inspect data types, missing values, duplicate rows,
repeated customer IDs, category labels, dates, and numerical ranges.
Document the cleaning decisions and save the cleaned dataset in a separate file.

### 2. Channel performance analysis

Group customers by `Marketing_Channel` and compare:

- Number of distinct customers
- Average customer acquisition cost (CAC)
- Median estimated customer lifetime value (CLV)
- Median customer-level CLV-to-CAC ratio

For the ratio, divide each customer's estimated CLV by their acquisition cost
when the cost is positive, then calculate the median within each group.
Repeat the comparison by `First_Purchase_Channel` and report the number of
customers behind each result.

### 3. Customer segmentation and evaluation

Standardize `Purchase_Frequency`, `Days_Since_Last_Purchase`, and
`Engagement_Score`, then use K-means to group customers with similar behavior.
Compare candidate numbers of clusters using silhouette scores, segment sizes,
and how clearly the groups can be interpreted.

After clustering, compare the segments using `Customer_Lifetime_Value`,
`Customer_Satisfaction`, `Loyalty_Score`, and the proportion with `Churned = 1`.
Use these comparisons to identify groups that may warrant retention follow-up.

### 4. Dash dashboard and automation

Build two dashboard pages: one for channel performance and one for customer
segments. Use the course BatchSimulator to simulate incoming records.
GitHub Actions will run the cleaning and analysis scripts and generate an
updated report. New records will be assigned to segments using the same
scaler and clustering model so that segment labels remain comparable across batches.

## Data Quality Notes

- **Customer IDs:** Check whether each `CustomerID` appears once before
  calculating customer-level metrics. Inspect repeated IDs before deciding
  how to handle them.
- **Dates:** Some sampled records have a `Purchase_Date` earlier than their
  `Acquisition_Date`. Count and flag these cases before using date-based indicators.
- **Revenue:** In some sampled records, `Revenue` differs from
  `Quantity × Unit_Price`. Check the field definitions before interpreting
  revenue as an order total. Also compare `Net_Revenue` with
  `Revenue × (1 − Discount_Applied)`.
- **CLV and acquisition cost:** Treat `Customer_Lifetime_Value` as an estimate.
  Check the supplied `CLV_to_CAC_Ratio` against the recalculated value,
  allowing for rounding.
- **Churn:** Use `Churned` to describe and compare the completed segments.
  Keep it out of the clustering inputs.

## Collaboration

Team members work on separate task branches and submit pull requests for review
before merging into `main`. All analyses use the same cleaned dataset and
documented metric definitions.

## Part C · Customer segments and retention (RQ2)

The reference analysis covers 50,000 unique customers. Four clusters gave the
highest sampled silhouette (0.328 among 2–8 candidates). The frequent-buying
segment has a median purchase frequency of 11, a median gap of 18 days and
median engagement of 0.521. The longest-gap and lowest-engagement segments each
have a median frequency of 3. No single cluster captures the RQ2 behavior.

To count individual customers who fit the question, we separately flag purchase
frequency ≥ 7 **and** either a gap ≥ 41 days **or** engagement ≤ 0.403. These
cutoffs are the reference upper quartiles for frequency and gap and the lower
quartile for engagement, including ties. The flag spans all four clusters.
Churn, lifetime value, satisfaction and loyalty are compared after clustering.

| Reference cohort | Customers | Median estimated CLV | Median satisfaction | Median loyalty | Marked as churned |
| --- | ---: | ---: | ---: | ---: | ---: |
| Meet the RQ2 rule | 5,408 | 14,676 | 1.9 | 25.8 | 4,149 / 5,408 (76.72%) |
| All others | 44,592 | 14,791.5 | 1.9 | 26.5 | 34,135 / 44,592 (76.55%) |

The rule identifies 10.82% of customers. Their observed churn share is only
0.17 percentage points higher. The data support a behavioral distinction;
they give little reason to rank this cohort above the others by observed churn.

### Run the customer-segment page

From the repository root in a Python 3.12 or 3.13 environment:

```sh
python -m pip install -r requirements-segments.txt
python segment_dashboard.py
```

Open http://127.0.0.1:8050. The checked-in outputs let the page open immediately.
For the shared Dash app, import `build_layout` from `segment_dashboard` and
mount `build_layout()` on the customer-segment route. Styles are scoped in
`assets/segments.css`. An optional results directory selects a batch snapshot.

### Score incoming customers

After the existing batch workflow produces its cumulative cleaned file, run:

```sh
python segmentation.py score --input simulator_batches/customer_clean.csv --output reports/batch_segments
python segment_dashboard.py --results reports/batch_segments
```

Each input is a snapshot with one row per customer. Re-running a snapshot
replaces its output tables. Score a single batch by passing its CSV instead.
Scoring reuses the reference feature order, scaler, centroids, names and
quartile cutoffs from [the model](models/customer_segments.json). New records
need a unique customer ID and all three valid behavior inputs. Outcome fields
may be missing. Churn shares use the count of known 0/1 labels as denominator.

[Customer assignments](reports/segments/assignments.csv) join back to the
cleaned table by `CustomerID`. [Segment profiles](reports/segments/profiles.csv)
and the [RQ2 cohort comparison](reports/segments/rq2_comparison.csv) provide the
dashboard and report inputs. To reproduce the full reference model and outputs:

```sh
python segmentation.py train
python -m unittest discover -s tests
```

Training uses ten K-means initializations per candidate and seed 42. All
candidates use the same random sample of 3,000 customers for silhouette scoring.
The reference fit includes the three supplied behavior fields, including
`Days_Since_Last_Purchase` as reported in the source. The reference keeps rows
flagged for purchase/acquisition date order, as agreed in the cleaning step.
