# Executive Summary

**Acquisition scale.** Search leads customer volume with 15,034 customers (30.1% of the dataset).

**Acquisition cost.** Referral has the lowest mean CAC (104.23 source units).

**Estimated value.** Direct has the highest median estimated CLV (14,988.50 source units); Direct has the highest median customer-level CLV/CAC (171.31x).

## First-purchase consistency

- **App:** Social leads median CLV/CAC at 179.57x across a subgroup of 7,451 customers.

- **Online:** Direct leads median CLV/CAC at 177.11x across a subgroup of 25,306 customers.

- **Phone:** Referral leads median CLV/CAC at 177.28x across a subgroup of 2,393 customers.

- **Store:** Direct leads median CLV/CAC at 173.13x across a subgroup of 14,850 customers.

The CLV/CAC leader changes across first-purchase subgroups; the overall leader does not represent every purchase context.

## Decision implications

Use separate lenses for acquisition scale, cost, and estimated value. Review channels within the relevant first-purchase context before prioritizing further investigation. These comparisons do not estimate incremental marketing returns or an optimal budget allocation.

## Method and limitations

One row per unique customer. Counts retain all records; financial metrics use valid non-negative values. CLV/CAC is calculated per customer with positive CAC before taking the group median. Valid sample counts accompany each metric. Cross-tab customer shares are normalized within each first-purchase subgroup. CLV is estimated, currency units are unverified, subgroup sizes differ, and channel differences are descriptive. Date anomalies are logged without removing records; no date-based or active-customer filter is applied.

## Source record

Input: `customer_clean.csv`

SHA256: `9d07d0e29d282dfeaff064b6214ac3914d7f27d465d1bb68dcc970ac5f57f93a`

Kaggle source: https://www.kaggle.com/datasets/rifatalam3/multi-channel-e-commerce-with-behavioral-metrics/data
