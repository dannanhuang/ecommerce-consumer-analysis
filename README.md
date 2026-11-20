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
