# Channel Analysis — Lingjin SHAN

## Research question
How do customer numbers, acquisition costs, and estimated customer
lifetime value vary across marketing channels?
Do the same patterns hold across first-purchase channels?

## Input
Use the shared customer_clean.csv produced by Member A.
Confirm the final file path and cleaning rules before analysis.

## Metrics
Group by Marketing_Channel and calculate:
- Number of distinct customers
- Average customer acquisition cost (CAC)
- Median estimated customer lifetime value (CLV)
- Median customer-level CLV/CAC ratio, using positive CAC only

Repeat these comparisons by First_Purchase_Channel.
Display the customer count for every group.

## Dashboard
- Filter by first-purchase channel
- Compare customer numbers across marketing channels
- Compare acquisition cost and estimated customer value
- Summarize the main finding

## Deliverables
- channel_analysis.py
- Channel summary tables
- Dash channel page
- Integration with Member C's customer segment page
