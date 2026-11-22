"""Customer-segment page for the team dashboard. Run directly for a standalone view."""
import argparse
import json
from pathlib import Path

from dash import Dash, dcc, html
import pandas as pd
import plotly.graph_objects as go

from segmentation import MODEL, RESULTS

COLORS = {1: "#116a64", 2: "#9c4d2f", 3: "#536f9a", 4: "#706485"}
METRICS = [
    ("Churn", "churn_share", "Share marked as churned", ".1%"),
    ("Value", "median_clv", "Median estimated lifetime value", ",.0f"),
    ("Satisfaction", "median_satisfaction", "Median satisfaction score", ".1f"),
    ("Loyalty", "median_loyalty", "Median loyalty score", ".1f"),
]


def display(value, spec):
    return "—" if pd.isna(value) else format(value, spec)


def table(headers, rows):
    return html.Div(html.Table([
        html.Thead(html.Tr([html.Th(item, scope="col") for item in headers])),
        html.Tbody([html.Tr([html.Th(row[0], scope="row"), *[html.Td(v) for v in row[1:]]])
                    for row in rows]),
    ]), className="sg-table")


def outcome_chart(profiles, field, title, spec):
    ordered = profiles.sort_values("segment_id", ascending=False)
    chart = go.Figure(go.Bar(
        x=ordered[field], y=ordered["customer_segment"], orientation="h",
        marker_color=[COLORS[i] for i in ordered.segment_id],
        text=[display(value, spec) for value in ordered[field]], textposition="outside",
        cliponaxis=False, hovertemplate="%{y}<br>%{text}<extra></extra>",
    ))
    upper = ordered[field].max()
    axis_range = [0, 1] if field == "churn_share" else [0, max(1, upper * 1.22)]
    chart.update_layout(
        title=dict(text=title, font_size=16, x=.02), height=265, showlegend=False,
        template="plotly_white", paper_bgcolor="white", plot_bgcolor="white",
        font=dict(family="Arial, sans-serif", size=13, color="#213837"),
        margin=dict(l=12, r=50, t=54, b=32),
        xaxis=dict(range=axis_range, tickformat=spec, gridcolor="#e0e9e6", zeroline=False),
        yaxis=dict(automargin=True, fixedrange=True), bargap=.44,
    )
    if ordered[field].isna().all():
        chart.add_annotation(text="No recorded values for this outcome", x=.5, y=.5,
                             xref="paper", yref="paper", showarrow=False)
    return dcc.Graph(figure=chart, config={"displayModeBar": False}, responsive=True)


def build_layout(results=RESULTS):
    """B can mount this layout directly in the shared Dash app."""
    results = Path(results)
    model = json.loads(MODEL.read_text(encoding="utf-8"))
    profiles = pd.read_csv(results / "profiles.csv")
    comparison = pd.read_csv(results / "rq2_comparison.csv").set_index("rq2_match")
    total = int(profiles.customers.sum())
    matched = int(profiles.rq2_customers.sum())
    covered = int((profiles.rq2_customers > 0).sum())
    limits = model["thresholds"]
    plural = "segment" if covered == 1 else "segments"
    rule = (f"Purchase frequency ≥ {limits['frequent']:g} and either a purchase gap ≥ "
            f"{limits['long_gap']:g} days or engagement ≤ {limits['low_engagement']:.3f}. "
            "Cutoffs use the upper frequency and gap quartiles and lower engagement quartile "
            "of the reference dataset. Ties are included.")
    if covered > 1:
        finding = f"The customers who meet the rule appear in {covered} model segments."
    elif covered == 1:
        finding = "In this snapshot, the customers who meet the rule appear in one model segment."
    else:
        finding = "No customers in this snapshot meet the reference cutoffs."
    churn = comparison["churn_share"].reindex([True, False])
    churn_note = "Each churn share uses customers with a known 0/1 churn label."
    if churn.notna().all():
        gap = 100 * (churn.loc[True] - churn.loc[False])
        churn_note = (f"Observed churn gap (meet rule minus others): {gap:+.2f} percentage points. "
                      + churn_note)
    comparison_rows = []
    for label, key, spec in [
        ("Customers", "customers", ",.0f"), ("Median estimated CLV", "median_clv", ",.1f"),
        ("Median satisfaction", "median_satisfaction", ".1f"),
        ("Median loyalty", "median_loyalty", ".1f"), ("Share marked as churned", "churn_share", ".2%"),
        ("Known churn labels", "known_churn", ",.0f"),
    ]:
        values = comparison[key].reindex([True, False])
        if key in ("customers", "known_churn"):
            values = values.fillna(0)
        comparison_rows.append([label, *[display(v, spec) for v in values]])

    profile_rows = [[f"{p.segment_id} · {p.customer_segment}", f"{p.customers:,}",
                     f"{p.median_frequency:g}", f"{p.median_days:g}", f"{p.median_engagement:.3f}",
                     f"{p.rq2_customers:,}"] for p in profiles.itertuples()]
    selected = len(model["segments"])
    fit_rows = [[f"{row['k']} (selected)" if row["k"] == selected else str(row["k"]),
                 f"{row['silhouette']:.3f}", f"{row['smallest_group']:,}", f"{row['largest_group']:,}"]
                for row in model["evaluations"]]
    return html.Main([
        html.Header([
            html.H1("Customer segments and retention"),
            html.P("Which historically frequent buyers have gone quiet, and how do their outcomes compare?"),
        ], className="sg-header"),
        html.Section([
            html.Div([
                html.H2("Frequent buyers with a long gap or low engagement"),
                html.Div(f"{matched:,}", className="sg-number"),
                html.P(f"of {total:,} customers ({matched / total:.1%}) meet the RQ2 rule."),
                html.P(finding), html.P(rule, className="sg-note"),
            ], className="sg-finding"),
            html.Div([
                html.H2("How these customers compare"),
                table(["Measure", "Meet rule", "Others"], comparison_rows),
                html.P(churn_note, className="sg-note"),
            ], className="sg-comparison"),
        ], className="sg-pair"),
        html.Section([
            html.H2("Behavior by model segment"),
            table(["Segment", "Customers", "Median purchase frequency", "Median gap (days)",
                   "Median engagement", "Meet RQ2 rule"], profile_rows),
        ]),
        html.Section([
            html.H2("Outcomes by model segment"),
            dcc.Tabs([dcc.Tab(label=tab, value=field, children=outcome_chart(profiles, field, label, spec),
                              className="sg-tab", selected_className="sg-tab-active")
                      for tab, field, label, spec in METRICS], value="churn_share", mobile_breakpoint=0),
        ]),
        html.Details([
            html.Summary("Model evaluation and batch scoring"),
            html.P(f"K-means used standardized purchase frequency, purchase gap and engagement for "
                   f"{model['reference_rows']:,} reference customers. Each candidate used 10 starts. "
                   f"Silhouettes use the same seeded sample of {model['evaluation_sample']:,} customers. "
                   f"The selected {selected}-group model had the highest reference silhouette and "
                   "distinguished frequent buyers, long gaps and differences in engagement."),
            table(["Groups", "Silhouette", "Smallest group", "Largest group"], fit_rows),
            html.P("New batches use the saved reference scale, centers, group names and quartile cutoffs. "
                   "Churn labels and value measures are used only to describe the groups."),
        ]),
    ], className="sg-page")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--port", type=int, default=8050)
    args = parser.parse_args()
    app = Dash(__name__, title="Customer segments and retention")
    app.layout = lambda: build_layout(args.results)
    app.run(port=args.port)
