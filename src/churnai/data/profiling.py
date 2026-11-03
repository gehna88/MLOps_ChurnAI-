"""Reproducible exploratory analysis, expressed as testable functions.

This replaces an ad-hoc EDA notebook: the same analysis runs in CI and produces
a Markdown report, so findings (e.g. the ``Complain`` leakage) are documented
and re-checked every time the data changes.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.metrics import roc_auc_score

from churnai.data.schema import CATEGORICAL_COLUMNS, TARGET

# A single column separating the classes this well is almost certainly leakage.
SINGLE_FEATURE_LEAKAGE_AUC = 0.90
AUC_NEUTRAL = 0.5
DECIMALS = 4


@dataclass(frozen=True)
class LeakageFinding:
    column: str
    auc: float

    @property
    def suspicious(self) -> bool:
        return self.auc >= SINGLE_FEATURE_LEAKAGE_AUC


def single_feature_auc(df: pd.DataFrame, column: str, target: str = TARGET) -> float:
    """Direction-agnostic ROC-AUC of one column used alone as a score."""
    values = df[column]
    if values.dtype == object or column in CATEGORICAL_COLUMNS:
        # Score each category by its churn rate (target encoding).
        values = values.map(df.groupby(column)[target].mean())
    auc = roc_auc_score(df[target], values.fillna(values.median()))
    return round(max(auc, 1 - auc), DECIMALS)


def leakage_scan(df: pd.DataFrame, exclude: list[str], target: str = TARGET) -> list[LeakageFinding]:
    columns = [c for c in df.columns if c not in {target, *exclude}]
    findings = [LeakageFinding(c, single_feature_auc(df, c, target)) for c in columns]
    return sorted(findings, key=lambda f: f.auc, reverse=True)


def churn_rate_by(df: pd.DataFrame, column: str, target: str = TARGET) -> pd.DataFrame:
    grouped = df.groupby(column)[target].agg(customers="count", churn_rate="mean")
    return grouped.round(DECIMALS).reset_index()


def numeric_summary_by_target(df: pd.DataFrame, columns: list[str], target: str = TARGET) -> pd.DataFrame:
    return df.groupby(target)[columns].mean().T.round(2).rename(
        columns={0: "mean_retained", 1: "mean_churned"}
    )


def render_markdown(df: pd.DataFrame, exclude: list[str], numeric: list[str]) -> str:
    lines = [
        "# ChurnAI data profile",
        "",
        f"- Rows: **{len(df):,}**, columns: **{df.shape[1]}**",
        f"- Churn rate: **{df[TARGET].mean():.2%}**",
        f"- Missing values: **{int(df.isna().sum().sum())}**",
        f"- Zero balances: **{(df['Balance'] == 0).mean():.2%}** of customers",
        "",
        "## Single-feature leakage scan",
        "",
        f"Columns with ROC-AUC >= {SINGLE_FEATURE_LEAKAGE_AUC} on their own are flagged.",
        "",
        "| column | AUC | flag |",
        "|---|---|---|",
    ]
    for finding in leakage_scan(df, exclude):
        flag = "LEAKAGE SUSPECTED" if finding.suspicious else ""
        lines.append(f"| {finding.column} | {finding.auc:.4f} | {flag} |")

    lines += ["", "## Mean of numeric features by outcome", "",
              numeric_summary_by_target(df, numeric).to_markdown()]
    for column in CATEGORICAL_COLUMNS:
        lines += ["", f"## Churn rate by {column}", "",
                  churn_rate_by(df, column).to_markdown(index=False)]
    return "\n".join(lines) + "\n"
