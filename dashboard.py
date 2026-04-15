"""
Economic Regime Factor ETF Allocator -- Visual Dashboard
=========================================================
Run:  python dashboard.py
Open: http://127.0.0.1:8050

Panels
------
  * KPI cards   (CAGR, Sharpe, Max DD, Vol, Calmar -- vs benchmark)
  * Equity curve (strategy / equal-weight benchmark / SPY)
  * Drawdown chart
  * Regime timeline   (color-coded daily regime)
  * Risk-on score     (0..1 continuous signal)
  * Macro factor z-scores  (GDP, CPI, M2, Velocity, Yield-Curve)
  * Current holdings table (vol-scaled weights, as-of today)
  * Regime allocations     (raw optimised weights per regime)
  * Monthly returns heatmap
  * Year-by-year returns bar chart
  * Rolling 12-month Sharpe
"""

import sys
from datetime import datetime
from pathlib import Path

import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
from dash import dash_table, dcc, html

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
TICKERS = ["SPY", "GLD", "MTUM", "VLUE", "USMV", "QUAL", "IJR", "VIG"]
ASSETS = TICKERS + ["cash"]
# Extra tickers fetched only for comparison (not part of the strategy universe)
COMP_EXTRA = ["QQQ", "AGG", "IEF", "TLT"]
START_DATE = "2010-01-01"
VOL_LOOKBACK = 63
MIN_VOL = 0.05
MAX_VOL = 2.00
VOL_EPS = 1e-8
CASH_YIELD = (1.045) ** (1 / 252) - 1  # ~4.5 % annual
PORT = 8050

REGIME_COLORS = {
    "Recovery": "#3fb950",
    "Overheating": "#d29922",
    "Stagflation": "#a371f7",
    "Contraction": "#f85149",
    "Unknown": "#8b949e",
}
DARK_BG = "#0d1117"
CARD_BG = "#161b22"
BORDER = "#30363d"
ACCENT = "#58a6ff"
GREEN = "#3fb950"
RED = "#f85149"
YELLOW = "#d29922"
MUTED = "#8b949e"
TEXT = "#e6edf3"
TMPL = "plotly_dark"

_CL = {
    "template": TMPL,
    "paper_bgcolor": CARD_BG,
    "plot_bgcolor": CARD_BG,
    "font": {"color": TEXT},
}
_M = {"l": 55, "r": 20, "t": 50, "b": 40}


# ---------------------------------------------------------------------------
# 1.  DATA LOADING
# ---------------------------------------------------------------------------


def load_outputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load regime labels and optimal allocations from outputs/."""
    regime_path = ROOT / "outputs" / "regime_labels_expanded.csv"
    alloc_path = ROOT / "outputs" / "optimal_allocations.csv"

    regime_df = pd.read_csv(regime_path, parse_dates=["date"])
    regime_df.set_index("date", inplace=True)
    regime_df.index = regime_df.index.to_period("M").to_timestamp("M")
    regime_df["regime"] = regime_df["regime"].astype(str).str.strip()

    alloc_df = pd.read_csv(alloc_path)
    alloc_df["regime"] = alloc_df["regime"].astype(str).str.strip()
    alloc_df.set_index("regime", inplace=True)

    print(
        f"  Regime labels: {len(regime_df)} months  "
        f"{regime_df.index.min().date()} -> {regime_df.index.max().date()}"
    )
    return regime_df, alloc_df


def fetch_prices() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download adjusted close prices for strategy + comparison tickers.

    Returns
    -------
    prices : DataFrame  -- strategy tickers only (TICKERS)
    comp   : DataFrame  -- comparison-only tickers (COMP_EXTRA)
    """
    today = datetime.today().strftime("%Y-%m-%d")
    all_tickers = TICKERS + COMP_EXTRA
    print(f"  Downloading prices {START_DATE} -> {today} ...")
    raw = yf.download(
        all_tickers, start=START_DATE, end=today, progress=False, auto_adjust=True
    )
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"]
    else:
        close = raw
    close = close.dropna(how="all")
    prices = close[[t for t in TICKERS if t in close.columns]]
    comp = close[[t for t in COMP_EXTRA if t in close.columns]]
    print(
        f"  Prices: {len(prices)} trading days  "
        f"{prices.index.min().date()} -> {prices.index.max().date()}"
    )
    return prices, comp


# ---------------------------------------------------------------------------
# 2.  BACKTEST
# ---------------------------------------------------------------------------

REGIME_ALIASES = {"Expansion": "Overheating", "Slowdown": "Contraction"}
RISK_ON_REGIMES = {"Recovery", "Overheating"}
RISK_OFF_REGIMES = {"Contraction", "Stagflation"}


def _series(d: dict, cols: list) -> pd.Series:
    return pd.Series({c: float(d.get(c, 0.0)) for c in cols}, index=cols)


def _blend(w_off: dict, w_on: dict, alpha: float) -> dict:
    a = float(np.clip(alpha, 0.0, 1.0))
    out = {
        k: (1 - a) * float(w_off.get(k, 0)) + a * float(w_on.get(k, 0)) for k in ASSETS
    }
    s = sum(out.values())
    return (
        {k: v / s for k, v in out.items()}
        if s > 0
        else {k: 1 / len(ASSETS) for k in ASSETS}
    )


def _vol_scale(raw_w: pd.Series, trailing: pd.DataFrame) -> pd.Series:
    w = raw_w.copy()
    risky = [c for c in w.index if c != "cash" and c in trailing.columns]
    if not risky:
        return w / w.sum()
    vol = trailing[risky].std()
    med = vol.median()
    vol = vol.clip(lower=MIN_VOL * med, upper=MAX_VOL * med)
    vol = vol.replace(0.0, VOL_EPS).fillna(med)
    w[risky] = w[risky] / vol
    return w / w.sum()


def run_backtest(
    prices: pd.DataFrame, regime_df: pd.DataFrame, alloc_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Run the strategy backtest.  Returns (weights_df, port_rets, bench_rets, spy_rets)."""
    allocations = alloc_df.to_dict(orient="index")
    allocations = {
        str(k).strip(): {a: float(v) for a, v in d.items()}
        for k, d in allocations.items()
    }
    for d in allocations.values():
        d.setdefault("cash", 0.0)

    # Risk-on / risk-off anchor blends
    def _avg(regimes):
        regs = [r for r in regimes if r in allocations]
        out = dict.fromkeys(ASSETS, 0.0)
        for r in regs:
            for a in ASSETS:
                out[a] += float(allocations[r].get(a, 0.0))
        n = len(regs) or 1
        return {a: v / n for a, v in out.items()}

    w_on_base = _avg(RISK_ON_REGIMES)
    w_off_base = _avg(RISK_OFF_REGIMES)

    # Forward-fill monthly regime signals to daily index (covers Jan-Apr 2026)
    regime_daily = regime_df.reindex(prices.index, method="ffill")

    returns = prices.pct_change().dropna()
    returns["cash"] = CASH_YIELD

    port_list: list[float] = []
    bench_list: list[float] = []
    weight_rows: list[dict] = []
    prev_month = None
    cur_w: dict = {a: 1 / len(ASSETS) for a in ASSETS}

    bench_assets = TICKERS

    for date in returns.index:
        row = regime_daily.loc[date] if date in regime_daily.index else None
        if row is None or pd.isna(row.get("regime", None)):
            port_list.append(np.nan)
            bench_list.append(float(returns.loc[date, bench_assets].mean()))
            continue

        month = date.to_period("M")
        if prev_month is None or month != prev_month:
            risk_on_val = row.get("risk_on", None)
            if risk_on_val is not None and not pd.isna(risk_on_val):
                base = _blend(w_off_base, w_on_base, float(risk_on_val))
            else:
                key = REGIME_ALIASES.get(
                    str(row["regime"]).strip(), str(row["regime"]).strip()
                )
                base = allocations.get(key, {a: 1 / len(ASSETS) for a in ASSETS})

            raw_w = _series(base, ASSETS)
            trail = returns[TICKERS].loc[:date].tail(VOL_LOOKBACK)
            scaled = _vol_scale(raw_w, trail)
            cur_w = {str(k): float(v) for k, v in scaled.items()}
            prev_month = month

        port_ret = sum(returns.loc[date, a] * cur_w.get(a, 0.0) for a in ASSETS)
        bench_ret = float(returns.loc[date, bench_assets].mean())
        port_list.append(port_ret)
        bench_list.append(bench_ret)
        weight_rows.append({"date": date, **cur_w})

    port_rets = pd.Series(port_list, index=returns.index, name="strategy").dropna()
    bench_rets = pd.Series(bench_list, index=returns.index, name="benchmark").dropna()
    spy_rets = returns["SPY"].dropna()
    weights_df = (
        pd.DataFrame(weight_rows).set_index("date") if weight_rows else pd.DataFrame()
    )

    return weights_df, port_rets, bench_rets, spy_rets


# ---------------------------------------------------------------------------
# 3.  METRICS
# ---------------------------------------------------------------------------


def metrics(rets: pd.Series, rf: float = CASH_YIELD) -> dict:
    r = rets.dropna()
    if r.empty:
        return dict.fromkeys(("cagr", "sharpe", "max_dd", "vol", "calmar"), 0.0)
    eq = (1 + r).cumprod()
    n = len(r)
    ann = n / 252
    cagr = float(eq.iloc[-1] ** (1 / ann) - 1)
    vol = float(r.std() * np.sqrt(252))
    exc = r - rf
    sharpe = float(exc.mean() / exc.std() * np.sqrt(252)) if exc.std() > 0 else 0.0
    dd = (eq / eq.cummax() - 1).min()
    calmar = cagr / abs(float(dd)) if dd != 0 else 0.0
    return {
        "cagr": cagr,
        "sharpe": sharpe,
        "max_dd": float(dd),
        "vol": vol,
        "calmar": calmar,
    }


def rolling_sharpe(rets: pd.Series, window: int = 252) -> pd.Series:
    rf = CASH_YIELD
    exc = rets - rf
    return exc.rolling(window).mean() / exc.rolling(window).std() * np.sqrt(252)


def current_holdings(weights_df: pd.DataFrame) -> pd.DataFrame:
    if weights_df.empty:
        return pd.DataFrame()
    last = weights_df.iloc[-1].sort_values(ascending=False)
    df = last.reset_index()
    df.columns = ["Asset", "Weight"]
    df = df[df["Weight"] > 0.001]
    df["Weight %"] = df["Weight"].map(lambda x: f"{x:.1%}")
    df["Bar"] = df["Weight"].map(lambda x: "#" * max(1, int(x * 40)))
    return df


# ---------------------------------------------------------------------------
# 4.  CHARTS
# ---------------------------------------------------------------------------


def fig_equity(port: pd.Series, bench: pd.Series, spy: pd.Series) -> go.Figure:
    p_eq = (1 + port).cumprod()
    b_eq = (1 + bench).cumprod()
    s_eq = (1 + spy).cumprod()
    # normalise to 1.0 at common start
    start = max(p_eq.index.min(), b_eq.index.min(), s_eq.index.min())

    def norm(s):
        s = s[s.index >= start]
        return s / s.iloc[0]

    p_n, b_n, s_n = norm(p_eq), norm(b_eq), norm(s_eq)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=p_n.index,
            y=p_n.values,
            name="Strategy",
            line={"color": ACCENT, "width": 2.5},
            hovertemplate="%{x|%Y-%m-%d}<br>%{y:.3f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=b_n.index,
            y=b_n.values,
            name="EW Benchmark",
            line={"color": MUTED, "width": 1.5, "dash": "dot"},
            hovertemplate="%{x|%Y-%m-%d}<br>%{y:.3f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=s_n.index,
            y=s_n.values,
            name="SPY",
            line={"color": YELLOW, "width": 1.5, "dash": "dash"},
            hovertemplate="%{x|%Y-%m-%d}<br>%{y:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        **_CL,
        margin=_M,
        height=370,
        title="Equity Curve",
        legend={"orientation": "h", "y": 1.02, "x": 1, "xanchor": "right"},
    )
    return fig


def fig_drawdown(port: pd.Series, bench: pd.Series) -> go.Figure:
    p_eq = (1 + port).cumprod()
    b_eq = (1 + bench).cumprod()
    p_dd = (p_eq / p_eq.cummax() - 1) * 100
    b_dd = (b_eq / b_eq.cummax() - 1) * 100
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=p_dd.index,
            y=p_dd.values,
            fill="tozeroy",
            fillcolor="rgba(248,81,73,0.18)",
            line={"color": RED, "width": 1.5},
            name="Strategy",
            hovertemplate="%{x|%Y-%m-%d}<br>%{y:.1f}%<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=b_dd.index,
            y=b_dd.values,
            line={"color": MUTED, "width": 1, "dash": "dot"},
            name="EW Bench",
            hovertemplate="%{x|%Y-%m-%d}<br>%{y:.1f}%<extra></extra>",
        )
    )
    fig.update_layout(
        **_CL,
        margin=_M,
        height=210,
        title="Drawdown (%)",
        legend={"orientation": "h", "y": 1.02, "x": 1, "xanchor": "right"},
    )
    return fig


def _hex_to_rgba(hex_color: str, alpha: float = 0.55) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def fig_regime_timeline(regime_daily: pd.DataFrame) -> go.Figure:
    """Stacked area = 1 coloured by regime."""
    rd = regime_daily.dropna(subset=["regime"]).reset_index()
    rd.columns = ["date"] + list(rd.columns[1:])

    fig = go.Figure()
    for regime, color in REGIME_COLORS.items():
        mask = rd["regime"] == regime
        if not mask.any():
            continue
        y_vals = np.where(mask, 1, np.nan)
        fig.add_trace(
            go.Scatter(
                x=rd["date"],
                y=y_vals,
                fill="tozeroy",
                fillcolor=_hex_to_rgba(color, 0.55),
                line={"width": 0, "color": color},
                name=regime,
                mode="lines",
                stackgroup="regime",
                hovertemplate=f"{regime}: %{{x|%Y-%m-%d}}<extra></extra>",
            )
        )
    fig.update_layout(
        **_CL,
        margin=_M,
        height=130,
        title="Macro Regime",
        yaxis={"showticklabels": False, "range": [0, 1]},
        showlegend=True,
        legend={"orientation": "h", "y": 1.02, "x": 1, "xanchor": "right"},
    )
    return fig


def fig_risk_on(regime_daily: pd.DataFrame) -> go.Figure:
    rd = regime_daily.dropna(subset=["risk_on"]).reset_index()
    rd.columns = ["date"] + list(rd.columns[1:])
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=rd["date"],
            y=rd["risk_on"],
            fill="tozeroy",
            fillcolor="rgba(88,166,255,0.15)",
            line={"color": ACCENT, "width": 2},
            name="Risk-On Score",
            hovertemplate="%{x|%Y-%m-%d}<br>Risk-On: %{y:.3f}<extra></extra>",
        )
    )
    fig.add_hline(y=0.5, line_dash="dash", line_color="rgba(255,255,255,0.3)")
    fig.update_layout(
        **_CL,
        margin=_M,
        height=220,
        title="Risk-On Score (0 = fully defensive, 1 = fully aggressive)",
        yaxis={"range": [0, 1]},
    )
    return fig


def fig_macro_factors(regime_daily: pd.DataFrame) -> go.Figure:
    cols_map = {
        "gdp_z": ("#3fb950", "GDP Z"),
        "infl_z": ("#f85149", "Inflation Z"),
        "m2_z": ("#79c0ff", "M2 Z"),
        "vel_z": ("#ffa657", "Velocity Z"),
        "yield_level_z": ("#a371f7", "Yield Curve Z"),
    }
    rd = regime_daily.reset_index()
    rd.columns = ["date"] + list(rd.columns[1:])
    fig = go.Figure()
    for col, (color, label) in cols_map.items():
        if col not in rd.columns:
            continue
        fig.add_trace(
            go.Scatter(
                x=rd["date"],
                y=rd[col],
                name=label,
                line={"color": color, "width": 1.5},
                hovertemplate=f"{label}: %{{y:.2f}}<br>%{{x|%Y-%m-%d}}<extra></extra>",
            )
        )
    fig.add_hline(y=0, line_color="rgba(255,255,255,0.15)")
    fig.update_layout(
        **_CL,
        margin=_M,
        height=300,
        title="Macro Factor Z-Scores",
        legend={"orientation": "h", "y": 1.02, "x": 1, "xanchor": "right"},
    )
    return fig


def fig_monthly_heatmap(port: pd.Series, label: str = "Strategy") -> go.Figure:
    eq = (1 + port).cumprod()
    mthly = eq.resample("ME").last().pct_change().dropna()
    mthly.index = pd.to_datetime(mthly.index)
    months = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]
    col_labels = months + ["", "Full Yr"]  # blank spacer column for visual separation
    current_yr = datetime.today().year
    years = sorted(mthly.index.year.unique())
    z, txt = [], []
    for yr in years:
        row_z, row_t = [], []
        yr_factors = []
        for mo in range(1, 13):
            sub = mthly[(mthly.index.year == yr) & (mthly.index.month == mo)]
            if not sub.empty:
                v = float(sub.iloc[0])
                row_z.append(round(v * 100, 2))
                row_t.append(f"{v * 100:.1f}%")
                yr_factors.append(1 + v)
            else:
                row_z.append(None)
                row_t.append("")
        # spacer
        row_z.append(None)
        row_t.append("")
        # Full Year / YTD column
        if yr_factors:
            full_ret = float(np.prod(yr_factors)) - 1
            row_z.append(round(full_ret * 100, 2))
            label = (
                f"YTD {full_ret * 100:.1f}%"
                if yr == current_yr
                else f"{full_ret * 100:.1f}%"
            )
            row_t.append(label)
        else:
            row_z.append(None)
            row_t.append("")
        z.append(row_z)
        txt.append(row_t)

    fig = go.Figure(
        go.Heatmap(
            x=col_labels,
            y=[str(y) for y in years],
            z=z,
            text=txt,
            texttemplate="%{text}",
            textfont={"size": 10},
            colorscale=[[0, RED], [0.5, "#1c2128"], [1, GREEN]],
            zmid=0,
            zmin=-20,
            zmax=20,
            colorbar={"title": "Ret %", "thickness": 14},
            hovertemplate="Year: %{y}  Month: %{x}<br>%{z:.1f}%<extra></extra>",
        )
    )
    # Vertical divider between months and Full Yr column
    fig.add_vline(x=12.5, line_color="rgba(255,255,255,0.25)", line_width=1)
    fig.update_layout(
        **_CL,
        margin={"l": 55, "r": 80, "t": 50, "b": 60},
        height=max(340, len(years) * 28 + 90),
        title=f"{label}  —  Monthly Returns  (last column = Full Yr / YTD)",
    )
    return fig


def fig_annual_returns(
    port: pd.Series,
    bench: pd.Series,
    spy: pd.Series | None = None,
    qqq: pd.Series | None = None,
) -> go.Figure:
    """Year-by-year returns bar chart; includes SPY and QQQ when provided.

    Current partial year is shown as YTD.
    """
    current_yr = datetime.today().year

    def yr_rets(s: pd.Series) -> pd.Series:
        s2 = s.copy()
        s2.index = pd.to_datetime(s2.index)
        eq = (1 + s2).cumprod()
        ann = eq.resample("YE").last().pct_change().dropna()
        # Append current-year YTD if the series extends into it
        if s2.index.year.max() == current_yr:
            ytd_start = s2[s2.index.year == current_yr]
            if not ytd_start.empty:
                eq_ytd = (1 + ytd_start).cumprod()
                ytd_ret = float(eq_ytd.iloc[-1]) - 1.0
                ytd_idx = pd.DatetimeIndex([pd.Timestamp(f"{current_yr}-12-31")])
                ytd_s = pd.Series([ytd_ret], index=ytd_idx)
                ann = pd.concat([ann[ann.index.year < current_yr], ytd_s])
        return ann

    p_ann = yr_rets(port)
    b_ann = yr_rets(bench)
    spy_ann = yr_rets(spy) if spy is not None else pd.Series(dtype=float)
    qqq_ann = yr_rets(qqq) if qqq is not None else pd.Series(dtype=float)

    years = sorted(
        set(p_ann.index.year)
        | set(b_ann.index.year)
        | set(spy_ann.index.year)
        | set(qqq_ann.index.year)
    )

    def _vals(ann):
        return [
            float(ann[ann.index.year == y].iloc[0]) * 100
            if y in ann.index.year
            else None
            for y in years
        ]

    p_vals = _vals(p_ann)
    b_vals = _vals(b_ann)
    spy_vals = _vals(spy_ann)
    qqq_vals = _vals(qqq_ann)

    # Label current year as YTD
    yr_labels = [f"{y} YTD" if y == current_yr else str(y) for y in years]
    p_colors = [GREEN if (v is not None and v >= 0) else RED for v in p_vals]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=yr_labels,
            y=p_vals,
            name="Strategy",
            marker_color=p_colors,
            hovertemplate="%{x}: %{y:.1f}%<extra>Strategy</extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=yr_labels,
            y=b_vals,
            name="EW Bench",
            mode="markers+lines",
            marker={"color": MUTED, "size": 5},
            line={"color": MUTED, "width": 1.5, "dash": "dot"},
            hovertemplate="%{x}: %{y:.1f}%<extra>EW Bench</extra>",
        )
    )
    if spy is not None:
        fig.add_trace(
            go.Scatter(
                x=yr_labels,
                y=spy_vals,
                name="SPY",
                mode="markers+lines",
                marker={"color": YELLOW, "size": 6, "symbol": "diamond"},
                line={"color": YELLOW, "width": 1.8, "dash": "dash"},
                hovertemplate="%{x}: %{y:.1f}%<extra>SPY</extra>",
            )
        )
    if qqq is not None:
        fig.add_trace(
            go.Scatter(
                x=yr_labels,
                y=qqq_vals,
                name="QQQ",
                mode="markers+lines",
                marker={"color": GREEN, "size": 6, "symbol": "square"},
                line={"color": GREEN, "width": 1.8, "dash": "longdash"},
                hovertemplate="%{x}: %{y:.1f}%<extra>QQQ</extra>",
            )
        )

    fig.update_layout(
        **_CL,
        margin=_M,
        height=390,
        title="Year-by-Year Returns  (current year = YTD)",
        barmode="group",
        legend={"orientation": "h", "y": 1.02, "x": 1, "xanchor": "right"},
        yaxis_title="Return (%)",
    )
    return fig


def fig_rolling_sharpe(port: pd.Series, bench: pd.Series) -> go.Figure:
    rs_p = rolling_sharpe(port, window=252)
    rs_b = rolling_sharpe(bench, window=252)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=rs_p.index,
            y=rs_p.values,
            name="Strategy",
            line={"color": ACCENT, "width": 2},
            hovertemplate="%{x|%Y-%m-%d}<br>Sharpe: %{y:.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=rs_b.index,
            y=rs_b.values,
            name="EW Bench",
            line={"color": MUTED, "width": 1.5, "dash": "dot"},
            hovertemplate="%{x|%Y-%m-%d}<br>Sharpe: %{y:.2f}<extra></extra>",
        )
    )
    fig.add_hline(y=0, line_color="rgba(255,255,255,0.2)")
    fig.add_hline(y=1, line_dash="dash", line_color="rgba(63,185,80,0.4)")
    fig.update_layout(
        **_CL,
        margin=_M,
        height=260,
        title="Rolling 12-Month Sharpe",
        legend={"orientation": "h", "y": 1.02, "x": 1, "xanchor": "right"},
    )
    return fig


def fig_regime_allocations(alloc_df: pd.DataFrame) -> go.Figure:
    """Grouped bar chart showing raw optimised weights per regime."""
    regimes = alloc_df.index.tolist()
    asset_colors = {
        "SPY": "#58a6ff",
        "GLD": "#d29922",
        "MTUM": "#3fb950",
        "VLUE": "#ffa657",
        "USMV": "#a371f7",
        "QUAL": "#79c0ff",
        "IJR": "#ff7b72",
        "VIG": "#56d364",
        "cash": "#8b949e",
    }
    fig = go.Figure()
    for asset in ASSETS:
        if asset not in alloc_df.columns:
            continue
        vals = [
            float(alloc_df.loc[r, asset]) if r in alloc_df.index else 0 for r in regimes
        ]
        if max(vals) < 0.001:
            continue
        fig.add_trace(
            go.Bar(
                x=regimes,
                y=[v * 100 for v in vals],
                name=asset,
                marker_color=asset_colors.get(asset, MUTED),
                hovertemplate=f"{asset}: %{{y:.1f}}%<extra>{asset}</extra>",
            )
        )
    fig.update_layout(
        **_CL,
        margin={"l": 55, "r": 20, "t": 50, "b": 60},
        height=330,
        title="Optimised Allocations by Regime (%)",
        barmode="group",
        legend={"orientation": "h", "y": 1.02, "x": 1, "xanchor": "right"},
        yaxis_title="Weight (%)",
    )
    return fig


# ---------------------------------------------------------------------------
# 4b.  STRATEGY COMPARISON
# ---------------------------------------------------------------------------

_COMP_COLORS = {
    "Strategy": "#58a6ff",
    "EW Bench": "#8b949e",
    "SPY": "#d29922",
    "QQQ": "#3fb950",
    "GLD": "#ffa657",
    "AGG": "#a371f7",
    "IEF": "#79c0ff",
    "TLT": "#56d364",
    "60/40": "#ff7b72",
    "80/20": "#f0883e",
}
_COMP_DASH = {
    "Strategy": "solid",
    "EW Bench": "dot",
    "SPY": "dash",
    "QQQ": "longdash",
    "GLD": "dashdot",
    "AGG": "longdashdot",
    "IEF": "dot",
    "TLT": "dash",
    "60/40": "dot",
    "80/20": "dashdot",
}


def build_comparison_returns(
    prices: pd.DataFrame,
    comp: pd.DataFrame,
    port_rets: pd.Series,
    bench_rets: pd.Series,
    spy_rets: pd.Series,
) -> dict:
    """Return {label: daily_return_series} for every comparison strategy."""
    out: dict = {
        "Strategy": port_rets.rename("Strategy"),
        "EW Bench": bench_rets.rename("EW Bench"),
        "SPY": spy_rets.rename("SPY"),
    }
    # Individual tickers from comparison universe
    for t in COMP_EXTRA:
        if t in comp.columns:
            r = comp[t].pct_change().dropna().rename(t)
            out[t] = r
    # 60/40 blend  (SPY 60% + AGG 40%)
    if "AGG" in comp.columns and "SPY" in prices.columns:
        spy_r = prices["SPY"].pct_change().dropna()
        agg_r = comp["AGG"].pct_change().dropna()
        idx = spy_r.index.intersection(agg_r.index)
        out["60/40"] = (0.60 * spy_r[idx] + 0.40 * agg_r[idx]).rename("60/40")
    # 80/20 blend  (SPY 80% + AGG 20%)
    if "AGG" in comp.columns and "SPY" in prices.columns:
        spy_r = prices["SPY"].pct_change().dropna()
        agg_r = comp["AGG"].pct_change().dropna()
        idx = spy_r.index.intersection(agg_r.index)
        out["80/20"] = (0.80 * spy_r[idx] + 0.20 * agg_r[idx]).rename("80/20")
    return out


def build_comparison_df(comp_returns: dict) -> pd.DataFrame:
    """DataFrame of metrics for each strategy, ready for DataTable."""
    rows = []
    for label, rets in comp_returns.items():
        m = metrics(rets)
        rows.append(
            {
                "Strategy": label,
                "CAGR": f"{m['cagr'] * 100:.1f}%",
                "Sharpe": f"{m['sharpe']:.2f}",
                "Max DD": f"{m['max_dd'] * 100:.1f}%",
                "Ann. Vol": f"{m['vol'] * 100:.1f}%",
                "Calmar": f"{m['calmar']:.2f}",
                "_cagr": m["cagr"],
                "_sharpe": m["sharpe"],
                "_max_dd": m["max_dd"],
            }
        )
    return pd.DataFrame(rows)


def fig_comparison_equity(comp_returns: dict) -> go.Figure:
    """Normalised equity curves for all strategies on one chart."""
    starts = [s.index.min() for s in comp_returns.values() if not s.empty]
    start = max(starts) if starts else None
    fig = go.Figure()
    for label, rets in comp_returns.items():
        eq = (1 + rets).cumprod()
        if start is not None:
            eq = eq[eq.index >= start]
        if eq.empty:
            continue
        eq = eq / eq.iloc[0]
        is_strategy = label == "Strategy"
        fig.add_trace(
            go.Scatter(
                x=eq.index,
                y=eq.values,
                name=label,
                line={
                    "color": _COMP_COLORS.get(label, MUTED),
                    "width": 2.5 if is_strategy else 1.5,
                    "dash": _COMP_DASH.get(label, "solid"),
                },
                hovertemplate=f"{label}: %{{y:.3f}}<br>%{{x|%Y-%m-%d}}<extra></extra>",
            )
        )
    fig.update_layout(
        **_CL,
        margin=_M,
        height=390,
        title="Strategy Comparison -- Normalised Equity Curves",
        legend={"orientation": "h", "y": 1.02, "x": 1, "xanchor": "right"},
    )
    return fig


# ---------------------------------------------------------------------------
# 4c.  INTERACTIVE HEATMAP STORE + CALLBACKS
# ---------------------------------------------------------------------------

# Populated in main() before app.run(); read-only during callbacks
_HEATMAP_FIGS: dict = {}


def register_callbacks(app: dash.Dash) -> None:
    from dash import Input, Output  # local import to avoid circular issues at top

    @app.callback(
        Output("heatmap-graph", "figure"),
        Input("heatmap-tabs", "value"),
    )
    def switch_heatmap(selected: str):
        return _HEATMAP_FIGS.get(selected or "Strategy", go.Figure())


# ---------------------------------------------------------------------------
# 5.  DASH LAYOUT
# ---------------------------------------------------------------------------


def _kpi_card(title: str, val: str, sub: str = "", color: str = TEXT) -> dbc.Col:
    return dbc.Col(
        dbc.Card(
            dbc.CardBody(
                [
                    html.P(
                        title,
                        style={
                            "color": MUTED,
                            "fontSize": "0.72rem",
                            "textTransform": "uppercase",
                            "letterSpacing": "0.07em",
                            "marginBottom": "4px",
                        },
                    ),
                    html.H4(
                        val,
                        style={
                            "color": color,
                            "fontWeight": "700",
                            "fontSize": "1.5rem",
                            "marginBottom": "2px",
                        },
                    ),
                    html.Small(sub, style={"color": MUTED, "fontSize": "0.78rem"}),
                ]
            ),
            style={
                "backgroundColor": CARD_BG,
                "border": f"1px solid {BORDER}",
                "borderRadius": "8px",
            },
        ),
        xs=6,
        sm=4,
        md=3,
        lg=True,
    )


def build_layout(
    pm: dict,
    bm: dict,
    sm: dict,
    latest_regime: str,
    latest_risk_on: float,
    figs: dict,
    holdings_df: pd.DataFrame,
    alloc_df: pd.DataFrame,
    regime_daily: pd.DataFrame,
) -> html.Div:

    def pct(v):
        return f"{v * 100:.1f}%"

    def f2(v):
        return f"{v:.2f}"

    # Colour helpers
    def ccolor(v):
        return GREEN if v > 0 else RED

    def scolor(v):
        return GREEN if v > 1 else YELLOW if v > 0.5 else RED

    def ddcolor(v):
        return GREEN if v > -0.10 else YELLOW if v > -0.20 else RED

    regime_color = REGIME_COLORS.get(latest_regime, MUTED)
    as_of_date = (
        regime_daily.index.max().strftime("%Y-%m-%d")
        if not regime_daily.empty
        else "--"
    )

    kpi_row = dbc.Row(
        [
            _kpi_card(
                "CAGR",
                pct(pm["cagr"]),
                f"Bench: {pct(bm['cagr'])}  SPY: {pct(sm['cagr'])}",
                ccolor(pm["cagr"]),
            ),
            _kpi_card(
                "Sharpe",
                f2(pm["sharpe"]),
                f"Bench: {f2(bm['sharpe'])}  SPY: {f2(sm['sharpe'])}",
                scolor(pm["sharpe"]),
            ),
            _kpi_card(
                "Max DD",
                pct(pm["max_dd"]),
                f"Bench: {pct(bm['max_dd'])}  SPY: {pct(sm['max_dd'])}",
                ddcolor(pm["max_dd"]),
            ),
            _kpi_card("Ann. Vol", pct(pm["vol"]), f"Bench: {pct(bm['vol'])}", TEXT),
            _kpi_card("Calmar", f2(pm["calmar"]), "", TEXT),
        ],
        className="g-2 mb-3",
    )

    # Regime pill
    regime_pill = html.Span(
        latest_regime,
        style={
            "backgroundColor": regime_color,
            "color": "#000",
            "padding": "3px 10px",
            "borderRadius": "12px",
            "fontWeight": "700",
            "fontSize": "0.85rem",
        },
    )

    # Holdings table
    hold_cols = [{"name": c, "id": c} for c in ["Asset", "Weight %", "Bar"]]
    hold_data = holdings_df.to_dict("records") if not holdings_df.empty else []

    _tbl = {
        "style_header": {
            "backgroundColor": "#21262d",
            "color": MUTED,
            "fontWeight": "600",
            "fontSize": "0.78rem",
            "border": f"1px solid {BORDER}",
        },
        "style_cell": {
            "backgroundColor": CARD_BG,
            "color": TEXT,
            "fontSize": "0.82rem",
            "border": f"1px solid {BORDER}",
            "padding": "6px 10px",
        },
        "style_table": {
            "border": f"1px solid {BORDER}",
            "borderRadius": "6px",
            "overflow": "hidden",
        },
    }

    return dbc.Container(
        [
            # Header
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H2(
                                "Economic Regime Factor ETF Allocator",
                                style={
                                    "color": TEXT,
                                    "fontWeight": "700",
                                    "marginBottom": "3px",
                                },
                            ),
                            html.P(
                                [
                                    f"Tickers: {', '.join(TICKERS)}  |  Rebalance: monthly  |  "
                                    f"Backtest: {START_DATE} -> today  |  Regime as-of: {as_of_date}   ",
                                    regime_pill,
                                    f"   Risk-On: {latest_risk_on:.3f}",
                                ],
                                style={"color": MUTED, "fontSize": "0.83rem"},
                            ),
                        ]
                    ),
                ],
                className="mb-2 mt-2",
            ),
            html.Hr(style={"borderColor": BORDER, "margin": "8px 0 14px 0"}),
            # KPI cards
            kpi_row,
            # Rolling Sharpe
            dbc.Row(
                [
                    dbc.Col(
                        dcc.Graph(
                            figure=figs["rolling_sharpe"],
                            config={"displayModeBar": False},
                        ),
                        md=12,
                    ),
                ],
                className="g-2 mb-3",
            ),
            # ── Monthly returns heatmap with benchmark toggle ────────────────
            dbc.Row(
                [
                    dbc.Col(
                        [
                            # Tab-style toggle
                            dcc.Tabs(
                                id="heatmap-tabs",
                                value="Strategy",
                                children=[
                                    dcc.Tab(
                                        label=k,
                                        value=k,
                                        style={
                                            "backgroundColor": CARD_BG,
                                            "color": MUTED,
                                            "border": f"1px solid {BORDER}",
                                            "padding": "6px 18px",
                                            "fontSize": "0.83rem",
                                            "fontWeight": "600",
                                            "borderRadius": "6px 6px 0 0",
                                        },
                                        selected_style={
                                            "backgroundColor": ACCENT,
                                            "color": "#0d1117",
                                            "border": f"1px solid {ACCENT}",
                                            "padding": "6px 18px",
                                            "fontSize": "0.83rem",
                                            "fontWeight": "700",
                                            "borderRadius": "6px 6px 0 0",
                                        },
                                    )
                                    for k in list(_HEATMAP_FIGS.keys())
                                ],
                                colors={
                                    "border": BORDER,
                                    "primary": ACCENT,
                                    "background": DARK_BG,
                                },
                                style={"marginBottom": "0px"},
                            ),
                            dcc.Graph(
                                id="heatmap-graph",
                                figure=_HEATMAP_FIGS.get("Strategy", go.Figure()),
                                config={"displayModeBar": False},
                            ),
                        ],
                        md=12,
                    ),
                ],
                className="g-2 mb-3",
            ),
            # Current holdings table
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.P(
                                f"Current Holdings  (as-of {as_of_date}, forward-filled)",
                                style={
                                    "color": MUTED,
                                    "fontSize": "0.75rem",
                                    "textTransform": "uppercase",
                                    "letterSpacing": "0.06em",
                                    "marginBottom": "6px",
                                },
                            ),
                            dash_table.DataTable(
                                data=hold_data,
                                columns=hold_cols,
                                **_tbl,
                                style_data_conditional=[
                                    {
                                        "if": {"column_id": "Asset"},
                                        "fontWeight": "700",
                                        "color": ACCENT,
                                    },
                                    {
                                        "if": {"column_id": "Bar"},
                                        "fontFamily": "monospace",
                                        "color": GREEN,
                                        "fontSize": "0.7rem",
                                    },
                                ],
                            ),
                        ],
                        md=3,
                    ),
                ],
                className="g-2 mb-4",
            ),
            html.Hr(style={"borderColor": BORDER}),
            html.P(
                f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  |  "
                "Regime labels from FRED (forward-filled to today)  |  "
                "Vol-scaled monthly rebalancing  |  ~4.5% cash yield  |  "
                "Not financial advice.",
                style={
                    "color": MUTED,
                    "fontSize": "0.72rem",
                    "textAlign": "center",
                    "paddingBottom": "12px",
                },
            ),
        ],
        fluid=True,
        style={
            "backgroundColor": DARK_BG,
            "minHeight": "100vh",
            "paddingLeft": "20px",
            "paddingRight": "20px",
        },
    )


# ---------------------------------------------------------------------------
# 6.  MAIN
# ---------------------------------------------------------------------------


def main():
    print("=" * 60)
    print("  ECONOMIC REGIME FACTOR ETF ALLOCATOR -- DASHBOARD")
    print("=" * 60)

    print("\n[1/5] Loading outputs ...")
    regime_df, alloc_df = load_outputs()

    print("\n[2/5] Fetching prices ...")
    prices, comp_prices = fetch_prices()

    print("\n[3/5] Running backtest ...")
    weights_df, port_rets, bench_rets, spy_rets = run_backtest(
        prices, regime_df, alloc_df
    )
    print(
        f"  Returns: {len(port_rets)} days  "
        f"{port_rets.index.min().date()} -> {port_rets.index.max().date()}"
    )

    # Forward-fill regime to today's price dates for display
    regime_daily = regime_df.reindex(prices.index, method="ffill")

    print("\n[4/5] Computing metrics ...")
    pm = metrics(port_rets)
    bm = metrics(bench_rets)
    sm = metrics(spy_rets)

    latest = regime_df.iloc[-1]
    latest_regime = str(latest["regime"])
    latest_risk_on = float(latest.get("risk_on", 0.5))

    print("\n  -- Strategy Performance --")
    print(
        f"  CAGR      {pm['cagr'] * 100:+.1f}%  (Bench {bm['cagr'] * 100:+.1f}%  SPY {sm['cagr'] * 100:+.1f}%)"
    )
    print(
        f"  Sharpe    {pm['sharpe']:.2f}  (Bench {bm['sharpe']:.2f}  SPY {sm['sharpe']:.2f})"
    )
    print(
        f"  Max DD    {pm['max_dd'] * 100:.1f}%  (Bench {bm['max_dd'] * 100:.1f}%  SPY {sm['max_dd'] * 100:.1f}%)"
    )
    print(f"  Vol       {pm['vol'] * 100:.1f}%")
    print(f"  Calmar    {pm['calmar']:.2f}")
    print(f"  Latest regime: {latest_regime}  risk_on={latest_risk_on:.3f}")
    print(f"  Regime data through: {regime_df.index.max().date()}")
    print(f"  Price data through:  {prices.index.max().date()}")

    holdings_df = current_holdings(weights_df)

    # Build benchmark return series for heatmap toggle
    def _pct(col):
        return (
            comp_prices[col].pct_change().dropna()
            if col in comp_prices.columns
            else None
        )

    qqq_rets = _pct("QQQ")
    agg_rets = _pct("AGG")
    ief_rets = _pct("IEF")

    blend_6040 = None
    if agg_rets is not None:
        s = prices["SPY"].pct_change().dropna()
        idx = s.index.intersection(agg_rets.index)
        blend_6040 = (0.60 * s[idx] + 0.40 * agg_rets[idx]).rename("60/40")

    print("\n[5/5] Building charts ...")

    # Pre-compute all heatmap figures (benchmark toggle)
    global _HEATMAP_FIGS
    _heatmap_series = {
        "Strategy": port_rets,
        "SPY": spy_rets,
        "QQQ": qqq_rets,
        "EW Bench": bench_rets,
        "AGG": agg_rets,
        "IEF": ief_rets,
        "60/40": blend_6040,
    }
    _HEATMAP_FIGS = {
        label: fig_monthly_heatmap(rets, label=label)
        for label, rets in _heatmap_series.items()
        if rets is not None and not rets.empty
    }

    figs = {
        "rolling_sharpe": fig_rolling_sharpe(port_rets, bench_rets),
    }

    app = dash.Dash(
        __name__, external_stylesheets=[dbc.themes.DARKLY], title="Regime Allocator"
    )
    app.layout = build_layout(
        pm,
        bm,
        sm,
        latest_regime,
        latest_risk_on,
        figs,
        holdings_df,
        alloc_df,
        regime_daily,
    )
    register_callbacks(app)

    print("\n" + "=" * 60)
    print("  Dashboard ready ->  http://127.0.0.1:8050")
    print("  Press Ctrl+C to stop.")
    print("=" * 60 + "\n")
    app.run(debug=False, host="127.0.0.1", port=PORT)


if __name__ == "__main__":
    main()
