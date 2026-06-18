"""Monthly rebalance dry-run or paper order submission.

Orchestrates: load target weights, get current positions (IBKR or mock), run create_orders,
print/save summary. --dry-run (default): no orders. --submit-paper-orders: submit paper only
when trading_enabled and pre-trade checks pass. No live trading support.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from src.config import PROJECT_ROOT
from src.execution.create_orders import (
    DEFAULT_TAU,
    OrderPreviewRow,
    PositionRow,
    RebalancePreview,
    create_order_preview,
    load_prices,
    load_target_weights,
)
from src.execution.reporting import RunSummary, new_run_id, utc_now_str, write_run_summary
from src.execution.safety import (
    SafetyConfig,
    duplicate_run_refuse,
    pre_trade_checks,
    safety_config_from_paper_config,
    write_submission_marker,
)

logger = logging.getLogger(__name__)


def _load_paper_config() -> dict[str, Any]:
    """Load paper_trading.yaml with rebalance section."""
    try:
        import yaml
    except ImportError:
        raise ImportError("PyYAML required. Install with: uv add pyyaml") from None
    config_path = PROJECT_ROOT / "config" / "paper_trading.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Paper trading config not found: {config_path}")
    with open(config_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return raw


def _nav_from_summary(summary: list[dict[str, str]]) -> float:
    """Extract NAV from account summary (NetLiquidation or TotalCashValue fallback)."""
    for row in summary:
        tag = (row.get("tag") or "").strip()
        if tag == "NetLiquidation":
            try:
                return float(row.get("value") or 0)
            except (TypeError, ValueError):
                pass
    for row in summary:
        tag = (row.get("tag") or "").strip()
        if tag == "TotalCashValue":
            try:
                return float(row.get("value") or 0)
            except (TypeError, ValueError):
                pass
    return 0.0


def _adapter_positions_to_rows(positions: list[dict[str, Any]]) -> list[PositionRow]:
    """Convert IBKR adapter position dicts to PositionRow list (legacy: no market value/price)."""
    out: list[PositionRow] = []
    for p in positions:
        symbol = (p.get("symbol") or "").strip()
        if not symbol:
            continue
        pos = float(p.get("position", 0))
        avg = float(p.get("avgCost", 0.0))
        out.append(PositionRow(symbol=symbol, position=pos, avg_cost=avg))
    return out


def _portfolio_items_to_rows(portfolio: list[dict[str, Any]]) -> list[PositionRow]:
    """Convert adapter get_portfolio() items to PositionRow with market value/price for valuation-correct weights."""
    out: list[PositionRow] = []
    for p in portfolio:
        symbol = (p.get("symbol") or "").strip()
        if not symbol:
            continue
        pos = float(p.get("position", 0))
        avg = float(p.get("avgCost", 0.0))
        mv = p.get("marketValue")
        mp = p.get("marketPrice")
        market_value = float(mv) if mv is not None else None
        market_price = float(mp) if mp is not None else None
        if market_value is not None and market_value < 0:
            market_value = None
        if market_price is not None and market_price <= 0:
            market_price = None
        out.append(
            PositionRow(
                symbol=symbol,
                position=pos,
                avg_cost=avg,
                market_value=market_value,
                market_price=market_price,
            )
        )
    return out


def load_mock_positions(path: Path) -> tuple[list[PositionRow], float]:
    """Load mock positions and optional NAV from JSON or CSV. Returns (positions, nav)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Mock positions file not found: {path}")

    suffix = path.suffix.lower()
    positions: list[PositionRow] = []
    nav = 0.0

    if suffix == ".json":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        def _row(
            item: dict,
        ) -> PositionRow:
            s = (item.get("symbol") or item.get("ticker") or "").strip()
            pos = float(item.get("position", item.get("quantity", 0)))
            avg = float(item.get("avgCost", item.get("avg_cost", 0)) or 0)
            mv = item.get("marketValue", item.get("market_value"))
            mp = item.get("marketPrice", item.get("market_price"))
            market_value = float(mv) if mv is not None else None
            market_price = float(mp) if mp is not None else None
            if market_value is not None and market_value < 0:
                market_value = None
            if market_price is not None and market_price <= 0:
                market_price = None
            return PositionRow(
                symbol=s,
                position=pos,
                avg_cost=avg,
                market_value=market_value,
                market_price=market_price,
            )

        if isinstance(data, dict):
            nav = float(data.get("nav", data.get("NetLiquidation", 0)) or 0)
            for item in data.get("positions", data.get("positions_list", [])):
                s = (item.get("symbol") or item.get("ticker") or "").strip()
                if not s:
                    continue
                positions.append(_row(item))
        elif isinstance(data, list):
            for item in data:
                s = (item.get("symbol") or item.get("ticker") or "").strip()
                if not s:
                    continue
                positions.append(_row(item))
            # NAV must be provided separately or computed from positions
            nav = sum(p.position * p.avg_cost for p in positions)
            if nav <= 0:
                nav = 1.0  # safe default for dry-run
        else:
            raise ValueError("Mock positions JSON must be object or list")
    elif suffix == ".csv":
        import csv
        with open(path, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                s = (row.get("symbol") or row.get("ticker") or "").strip()
                if not s:
                    continue
                positions.append(
                    PositionRow(
                        symbol=s,
                        position=float(row.get("position", row.get("quantity", 0)) or 0),
                        avg_cost=float(row.get("avg_cost", row.get("avgCost", 0)) or 0),
                    )
                )
        nav = sum(p.position * p.avg_cost for p in positions) or 1.0
    else:
        raise ValueError(f"Unsupported mock positions format: {suffix}")

    if nav <= 0:
        nav = 1.0
    logger.info("Loaded mock positions: %d rows, NAV=%.2f from %s", len(positions), nav, path)
    return positions, nav


def fetch_live_positions_and_nav() -> tuple[list[PositionRow], float] | None:
    """Connect to IBKR, fetch portfolio (market value/price) and NAV. Returns None on failure."""
    try:
        from src.execution.ibkr_adapter import IBKRPaperAdapter
    except ImportError:
        logger.warning("IBKR adapter not available")
        return None
    adapter = IBKRPaperAdapter()
    try:
        adapter.connect()
    except Exception as e:
        logger.warning("Broker connection failed, will use mock if configured: %s", e)
        return None
    try:
        summary = adapter.get_account_summary()
        nav = _nav_from_summary(summary)
        # Prefer portfolio (market value/price) for valuation-correct weights and order sizing
        portfolio = adapter.get_portfolio()
        if portfolio:
            rows = _portfolio_items_to_rows(portfolio)
            if nav <= 0:
                nav = sum(
                    (p.market_value if p.market_value is not None else p.position * p.avg_cost)
                    for p in rows
                ) or 1.0
            logger.info("Using broker portfolio with market value/price for %d positions", len(rows))
            return rows, nav
        positions = adapter.get_positions()
        if nav <= 0:
            rows = _adapter_positions_to_rows(positions)
            nav = sum(p.position * p.avg_cost for p in rows) or 1.0
        return _adapter_positions_to_rows(positions), nav
    finally:
        adapter.disconnect()


def save_report(preview: RebalancePreview, report_dir: Path, run_id: str) -> list[Path]:
    """Write target vs current, proposed orders, turnover, and summary to report_dir. Returns paths."""
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    # Weights comparison
    weights_path = report_dir / f"rebalance_weights_{run_id}.json"
    with open(weights_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "nav": preview.nav,
                "tau": preview.tau,
                "current_weights": preview.current_weights,
                "target_weights": preview.target_weights,
                "weight_drift": preview.weight_drift,
                "tau_filtered_delta": preview.tau_filtered_delta,
            },
            f,
            indent=2,
        )
    saved.append(weights_path)

    # Proposed orders
    orders_path = report_dir / f"rebalance_orders_{run_id}.json"
    orders_data = [
        {
            "symbol": o.symbol,
            "side": o.side,
            "shares": o.shares,
            "approximate_dollar": o.approximate_dollar,
            "weight_delta": o.weight_delta,
            "price_used": o.price_used,
        }
        for o in preview.proposed_orders
    ]
    with open(orders_path, "w", encoding="utf-8") as f:
        json.dump(orders_data, f, indent=2)
    saved.append(orders_path)

    # Summary
    num_tradable = len(preview.target_weights) - (1 if "cash" in preview.target_weights else 0)
    summary_path = report_dir / f"rebalance_summary_{run_id}.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"Rebalance dry-run report — {run_id}\n")
        f.write("=" * 60 + "\n")
        f.write(f"NAV: {preview.nav:.2f}\n")
        f.write(f"Tau: {preview.tau}\n")
        f.write(f"Turnover (one-way): {preview.turnover_one_way:.4f}\n")
        f.write(f"Target weights: {len(preview.target_weights)} entries ({num_tradable} tradable + cash).\n")
        f.write(f"Proposed orders: {len(preview.proposed_orders)} (cash is not traded as a security).\n")
        if preview.symbols_missing_price:
            f.write(f"Symbols missing price (excluded from orders): {preview.symbols_missing_price}\n")
        f.write("\n--- Proposed orders ---\n")
        for o in preview.proposed_orders:
            f.write(f"  {o.side} {o.shares} {o.symbol} @ ~{o.price_used:.2f} (~${o.approximate_dollar:.2f})\n")
    saved.append(summary_path)
    logger.info("Saved report to %s: %s", report_dir, [p.name for p in saved])
    return saved


def run_dry_run(
    target_weights_path: Path,
    tau: float = DEFAULT_TAU,
    mock_positions_path: Path | None = None,
    prices_path: Path | None = None,
    report_dir: Path | None = None,
    use_mock_only: bool = False,
) -> RebalancePreview:
    """Load targets, get positions (broker or mock), build order preview, save report."""
    target_weights_path = Path(target_weights_path)
    target_weights = load_target_weights(target_weights_path)
    if not target_weights:
        raise ValueError("Target weights are empty")

    positions: list[PositionRow]
    nav: float

    if use_mock_only:
        path = Path(mock_positions_path) if mock_positions_path else None
        if not path or not path.exists():
            raise FileNotFoundError("use_mock_only=True but mock_positions_path not provided or missing")
        positions, nav = load_mock_positions(path)
        logger.info("Using mock positions only: %d positions, NAV=%.2f", len(positions), nav)
    else:
        live = fetch_live_positions_and_nav()
        if live is not None:
            positions, nav = live
            logger.info("Using live broker data: %d positions, NAV=%.2f", len(positions), nav)
        else:
            mock_path = Path(mock_positions_path) if mock_positions_path else None
            if mock_path and mock_path.exists():
                positions, nav = load_mock_positions(mock_path)
                logger.info("Using mock positions (broker unavailable): %s", mock_path)
            else:
                positions = []
                nav = 1.0
                logger.warning("Broker unavailable and no mock file; using empty positions, NAV=1")

    prices = None
    if prices_path:
        pp = Path(prices_path)
        if pp.suffix and pp.exists():
            prices = load_prices(pp)
            logger.info("Loaded %d prices from %s", len(prices or {}), pp)
    preview = create_order_preview(
        target_weights=target_weights,
        current_positions=positions,
        nav=nav,
        tau=tau,
        prices=prices,
    )

    if report_dir:
        run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        save_report(preview, Path(report_dir), run_id)

    return preview


def _smoke_test_checks(config: dict[str, Any]) -> None:
    """Raise ValueError if config does not allow smoke-test submission (trading_enabled, paper_only)."""
    if not config.get("trading_enabled", False):
        raise ValueError("Smoke-test submission refused: trading_enabled is false in config. Set to true to allow.")
    if not config.get("paper_only", True):
        raise ValueError("Smoke-test submission refused: paper_only must be true. Live trading is not supported.")


def _run_smoke_test_submission(
    symbol: str,
    shares: int,
    side: str,
    report_dir: Path,
    config: dict[str, Any],
    rebal: dict[str, Any],
) -> int:
    """Submit a single small paper order (smoke-test). Log and write smoke report; optionally fetch positions."""
    from src.execution.submit_orders import submit_paper_orders as do_submit
    side_upper = (side or "BUY").strip().upper()
    if side_upper not in ("BUY", "SELL"):
        raise ValueError(f"Smoke-test side must be BUY or SELL, got: {side}")
    if shares <= 0 or shares > 10:
        raise ValueError(f"Smoke-test shares must be 1-10, got: {shares}")
    symbol = (symbol or "").strip()
    if not symbol:
        raise ValueError("Smoke-test symbol is required (e.g. --smoke-test-symbol SPY)")
    # One order: market uses price_used 0; submit_orders uses MKT so price not needed
    smoke_order = [{"symbol": symbol, "side": side_upper, "shares": shares, "price_used": 0.0}]
    order_type = rebal.get("order_type", "MKT")
    account = config.get("account") or ""
    logger.info("Smoke-test: submitting single paper order %s %s %s", side_upper, shares, symbol)
    results = do_submit(smoke_order, order_type=order_type, account=account)
    run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    smoke_path = report_dir / f"smoke_test_{run_id}.json"
    report_data = {
        "run_id": run_id,
        "mode": "smoke_test",
        "symbol": symbol,
        "side": side_upper,
        "shares": shares,
        "results": [{"symbol": r.symbol, "side": r.side, "shares": r.shares, "order_id": r.order_id, "status": r.status, "message": r.message} for r in results],
    }
    try:
        from src.execution.reconcile_post_trade import fetch_post_trade_positions_and_nav
        recon_client_id = config.get("reconciliation_client_id") or (config.get("client_id", 1) + 1)
        pos_nav = fetch_post_trade_positions_and_nav(client_id_override=recon_client_id)
        if pos_nav:
            positions, nav = pos_nav
            report_data["post_submission_nav"] = nav
            report_data["post_submission_positions_count"] = len(positions)
    except Exception as e:
        logger.warning("Smoke-test: could not fetch post-submission positions: %s", e)
    with open(smoke_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
    logger.info("Smoke-test report saved: %s", smoke_path)
    print("\n--- Smoke-test: paper order submitted ---")
    for r in results:
        print(f"  {r.side} {r.shares} {r.symbol} orderId={r.order_id} status={r.status}")
    print("\nDone. Smoke-test complete. No full rebalance submitted.")
    return 0


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Monthly rebalance: dry-run (default) or paper order submission")
    p.add_argument("--dry-run", action="store_true", default=True, help="Dry run only (default)")
    p.add_argument("--no-dry-run", action="store_false", dest="dry_run", help="Allow submission when combined with --submit-paper-orders")
    p.add_argument("--submit-paper-orders", action="store_true", help="Submit paper orders (requires trading_enabled, live broker, no duplicate run)")
    p.add_argument("--smoke-test-symbol", type=str, default="", help="Smoke-test: submit one small order for this symbol only (e.g. SPY)")
    p.add_argument("--smoke-test-shares", type=int, default=1, help="Smoke-test: number of shares (1-10, default 1)")
    p.add_argument("--smoke-test-side", type=str, default="BUY", help="Smoke-test: BUY or SELL (default BUY)")
    p.add_argument("--target-weights", type=str, default="", help="Path to target weights CSV/JSON")
    p.add_argument("--mock-positions", type=str, default="", help="Path to mock positions JSON/CSV (fallback; not used when submitting)")
    p.add_argument("--use-mock-only", action="store_true", help="Skip broker; use mock positions only (incompatible with --submit-paper-orders)")
    p.add_argument("--prices", type=str, default="", help="Optional JSON symbol->price for share conversion")
    p.add_argument("--report-dir", type=str, default="", help="Directory for report artifacts")
    return p


def _validate_arg_combinations(args, submit_paper: bool, smoke_mode: bool) -> int | None:
    """Return non-zero exit code if combination is invalid; None if OK."""
    if submit_paper and args.use_mock_only:
        print("FAILED: --submit-paper-orders cannot be used with --use-mock-only. Submission requires live broker.", file=sys.stderr)
        return 1
    if submit_paper and args.dry_run:
        print("FAILED: --submit-paper-orders requires --no-dry-run.", file=sys.stderr)
        return 1
    if smoke_mode and (not submit_paper or args.dry_run):
        print("FAILED: Smoke-test requires --submit-paper-orders and --no-dry-run.", file=sys.stderr)
        return 1
    return None


def _resolve_input_paths(args, rebal: dict[str, Any], root: Path) -> dict[str, Any]:
    """Resolve target / mock / prices paths from CLI overrides + config defaults."""
    raw_target = args.target_weights or rebal.get(
        "target_weights_path", "outputs/rebalance/target_weights.csv"
    )
    target_path = Path(raw_target)
    if not target_path.is_absolute():
        target_path = root / target_path
    raw_mock = args.mock_positions or rebal.get("mock_positions_path")
    mock_path = None
    if raw_mock and (isinstance(raw_mock, Path) or str(raw_mock).strip()):
        mock_path = root / raw_mock if not isinstance(raw_mock, Path) else raw_mock
        if not mock_path.is_absolute():
            mock_path = root / mock_path
    prices_path = args.prices or rebal.get("prices_path")
    if prices_path and not isinstance(prices_path, Path):
        prices_path = root / prices_path if prices_path else None
    return {"target_path": target_path, "mock_path": mock_path, "prices_path": prices_path}


def _gate_submission(preview, config, rebal, report_dir, summary, safety_cfg) -> "Path | None":
    """Run safety + duplicate-run gates; return marker_path if OK, None if blocked."""
    pre_trade_checks(
        preview,
        trading_enabled=bool(config.get("trading_enabled", False)),
        paper_only=bool(config.get("paper_only", True)),
        cfg=safety_cfg,
    )
    last_submission_file = rebal.get("last_submission_file", "last_paper_submission.json")
    marker_path = (
        Path(last_submission_file)
        if Path(last_submission_file).is_absolute()
        else Path(report_dir) / Path(last_submission_file).name
    )
    if rebal.get("duplicate_run_protection", True) and duplicate_run_refuse(marker_path):
        print("FAILED: Duplicate run protection. Paper submission already performed today.", file=sys.stderr)
        summary.mode = "safety_fail"
        summary.status = "blocked"
        summary.message = "Duplicate run protection triggered"
        return None
    return marker_path


def _execute_paper_submission(
    preview, config, rebal, report_dir, run_id, summary, safety_cfg
) -> int:
    """Run safety + submission + reconciliation. Returns process exit code."""
    marker_path = _gate_submission(preview, config, rebal, report_dir, summary, safety_cfg)
    if marker_path is None:
        return 1
    from src.execution.submit_orders import submit_paper_orders as do_submit
    order_type = rebal.get("order_type", "MKT")
    account = config.get("account") or ""
    results = do_submit(preview.proposed_orders, order_type=order_type, account=account)
    write_submission_marker(marker_path, run_id=run_id, submitted_order_count=len(results), dry_run=False)
    _post_trade_reconcile(preview, config, report_dir, summary)
    print("\n--- Paper orders submitted ---")
    for r in results:
        print(f"  {r.side} {r.shares} {r.symbol} orderId={r.order_id} status={r.status}")
    print("\nDone. Paper orders submitted.")
    return 0


def _post_trade_reconcile(preview, config, report_dir, summary) -> None:
    """Best-effort post-trade reconciliation; failures are non-fatal."""
    try:
        from src.execution.reconcile_post_trade import run_reconciliation
        recon_client_id = config.get("reconciliation_client_id") or (config.get("client_id", 1) + 1)
        rec = run_reconciliation(
            preview.target_weights, report_dir=report_dir, client_id_override=recon_client_id
        )
        logger.info("Reconciliation max_abs_delta=%.4f", rec.max_abs_delta)
        summary.metrics["reconciliation_max_abs_delta"] = rec.max_abs_delta
    except Exception as e:
        logger.warning("Post-trade reconciliation failed (non-fatal): %s", e)
        summary.outputs["reconciliation_error"] = str(e)


def _print_dry_run_summary(preview) -> None:
    num_target_entries = len(preview.target_weights)
    num_tradable = num_target_entries - (1 if "cash" in preview.target_weights else 0)
    print("\n" + "=" * 60)
    print("Rebalance dry-run summary")
    print("=" * 60)
    print(f"NAV: {preview.nav:.2f}")
    print(f"Tau: {preview.tau}")
    print(f"Turnover (one-way): {preview.turnover_one_way:.4f}")
    print(f"Target weights: {num_target_entries} entries ({num_tradable} tradable + cash).")
    print(f"Proposed orders: {len(preview.proposed_orders)} (cash is not traded as a security).")
    if preview.symbols_missing_price:
        print(f"Symbols missing price: {preview.symbols_missing_price}")
    print("\n--- Proposed orders ---")
    for o in preview.proposed_orders:
        print(f"  {o.side} {o.shares} {o.symbol} @ ~{o.price_used:.2f} (~${o.approximate_dollar:.2f})")
    print("\nDone. No orders submitted.")


def _init_run_summary(args) -> "RunSummary":
    return RunSummary(
        run_id=new_run_id(),
        runner="monthly_rebalance_runner",
        mode="dry_run" if getattr(args, "dry_run", True) else "live",
        status="ok",
        started_utc=utc_now_str(),
        inputs={},
        metrics={},
        outputs={},
    )


def _record_inputs(summary, rebal: dict[str, Any], safety_cfg) -> None:
    summary.inputs.update({
        "tau": float(rebal.get("tau", DEFAULT_TAU)),
        "order_type": str(rebal.get("order_type", "MKT")),
        "allow_leverage": safety_cfg.allow_leverage,
        "max_notional_per_order": safety_cfg.max_notional_per_order,
        "max_total_turnover": safety_cfg.max_total_turnover,
        "max_weight_delta": safety_cfg.max_weight_delta,
    })


def _record_preview_metrics(summary, preview) -> None:
    summary.metrics.update({
        "target_weight_sum": sum(preview.target_weights.values()),
        "turnover_one_way": preview.turnover_one_way,
        "order_count": len(preview.proposed_orders),
    })


def _do_main_flow(args, smoke_mode: bool, smoke_symbol: str, submit_paper: bool, summary):
    """Inner body of main(): returns (exit_code | None, preview | None, report_dir)."""
    config = _load_paper_config()
    rebal = config.get("rebalance") or {}
    safety_cfg = safety_config_from_paper_config(config)
    _record_inputs(summary, rebal, safety_cfg)
    report_dir = args.report_dir or rebal.get("report_dir", "outputs/rebalance/reports")
    if not isinstance(report_dir, Path):
        report_dir = PROJECT_ROOT / report_dir
    if smoke_mode:
        _smoke_test_checks(config)
        code = _run_smoke_test_submission(
            symbol=smoke_symbol,
            shares=getattr(args, "smoke_test_shares", 1) or 1,
            side=getattr(args, "smoke_test_side", "BUY") or "BUY",
            report_dir=report_dir, config=config, rebal=rebal,
        )
        return code, None, report_dir
    paths = _resolve_input_paths(args, rebal, PROJECT_ROOT)
    preview = run_dry_run(
        target_weights_path=paths["target_path"],
        tau=float(rebal.get("tau", DEFAULT_TAU)),
        mock_positions_path=paths["mock_path"],
        prices_path=paths["prices_path"],
        report_dir=report_dir, use_mock_only=args.use_mock_only,
    )
    _record_preview_metrics(summary, preview)
    if submit_paper:
        return _execute_paper_submission(
            preview, config, rebal, report_dir, summary.run_id, summary, safety_cfg
        ), preview, report_dir
    return None, preview, report_dir


def main() -> int:
    args = _build_arg_parser().parse_args()
    submit_paper = getattr(args, "submit_paper_orders", False)
    smoke_symbol = (args.smoke_test_symbol or "").strip()
    smoke_mode = bool(smoke_symbol)
    invalid = _validate_arg_combinations(args, submit_paper, smoke_mode)
    if invalid is not None:
        return invalid
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    summary = _init_run_summary(args)
    report_dir = PROJECT_ROOT / "outputs"
    preview = None
    code: int | None = None
    try:
        code, preview, report_dir = _do_main_flow(args, smoke_mode, smoke_symbol, submit_paper, summary)
    except (FileNotFoundError, ValueError, KeyError, ConnectionError) as e:
        logger.exception("Run failed: %s", e)
        print(f"FAILED: {e}", file=sys.stderr)
        summary.mode = "error"
        summary.status = "error"
        summary.message = str(e)
        return 1
    finally:
        summary.finished_utc = utc_now_str()
        try:
            write_run_summary(Path(report_dir), summary)
        except Exception as e:
            logger.warning("Could not write run summary: %s", e)
    if code is not None:
        return code
    if preview is not None:
        _print_dry_run_summary(preview)
    return 0


if __name__ == "__main__":
    sys.exit(main())
