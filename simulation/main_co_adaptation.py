import argparse
import copy
import json
import logging
from pathlib import Path
from typing import List, Sequence

from Client import Client
from Server import Server
from co_adaptation_strategy import ActiveCoAdaptationRunner, StrategyConfig
from utils.utils import arg as BaseArgs
from utils.utils import pkl_load

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # pragma: no cover
    SummaryWriter = None


MODEL_DEFAULTS = {
    "MLP": {"seq_len": 168},
    "LSTM": {"seq_len": 168},
    "CNN": {"seq_len": 168},
    "DLinear": {"seq_len": 96},
    "PatchTST": {
        "seq_len": 168,
        "online_lr": 1e-4,
        "d_model": 128,
        "d_ff": 256,
        "n_heads": 4,
        "e_layers": 3,
        "drop_out": 0.1,
        "patch_len": 16,
        "stride": 8,
        "train_epochs": 10,
        "fed_rounds": 10,
        "fed_local_epochs": 1,
        "energy_loss_weight": 0.0,
    },
}


def _model_default(model_name: str, key: str, fallback):
    return MODEL_DEFAULTS.get(model_name, {}).get(key, fallback)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Active cloud-edge co-adaptation runner")
    parser.add_argument("--model", default="LSTM", choices=list(MODEL_DEFAULTS.keys()))
    parser.add_argument("--fed-type", default="none", choices=["none", "fedavgm", "fedprox", "fedsgd"])
    parser.add_argument("--root-path", default="data")
    parser.add_argument("--clients", nargs="*", default=None)
    parser.add_argument("--client-limit", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--pred-len", type=int, default=24)
    parser.add_argument("--label-len", type=int, default=0)
    parser.add_argument("--hidden-dims", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--online-lr", type=float, default=1e-3)
    parser.add_argument("--energy-loss-weight", type=float, default=None)
    parser.add_argument("--energy-loss-relative", action="store_true")
    parser.add_argument("--train-epochs", type=int, default=None)
    parser.add_argument("--fed-rounds", type=int, default=None)
    parser.add_argument("--fed-local-epochs", type=int, default=None)
    parser.add_argument("--global-decay-lr", type=float, default=0.1)
    parser.add_argument("--local-decay-lr", type=float, default=0.1)
    parser.add_argument("--long-w-lr", type=float, default=0.5)
    parser.add_argument("--short-w-lr", type=float, default=1e-4)
    parser.add_argument("--fedavgm-lr", type=float, default=0.1)
    parser.add_argument("--fedavgm-beta", type=float, default=0.9)
    parser.add_argument("--fedsgd-lr", type=float, default=1.0)
    parser.add_argument("--fedprox-reg", type=float, default=0.01)
    parser.add_argument("--model-buffer-size", type=int, default=7)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-threads", type=int, default=8)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-ratio", type=float, default=None)
    parser.add_argument("--d-model", type=int, default=None)
    parser.add_argument("--d-ff", type=int, default=None)
    parser.add_argument("--n-heads", type=int, default=None)
    parser.add_argument("--e-layers", type=int, default=None)
    parser.add_argument("--drop-out", type=float, default=None)
    parser.add_argument("--patch-len", type=int, default=None)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--consider-error", action="store_true")
    parser.add_argument("--warmup-ensemble", action="store_true")
    parser.add_argument("--ensemble-epochs", type=int, default=2)
    parser.add_argument("--train-if-missing", action="store_true")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--output-root", default="results/co_adaptation")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=72)
    parser.add_argument("--edge-adapter", default="online", choices=["online", "agem", "mir", "der"])
    parser.add_argument("--edge-memory-size", type=int, default=7)
    parser.add_argument("--edge-replay-size", type=int, default=2)
    parser.add_argument("--edge-replay-weight", type=float, default=0.5)
    parser.add_argument("--edge-distill-weight", type=float, default=0.5)
    parser.add_argument("--edge-adapter-seed", type=int, default=0)
    parser.add_argument("--edge-only", action="store_true")
    parser.add_argument("--recent-window", type=int, default=6)
    parser.add_argument("--resource-window", type=int, default=12)
    parser.add_argument("--cloud-memory-size", type=int, default=5)
    parser.add_argument("--sync-budget-clients", type=int, default=4)
    parser.add_argument("--max-sync-budget-clients", type=int, default=None)
    parser.add_argument("--adaptive-budget-gain", type=float, default=0.0)
    parser.add_argument("--residual-priority-weight", type=float, default=0.0)
    parser.add_argument("--forced-refresh-rounds", type=int, default=0)
    parser.add_argument("--sync-threshold", type=float, default=0.80)
    parser.add_argument("--downlink-drift-threshold", type=float, default=0.60)
    parser.add_argument("--downlink-utility-threshold", type=float, default=0.01)
    parser.add_argument("--stale-penalty", type=float, default=0.01)
    parser.add_argument("--stale-gamma", type=float, default=1.0)
    parser.add_argument("--server-lambda", type=float, default=0.5)
    parser.add_argument("--eta-min-scale", type=float, default=0.5)
    parser.add_argument("--eta-max-scale", type=float, default=2.0)
    parser.add_argument("--eta-slope", type=float, default=2.0)
    parser.add_argument("--eta-bias", type=float, default=0.5)
    parser.add_argument("--mu-max", type=float, default=0.05)
    parser.add_argument("--mu-cloud-gain", type=float, default=4.0)
    parser.add_argument("--mu-drift-penalty", type=float, default=2.5)
    parser.add_argument("--rho-m", type=float, default=0.9)
    parser.add_argument("--rho-v", type=float, default=0.9)
    parser.add_argument("--age-horizon", type=int, default=12)
    parser.add_argument("--beta-resource", type=float, default=1.0)
    parser.add_argument("--drift-weight-loss", type=float, default=0.40)
    parser.add_argument("--drift-weight-disagreement", type=float, default=0.30)
    parser.add_argument("--drift-weight-volatility", type=float, default=0.20)
    parser.add_argument("--drift-weight-age", type=float, default=0.10)
    parser.add_argument("--agg-beta-drift", type=float, default=1.0)
    parser.add_argument("--agg-beta-utility", type=float, default=0.75)
    parser.add_argument("--agg-beta-volatility", type=float, default=0.50)
    parser.add_argument("--agg-beta-staleness", type=float, default=0.25)
    parser.add_argument("--grad-clip-norm", type=float, default=None)
    parser.add_argument("--snapshot-blend", type=float, default=1.0)
    parser.add_argument("--performance-fallback-mix", type=float, default=0.0)
    parser.add_argument("--unsynced-global-lr-scale", type=float, default=1.0)
    parser.add_argument("--post-sync-broadcast", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-latest-cloud-snapshot", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--retrieval-policy",
        default="historical_match",
        choices=["historical_match", "latest", "random"],
    )
    parser.add_argument("--retrieval-seed", type=int, default=0)
    parser.add_argument("--downlink-loss-rate", type=float, default=0.0)
    parser.add_argument("--network-seed", type=int, default=0)
    parser.add_argument("--edge-only-on-downlink-failure", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def create_output_dir(output_root: str, run_name: str = None) -> Path:
    base = Path(output_root)
    base.mkdir(parents=True, exist_ok=True)
    if run_name is None:
        run_name = "run"
        next_idx = 1
        while (base / f"{run_name}_{next_idx:03d}").exists():
            next_idx += 1
        run_name = f"{run_name}_{next_idx:03d}"
    output_dir = base / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def setup_logging(output_dir: Path) -> logging.Logger:
    logger = logging.getLogger("co_adaptation")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(output_dir / "run.log")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def discover_clients(root_path: str, requested: Sequence[str] = None) -> List[str]:
    root = Path(root_path)
    if requested:
        return list(requested)

    data_clients = {path.stem for path in root.glob("*.csv")}
    save_root = Path("save")
    saved_clients = {
        path.name
        for path in save_root.iterdir()
        if path.is_dir() and path.name != "offline federated"
    }
    candidates = sorted(data_clients & saved_clients)
    if candidates:
        return candidates
    return sorted(data_clients)


def build_base_args(cli_args: argparse.Namespace) -> BaseArgs:
    fed_type = None if cli_args.fed_type == "none" else cli_args.fed_type
    seq_len = cli_args.seq_len or _model_default(cli_args.model, "seq_len", 96)
    online_lr = getattr(cli_args, "online_lr", 1e-3)
    if online_lr == 1e-3 and cli_args.model == "PatchTST":
        online_lr = _model_default(cli_args.model, "online_lr", online_lr)
    return BaseArgs(
        batch_size=cli_args.batch_size,
        lr=cli_args.lr,
        online_lr=online_lr,
        global_decay_lr=cli_args.global_decay_lr,
        local_decay_lr=cli_args.local_decay_lr,
        long_w_lr=cli_args.long_w_lr,
        short_w_lr=cli_args.short_w_lr,
        fedprox_reg=cli_args.fedprox_reg,
        fedsgd_lr=cli_args.fedsgd_lr,
        fedavgm_lr=cli_args.fedavgm_lr,
        fedavgm_beta=cli_args.fedavgm_beta,
        consider_error=cli_args.consider_error,
        gpu=cli_args.gpu,
        seed=cli_args.seed,
        max_threads=cli_args.max_threads,
        hidden_dims=cli_args.hidden_dims,
        pred_len=cli_args.pred_len,
        label_len=cli_args.label_len,
        seq_len=seq_len,
        model_buffer_size=cli_args.model_buffer_size,
        root_path=cli_args.root_path,
        num_workers=cli_args.num_workers,
        ensemble_epochs=cli_args.ensemble_epochs,
        train_epochs=getattr(cli_args, "train_epochs", None) or _model_default(cli_args.model, "train_epochs", 10),
        fed_type=fed_type,
        fed_rounds=getattr(cli_args, "fed_rounds", None) or _model_default(cli_args.model, "fed_rounds", 10),
        fed_local_epochs=getattr(cli_args, "fed_local_epochs", None) or _model_default(cli_args.model, "fed_local_epochs", 1),
        train_ratio=cli_args.train_ratio,
        energy_loss_weight=getattr(cli_args, "energy_loss_weight", None) if getattr(cli_args, "energy_loss_weight", None) is not None else _model_default(cli_args.model, "energy_loss_weight", 0.0),
        energy_loss_relative=getattr(cli_args, "energy_loss_relative", False),
        d_model=getattr(cli_args, "d_model", None) or _model_default(cli_args.model, "d_model", 512),
        d_ff=getattr(cli_args, "d_ff", None) or _model_default(cli_args.model, "d_ff", 2048),
        n_heads=getattr(cli_args, "n_heads", None) or _model_default(cli_args.model, "n_heads", 8),
        e_layers=getattr(cli_args, "e_layers", None) or _model_default(cli_args.model, "e_layers", 2),
        drop_out=getattr(cli_args, "drop_out", None) if getattr(cli_args, "drop_out", None) is not None else _model_default(cli_args.model, "drop_out", 0.1),
        patch_len=getattr(cli_args, "patch_len", None) or _model_default(cli_args.model, "patch_len", 16),
        stride=getattr(cli_args, "stride", None) or _model_default(cli_args.model, "stride", 8),
    )


def build_strategy_config(cli_args: argparse.Namespace) -> StrategyConfig:
    return StrategyConfig(
        edge_adapter=cli_args.edge_adapter,
        edge_memory_size=cli_args.edge_memory_size,
        edge_replay_size=cli_args.edge_replay_size,
        edge_replay_weight=cli_args.edge_replay_weight,
        edge_distill_weight=cli_args.edge_distill_weight,
        edge_adapter_seed=cli_args.edge_adapter_seed,
        enable_cloud_collaboration=not cli_args.edge_only,
        recent_window=cli_args.recent_window,
        resource_window=cli_args.resource_window,
        cloud_memory_size=cli_args.cloud_memory_size,
        sync_budget_clients=cli_args.sync_budget_clients,
        max_sync_budget_clients=getattr(cli_args, "max_sync_budget_clients", None),
        adaptive_budget_gain=getattr(cli_args, "adaptive_budget_gain", 0.0),
        residual_priority_weight=getattr(cli_args, "residual_priority_weight", 0.0),
        forced_refresh_rounds=getattr(cli_args, "forced_refresh_rounds", 0),
        sync_threshold=cli_args.sync_threshold,
        downlink_drift_threshold=cli_args.downlink_drift_threshold,
        downlink_utility_threshold=cli_args.downlink_utility_threshold,
        stale_penalty=cli_args.stale_penalty,
        stale_gamma=cli_args.stale_gamma,
        server_lambda=cli_args.server_lambda,
        eta_min_scale=cli_args.eta_min_scale,
        eta_max_scale=cli_args.eta_max_scale,
        eta_slope=cli_args.eta_slope,
        eta_bias=cli_args.eta_bias,
        mu_max=cli_args.mu_max,
        mu_cloud_gain=cli_args.mu_cloud_gain,
        mu_drift_penalty=cli_args.mu_drift_penalty,
        rho_m=cli_args.rho_m,
        rho_v=cli_args.rho_v,
        age_horizon=cli_args.age_horizon,
        beta_resource=cli_args.beta_resource,
        drift_weight_loss=cli_args.drift_weight_loss,
        drift_weight_disagreement=cli_args.drift_weight_disagreement,
        drift_weight_volatility=cli_args.drift_weight_volatility,
        drift_weight_age=cli_args.drift_weight_age,
        agg_beta_drift=cli_args.agg_beta_drift,
        agg_beta_utility=cli_args.agg_beta_utility,
        agg_beta_volatility=cli_args.agg_beta_volatility,
        agg_beta_staleness=cli_args.agg_beta_staleness,
        grad_clip_norm=cli_args.grad_clip_norm,
        snapshot_blend=cli_args.snapshot_blend,
        performance_fallback_mix=cli_args.performance_fallback_mix,
        unsynced_global_lr_scale=cli_args.unsynced_global_lr_scale,
        post_sync_broadcast=getattr(cli_args, "post_sync_broadcast", True),
        use_latest_cloud_snapshot=getattr(cli_args, "use_latest_cloud_snapshot", False),
        retrieval_policy=getattr(cli_args, "retrieval_policy", "historical_match"),
        retrieval_seed=getattr(cli_args, "retrieval_seed", 0),
        downlink_loss_rate=getattr(cli_args, "downlink_loss_rate", 0.0),
        network_seed=getattr(cli_args, "network_seed", 0),
        edge_only_on_downlink_failure=getattr(cli_args, "edge_only_on_downlink_failure", False),
        max_steps=cli_args.max_steps,
        log_every=cli_args.log_every,
    )


def _json_safe_dict(raw: dict) -> dict:
    safe = {}
    for key, value in raw.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            safe[key] = value
        else:
            safe[key] = str(value)
    return safe


def load_or_prepare_models(
    clients: Sequence[Client],
    server: Server,
    model_name: str,
    fed_type: str,
    train_if_missing: bool,
    logger: logging.Logger,
) -> None:
    global_filename = "model.pkl" if fed_type == "none" else f"{fed_type}_model.pkl"
    global_model_path = Path("save") / "offline federated" / model_name / global_filename

    if global_model_path.exists():
        server.set_model(pkl_load(str(global_model_path)).state_dict())
        logger.info("Loaded global model from %s", global_model_path)
    elif train_if_missing:
        logger.info("Global model not found. Training federated initialization.")
        server.offline_fed()
    else:
        raise FileNotFoundError(f"Missing global model: {global_model_path}")

    for client in clients:
        local_model_path = Path("save") / client.data_name / model_name / "checkpoints" / "model.pkl"
        if local_model_path.exists():
            client.set_model(pkl_load(str(local_model_path)).state_dict(), type="local")
        elif train_if_missing:
            logger.info("Local model missing for %s. Training local initialization.", client.data_name)
            client.train(save=True, type="local")
        else:
            raise FileNotFoundError(f"Missing local model for {client.data_name}: {local_model_path}")
        client.set_model(server.global_model.state_dict(), type="global")


def maybe_warmup_ensemble(clients: Sequence[Client], logger: logging.Logger) -> None:
    for client in clients:
        logger.info("Warmup ensemble for %s", client.data_name)
        client.train_ensemble()


def main() -> None:
    cli_args = parse_args()
    output_dir = create_output_dir(cli_args.output_root, cli_args.run_name)
    logger = setup_logging(output_dir)
    base_args = build_base_args(cli_args)
    strategy_config = build_strategy_config(cli_args)

    clients_to_use = discover_clients(cli_args.root_path, requested=cli_args.clients)
    if cli_args.client_limit is not None:
        clients_to_use = clients_to_use[: cli_args.client_limit]
    if not clients_to_use:
        raise RuntimeError("No clients were discovered for co-adaptation.")

    logger.info("Using clients: %s", ", ".join(clients_to_use))

    clients = [
        Client(data_name=name, args=copy.deepcopy(base_args), model_name=cli_args.model)
        for name in clients_to_use
    ]
    server = Server(clients=clients, args=base_args)
    load_or_prepare_models(
        clients=clients,
        server=server,
        model_name=cli_args.model,
        fed_type=cli_args.fed_type,
        train_if_missing=cli_args.train_if_missing,
        logger=logger,
    )

    if cli_args.warmup_ensemble:
        maybe_warmup_ensemble(clients, logger)

    writer = SummaryWriter(log_dir=output_dir / "tensorboard") if SummaryWriter else None
    runner = ActiveCoAdaptationRunner(
        clients=clients,
        server=server,
        args=base_args,
        strategy_config=strategy_config,
        output_dir=output_dir,
        writer=writer,
        logger=logger,
    )

    with open(output_dir / "config.json", "w", encoding="utf-8") as fp:
        json.dump(
            {
                "cli_args": vars(cli_args),
                "base_args": _json_safe_dict(base_args.__dict__),
                "strategy_config": vars(strategy_config),
            },
            fp,
            indent=2,
        )

    results = runner.run()
    summary = json.loads((output_dir / "run_summary.json").read_text(encoding="utf-8"))
    logger.info("Finished run. Summary: %s", summary)
    logger.info("Results saved under %s", output_dir)
    logger.info("Metric columns: %s", ", ".join(results["metrics"].columns))


if __name__ == "__main__":
    main()
