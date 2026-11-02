# Methodology ↔ Code Mapping

This document maps the equations of the paper to the concrete implementation.
All simulation code lives in [`simulation/`](../simulation/); the deployed
equivalents live in `hardware/shared/`, `hardware/server_app/`,
`hardware/client_app/` and `pi_client/client_runtime/`.

## System roles

| Paper | Simulation | Hardware runtime |
|-------|-----------|------------------|
| Client $i$, edge model $\theta^e_{i,t}$ | `Client.py` → `client.local_model` / `local_trainer` | `pi_client/client_runtime/agent.py` (edge model) |
| Client-side cloud branch $\theta^g_{i,t}$ | `Client.py` → `client.global_model` / `global_trainer` | `client_runtime/agent.py` (global branch) |
| Server global model $\theta^c_t$ + memory bank $\mathcal{M}_t$ | `Server.py` + `co_adaptation_strategy.py::ActiveCoAdaptationRunner.cloud_memory` | `hardware/server_app/coordinator.py` + `snapshot_store.py` |
| Per-cycle orchestration | `ActiveCoAdaptationRunner.run()` | `pi_client/client_runtime/agent.py::EdgeAgent.run()` |

## 1. Drift-aware communication

### Standardized degradation score (Eqs. 4–6)

$$s_{i,t} = \left[\frac{\ell^e_{i,t} - m_{i,t}}{\sqrt{v_{i,t}} + \epsilon}\right]_+$$

- `co_adaptation_strategy.py::ActiveCoAdaptationRunner._update_loss_shock()`
- EWA statistics with forgetting factors `rho_m` / `rho_v`
  (`StrategyConfig.rho_m`, `rho_v`).

### Composite drift score (Eqs. 7–12)

$$D_{i,t} = \omega_1 s_{i,t} + \omega_2 d_{i,t} + \omega_3 \nu_{i,t} + \omega_4 a_{i,t}$$

| Term | Meaning | Implementation |
|------|---------|----------------|
| $s_{i,t}$ | loss degradation | `_update_loss_shock()` |
| $d_{i,t}$ | cloud-edge disagreement (stabilized streams) | `_compute_disagreement()` |
| $\nu_{i,t}$ | residual volatility over window $P$ | `_compute_residual_volatility()` |
| $a_{i,t}$ | synchronization age | `sync_age = (t - last_sync_step) / age_horizon` in `_collect_step_context()` |

Weights: `drift_weight_loss / drift_weight_disagreement / drift_weight_volatility / drift_weight_age`
(default 0.40 / 0.30 / 0.20 / 0.10).

### Urgency and budgeted client selection (Eqs. 13–14)

$$u_{i,t} = \frac{D_{i,t}}{1+\beta_c c_{i,t}}, \qquad \mathcal{S}_t = \arg\max \sum_{i \in \mathcal{S}} u_{i,t} \ \text{s.t. budget}$$

- Urgency: `_collect_step_context()` (resource pressure `c_{i,t}` is the client's
  recent sync ratio over `resource_window`, `beta_resource`).
- Request when `u_{i,t} > sync_threshold`.
- Budgeted top-k selection: `_select_clients()` (`sync_budget_clients`).
- Hardware: `hardware/shared/strategy/scheduler.py::select_clients()` +
  `drift.py`.

### Selective downlink retrieval (Eqs. 15–19)

$$j^\star_i = \arg\min_j \Big[\tfrac{1}{P}\sum_{p}\ell(\theta^c_{t_j}; x_{i,t-p},y_{i,t-p}) + \lambda_{stale}(t-t_j)\Big]$$

$$g_{i,t} = \frac{[\bar\ell^e_{i,t}-\bar\ell^{c,\star}_{i,t}]_+}{\bar\ell^e_{i,t}+\epsilon},
\qquad q^{\mathrm{down}}_{i,t} = \mathbb{I}(i\in\mathcal{S}_t)\mathbb{I}(D_{i,t}>\tau_d)\mathbb{I}(g_{i,t}>\tau_g)$$

- Snapshot search over the memory bank: `_select_cloud_snapshot()`
  (`retrieval_policy = "historical_match"`; `stale_penalty` = $\lambda_{stale}$;
  `latest` / `random` policies provided for ablations).
- Utility gating: `cloud_utility` returned by `_select_cloud_snapshot()`;
  downlink decision in `_adapt_client()` using `downlink_drift_threshold` and
  `downlink_utility_threshold`.
- Simulated downlink loss: `downlink_loss_rate` + `_deliver_cloud_snapshot()`.

## 2. Robust cloud-edge co-evolution

### Utility-modulated edge evolution (Eqs. 20–24)

$$\eta^e_{i,t} = \eta_{\min}+(\eta_{\max}-\eta_{\min})\sigma(a_\eta(D_{i,t}-b_\eta)), \qquad
\mu_{i,t} = \mu_{\max}\sigma(a_g g_{i,t} - a_d D_{i,t})$$

$$\theta^e_{i,t+1} = \theta^e_{i,t} - \eta^e_{i,t}\left[\nabla\ell(\theta^e_{i,t}) + q^{\mathrm{down}}_{i,t}\mu_{i,t}(\theta^e_{i,t}-\bar\theta^c_{i,t})\right]$$

- `_adaptive_local_lr()` (`eta_min_scale`, `eta_max_scale`, `eta_slope`, `eta_bias`)
- `_adaptive_mu()` (`mu_max`, `mu_cloud_gain`, `mu_drift_penalty`)
- `_online_update()` with `proximal_state` = the retrieved cloud anchor
  (`mu = 0` when no downlink), plus optional gradient clipping
  (`grad_clip_norm`) and optional energy loss (`energy_loss_weight`).
- Alternative edge adapters for ablations (online / AGEM / MIR / DER replay):
  `online_adaptation.py::EdgeAdapter` (`--edge-adapter`).

### Reliability-calibrated cloud evolution (Eqs. 25–29)

$$\theta^{g,+}_{i,t} = q^{\mathrm{down}}_{i,t}\bar\theta^c_{i,t} + (1-q^{\mathrm{down}}_{i,t})\theta^g_{i,t}, \qquad
\theta^g_{i,t+1} = \theta^{g,+}_{i,t} - \eta_g\nabla\ell(\theta^{g,+}_{i,t})$$

$$\tilde\Delta_{i,t} = (1+\tau_{i,t})^{-\gamma}\Delta_{i,t}, \qquad
\omega_{i,t} \propto \exp(\beta_D D_{i,t} + \beta_g g_{i,t} - \beta_\nu\nu_{i,t} - \beta_\tau\tau_{i,t})$$

$$\theta^c_{t+1} = \theta^c_t + \frac{2}{\lambda_c+2}\sum_{i\in\mathcal{S}_t}\omega_{i,t}\tilde\Delta_{i,t}$$

- Anchor installation into the cloud branch: `_deliver_cloud_snapshot()`
  (`snapshot_blend` allows soft blending).
- Cloud-branch online update: `_adapt_client()` →
  `_online_update(client.global_trainer, ...)` (`unsynced_global_lr_scale`
  slows the branch for non-selected clients).
- Innovation $\Delta$: `_state_dict_difference()`; staleness in communication
  rounds: `staleness = (t - anchor_step) / pred_len`; discount `stale_gamma`.
- Reliability weights: `_compute_aggregation_weights()`
  (`agg_beta_drift/utility/volatility/staleness`).
- Server update with stabilization $\lambda_c$: `_aggregate_cloud_updates()`
  (`server_lambda`), then the new snapshot is appended to `cloud_memory`
  (`cloud_memory_size` = $M$) and optionally pushed back
  (`post_sync_broadcast`).
- Hardware: `hardware/server_app/services/aggregation_service.py`,
  `dispatch_service.py`.

## 3. Hierarchical prediction-space fusion

### Temporal stabilization (Eqs. 30–33)

$$\tilde y^k_{i,t} = \sum_{\tau=t-B+1}^{t} w^k_{\tau,t}(\lambda^k_{i,t}) f^k_{i,\tau}(x_{i,t}),
\qquad \lambda^k_{i,t+1} = \lambda^k_{i,t} - \eta_\lambda\nabla_\lambda \ell(\tilde y^k_{i,t}, y_{i,t})$$

- Predictor buffers: `Client.local_model_buffer` / `global_model_buffer`
  (size `model_buffer_size` = $B$).
- Exponentially decaying temporal weights: `Client.incremental_model_ensemble()`;
  the decay rate is a learned parameter updated by gradient descent when
  `decay_update=True` (`local_decay_rate` / `global_decay_rate`,
  `local_decay_lr` / `global_decay_lr`).

### Multi-view cloud-edge fusion (Eqs. 34–40)

- Long-term reliability view $\alpha^{\mathrm{lt}}$ (online gradient descent on
  the gate): `utils/trainer.py::Ensemble_Trainer.online_step()`
  (`long_weight`, `long_w_lr`).
- Short-term context view $\alpha^{\mathrm{st}}$ (context MLP on stabilized
  predictions + realized targets): `models/Ensemble_MLP.py`
  (`short_ensembler`, width 32, 3 layers, Tanh; `short_w_lr`).
- Recent-performance view $\alpha^{\mathrm{pf}}$ (error-mean + error-std
  scores): `co_adaptation_strategy.py::_run_fusion_step()`
  (`consider_error=True`; `performance_fallback_mix` interpolates this view
  into the final gate).
- Final fused forecast: $\hat y_{i,t} = \alpha_{i,t}\tilde y^c_{i,t} + (1-\alpha_{i,t})\tilde y^e_{i,t}$,
  implemented at the end of `_run_fusion_step()`.

## Offline initialization

- Local warm-start training: `utils/trainer.py::Trainer.train()` (Adam + MSE,
  early stopping, checkpointing).
- Federated initialization: `Server.offline_fed()` — standard FedAvg, with
  optional FedProx / FedSGD / FedAvgM variants (`--fed-type`).
- Ensemble (fusion) warm-up: `utils/trainer.py::Ensemble_Trainer.train()`
  (`--warmup-ensemble`, `--ensemble-epochs`).
- Checkpoint loading: `main_co_adaptation.py::load_or_prepare_models()`
  (`--train-if-missing` trains missing initialization from scratch).

## Hyperparameters of the DCE control loop

Defaults of `StrategyConfig` (all overridable via CLI flags in
`main_co_adaptation.py`):

| Parameter | Default | Role |
|-----------|---------|------|
| `sync_threshold` ($\tau_{up}$) | 0.80 | uplink trigger |
| `downlink_drift_threshold` ($\tau_d$) | 0.60 | drift gate for downlink |
| `downlink_utility_threshold` ($\tau_g$) | 0.01 | utility gate for downlink |
| `sync_budget_clients` ($B_t$) | 4 | per-cycle client budget |
| `cloud_memory_size` ($M$) | 5 | server snapshot memory bank |
| `stale_penalty` ($\lambda_{stale}$) | 0.01 | snapshot-age penalty |
| `stale_gamma` ($\gamma$) | 1.0 | upload staleness discount |
| `server_lambda` ($\lambda_c$) | 0.5 | server stabilization |
| `eta_min/max_scale` | 0.5 / 2.0 | edge LR bounds (× `online_lr`) |
| `mu_max`, `mu_cloud_gain`, `mu_drift_penalty` | 0.05 / 4.0 / 2.5 | tethering strength |
| `rho_m`, `rho_v` | 0.9 / 0.9 | EWA forgetting factors |
| `recent_window` ($P$) | 6 | residual / retrieval window |
| `age_horizon` ($A_{max}$) | 12 | sync-age normalization |
| `beta_resource` ($\beta_c$) | 1.0 | communication-ratio damping |
| `model_buffer_size` ($B$) | 7 | predictor buffer for temporal stabilization |

## Key outputs

`ActiveCoAdaptationRunner._finalize_outputs()` writes, per run:

- `co_adaptation_metrics.csv` — per-client MSE/MAE/EPE for ensemble, local-only
  and global-only streams
- `communication_summary.csv` — sync counts, upload/download MB, downlink
  success/failure counts
- `sync_events.csv` — per-event drift/utility/staleness diagnostics
- `step_metrics.csv`, `client_step_metrics.csv`, `run_summary.json`
