# Settings

The encoder's settings live in gym-CDA's train config under `encoder_specs.jepa`, beside the other
encoders' blocks, and every one has a default (`JEPA_DEFAULTS` in `cda_jepa/encoder.py`). A block
of `{}` is the defaults. They are structural: the encoder fingerprint covers them, so a resume
cannot change one, and a pretrained encoder loads only into the settings it was trained with.

| Key | Default | What it does |
|---|---|---|
| `tokenization` | `"both"` | How the book grid becomes tokens: `both` (one per level per snapshot), `time` (one per snapshot), `level` (one per level, latest snapshot). Means what it does for gym-CDA's `transformer` |
| `d_model` | `128` | Token width, as for `transformer` |
| `num_heads` | `4` | Attention heads, as for `transformer` |
| `num_layers` | `2` | Trunk layers, as for `transformer` |
| `ff_dim` | `256` | Feed-forward width, as for `transformer` |
| `dropout` | `0.0` | As for `transformer` |
| `pool` | `"attention"` | How tokens pool to the policy latent, as for `transformer` |
| `mask_axis` | `"level"` | What a mask hides: `level` (a depth band in every snapshot), `time` (whole snapshots), `random` (the ablation baseline) |
| `mask_ratio` | `0.4` | Fraction of tokens hidden from the context |
| `predictor_layers` | `1` | Layers of the narrow predictor |
| `predictor_dim` | `64` | Width of the predictor |
| `ema_decay` | `0.996` | How slowly the target trunk follows the online one |
| `aux_loss_coeff` | `0.1` | Weight on the latent-prediction loss in PPO's total |
| `variance_coeff` | `1.0` | Weight on the VICReg-style variance hinge against collapse |
| `world_model` | `false` | Adds the action-conditioned term; needs `Columns.NEXT_OBS`, attached only when this is on |
| `world_model_coeff` | `0.1` | Weight on the world-model loss |

## Notes

These were the `_note` texts of gym-CDA's `encoder_specs.jepa` block.

**What it is.** Joint-Embedding Predictive Architecture. The transformer keys mean what they do in that block; the rest configure the self-supervised objective. Predicts MASKED TOKENS' LATENTS from the visible ones, scored against an EMA copy of the encoder - nothing reconstructs the input, which matters here because a reconstruction loss would be dominated by sqrt(volume) jitter (measured channel sigmas [1.27, 8.17, 0.046, 9.52]).

**The policy latent.** The POLICY latent is always computed from the UNMASKED observation - the agent acts on everything it was given. Masking exists only for the auxiliary loss, and runs in train mode only, so a training step costs two trunk passes plus the target's while inference costs exactly one, the same as `transformer`.

**`mask_axis`.** What a mask hides. 'level' hides a contiguous depth band in every snapshot (what depth is consistent with this touch?); 'time' hides whole snapshots (where is the book heading?); 'random' uses no structure and is the ablation baseline. Under 'level' tokenisation there is only one snapshot, so all three degenerate to random.

**Collapse.** WATCH jepa_latent_std IN THE LEARNER METRICS. A collapsed JEPA drives its prediction loss to ZERO, which reads as success - loss and throughput look identical to a working one. latent_std goes to 0 exactly when every observation maps to the same latent and should sit near 1.0. variance_coeff weights a VICReg-style hinge that pushes back once it starts falling. Both the metric and the hinge are computed on the ONLINE side (the predictor's output), not on the EMA target: read off the target they carried no gradient at all and the term was a constant added to the loss - finding S2-9; jepa_offdiag_cov catches dimensional collapse, where variance holds up but the dimensions become redundant.

**`ema_decay`.** How slowly the target encoder follows the online one. The asymmetry - a frozen, lagging target the predictor has to chase - is the primary anti-collapse mechanism; a constant encoder cannot satisfy a moving target. Higher is slower and more stable; too high and the target carries no signal.

**`world_model`.** world_model turns on the ACTION-CONDITIONED term: predict the next observation's latent from this one's plus the action taken, z_hat_t+1 = P(z_t, a_t), scored against the EMA target encoder's view of o_t+1. What it learns is the latent market impact of an order - how the book responds to a market order versus a passive quote versus a cancel - which is a microstructure quantity this env generates endogenously. Off by default because it needs Columns.NEXT_OBS in the train batch, which PPO does not add; train.py attaches RLlib's learner connector for it ONLY when this is on, so no other architecture pays for a column it never reads. Honest bound: z_t+1 depends on every agent's action and this conditions on one of them, so the predictor fits a conditional expectation over the opponents. The loss therefore has a non-zero floor which is NOT underfitting - it is an opponent model, and tuning it away would mean overfitting to noise.

## Rules

- `mask_axis` must be one of `level`, `time`, `random`, and `mask_ratio` must be strictly between
  0 and 1. The number of hidden tokens is clamped so at least one is hidden and one stays visible.
- `num_layers` and `predictor_layers` are at least 1.
- A tokenisation has only the axes it has: `level` tokenisation is one snapshot and `time`
  tokenisation makes a snapshot one token, so asking for the missing axis gives a random mask.
- The pretrainer's own settings (steps, batch size, corpus) are not here: they are its CLI flags,
  with defaults in `cda_jepa/config.py` ([pretraining.md](pretraining.md)).
