# Metrics

`CDAJEPALearner` logs these per trained module, beside PPO's own, under `learners/<module_id>/`.
They are computed on the actor's encoder only: with `vf_share_layers` false the critic has its own
encoder instance, and adding its term too would double the loss under that setting alone.

| Metric | What it is | How to read it |
|---|---|---|
| `jepa_aux_loss` | The latent-prediction loss plus `variance_coeff` times the variance hinge. PPO's total gains `aux_loss_coeff` times this | Present and above 0. Missing means the module stopped forwarding it, and training then looks entirely normal |
| `jepa_predict_loss` | The smooth-L1 latent-prediction loss alone, before its coefficient | Falls as the encoder learns; **a fall to 0 is a collapse, not a success** |
| `jepa_latent_std` | The per-dimension standard deviation across the batch of the layer-normed predictor output, averaged; NaN (not measured) for a batch of one row | Near 1 (the targets are layer-normed). Near 0 means every observation maps to one latent |
| `jepa_offdiag_cov` | Mean absolute off-diagonal covariance of the same latents | Rising while `jepa_latent_std` holds means dimensional collapse: the dimensions become redundant |
| `jepa_world_loss` | The action-conditioned next-latent loss. PPO's total gains `world_model_coeff` times this | Only with `"world_model": true`. It has a non-zero floor: the next book depends on every agent's action and this conditions on one, so the floor is an opponent model, not underfitting |

`jepa_latent_std` and the hinge are read on the online side, the predictor's output. Read off the
EMA target they carried no gradient and the hinge was a constant (gym-CDA's finding S2-9, in its
[docs/29](https://github.com/ChuaCheowHuan/gym-continuousDoubleAuction/blob/master/docs/29_fixed_issues.md)).

The pretrainer reports the same `latent_std` and exits non-zero on a collapse
([pretraining.md](pretraining.md)).
