# Future Extensions

Forward-looking design ideas for the regime-aware ETF allocator. None of these are implemented yet — they describe paths from the current rule-based system toward a more ML-flavored variant.

## Replace the regime classifier with a Hidden Markov Model

The current classifier is deterministic: z-scores on macro indicators cross threshold → regime label. An HMM would:

- Treat the regime as a hidden discrete state that evolves over time.
- Estimate (a) transition probabilities between regimes and (b) emission distributions of macro indicators *given* each regime.
- Parameters are fit by Baum-Welch / EM.
- Output: a *probability distribution* over regimes each month, not a single hard label.

Benefits: smoother regime transitions (no thresholding artifacts), uncertainty quantification (you can say "70% Stagflation / 30% Contraction"), can be used to size positions by certainty.

Risks: HMM is sensitive to initialization and number of states; overfitting on 100+ months is real. Would need walk-forward HMM refitting (much more compute than the current rule).

## Train a neural net to predict next-month returns from macro features

Current model uses past returns directly as the optimizer's input. Replace with:

- Input: macro feature vector (the same FRED indicators currently used).
- Output: predicted next-month return for each ETF, OR predicted next-month covariance matrix.
- Architecture: small MLP or LSTM is enough — financial data doesn't need transformers.
- Loss: MSE on realized returns, or a portfolio-aware loss (e.g., negative Sharpe of the resulting allocation).

Feed predictions into the existing Sortino optimizer instead of historical means.

Benefits: lets the model condition on regime-relevant information, not just historical averages. Could pick up nonlinear macro-return relationships the current linear classifier misses.

Risks: large overfitting risk on ~10 years of monthly data. Need very strong regularization, dropout, ensembling. Walk-forward retraining is essential. Performance gain over the simpler model is usually modest in studies.

## Meta-learner for `hybrid_macro_weight`

Current `hybrid_macro_weight=0.5` is a fixed hyperparameter — equal weight on the macro regime signal vs the momentum signal. A meta-learner would:

- Take recent strategy performance and recent regime stability as input.
- Output a time-varying `hybrid_macro_weight ∈ [0, 1]`.
- Train via reinforcement learning OR by walk-forward gridding over recent windows.

Benefits: model becomes adaptive — leans on momentum during stable regimes, leans on macro during regime transitions.

Risks: introduces another hyperparameter surface to overfit. Hard to validate cleanly with walk-forward; the meta-learner sees signals derived from data the inner model already saw.

## Priority order if pursuing any of these

1. **HMM regime classifier** — highest pedagogical and interpretability value. Cleanest swap-in for the existing classifier.
2. **Meta-learner for hybrid weight** — moderate compute, moderate benefit, easiest to A/B against current.
3. **Neural-net return prediction** — biggest engineering lift, easiest to overfit, most exciting on a resume. Only do this if you have a clear plan for regularization and OOS validation.

## Honest caveat

For all three: the realistic Sharpe improvement is in the 0.1–0.2 range over the current model, with high variance. None of these are guaranteed to be statistically distinguishable from the current model on a 10-year sample. Pursue them for the *learning* or because you find a specific failure mode to fix (e.g., the 2021 Overheating-regime drag) — not because you expect headline-grabbing alpha.
