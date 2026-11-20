# PM2.5 / EV mice: behaviour vs brain markers

Does memory change follow **neuron loss** (NeuN) or **glial activation** (IBA1, GFAP)?
Answer: it fits the **glia** pattern better than the NeuN pattern. N EV has low NeuN, near-Saline glia, and short-delay memory not detectably different from Saline. Exploratory: Holm-corrected p = 0.051–0.083.

Full results, methods and figures: [RESULTS.md](RESULTS.md).

## Run

```bash
pip install -r requirements.txt
python run_all.py            # everything (~30 min)
python run_all.py --quick    # fast check (~3 min)
python run_all.py --list     # all steps
python run_all.py --only pattern_contrast   # one module
python -m pytest tests       # tests (~3 s)
```

Input: `data/raw/ETH25-10947 machine learning. AD (new).xlsx`. Output: `results/` (tables, figures, log).

## Data

| Block | Mice |
|---|---|
| Behaviour: T-maze 15/60 s, NOR, EPM at weeks 4 and 12 | 48 (12 per group) |
| IHC: GFAP, NeuN, IBA1 × CA1, CA2, CA3, DG, cortex | 5–7 per group |
| Proteomics (~4,800 proteins) | 24 (6 per group) |

Groups: Saline 1–12, PM2.5 13–24, PM2.5 EV 25–36, N EV 37–48.

## Layout

```
config.py      settings (paths, groups, columns, resampling counts, seed)
run_all.py     runs all steps in order
data/          raw file + loader
features/      analysis table, indices, proteomics prep, protein sets, T-maze trials
utils/         io, stats helpers, permutation helpers
qc/            data checks
core/          main analysis: glia pattern vs NeuN pattern
robustness/    permutation, bootstrap, Bayes, GEE, mixed model, confounds, mediation
ml/            nested leave-one-out models
advanced/      small-n / high-dimensional methods (proteomics, PLS, block tests)
plots/         figures 1–4
tests/         data, features, helpers, key results, import rules
```

Rules: a module imports only from `features/`, `utils/`, its own layer or an earlier one. Each module has a `run()` and starts with a short docstring.

## Definitions

- **Marker index:** z-score each of 5 regions, then average. **Glia index** = mean of GFAP and IBA1 indices.
- **Memory composite:** mean z of T-maze 15 s, T-maze 60 s and NOR.
- **Glia pattern:** PM2.5 and PM2.5 EV. **NeuN pattern:** all treated groups. They differ only in N EV.
- **Primary test:** `y ~ glia_high + neun_low` on 48 mice, Holm-corrected over 4 outcomes × 2 effects.
- **Group-adjusted correlation:** rank, subtract the group mean rank, correlate.

## Caveats

NeuN is staining intensity, not a cell count. Week 4 is not a baseline. Within-group tests have low power (5–6 stained mice per group). Results show association, not cause.
