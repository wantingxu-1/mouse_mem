"""
Run the whole analysis from the raw Excel file to results/.

Usage (from inside the analysis/ folder):
    python run_all.py                    # everything, full resampling counts (~30 min)
    python run_all.py --quick            # fewer permutations/bootstraps (fast check)
    python run_all.py core plots         # only these layers, in pipeline order
    python run_all.py --only sex_sensitivity tmaze_chance_level   # single modules
    python run_all.py --list             # show every step and exit

Layers run in this order: qc, core, robustness, ml, advanced, plots.
Every step writes its tables to results/tables, figures to results/figures,
and its key numbers to results/log.txt. Figures 3-4 read tables written by
earlier layers, so run those layers first (a full run does this).
A failing step stops the run with a non-zero exit code and names the step.
Warnings raised inside a step (e.g. convergence) are summarised in the log.
"""
import argparse
from collections import Counter
import sys
import time
import traceback
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
# Library API notices are not about the analysis; every other warning is recorded per step.
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

from qc import qc_checks, proteomics_integrity
from core import (group_behaviour, group_ihc, pattern_contrast, exclusion_sensitivity,
                  individual_correlation, model_comparison, proteomics_markers,
                  glia_neuron_ratio, nev_heterogeneity, composition_sensitivity)
from robustness import (permutation_tests, bootstrap_ci, leave_one_out_influence,
                        holm_primary, paired_weeks, multiregion_profile,
                        bayes_equivalence, tmaze_binomial_model, ihc_mixed_model,
                        activity_confound, mediation_analysis,
                        sex_sensitivity, selection_bias, tmaze_chance_level)
from ml import ridge_ihc_behaviour, elasticnet_proteomics, logistic_ihc_group, model_zoo, incremental_value
from advanced import (protein_sets, stability_selection, multitask_learning, ensemble_stacking,
                      multiblock_association, pls_integration, coexpression_modules,
                      legacy_claims_recheck)
from plots import fig_group_dissociation, fig_individual_mice, fig_evidence_summary, fig_mechanism
from utils.io import start_log, log

LAYERS = ["qc", "core", "robustness", "ml", "advanced", "plots"]
QUICK_PERM = 499


def registry():
    """(layer, module, keyword arguments used by --quick). Order = execution order."""
    P = {"n_perm": QUICK_PERM}
    return [
        ("qc", qc_checks, {}), ("qc", proteomics_integrity, {}),
        ("core", group_behaviour, {}), ("core", group_ihc, {}), ("core", pattern_contrast, {}),
        ("core", exclusion_sensitivity, {}), ("core", individual_correlation, {}), ("core", model_comparison, {}),
        ("core", proteomics_markers, {}), ("core", glia_neuron_ratio, P), ("core", nev_heterogeneity, {}),
        ("core", composition_sensitivity, {}),
        ("robustness", permutation_tests, P), ("robustness", bootstrap_ci, {"n_boot": 300}),
        ("robustness", leave_one_out_influence, {}), ("robustness", holm_primary, P), ("robustness", paired_weeks, {}),
        ("robustness", multiregion_profile, P), ("robustness", bayes_equivalence, {}),
        ("robustness", tmaze_binomial_model, {}), ("robustness", ihc_mixed_model, {}),
        ("robustness", activity_confound, P), ("robustness", mediation_analysis, {}),
        ("robustness", sex_sensitivity, P), ("robustness", selection_bias, P), ("robustness", tmaze_chance_level, {}),
        ("ml", ridge_ihc_behaviour, {"n_perm": 50}), ("ml", elasticnet_proteomics, {}),
        ("ml", logistic_ihc_group, {"n_perm": 20}), ("ml", model_zoo, {"n_perm": 10}), ("ml", incremental_value, {"n_perm": 99}),
        ("advanced", protein_sets, {}), ("advanced", stability_selection, {}), ("advanced", multitask_learning, {}),
        ("advanced", ensemble_stacking, {}), ("advanced", multiblock_association, {"n_perm": 499}),
        ("advanced", pls_integration, {"n_perm": 199}), ("advanced", coexpression_modules, {}),
        ("advanced", legacy_claims_recheck, {"n_perm": 100}),
        ("plots", fig_group_dissociation, {}), ("plots", fig_individual_mice, {}),
        ("plots", fig_evidence_summary, {}), ("plots", fig_mechanism, {}),
    ]


def summarise(caught):
    """One line per distinct warning (category + first 90 characters), with its count."""
    if not caught:
        return ""
    counts = Counter(f"{w.category.__name__}: {str(w.message).splitlines()[0][:90]}" for w in caught)
    return "\n" + "\n".join(f"    warning x{n}  {msg}" for msg, n in counts.most_common())


def short(module):
    return module.__name__.split(".")[-1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("layers", nargs="*", metavar="LAYER",
                    help="layers to run (default: all): " + " ".join(LAYERS))
    ap.add_argument("--only", nargs="+", metavar="MODULE", help="run only these modules (file names, no .py)")
    ap.add_argument("--quick", action="store_true", help="fewer permutations / bootstraps")
    ap.add_argument("--list", action="store_true", help="list all steps and exit")
    args = ap.parse_args()

    bad = set(args.layers) - set(LAYERS)
    if bad:
        ap.error(f"unknown layer(s): {', '.join(sorted(bad))}; choose from {' '.join(LAYERS)}")
    steps = registry()
    if args.list:
        for layer, mod, _ in steps:
            print(f"{layer:11s} {short(mod):28s} {(mod.__doc__ or '').strip().splitlines()[0]}")
        return 0
    if args.only:
        unknown = set(args.only) - {short(m) for _, m, _ in steps}
        if unknown:
            ap.error(f"unknown module(s): {', '.join(sorted(unknown))}  (see --list)")
        steps = [s for s in steps if short(s[1]) in args.only]
    else:
        steps = [s for s in steps if s[0] in (args.layers or LAYERS)]

    start_log()
    t0 = time.time()
    for layer, mod, quick_kw in steps:
        t1 = time.time()
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                warnings.filterwarnings("ignore", category=FutureWarning)
                warnings.filterwarnings("ignore", category=DeprecationWarning)
                mod.run(**(quick_kw if args.quick else {}))
        except Exception:
            log(f"\nFAILED in step {layer}/{short(mod)}:\n{traceback.format_exc()}")
            return 1
        log(f"  [{layer}/{short(mod)} done in {time.time() - t1:.0f} s]{summarise(caught)}")
    log(f"\nFinished {len(steps)} steps in {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
