"""
Central configuration for the whole project.

Every path, group definition, column mapping, colour and random-number setting
lives here. If the group assignment or the Excel file changes, edit this file
only; all other modules read from it.
"""
from pathlib import Path

# ---------------------------------------------------------------- paths
PROJECT_DIR = Path(__file__).resolve().parent
RAW_EXCEL = PROJECT_DIR / "data" / "raw" / "ETH25-10947 machine learning. AD (new).xlsx"
RESULTS_DIR = PROJECT_DIR / "results"
TABLE_DIR = RESULTS_DIR / "tables"
FIGURE_DIR = RESULTS_DIR / "figures"
LOG_FILE = RESULTS_DIR / "log.txt"

# ---------------------------------------------------------------- experimental design
# Mouse IDs are in column B of the sheet. Groups are given by cell colour,
# 12 consecutive mice per group (confirmed by the labels in column A).
GROUPS = ["Saline", "PM2.5", "PM2.5 EV", "N EV"]
GROUP_OF_MOUSE = {i: GROUPS[(i - 1) // 12] for i in range(1, 49)}
N_MICE = 48
REFERENCE_GROUP = "Saline"

# Group patterns observed in the IHC (used by core.pattern_contrast):
#   glia pattern: IBA1/GFAP raised in these groups, normal in the others
#   NeuN pattern: NeuN lowered in every group except the reference group
GLIA_HIGH_GROUPS = ["PM2.5", "PM2.5 EV"]
NEUN_LOW_GROUPS = ["PM2.5", "PM2.5 EV", "N EV"]

# ---------------------------------------------------------------- sheet layout
HEADER_ROW = 1          # 0-based row index of the column names in the sheet
FIRST_BEHAVIOUR_COL = 2  # column C
FIRST_IHC_COL = 15       # column P
FIRST_PROTEIN_COL = 30   # column AE (first proteomics column)

# Behaviour columns (as written in the sheet, whitespace stripped) -> short names
BEHAVIOUR_COLUMNS = {
    "Tmaze week 4 - 15 seconds": "T4_15",
    "Tmaze week 4 - 60 seconds": "T4_60",
    "NOR week 4": "NOR4",
    "EPM closed week 4": "EPMclosed4",
    "EPM crossing week 4": "EPMcentre4",
    "EPM Open week 4": "EPMopen4",
    "Tmaze week 12 - 15 seconds": "T12_15",
    "Tmaze week 12 - 60 seconds": "T12_60",
    "NOR week 12": "NOR12",
    "EPM closed week 12": "EPMclosed12",
    "EPM crossing week 12": "EPMcentre12",
    "EPM Open week 12": "EPMopen12",
    "EPM crossing freq week 12": "EPMentries12",
}

# Human-readable labels for tables and figures
LABELS = {
    "T4_15": "T-maze wk4 15s", "T4_60": "T-maze wk4 60s", "NOR4": "NOR wk4",
    "EPMclosed4": "EPM closed time wk4", "EPMcentre4": "EPM centre time wk4", "EPMopen4": "EPM open time wk4",
    "T12_15": "T-maze wk12 15s", "T12_60": "T-maze wk12 60s", "NOR12": "NOR wk12",
    "EPMclosed12": "EPM closed time wk12", "EPMcentre12": "EPM centre time wk12", "EPMopen12": "EPM open time wk12",
    "EPMentries12": "EPM crossing freq wk12",
    "Mem4": "Memory composite wk4", "Mem12": "Memory composite wk12",
    "T12_diff": "T-maze wk12 60s minus 15s (not recommended)",
    "NeuN_idx": "NeuN index", "GFAP_idx": "GFAP index", "IBA1_idx": "IBA1 index", "Glia_idx": "Glia index",
}

# IHC markers and regions (column names are "<MARKER> <Region>")
IHC_MARKERS = ["GFAP", "NeuN", "IBA1"]
IHC_REGIONS = ["CA1", "CA2", "CA3", "DG", "Cortex"]

# Memory composite = mean of z-scores of these items (available items only)
MEMORY_ITEMS = {"Mem4": ["T4_15", "T4_60", "NOR4"], "Mem12": ["T12_15", "T12_60", "NOR12"]}

# Outcomes analysed in most modules
MEMORY_OUTCOMES = ["Mem12", "T12_15", "T12_60", "NOR12", "Mem4", "T4_15", "T4_60", "NOR4"]
ANXIETY_OUTCOMES = ["EPMopen4", "EPMopen12", "EPMentries12"]
PRIMARY_OUTCOMES = ["Mem12", "T12_15", "T12_60", "NOR12"]   # fixed family for Holm

# ---------------------------------------------------------------- proteomics
MIN_DETECTED_PER_GROUP = 4     # keep a protein if detected in >=4 of 6 mice in every group
MIN_PROTEINS_PER_MOUSE = 100   # a mouse "has proteomics" if >100 proteins detected
CELL_TYPE_MARKERS = {
    "Astrocyte": ["Gfap", "Aqp4", "Aldh1l1", "S100b", "Vim", "Slc1a3", "Slc1a2", "Glul"],
    "Microglia": ["Hexb", "Tmem119", "P2ry12", "C1qb", "Aif1"],
    "Neuron/synapse": ["Rbfox3", "Syp", "Syn1", "Dlg4", "Snap25", "Map2", "Nefl",
                       "Camk2a", "Nrgn", "Gap43", "Calb1"],
    "Oligodendrocyte": ["Mbp", "Plp1"],
}
MIN_MICE_FOR_MARKER_SCORE = 20  # use a marker in a score only if detected in >=20 mice
# Protein <-> IHC pairs for the agreement check
PROTEIN_IHC_PAIRS = [("Gfap", "GFAP_idx"), ("Hexb", "IBA1_idx"), ("Aif1", "IBA1_idx"), ("Rbfox3", "NeuN_idx")]

# ---------------------------------------------------------------- resampling settings
SEED = 20261118
N_PERMUTATIONS = 9999
N_BOOTSTRAP = 2000
N_ML_PERMUTATIONS = 100
N_ML_ZOO_PERMUTATIONS = 50        # ml/model_zoo.py (8 learners x 4 settings)
N_ML_LOGISTIC_PERMUTATIONS = 100  # ml/logistic_ihc_group.py
N_PERM_BLOCK = 4999               # advanced/multiblock_association.py, pls_integration.py
N_PERM_PROTEOME = 1000            # advanced/legacy_claims_recheck.py (proteome-wide max statistic)
N_BOOT_PLS = 1000                 # advanced/pls_integration.py
N_BOOT_MODULES = 100              # advanced/coexpression_modules.py (module stability)
N_BOOT_MEDIATION = 5000           # robustness/mediation_analysis.py
N_SHUFFLE_ALIGNMENT = 10000       # qc/proteomics_integrity.py (sample alignment)
N_NULL_COUNT_SHUFFLES = 200       # qc/proteomics_integrity.py (null of the ANOVA p<0.05 count)

# ---------------------------------------------------------------- figure colours
GROUP_COLOURS = {"Saline": "#2a78d6", "PM2.5": "#eb6834", "PM2.5 EV": "#1baf7a", "N EV": "#eda100"}
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"

# ---------------------------------------------------------------- knowledge-guided protein sets (advanced/)
# Idea borrowed from MetaGeno (features grouped by prior structure, e.g. chromosomes)
# and PRS / transfer learning (weights from outside the data). Here: proteins are
# grouped by literature-defined cell type / pathway; each set becomes ONE score,
# so 3,500 proteins shrink to ~16 pre-specified features without looking at outcomes.
# Lists = explicit genes; strings starting with "re:" = regular expression on gene symbol.
PROTEIN_SETS = {
    "Microglia (Hexb, Tmem119, C1q)": ["Hexb", "Tmem119", "C1qb", "C1qc", "P2ry12", "Aif1", "Cx3cr1", "Csf1r"],
    "Activated microglia / DAM": ["Apoe", "Cd9", "Ctsb", "Ctsd", "Fth1", "Ftl1", "Trem2", "Tyrobp", "Lgals3", "Cd68", "Spp1", "Cst7"],
    "Reactive astrocyte": ["Gfap", "Vim", "Cd44", "Serpina3n", "Lcn2", "Osmr", "Timp1", "S100a10"],
    "Astrocyte homeostatic": ["Aqp4", "Aldh1l1", "Slc1a2", "Slc1a3", "Glul", "Gja1", "Atp1a2", "Sparcl1", "Aldoc", "Slc4a4"],
    "Presynaptic": ["Syp", "Syn1", "Syn2", "Snap25", "Stx1a", "Vamp2", "Syt1", "Cplx1", "Cplx2", "Rab3a", "Stxbp1", "Nsf"],
    "Postsynaptic": ["Dlg4", "Shank1", "Shank2", "Shank3", "Homer1", "Gria1", "Gria2", "Grin1", "Grin2b", "Camk2a", "Nrgn", "Dlgap1", "Syngap1"],
    "Neuronal structure": ["Rbfox3", "Map2", "Tubb3", "Nefl", "Nefm", "Nefh", "Eno2", "Uchl1", "Gap43", "Stmn2", "Mapt", "Ina"],
    "Myelin / oligodendrocyte": ["Mbp", "Plp1", "Mog", "Mag", "Cnp", "Mobp", "Cldn11"],
    "Lysosome": ["Ctsb", "Ctsd", "Ctsl", "Lamp1", "Lamp2", "Hexa", "Hexb", "Gba", "Npc2", "Psap", "Atp6v1a"],
    "Oxidative stress": ["Sod1", "Sod2", "Cat", "Gpx1", "Prdx1", "Prdx2", "Prdx3", "Prdx5", "Prdx6", "Hmox1", "Txn1", "Gsr"],
    "AD-related": ["App", "Apoe", "Clu", "Mapt", "Bin1", "Picalm", "Sorl1", "Psen1", "Bace1", "Snca"],
    "Mitochondrial OXPHOS": "re:^(Nduf|Sdh[abcd]$|Uqcr|Cox[4-8]|Atp5)",
    "Ribosome": r"re:^(Rpl|Rps)\d",
    "Proteasome": r"re:^Psm[abcd]\d",
}
MIN_SET_MEMBERS = 3
N_RANDOM_SETS = 1000

# Stability selection (Stabl-like): subsample half the mice, Lasso path, count selections;
# decoy (row-permuted) copies of every protein set the reliability threshold.
STABILITY_SUBSAMPLES = 200
STABILITY_TARGETS = ["Glia_idx", "IBA1_idx", "GFAP_idx", "NeuN_idx", "Mem12", "T12_15"]
