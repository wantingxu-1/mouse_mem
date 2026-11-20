"""Output helpers: logging, saving tables and figures to results/."""
import datetime
import config


def ensure_dirs():
    for d in (config.RESULTS_DIR, config.TABLE_DIR, config.FIGURE_DIR):
        d.mkdir(parents=True, exist_ok=True)


def log(*parts):
    """Print a message and append it to results/log.txt."""
    ensure_dirs()
    msg = " ".join(str(p) for p in parts)
    print(msg)
    with open(config.LOG_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def start_log():
    """Start a fresh log file (called by run_all.py)."""
    ensure_dirs()
    config.LOG_FILE.write_text(f"Run started {datetime.datetime.now():%Y-%m-%d %H:%M:%S}\n", encoding="utf-8")


def header(title):
    log("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


def save_table(df, name, index=False, round_to=4):
    """Save a DataFrame as results/tables/<name>.csv (numbers rounded)."""
    ensure_dirs()
    path = config.TABLE_DIR / f"{name}.csv"
    out = df.copy()
    num = out.select_dtypes("number").columns
    out[num] = out[num].round(round_to)
    out.to_csv(path, index=index)
    log(f"  -> saved table {path.relative_to(config.PROJECT_DIR)}")
    return path


def load_table(name):
    """Read results/tables/<name>.csv written by an earlier step, with a clear error if missing."""
    import pandas as pd
    path = config.TABLE_DIR / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path.relative_to(config.PROJECT_DIR)} not found: run the step that writes it first "
                                f"(e.g. python run_all.py, or the matching layer) before this one.")
    return pd.read_csv(path)


def save_figure(fig, name, dpi=200):
    ensure_dirs()
    path = config.FIGURE_DIR / f"{name}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    log(f"  -> saved figure {path.relative_to(config.PROJECT_DIR)}")
    return path
