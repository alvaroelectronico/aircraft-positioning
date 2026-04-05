# -*- coding: utf-8 -*-
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# CONFIG (EDITA AQUÍ)
# =========================
NEW_RESULTS_XLSX = Path(r"C:\Users\FX516\OneDrive - Universidad Politécnica de Madrid\UNIVERSIDAD\MASTER\TFM\CODE\aircraft-positioning\results_batch_bueno.xlsx")
OLD_RESULTS_XLSX = Path(r"C:\Users\FX516\OneDrive - Universidad Politécnica de Madrid\UNIVERSIDAD\MASTER\TFM\CODE\aircraft-positioning\results_old.xlsx")
OUTDIR = Path(r"C:\Users\FX516\OneDrive - Universidad Politécnica de Madrid\UNIVERSIDAD\MASTER\TFM\CODE\aircraft-positioning\results_analysis")

NEW_LABEL = "final"
OLD_LABEL = "initial"
SHEET = 0
# =========================


# -----------------------
# Helpers
# -----------------------
def read_results(path: Path, sheet=0) -> pd.DataFrame:
    suf = path.suffix.lower()
    if suf == ".csv":
        return pd.read_csv(path)
    if suf in (".xlsx", ".xls"):
        return pd.read_excel(path, sheet_name=sheet)
    raise ValueError(f"Formato no soportado: {path}")


def parse_scenario_name(name: str) -> dict:
    base = (name or "").lower()
    typ = "many" if "many" in base else ("few" if "few" in base else "unknown")
    win = "tight" if "tight" in base else ("loose" if "loose" in base else "mix")

    import re
    jobs_pp = None
    mP = re.search(r"_p(\d+)", base)
    if mP:
        jobs_pp = int(mP.group(1))

    positions = None
    mPL = re.search(r"_pl(\d+)", base)
    if mPL:
        positions = int(mPL.group(1))

    seed = None
    mSeed = re.search(r"_seed(\d+)", base)
    if mSeed:
        seed = int(mSeed.group(1))

    return {
        "scenario_type": typ,
        "window_tightness": win,
        "jobs_per_plane_hint": jobs_pp,
        "positions_hint": positions,
        "seed": seed
    }


def coerce_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan


def infer_feasible(status: str, term: str, objective) -> bool:
    s = (status or "").lower()
    t = (term or "").lower()
    if "infeasible" in t:
        return False
    if "error" in s or "error" in t:
        return False
    # Si hay objetivo numérico, normalmente hay incumbente
    if np.isfinite(coerce_float(objective)):
        return True
    return ("optimal" in t) or ("feasible" in t)


def gap_from_bound(obj, best_bound):
    objf = coerce_float(obj)
    bbf = coerce_float(best_bound)
    if not np.isfinite(objf) or not np.isfinite(bbf):
        return np.nan
    denom = max(abs(objf), abs(bbf), 1.0)
    return abs(objf - bbf) / denom


def ecdf(x):
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.array([]), np.array([])
    xs = np.sort(x)
    ys = np.arange(1, len(xs) + 1) / len(xs)
    return xs, ys


def agg_block(g: pd.DataFrame) -> pd.Series:
    g = g.copy()
    n = len(g)
    feas = g["feasible"].fillna(False)
    feasible_rate = feas.mean() if n > 0 else np.nan

    gf = g[feas].copy()
    out = {
        "n_cases": n,
        "feasible_rate": feasible_rate,
        "time_p50_s": gf["solve_time_s"].median() if len(gf) else np.nan,
        "time_p90_s": gf["solve_time_s"].quantile(0.90) if len(gf) else np.nan,
        "gap_median": gf["mipgap_filled"].median() if len(gf) else np.nan,
        "Tmax_p50": gf["Tmax"].median() if "Tmax" in gf and len(gf) else np.nan,
        "delay_p50": gf["client_delay_sum"].median() if "client_delay_sum" in gf and len(gf) else np.nan,
        "violC_p50": gf["viol_C_sum"].median() if "viol_C_sum" in gf and len(gf) else np.nan,
        "switch_p50": gf["client_switches"].median() if "client_switches" in gf and len(gf) else np.nan,
        "idle_p50": gf["idle_sum"].median() if "idle_sum" in gf and len(gf) else np.nan,
        "presence_p50": gf["presence_sum"].median() if "presence_sum" in gf and len(gf) else np.nan,
    }
    # tamaño modelo si existe
    out["n_bin_med"] = gf["n_bin"].median() if "n_bin" in gf and len(gf) else np.nan
    out["n_vars_med"] = gf["n_vars"].median() if "n_vars" in gf and len(gf) else np.nan
    out["n_cons_med"] = gf["n_cons"].median() if "n_cons" in gf and len(gf) else np.nan

    # tasas de incidencias (sobre factibles)
    if len(gf):
        out["pct_delay_pos"] = float((gf.get("client_delay_sum", 0) > 1e-9).mean()) if "client_delay_sum" in gf else np.nan
        out["pct_violC_pos"] = float((gf.get("viol_C_sum", 0) > 1e-9).mean()) if "viol_C_sum" in gf else np.nan
        out["pct_switch_pos"] = float((gf.get("client_switches", 0) > 1e-9).mean()) if "client_switches" in gf else np.nan
    else:
        out["pct_delay_pos"] = np.nan
        out["pct_violC_pos"] = np.nan
        out["pct_switch_pos"] = np.nan

    return pd.Series(out)


def ensure_cols(df: pd.DataFrame, cols):
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    return df


def save_boxplot(dfb, value_col, title, ylabel, outpath, modes_order=None):
    if value_col not in dfb.columns:
        return
    dfb = dfb.copy()
    if modes_order is None:
        modes_order = sorted([m for m in dfb["mode"].dropna().unique()])
    models = [OLD_LABEL, NEW_LABEL]

    data, labels = [], []
    for m in modes_order:
        for mid in models:
            s = dfb[(dfb["mode"] == m) & (dfb["model_id"] == mid)][value_col].dropna()
            data.append(s.values)
            labels.append(f"{m}\n{mid}")

    plt.figure(figsize=(10, 5))
    plt.boxplot(data, labels=labels, showfliers=False)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=220)
    plt.close()


def save_bar_means(dfb, value_col, title, ylabel, outpath, modes_order=None):
    if value_col not in dfb.columns:
        return
    dfb = dfb.copy()
    if modes_order is None:
        modes_order = sorted([m for m in dfb["mode"].dropna().unique()])
    models = [OLD_LABEL, NEW_LABEL]

    # media sobre factibles por modo y modelo
    rows = []
    for m in modes_order:
        for mid in models:
            g = dfb[(dfb["mode"] == m) & (dfb["model_id"] == mid)]
            rows.append({"mode": m, "model_id": mid, "mean": float(g[value_col].mean()) if len(g) else np.nan})
    t = pd.DataFrame(rows)

    x = np.arange(len(modes_order))
    width = 0.35

    y_old = [t[(t["mode"] == m) & (t["model_id"] == OLD_LABEL)]["mean"].values[0] for m in modes_order]
    y_new = [t[(t["mode"] == m) & (t["model_id"] == NEW_LABEL)]["mean"].values[0] for m in modes_order]

    plt.figure(figsize=(9, 4.5))
    plt.bar(x - width/2, y_old, width, label=OLD_LABEL)
    plt.bar(x + width/2, y_new, width, label=NEW_LABEL)
    plt.xticks(x, modes_order)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=220)
    plt.close()


# -----------------------
# Main
# -----------------------
def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)

    df_new = read_results(NEW_RESULTS_XLSX, sheet=SHEET)
    df_old = read_results(OLD_RESULTS_XLSX, sheet=SHEET)

    df_new["model_id"] = NEW_LABEL
    df_old["model_id"] = OLD_LABEL
    df = pd.concat([df_new, df_old], ignore_index=True)

    # Normalizar nombres "por si acaso"
    rename_map = {
        "Scenario": "scenario",
        "Mode": "mode",
        "Status": "status",
        "Termination": "termination",
        "SolveTime": "solve_time_s",
        "SolveTime_s": "solve_time_s",
        "MIPGap": "mipgap",
        "BestBound": "best_bound",
        "Objective": "objective",
        "TMAX": "Tmax",
    }
    df = df.rename(columns=rename_map)

    # Asegurar columnas mínimas (con tus headers ya deberían existir)
    base_cols = ["scenario", "mode", "status", "termination", "solve_time_s",
                 "mipgap", "best_bound", "objective", "Tmax",
                 "client_delay_sum", "viol_C_sum", "client_switches",
                 "positions_slack_used", "switches_slot_sum", "presence_sum", "idle_sum",
                 "n_jobs", "n_planes", "n_clients", "n_positions", "H",
                 "n_bin", "n_vars", "n_cons"]
    df = ensure_cols(df, base_cols)

    # Enriquecer desde el nombre del escenario
    parsed = df["scenario"].astype(str).apply(parse_scenario_name).apply(pd.Series)
    df = pd.concat([df, parsed], axis=1)

    # Numerizar
    num_cols = ["solve_time_s", "mipgap", "best_bound", "objective", "Tmax",
                "client_delay_sum", "viol_C_sum", "client_switches",
                "positions_slack_used", "switches_slot_sum", "presence_sum", "idle_sum",
                "n_jobs", "n_planes", "n_clients", "n_positions", "H", "n_bin", "n_vars", "n_cons"]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Factibilidad + gap rellenado si falta
    df["feasible"] = df.apply(lambda r: infer_feasible(str(r["status"]), str(r["termination"]), r["objective"]), axis=1)
    df["mipgap_filled"] = df["mipgap"]
    mask = ~np.isfinite(df["mipgap_filled"])
    df.loc[mask, "mipgap_filled"] = df.loc[mask].apply(lambda r: gap_from_bound(r["objective"], r["best_bound"]), axis=1)

    # Guardar enriquecido
    df.to_csv(OUTDIR / "results_enriched.csv", index=False)

    # -----------------------
    # 1) Resúmenes agregados
    # -----------------------
    by_model_mode = df.groupby(["model_id", "mode"], dropna=False).apply(agg_block).reset_index()
    by_model_mode.to_csv(OUTDIR / "summary_by_model_mode.csv", index=False)

    by_model_mode_type = df.groupby(["model_id", "mode", "scenario_type", "window_tightness"], dropna=False).apply(agg_block).reset_index()
    by_model_mode_type.to_csv(OUTDIR / "summary_by_model_mode_type.csv", index=False)

    # -----------------------
    # 2) Comparativa pareada (IMPORTANTE: no hay preset)
    # Emparejamos por (scenario, mode)
    # -----------------------
    key_cols = ["scenario", "mode"]

    keep = ["solve_time_s", "mipgap_filled", "feasible", "objective", "Tmax",
            "client_delay_sum", "viol_C_sum", "client_switches",
            "n_planes", "n_jobs", "n_positions", "H", "scenario_type", "window_tightness",
            "n_bin", "n_vars", "n_cons"]

    dfn = df[df["model_id"] == NEW_LABEL][key_cols + keep].copy().add_prefix("new_")
    dfo = df[df["model_id"] == OLD_LABEL][key_cols + keep].copy().add_prefix("old_")

    # recuperar keys sin prefijo (para merge)
    for k in key_cols:
        dfn[k] = df[df["model_id"] == NEW_LABEL][k].values
        dfo[k] = df[df["model_id"] == OLD_LABEL][k].values

    paired = pd.merge(dfn, dfo, on=key_cols, how="inner")
    paired["both_feasible"] = paired["new_feasible"] & paired["old_feasible"]

    # ratios/diffs (solo con ambos factibles)
    paired["time_ratio_old_over_new"] = paired["old_solve_time_s"] / paired["new_solve_time_s"]
    paired["log_time_ratio"] = np.log(paired["time_ratio_old_over_new"])
    paired["gap_diff_old_minus_new"] = paired["old_mipgap_filled"] - paired["new_mipgap_filled"]
    paired["Tmax_diff_old_minus_new"] = paired["old_Tmax"] - paired["new_Tmax"]
    paired["delay_diff_old_minus_new"] = paired["old_client_delay_sum"] - paired["new_client_delay_sum"]
    paired["violC_diff_old_minus_new"] = paired["old_viol_C_sum"] - paired["new_viol_C_sum"]
    paired["switch_diff_old_minus_new"] = paired["old_client_switches"] - paired["new_client_switches"]

    paired["win_time_new"] = paired["both_feasible"] & (paired["time_ratio_old_over_new"] > 1.0)
    paired["win_gap_new"] = paired["both_feasible"] & (paired["gap_diff_old_minus_new"] > 0.0)
    paired["win_Tmax_new"] = paired["both_feasible"] & (paired["Tmax_diff_old_minus_new"] > 0.0)
    paired["win_delay_new"] = paired["both_feasible"] & (paired["delay_diff_old_minus_new"] > 0.0)
    paired["win_violC_new"] = paired["both_feasible"] & (paired["violC_diff_old_minus_new"] > 0.0)
    paired["win_switch_new"] = paired["both_feasible"] & (paired["switch_diff_old_minus_new"] > 0.0)

    paired.to_csv(OUTDIR / "paired_raw.csv", index=False)

    wins = pd.DataFrame([{
        "n_pairs": len(paired),
        "n_both_feasible": int(paired["both_feasible"].sum()),
        "feasible_rate_new": float(paired["new_feasible"].mean()) if len(paired) else np.nan,
        "feasible_rate_old": float(paired["old_feasible"].mean()) if len(paired) else np.nan,
        "wins_time_new": int(paired["win_time_new"].sum()),
        "wins_gap_new": int(paired["win_gap_new"].sum()),
        "wins_Tmax_new": int(paired["win_Tmax_new"].sum()),
        "wins_delay_new": int(paired["win_delay_new"].sum()),
        "wins_violC_new": int(paired["win_violC_new"].sum()),
        "wins_switch_new": int(paired["win_switch_new"].sum()),
    }])
    wins.to_csv(OUTDIR / "paired_wins_summary.csv", index=False)

    # speedup por modo (solo pares factibles)
    pf = paired[paired["both_feasible"]].copy()
    if len(pf) > 0:
        sp_mode = pf.groupby("mode").agg(
            n=("scenario", "count"),
            speedup_p50=("time_ratio_old_over_new", "median"),
            speedup_p90=("time_ratio_old_over_new", lambda s: np.quantile(s.dropna(), 0.90) if len(s.dropna()) else np.nan),
            gap_diff_med=("gap_diff_old_minus_new", "median"),
            Tmax_diff_med=("Tmax_diff_old_minus_new", "median"),
            violC_diff_med=("violC_diff_old_minus_new", "median"),
            switch_diff_med=("switch_diff_old_minus_new", "median"),
        ).reset_index()
    else:
        sp_mode = pd.DataFrame(columns=["mode","n","speedup_p50","speedup_p90","gap_diff_med","Tmax_diff_med","violC_diff_med","switch_diff_med"])
    sp_mode.to_csv(OUTDIR / "paired_speedup_by_mode.csv", index=False)

    # -----------------------
    # 3) FIGURAS POTENTES
    # -----------------------
    dfb = df[df["feasible"]].copy()
    modes_order = sorted([m for m in df["mode"].dropna().unique()])

    # 3.1 ECDF tiempos
    plt.figure()
    for mid in [OLD_LABEL, NEW_LABEL]:
        g = dfb[dfb["model_id"] == mid]
        xs, ys = ecdf(g["solve_time_s"].values)
        if len(xs) > 0:
            plt.step(xs, ys, where="post", label=mid)
    plt.xlabel("Tiempo de resolución (s)")
    plt.ylabel("ECDF")
    plt.title("Distribución acumulada de tiempos (casos factibles)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTDIR / "fig_ecdf_solve_time.png", dpi=220)
    plt.close()

    # 3.2 Boxplots clave
    save_boxplot(dfb, "solve_time_s", "Tiempos por modo y modelo (factibles)", "Tiempo (s)",
                 OUTDIR / "fig_box_time_by_mode_model.png", modes_order)

    save_boxplot(dfb, "mipgap_filled", "MIPGap por modo y modelo (factibles)", "MIPGap",
                 OUTDIR / "fig_box_gap_by_mode_model.png", modes_order)

    save_boxplot(dfb, "Tmax", "Makespan (Tmax) por modo y modelo (factibles)", "Tmax (días)",
                 OUTDIR / "fig_box_Tmax_by_mode_model.png", modes_order)

    # 3.3 Barras (medias) KPIs operativos: violaciones, cambios, retraso
    save_bar_means(dfb, "viol_C_sum", "Violaciones cliente→posición (media) por modo y modelo", "viol_C_sum",
                   OUTDIR / "fig_bar_violC_mean_by_mode.png", modes_order)

    save_bar_means(dfb, "client_switches", "Cambios de cliente (media) por modo y modelo", "client_switches",
                   OUTDIR / "fig_bar_switch_mean_by_mode.png", modes_order)

    save_bar_means(dfb, "client_delay_sum", "Retraso total por cliente (media) por modo y modelo", "client_delay_sum",
                   OUTDIR / "fig_bar_delay_mean_by_mode.png", modes_order)

    # 3.4 Scatter: tiempo vs binarias (si existen)
    if dfb["n_bin"].notna().any():
        plt.figure()
        for mid in [OLD_LABEL, NEW_LABEL]:
            g = dfb[(dfb["model_id"] == mid) & (dfb["n_bin"].notna())]
            if len(g) > 0:
                plt.scatter(g["n_bin"].values, g["solve_time_s"].values, alpha=0.5, label=mid)
        plt.xlabel("# variables binarias (n_bin)")
        plt.ylabel("Tiempo (s)")
        plt.title("Escalabilidad: tiempo vs #binarias (factibles)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUTDIR / "fig_scatter_time_vs_bin.png", dpi=220)
        plt.close()

    # 3.5 Histograma speedup (solo pares factibles)
    if len(pf) > 0 and pf["log_time_ratio"].notna().any():
        plt.figure()
        plt.hist(pf["log_time_ratio"].dropna().values, bins=25)
        plt.xlabel("log( tiempo_old / tiempo_new )")
        plt.ylabel("Nº de pares")
        plt.title("Speedup (pares factibles): >0 implica que el modelo final es más rápido")
        plt.tight_layout()
        plt.savefig(OUTDIR / "fig_hist_log_speedup.png", dpi=220)
        plt.close()

    # 3.6 Heatmap factibilidad por tipo/tightness y modelo
    pivot = df.groupby(["model_id", "scenario_type", "window_tightness"])["feasible"].mean().reset_index()
    for mid in [OLD_LABEL, NEW_LABEL]:
        p = pivot[pivot["model_id"] == mid].pivot(index="scenario_type", columns="window_tightness", values="feasible")
        plt.figure(figsize=(6, 4))
        arr = p.values if p.size else np.zeros((1, 1))
        plt.imshow(arr, aspect="auto")
        plt.xticks(range(len(p.columns)), list(p.columns))
        plt.yticks(range(len(p.index)), list(p.index))
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                v = arr[i, j]
                if np.isfinite(v):
                    plt.text(j, i, f"{100*v:.0f}%", ha="center", va="center")
        plt.colorbar(label="Factibilidad")
        plt.title(f"Factibilidad por tipo de escenario ({mid})")
        plt.tight_layout()
        plt.savefig(OUTDIR / f"fig_heatmap_feasible_{mid}.png", dpi=220)
        plt.close()

    # 3.7 Trade-off: cumplimiento vs tiempo (por modo)
    #     (a) viol_C_sum vs solve_time_s
    #     (b) client_switches vs solve_time_s

    def _scatter_tradeoff_per_mode(dfb, y_col, y_label, fname_prefix):
        if y_col not in dfb.columns:
            return
        for m in modes_order:
            g = dfb[dfb["mode"] == m].copy()
            # Solo si hay datos
            if len(g) == 0:
                continue

            plt.figure(figsize=(6.5, 4.5))
            for mid in [OLD_LABEL, NEW_LABEL]:
                gg = g[g["model_id"] == mid].copy()
                # Filtrar NaNs
                gg = gg[gg["solve_time_s"].notna() & gg[y_col].notna()]
                if len(gg) == 0:
                    continue
                plt.scatter(gg["solve_time_s"].values, gg[y_col].values, alpha=0.6, label=mid)

            plt.xlabel("Tiempo de resolución (s)")
            plt.ylabel(y_label)
            plt.title(f"Trade-off por modo={m}: {y_label} vs tiempo (factibles)")
            plt.legend()
            plt.tight_layout()
            plt.savefig(OUTDIR / f"{fname_prefix}_{m}.png", dpi=220)
            plt.close()

    # Nota: dfb ya está filtrado a factibles
    _scatter_tradeoff_per_mode(dfb, "viol_C_sum", "viol_C_sum (violaciones cliente→posición)", "fig_tradeoff_violC_vs_time")
    _scatter_tradeoff_per_mode(dfb, "client_switches", "client_switches (cambios de cliente)", "fig_tradeoff_switch_vs_time")


    # -----------------------
    # 4) TXT con conclusiones listas para pegar
    # -----------------------
    def fmt(x, nd=3):
        if x is None or (isinstance(x, float) and not np.isfinite(x)):
            return "NA"
        return f"{x:.{nd}f}"

    lines = []
    lines.append("CONCLUSIONES AUTOMÁTICAS (comparativa modelo inicial vs modelo final)\n")
    lines.append(f"Modelos: {OLD_LABEL} (inicial) vs {NEW_LABEL} (final)")
    lines.append("Comparativa pareada por (scenario, mode). No existe columna 'preset' en los resultados.\n")

    lines.append(f"- Nº de pares comparados: {len(paired)}")
    lines.append(f"- Nº de pares con ambos factibles: {int(paired['both_feasible'].sum())}")
    lines.append(f"- Factibilidad (final): {100*paired['new_feasible'].mean():.1f}%")
    lines.append(f"- Factibilidad (inicial): {100*paired['old_feasible'].mean():.1f}%\n")

    if len(pf) > 0:
        lines.append("Rendimiento (solo pares con ambos factibles):")
        lines.append(f"- Speedup p50 (old/new) en tiempo: {fmt(pf['time_ratio_old_over_new'].median(), 2)}×")
        lines.append(f"- Speedup p90 (old/new) en tiempo: {fmt(np.quantile(pf['time_ratio_old_over_new'].dropna(), 0.90), 2)}×")
        lines.append(f"- % de wins en tiempo (final más rápido): {100*pf['win_time_new'].mean():.1f}%")
        lines.append(f"- Diferencia mediana de GAP (old - new): {fmt(pf['gap_diff_old_minus_new'].median(), 4)}")
        lines.append(f"- % de wins en GAP (final menor GAP): {100*pf['win_gap_new'].mean():.1f}%")
        lines.append(f"- Diferencia mediana de Tmax (old - new): {fmt(pf['Tmax_diff_old_minus_new'].median(), 3)}")
        lines.append(f"- % de wins en Tmax (final menor makespan): {100*pf['win_Tmax_new'].mean():.1f}%")
        lines.append(f"- Diferencia mediana viol_C_sum (old - new): {fmt(pf['violC_diff_old_minus_new'].median(), 3)}")
        lines.append(f"- % de wins en viol_C_sum (final menos violaciones): {100*pf['win_violC_new'].mean():.1f}%")
        lines.append(f"- Diferencia mediana client_switches (old - new): {fmt(pf['switch_diff_old_minus_new'].median(), 3)}")
        lines.append(f"- % de wins en client_switches (final menos cambios): {100*pf['win_switch_new'].mean():.1f}%\n")

    lines.append("Figuras generadas (recomendadas para memoria):")
    lines.append("- fig_ecdf_solve_time.png (robustez/tiempos)")
    lines.append("- fig_box_time_by_mode_model.png (tiempo por política)")
    lines.append("- fig_box_gap_by_mode_model.png (gap por política)")
    lines.append("- fig_box_Tmax_by_mode_model.png (makespan por política)")
    lines.append("- fig_bar_violC_mean_by_mode.png (cumplimiento cliente→posición)")
    lines.append("- fig_bar_switch_mean_by_mode.png (mezcla/cambios de cliente)")
    lines.append("- fig_scatter_time_vs_bin.png (escalabilidad vs tamaño)")
    lines.append("- fig_heatmap_feasible_initial/final.png (factibilidad por tipo de escenario)")
    lines.append("- fig_hist_log_speedup.png (mejora sistemática)\n")
    lines.append("- fig_tradeoff_violC_vs_time_<modo>.png (trade-off: cumplimiento vs tiempo)")
    lines.append("- fig_tradeoff_switch_vs_time_<modo>.png (trade-off: mezcla vs tiempo)")


    (OUTDIR / "conclusions_auto.txt").write_text("\n".join(lines), encoding="utf-8")

    print("OK. Análisis profundo generado en:", OUTDIR)


if __name__ == "__main__":
    main()
