#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch Runner para ejecutar escenarios .xlsx con el modelo "antiguo"
(model_func1_sh_client_policy.py) y exportar resultados a CSV.

CORRECCIÓN CLAVE (escenarios de scenario_maker):
- Los escenarios tienen SOLO hoja 'case' (sin 'Planes2').
- El modelo antiguo read_input(...) requiere planes_sheet (por defecto 'Planes2').
- Este runner crea automáticamente una hoja de aviones (Planes2) a partir de 'case'
  usando: early_start = min(es) y late_finish = max(lf) por avión, y ejecuta sobre
  un Excel temporal para no tocar tus ficheros.

Uso típico:
python batch_runner.py \
  --model-module model_func1_sh_client_policy \
  --scenarios-dir scenarios \
  --pattern "*.xlsx" \
  --sheet "case" \
  --planes-sheet "Planes2" \
  --modes soft12 soft medium hard \
  --solver gurobi \
  --timelimit 300 \
  --mipgap 0.02 \
  --n-positions 5 \
  --w-client-pos 1e6 \
  --wms 0.0 \
  --out results_old.csv
"""

import argparse
import time
from pathlib import Path
from typing import List, Optional, Dict, Any
import importlib
import tempfile
import os

import pandas as pd
import pyomo.environ as pyo


# -----------------------
# Utils
# -----------------------

def parse_positions_arg(positions_csv: Optional[str], n_positions: Optional[int], prefix: str) -> List[str]:
    if positions_csv:
        return [s.strip() for s in positions_csv.split(",") if s.strip()]
    if n_positions:
        return [f"{prefix}{i}" for i in range(1, int(n_positions) + 1)]
    return []


def find_scenarios(scenarios_dir: Path, pattern: str) -> List[Path]:
    if not scenarios_dir.exists():
        raise FileNotFoundError(f"No existe la carpeta de escenarios: {scenarios_dir}")
    files = sorted(scenarios_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No se han encontrado escenarios con patrón '{pattern}' en {scenarios_dir}")
    return files


def solver_factory(name: str):
    solver = pyo.SolverFactory(name)
    if not solver.available(False):
        raise RuntimeError(f"Solver '{name}' no está disponible en este entorno.")
    return solver


def safe_val(v) -> float:
    try:
        if hasattr(v, "value"):
            return float(v.value) if v.value is not None else 0.0
        return float(pyo.value(v))
    except Exception:
        return 0.0


def sum_indexed(var) -> float:
    try:
        return float(sum(safe_val(var[idx]) for idx in var))
    except Exception:
        return 0.0


def _extract_solver_mipgap(results) -> Optional[float]:
    try:
        if hasattr(results, "solver") and hasattr(results.solver, "gap") and results.solver.gap is not None:
            return float(results.solver.gap)
    except Exception:
        pass
    try:
        if hasattr(results, "solver") and hasattr(results.solver, "get"):
            g = results.solver.get("gap", None)
            if g is not None:
                return float(g)
    except Exception:
        pass
    try:
        prob = getattr(results, "problem", None)
        if prob and len(prob) > 0:
            rec = prob[0]
            for k in ("MIPGap", "mipgap", "gap"):
                if k in rec and rec[k] is not None:
                    return float(rec[k])
    except Exception:
        pass
    return None


def _extract_best_bound(results) -> Optional[float]:
    try:
        if hasattr(results, "solver") and hasattr(results.solver, "get"):
            for k in ("best_bound", "BestBound", "bestbound", "dual_bound", "DualBound"):
                bb = results.solver.get(k, None)
                if bb is not None:
                    return float(bb)
    except Exception:
        pass
    try:
        prob = getattr(results, "problem", None)
        if prob and len(prob) > 0:
            rec = prob[0]
            for k in ("UpperBound", "LowerBound", "dual_bound", "best_bound"):
                if k in rec and rec[k] is not None:
                    return float(rec[k])
    except Exception:
        pass
    return None


def objective_from_results(results) -> Optional[float]:
    try:
        prob = getattr(results, "problem", None)
        if prob and len(prob) > 0:
            rec = prob[0]
            for k in ("UpperBound", "LowerBound", "Objective", "objective"):
                if k in rec and rec[k] is not None:
                    return float(rec[k])
    except Exception:
        pass
    return None


def apply_preset_options(solver, preset: str):
    preset = (preset or "base").lower()
    if preset == "base":
        return
    if preset == "fast":
        try:
            solver.options["MIPFocus"] = 1
        except Exception:
            pass
        try:
            solver.options["Heuristics"] = 0.2
        except Exception:
            pass
        return
    if preset == "feasible":
        try:
            solver.options["MIPFocus"] = 1
        except Exception:
            pass
        try:
            solver.options["Heuristics"] = 0.5
        except Exception:
            pass
        return


def _model_size_metrics(m: pyo.ConcreteModel) -> Dict[str, Optional[int]]:
    try:
        n_vars = int(sum(1 for _ in m.component_data_objects(pyo.Var, active=True)))
    except Exception:
        n_vars = None
    try:
        n_cons = int(sum(1 for _ in m.component_data_objects(pyo.Constraint, active=True)))
    except Exception:
        n_cons = None
    try:
        n_bin = int(sum(
            1 for v in m.component_data_objects(pyo.Var, active=True)
            if getattr(v, "is_binary", lambda: False)()
        ))
    except Exception:
        n_bin = None
    return {"n_vars": n_vars, "n_cons": n_cons, "n_bin": n_bin}


# -----------------------
# Excel compatibility for scenario_maker
# -----------------------

def _build_planes_sheet_from_case(df_case: pd.DataFrame) -> pd.DataFrame:
    """
    Escenarios de scenario_maker: hoja 'case' con columnas:
      plane, task, job, duration, client, es, lf
    Creamos df_planes con columnas:
      plane, early_start, late_finish
    Donde:
      early_start = min(es) por avión
      late_finish = max(lf) por avión
    """
    required = {"plane", "es", "lf"}
    missing = required - set(df_case.columns)
    if missing:
        # fallback conservador si el escenario no tiene es/lf
        # (no debería ocurrir con tus escenarios)
        planes = pd.to_numeric(df_case["plane"], errors="coerce").dropna().astype(int).unique()
        return pd.DataFrame({"plane": planes, "early_start": 0.0, "late_finish": 1e6})

    tmp = df_case.copy()
    tmp["plane"] = pd.to_numeric(tmp["plane"], errors="coerce")
    tmp = tmp[tmp["plane"].notna()].copy()
    tmp["plane"] = tmp["plane"].astype(int)
    tmp["es"] = pd.to_numeric(tmp["es"], errors="coerce")
    tmp["lf"] = pd.to_numeric(tmp["lf"], errors="coerce")

    agg = tmp.groupby("plane", as_index=False).agg(
        early_start=("es", "min"),
        late_finish=("lf", "max")
    )
    return agg


def _ensure_planes_sheet_temp_xlsx(xlsx_path: Path, case_sheet: str, planes_sheet: str) -> Path:
    """
    Si el Excel no tiene planes_sheet, crea un Excel temporal con:
      - case_sheet (copia exacta)
      - planes_sheet (derivada de case: early_start / late_finish)
    """
    xls = pd.ExcelFile(xlsx_path)
    if planes_sheet in xls.sheet_names:
        return xlsx_path  # ya está

    if case_sheet not in xls.sheet_names:
        raise ValueError(f"El escenario {xlsx_path.name} no contiene la hoja '{case_sheet}'.")

    df_case = pd.read_excel(xlsx_path, sheet_name=case_sheet)
    df_planes = _build_planes_sheet_from_case(df_case)

    # Crear temporal
    tmp = tempfile.NamedTemporaryFile(prefix="tmp_with_planes_", suffix=".xlsx", delete=False)
    tmp_path = Path(tmp.name)
    tmp.close()

    with pd.ExcelWriter(tmp_path, engine="openpyxl") as writer:
        df_case.to_excel(writer, sheet_name=case_sheet, index=False)
        df_planes.to_excel(writer, sheet_name=planes_sheet, index=False)

    return tmp_path


# -----------------------
# Core runner
# -----------------------

def _read_case(model_mod, xlsx_path: Path, case_sheet: str, planes_sheet: str):
    """
    Compatibilidad:
    - read_case_single_sheet(xlsx_path, sheet_name=..., planning_start=...)
    - read_input(xlsx_path, case_sheet, planning_start=..., planes_sheet=...)

    Para el modelo antiguo confirmado: read_input + requiere planes_sheet.
    Si el Excel no la tiene, se crea un temporal con esa hoja.
    """
    planning_start = getattr(model_mod, "PLANNING_START", None)

    if hasattr(model_mod, "read_case_single_sheet"):
        return model_mod.read_case_single_sheet(
            str(xlsx_path),
            sheet_name=case_sheet,
            planning_start=planning_start
        )

    if hasattr(model_mod, "read_input"):
        tmp_path = None
        try:
            # Crear temporal si falta planes_sheet
            tmp_path = _ensure_planes_sheet_temp_xlsx(xlsx_path, case_sheet=case_sheet, planes_sheet=planes_sheet)
            return model_mod.read_input(
                str(tmp_path),
                case_sheet,
                planning_start=planning_start,
                planes_sheet=planes_sheet
            )
        finally:
            # Si se creó un temporal, borrarlo
            try:
                if tmp_path is not None and Path(tmp_path) != Path(xlsx_path) and Path(tmp_path).exists():
                    os.remove(tmp_path)
            except Exception:
                pass

    raise AttributeError("El módulo del modelo debe implementar read_case_single_sheet(...) o read_input(...).")


def _build_model(model_mod, d: dict, mode: str, w_client_pos: float, wms: float):
    if not hasattr(model_mod, "build_model"):
        raise AttributeError("El módulo del modelo debe implementar build_model(d, ...).")

    try:
        return model_mod.build_model(d, client_pos_policy=mode, w_client_pos=w_client_pos, wms=wms)
    except TypeError:
        if hasattr(model_mod, "CLIENT_POS_POLICY"):
            setattr(model_mod, "CLIENT_POS_POLICY", mode)
        if hasattr(model_mod, "W_CLIENT_POS"):
            setattr(model_mod, "W_CLIENT_POS", w_client_pos)
        if hasattr(model_mod, "W_MAKESPAN"):
            setattr(model_mod, "W_MAKESPAN", wms)
        return model_mod.build_model(d)


def build_and_solve(model_mod,
                    xlsx_path: Path,
                    case_sheet: str,
                    planes_sheet: str,
                    mode: str,
                    w_client_pos: float,
                    wms: float,
                    solver_name: str,
                    timelimit: Optional[float],
                    mipgap: Optional[float],
                    preset: str,
                    tee: bool = False) -> Dict[str, Any]:
    t0 = time.time()

    d = _read_case(model_mod, xlsx_path, case_sheet=case_sheet, planes_sheet=planes_sheet)
    m = _build_model(model_mod, d, mode=mode, w_client_pos=w_client_pos, wms=wms)

    solver = solver_factory(solver_name)
    apply_preset_options(solver, preset=preset)

    if timelimit is not None:
        try:
            solver.options["TimeLimit"] = float(timelimit)
        except Exception:
            try:
                solver.options["timelimit"] = float(timelimit)
            except Exception:
                pass

    if mipgap is not None:
        try:
            solver.options["MIPGap"] = float(mipgap)
        except Exception:
            try:
                solver.options["mipgap"] = float(mipgap)
            except Exception:
                pass

    results = solver.solve(m, tee=tee)
    solve_time = time.time() - t0

    status = str(getattr(results.solver, "status", "NA"))
    term = str(getattr(results.solver, "termination_condition", "NA"))

    mipgap_out = _extract_solver_mipgap(results)
    best_bound = _extract_best_bound(results)

    objective_val = objective_from_results(results)
    if objective_val is None:
        try:
            objective_val = float(pyo.value(m.obj))
        except Exception:
            objective_val = None

    metrics = {
        "status": status,
        "termination": term,
        "solve_time_s": round(float(solve_time), 3),
        "mipgap": float(mipgap_out) if mipgap_out is not None else None,
        "best_bound": float(best_bound) if best_bound is not None else None,
        "objective": objective_val,
        "Tmax": safe_val(m.Tmax) if hasattr(m, "Tmax") else None,
        "client_delay_sum": sum_indexed(m.vClientDelay) if hasattr(m, "vClientDelay") else None,
        "viol_C_sum": sum_indexed(m.viol_C) if hasattr(m, "viol_C") else None,
        "client_switches": sum_indexed(m.client_change) if hasattr(m, "client_change") else None,
        "positions_slack_used": sum_indexed(m.u_pos) if hasattr(m, "u_pos") else None,
        "switches_slot_sum": sum_indexed(m.v01SwitchPlanes) if hasattr(m, "v01SwitchPlanes") else None,
        "presence_sum": sum_indexed(m.vPresence) if hasattr(m, "vPresence") else None,
        "idle_sum": sum_indexed(m.vIdle) if hasattr(m, "vIdle") else None,
        "n_jobs": len(list(m.J)) if hasattr(m, "J") else None,
        "n_planes": len(list(m.R)) if hasattr(m, "R") else None,
        "n_clients": len(list(m.C)) if hasattr(m, "C") else None,
        "n_positions": len(list(m.P)) if hasattr(m, "P") else None,
        "H": float(pyo.value(m.H)) if hasattr(m, "H") else None,
    }

    metrics.update(_model_size_metrics(m))
    return metrics


# -----------------------
# CLI
# -----------------------

def main():
    ap = argparse.ArgumentParser(description="Batch runner para escenarios aircraft-positioning (modelo antiguo).")
    ap.add_argument("--model-module", type=str, default="model_func1_sh_client_policy",
                    help="Nombre del módulo Python del modelo (sin .py).")

    ap.add_argument("--scenarios-dir", type=str, default="scenarios",
                    help="Carpeta que contiene los escenarios .xlsx.")
    ap.add_argument("--pattern", type=str, default="*.xlsx",
                    help="Patrón de búsqueda de escenarios (glob).")

    ap.add_argument("--sheet", type=str, default="case",
                    help="Nombre de hoja con el caso (case_sheet).")
    ap.add_argument("--planes-sheet", type=str, default="Planes2",
                    help="Nombre de hoja con aviones que requiere el modelo antiguo; si no existe, se crea temporalmente.")

    ap.add_argument("--modes", type=str, nargs="+", default=["soft12", "soft", "medium", "hard"],
                    help="Modos de política a evaluar.")
    ap.add_argument("--solver-presets", type=str, nargs="+",
                    default=["base"],
                    choices=["base", "fast", "feasible"],
                    help="Presets del solver a cruzar con los modos.")
    ap.add_argument("--solver", type=str, default="gurobi",
                    help="Solver (gurobi, cplex, cbc, glpk, ...).")
    ap.add_argument("--timelimit", type=float, default=300.0,
                    help="Límite de tiempo en segundos.")
    ap.add_argument("--mipgap", type=float, default=0.02,
                    help="MIP gap objetivo.")

    ap.add_argument("--w-client-pos", type=float, default=1e6,
                    help="Peso para la parte blanda de la política cliente-posición.")
    ap.add_argument("--wms", type=float, default=0.0,
                    help="Peso del makespan (si el modelo lo usa).")

    ap.add_argument("--positions", type=str, default=None,
                    help="Lista explícita de posiciones separadas por comas. Ej: position1,position2,...")
    ap.add_argument("--n-positions", type=int, default=None,
                    help="Número de posiciones (si no se pasa --positions).")
    ap.add_argument("--pos-prefix", type=str, default="position",
                    help="Prefijo para generar posiciones si se usa --n-positions.")

    ap.add_argument("--out", type=str, default="results_old.csv",
                    help="CSV de salida.")
    ap.add_argument("--tee", action="store_true",
                    help="Muestra el log del solver (tee=True).")

    args = ap.parse_args()

    model_mod = importlib.import_module(args.model_module)

    pos_list = parse_positions_arg(args.positions, args.n_positions, args.pos_prefix)
    if pos_list:
        setattr(model_mod, "POSITIONS", pos_list)

    scenarios_dir = Path(args.scenarios_dir)
    files = find_scenarios(scenarios_dir, args.pattern)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "scenario", "mode", "preset",
        "status", "termination", "solve_time_s", "mipgap", "best_bound",
        "objective", "Tmax", "client_delay_sum", "viol_C_sum", "client_switches",
        "positions_slack_used", "switches_slot_sum", "presence_sum", "idle_sum",
        "n_jobs", "n_planes", "n_clients", "n_positions", "H",
        "n_vars", "n_cons", "n_bin"
    ]

    import csv
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for fp in files:
            for mode in args.modes:
                for preset in args.solver_presets:
                    try:
                        metrics = build_and_solve(
                            model_mod,
                            xlsx_path=fp,
                            case_sheet=args.sheet,
                            planes_sheet=args.planes_sheet,
                            mode=mode,
                            w_client_pos=args.w_client_pos,
                            wms=args.wms,
                            solver_name=args.solver,
                            timelimit=args.timelimit,
                            mipgap=args.mipgap,
                            preset=preset,
                            tee=args.tee
                        )
                        row = {"scenario": fp.name, "mode": mode, "preset": preset}
                        row.update(metrics)
                        writer.writerow(row)
                        print(f"[OK] {fp.name} | {mode} | {preset}")
                    except Exception as e:
                        row = {"scenario": fp.name, "mode": mode, "preset": preset,
                               "status": "ERROR", "termination": str(e)}
                        writer.writerow(row)
                        print(f"[ERR] {fp.name} | {mode} | {preset} -> {e}")

    print("\nListo. CSV generado:", out_path.resolve())


if __name__ == "__main__":
    main()
