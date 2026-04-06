#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch Runner para el modelo aircraft-positioning.

Propósito
---------
Automatizar un estudio experimental: para cada escenario Excel, cada modo de
política de cliente y cada preset del solver, construye el MIP, lo resuelve y
registra métricas en CSV más un log detallado del solver (Gurobi).

Flujo típico
------------
1. Descubre o recibe la lista de ficheros .xlsx (hoja única ``case`` por defecto).
2. Importa dinámicamente el módulo del modelo (``--model-module``).
3. Opcionalmente inyecta ``POSITIONS`` y ``PLANNING_START`` en ese módulo antes
   de llamar a ``read_case_single_sheet`` / ``build_model``.
4. Bucle triple: escenario × modo × preset → ``build_and_solve`` → fila CSV.

Entorno y rutas
---------------
- ``AP_DATA_DIR``: raíz de datos; si no existe, se usa el directorio hermano
  ``aircraft-positioning-data`` respecto al repo. Bajo ese árbol suelen vivir
  ``input/scenarios_suite``, ``output/results`` y ``output/logs``.

Salida
------
- CSV (``results_batch.csv`` o ``--out``): una fila por corrida con estado del
  solver, tiempos, gap, límites, tamaño del modelo y KPIs específicos del
  dominio (retrasos, violaciones, cambios de cliente, etc.).
- Si el solver es Gurobi: un ``.gurobi.log`` por corrida en ``--log-dir``.

Contrato del módulo de modelo (p.ej. ``src.solvers.client_policy``)
-------------------------------------------------------------------
- ``POSITIONS`` (lista de str): opcional; se puede fijar desde CLI con
  ``--positions`` o ``--n-positions``.
- ``read_case_single_sheet(xlsx_path, sheet_name="case", planning_start=...)``
  → diccionario/estructura ``d`` consumida por ``build_model``.
- ``build_model(d, client_pos_policy, w_client_pos, wms)`` → modelo Pyomo ``m``
  con componentes opcionales que este script intenta leer (``Tmax``, ``J``, ``R``,
  etc.); si falta un componente, el KPI correspondiente queda ``None``.
"""

import argparse
import csv
import os
import sys
import time
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any

# Raíz del repo: al ejecutar el script desde cualquier cwd, Python debe poder
# resolver imports como ``src.solvers...`` sin instalar el paquete en modo editable.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pyomo.environ as pyo
import importlib

# Directorio de datos externo. Configurable via variable de entorno AP_DATA_DIR.
# Por defecto apunta al directorio hermano aircraft-positioning-data.
_DATA_DIR = Path(os.environ.get("AP_DATA_DIR", str(_REPO_ROOT.parent / "aircraft-positioning-data")))


# -----------------------
# Defaults (customizados para tu TFM)
# -----------------------

# Shortlist usada cuando NO se pasa ``--no-only-scenarios``: evita barrer todo
# el directorio de escenarios y acorta iteraciones de prueba / TFM.
DEFAULT_ONLY_SCENARIOS = [
    "scn_few-loose_seed1_P3_pl5.xlsx",
    "scn_many-medium_seed1_P5_pl20.xlsx",
    "scn_many-medium_seed9_P5_pl20.xlsx",
    "scn_heavy-tight_seed8_P4_pl30.xlsx",
]


# -----------------------
# Utils
# -----------------------

def parse_positions_arg(positions_csv: Optional[str], n_positions: Optional[int], prefix: str) -> List[str]:
    """Construye la lista de nombres de posición para inyectar en el módulo del modelo.

    Prioridad: ``positions_csv`` (coma-separada) sobre ``n_positions`` (genera
    ``prefix1``..``prefixN``). Usado solo si el usuario pasa ``--positions`` o
    ``--n-positions`` en CLI.
    """
    if positions_csv:
        return [s.strip() for s in positions_csv.split(",") if s.strip()]
    if n_positions:
        return [f"{prefix}{i}" for i in range(1, int(n_positions) + 1)]
    raise ValueError("You must provide either --positions or --n-positions.")


def find_scenarios(scenarios_dir: str, pattern: str) -> List[Path]:
    """Lista ordenada de ``Path`` que coinciden con ``pattern`` (glob) bajo ``scenarios_dir``."""
    base = Path(scenarios_dir)
    if not base.exists():
        raise FileNotFoundError(f"Scenarios directory not found: {base}")
    return sorted(base.glob(pattern))


def solver_factory(name: str):
    """Crea el solver Pyomo y falla pronto si no está instalado / licenciado."""
    solver = pyo.SolverFactory(name)
    if not solver.available(False):
        raise RuntimeError(f"Solver '{name}' is not available in this environment.")
    return solver


def safe_val(v) -> float:
    """Devuelve 0.0 si la variable no está inicializada (value=None) en vez de romper."""
    try:
        if hasattr(v, "value"):
            return float(v.value) if v.value is not None else 0.0
        return float(pyo.value(v))
    except Exception:
        return 0.0


def sum_indexed(var_like) -> float:
    """Suma valores indexados de Pyomo omitiendo entradas con ``value is None``.

    Intenta primero ``.values()`` (dict-like); si falla, itera por índices. Cualquier
    excepción devuelve 0.0 — preferible a romper el batch por una variable huérfana.
    """
    total = 0.0
    try:
        for v in var_like.values():
            val = getattr(v, "value", None)
            if val is not None:
                total += float(val)
        return total
    except Exception:
        pass
    try:
        for idx in var_like:
            v = var_like[idx]
            val = getattr(v, "value", None)
            if val is not None:
                total += float(val)
    except Exception:
        pass
    return total


def count_model_sizes(m) -> Dict[str, Optional[int]]:
    """Cuenta variables y restricciones activas del modelo Pyomo (a nivel de *data*).

    Devuelve:
      - n_constraints: nº de ConstraintData activas
      - n_vars: nº de VarData activas
      - n_bin / n_int / n_cont: desglose de variables por dominio
    """
    try:
        n_constraints = sum(
            1 for _ in m.component_data_objects(pyo.Constraint, active=True, descend_into=True)
        )
    except Exception:
        n_constraints = None

    try:
        n_vars = 0
        n_bin = 0
        n_int = 0
        n_cont = 0
        for v in m.component_data_objects(pyo.Var, active=True, descend_into=True):
            n_vars += 1
            if v.is_binary():
                n_bin += 1
            elif v.is_integer():
                n_int += 1
            else:
                n_cont += 1
    except Exception:
        n_vars = n_bin = n_int = n_cont = None

    return {
        "n_constraints": n_constraints,
        "n_vars": n_vars,
        "n_bin": n_bin,
        "n_int": n_int,
        "n_cont": n_cont,
    }

def _extract_solver_mipgap(results) -> Optional[float]:
    """Intenta extraer el MIP gap del objeto ``results`` de Pyomo.

    Los solvers reportan el gap en sitios distintos según versión y interfaz;
    se prueba primero ``results.solver`` y luego el primer registro de ``problem``.
    """
    try:
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
    """Mejor cota conocida (típicamente dual / LB en minimización según el solver)."""
    try:
        bb = results.solver.get("best_bound", None)
        if bb is not None:
            return float(bb)
    except Exception:
        pass
    try:
        prob = getattr(results, "problem", None)
        if prob and len(prob) > 0:
            rec = prob[0]
            for k in ("Lower bound", "lower_bound", "Best bound", "best_bound"):
                if k in rec and rec[k] is not None:
                    return float(rec[k])
    except Exception:
        pass
    return None


def objective_from_results(results) -> Optional[float]:
    """Valor del objetivo reportado por el solver, sin reevaluar la expresión Pyomo en Python.

    Útil cuando el modelo es grande o hay componentes que disparan warnings al acceder al objetivo.
    """
    try:
        obj = results.solver.get("objective", None)
        if obj is not None:
            return float(obj)
    except Exception:
        pass
    try:
        prob = getattr(results, "problem", None)
        if prob and len(prob) > 0:
            rec = prob[0]
            for k in ("Upper bound", "upper_bound", "Objective", "objective"):
                if k in rec and rec[k] is not None:
                    return float(rec[k])
    except Exception:
        pass
    return None


def apply_preset_options(solver, solver_name: str, preset: str, timelimit: Optional[float], mipgap: Optional[float]):
    """
    Ajusta opciones del solver según preset y aplica TimeLimit / MIP gap.

    Implementación:
      - Gurobi: nombres de opciones oficiales (``TimeLimit``, ``MIPGap``, etc.).
      - CPLEX / CBC: claves típicas en la interfaz Pyomo (pueden variar según versión).
      - Otros: ``timelimit`` / ``mipgap`` genéricos como último recurso.

    Presets (solo amplían ajustes cuando ``solver_name`` es Gurobi):
      - ``base``: solo límites; sin tocar ``MIPFocus`` ni heurística.
      - ``fast``: ``MIPFocus=1`` — énfasis en encontrar soluciones incumbentes rápido.
      - ``feasible``: ``MIPFocus=3`` — énfasis en hallar una factible cuando el modelo es duro.
    """
    sname = solver_name.lower()
    opts = solver.options

    # Timelimit / gap genéricos por solver
    try:
        if sname.startswith("gurobi"):
            if timelimit is not None and timelimit > 0:
                opts["TimeLimit"] = float(timelimit)
            if mipgap is not None:
                opts["MIPGap"] = float(mipgap)
        elif sname.startswith("cplex"):
            if timelimit is not None and timelimit > 0:
                opts["timelimit"] = float(timelimit)
            if mipgap is not None:
                opts["mip tolerances mipgap"] = float(mipgap)
        elif sname == "cbc":
            if timelimit is not None and timelimit > 0:
                opts["seconds"] = float(timelimit)
            if mipgap is not None:
                opts["ratioGap"] = float(mipgap)
        else:
            if timelimit is not None and timelimit > 0:
                opts["timelimit"] = float(timelimit)
            if mipgap is not None:
                opts["mipgap"] = float(mipgap)
    except Exception:
        pass

    # Presets específicos (Gurobi)
    if not sname.startswith("gurobi"):
        return

    preset = (preset or "base").lower().strip()
    if preset == "base":
        return
    if preset == "fast":
        opts["MIPFocus"] = 1
        opts["Heuristics"] = 0.2
        opts["Cuts"] = 1
        opts["Presolve"] = 2
        return
    if preset == "feasible":
        opts["MIPFocus"] = 3
        opts["Heuristics"] = 0.5
        opts["Cuts"] = 1
        opts["Presolve"] = 2
        return


def _safe_filename(s: str) -> str:
    """Nombre de fichero seguro para logs: reemplaza caracteres problemáticos en Windows/Unix."""
    return "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in s)


# -----------------------
# Core
# -----------------------

def build_and_solve(model_mod,
                    xlsx_path: Path,
                    sheet_name: str,
                    mode: str,
                    w_client_pos: float,
                    wms: float,
                    solver_name: str,
                    timelimit: Optional[float],
                    mipgap: Optional[float],
                    preset: str,
                    log_file: Optional[Path] = None,
                    tee: bool = False) -> Dict[str, Any]:
    """Pipeline completo de una corrida: leer Excel → Pyomo → solver → dict de KPIs.

    ``mode`` se pasa como ``client_pos_policy`` al constructor del modelo (p.ej.
    ``soft12``, ``hard``): define cómo se penalizan o relajan asignaciones cliente–posición.

    Las posiciones del hangar vienen del **módulo** (variable global ``POSITIONS``
    inyectada en ``main`` si el usuario lo pidió), no del Excel; por eso el lector
    no recibe un argumento ``positions`` explícito aquí.

    Returns:
        Diccionario listo para fusionar en una fila CSV (sin claves ``scenario``,
        ``mode``, ``preset``; esas las añade ``main``).
    """
    # 1) Leer datos del escenario (NO pasar 'positions' al lector)
    d = model_mod.read_case_single_sheet(
        str(xlsx_path),
        sheet_name=sheet_name,
        planning_start=getattr(model_mod, "PLANNING_START", None)
    )

    # 2) Construir modelo
    m = model_mod.build_model(d, client_pos_policy=mode, w_client_pos=w_client_pos, wms=wms)

    # 3) Solver + presets + opciones
    solver = solver_factory(solver_name)
    apply_preset_options(solver, solver_name, preset, timelimit, mipgap)

    # 3.1) LogFile por corrida (solo Gurobi)
    sname = solver_name.lower()
    if log_file is not None and sname.startswith("gurobi"):
        log_file.parent.mkdir(parents=True, exist_ok=True)
        solver.options["LogFile"] = str(log_file)

    # Tras la resolución, al leer valores de variables, Pyomo puede loguear
    # advertencias masivas; subimos el umbral solo durante esta corrida.
    logging.getLogger("pyomo.core").setLevel(logging.ERROR)
    logging.getLogger("pyomo").setLevel(logging.ERROR)

    # 4) Resolver y medir tiempo (wall-clock de la llamada a solve, no CPU interna)
    t0 = time.perf_counter()
    results = solver.solve(m, tee=bool(tee))
    t1 = time.perf_counter()
    solve_time = t1 - t0

    # 5) Estado/gap/objetivo/bound
    term = str(getattr(results.solver, "termination_condition", ""))
    status = str(getattr(results.solver, "status", ""))

    mipgap_out = _extract_solver_mipgap(results)
    best_bound = _extract_best_bound(results)
    objective_val = objective_from_results(results)

    # 5.1) Tamaño del modelo (Pyomo)
    sizes = count_model_sizes(m)

    # 6) KPIs del modelo: cada ``hasattr`` permite modelos que no declaren algún
    #    bloque (p.ej. sin holgura de posiciones); ``safe_val`` / ``sum_indexed``
    #    evitan fallar si el solver dejó componentes en None.
    metrics = {
        "status": status,
        "termination": term,
        "solve_time_s": round(solve_time, 3),
        "mipgap": mipgap_out,
        "best_bound": best_bound,
        "objective": objective_val,
        "n_constraints": sizes.get("n_constraints"),
        "n_vars": sizes.get("n_vars"),
        "n_bin": sizes.get("n_bin"),
        "n_int": sizes.get("n_int"),
        "n_cont": sizes.get("n_cont"),
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
    return metrics


# -----------------------
# CLI
# -----------------------

def main():
    """Punto de entrada: parsea CLI, resuelve lista de escenarios, escribe CSV y logs."""
    ap = argparse.ArgumentParser(description="Batch runner for aircraft-positioning scenarios (con logs por corrida).")
    ap.add_argument("--model-module", type=str, default="src.solvers.client_policy",
                    help="Nombre del módulo Python del modelo (sin .py). "
                         "Por defecto: src.solvers.client_policy. "
                         "Alternativa: src.solvers.standard")

    # Aceptar ambos nombres (--scenarios y --scenarios-dir)
    ap.add_argument("--scenarios-dir", type=str,
                    default=str(_DATA_DIR / "input" / "scenarios_suite"),
                    help="Carpeta donde están los .xlsx. Por defecto: AP_DATA_DIR/input/scenarios_suite")
    ap.add_argument("--pattern", type=str, default="*.xlsx",
                    help="Patrón glob para seleccionar escenarios (si NO usas --only-scenarios).")
    ap.add_argument("--sheet", type=str, default="case",
                    help="Nombre de la hoja dentro del .xlsx.")

    # NUEVO: ejecutar solo una lista explícita (por defecto, tu shortlist del TFM)
    ap.add_argument("--only-scenarios", type=str, nargs="+", default=DEFAULT_ONLY_SCENARIOS,
                    help="Lista explícita de ficheros .xlsx a ejecutar (relativos a --scenarios-dir). "
                         "Por defecto, usa la shortlist fijada en el script.")
    ap.add_argument("--no-only-scenarios", action="store_true",
                    help="Desactiva la shortlist por defecto y usa --pattern para seleccionar escenarios.")

    # NUEVO: logs por corrida
    ap.add_argument("--log-dir", type=str,
                    default=str(_DATA_DIR / "output" / "logs"),
                    help="Carpeta donde guardar los logs de Gurobi (uno por corrida). "
                         "Por defecto: AP_DATA_DIR/output/logs")
    ap.add_argument("--tee", action="store_true",
                    help="Si se activa, imprime salida del solver por pantalla además de guardar log.")

    ap.add_argument("--modes", type=str, nargs="+", default=["soft12"],
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
                    help="Peso para la parte blanda de la política cliente→posición.")
    ap.add_argument("--wms", type=float, default=1.0,
                    help="Peso del makespan.")

    ap.add_argument("--positions", type=str, default=None,
                    help="Lista de posiciones separadas por comas (e.g., position1,position2,position3).")
    ap.add_argument("--n-positions", type=int, default=None,
                    help="Si no se pasa --positions, genera este número con el prefijo.")
    ap.add_argument("--pos-prefix", type=str, default="position",
                    help="Prefijo para generar posiciones con --n-positions.")

    # Aceptar --results-dir además de --out
    ap.add_argument("--results-dir", type=str,
                    default=str(_DATA_DIR / "output" / "results"),
                    help="Carpeta donde guardar el CSV (se llamará results_batch.csv). "
                         "Por defecto: AP_DATA_DIR/output/results")
    ap.add_argument("--out", type=str, default=None,
                    help="Ruta del CSV de salida. Si se pasa, tiene prioridad sobre --results-dir.")

    args = ap.parse_args()

    # Carpeta de los .xlsx. El parser define ``--scenarios-dir`` con default; la
    # expresión ``or args.scenarios`` es heredada por si existiera alias antiguo
    # (no registrado hoy; en uso normal ``scenarios_dir`` ya viene informado).
    scenarios_dir = args.scenarios_dir or args.scenarios or "scenarios"

    # Import dinámico: el mismo script sirve para ``client_policy``, ``standard``, etc.
    model_mod = importlib.import_module(args.model_module)

    # ``POSITIONS`` debe existir en el módulo antes de ``build_model`` si el modelo
    # indexa posiciones por nombre; sin CLI, el módulo usa su valor por defecto.
    if args.positions or args.n_positions:
        pos_list = parse_positions_arg(args.positions, args.n_positions, args.pos_prefix)
        setattr(model_mod, "POSITIONS", pos_list)

    # Asegura PLANNING_START por si el lector lo usa
    if not hasattr(model_mod, "PLANNING_START"):
        setattr(model_mod, "PLANNING_START", None)

    # Dos modos: (A) shortlist explícita por nombre —comportamiento por defecto—;
    # (B) barrido por glob con ``--pattern`` tras ``--no-only-scenarios``.
    files: List[Path] = []
    if args.no_only_scenarios:
        files = find_scenarios(scenarios_dir, args.pattern)
    else:
        base = Path(scenarios_dir)
        for name in (args.only_scenarios or []):
            p = base / name
            if not p.exists():
                raise FileNotFoundError(f"Scenario not found: {p}")
            files.append(p)
        files = sorted(files)

    if not files:
        print(f"No .xlsx files found in {scenarios_dir} (only_scenarios={args.only_scenarios}, pattern={args.pattern})")
        sys.exit(1)

    # Ruta del CSV: ``--out`` gana; si no, subcarpeta fija bajo resultados.
    if args.out:
        out_path = Path(args.out)
    elif args.results_dir:
        out_path = Path(args.results_dir) / "results_batch.csv"
    else:
        out_path = Path("results_batch.csv")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Orden de columnas fijo para comparar corridas y concatenar resultados en análisis.
    fieldnames = [
        "scenario", "mode", "preset",
        "status", "termination", "solve_time_s", "mipgap", "best_bound",
        "objective", "n_constraints", "n_vars", "n_bin", "n_int", "n_cont", "Tmax", "client_delay_sum", "viol_C_sum", "client_switches",
        "positions_slack_used", "switches_slot_sum", "presence_sum", "idle_sum",
        "n_jobs", "n_planes", "n_clients", "n_positions", "H"
    ]

    log_dir = Path(args.log_dir)

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for fp in files:
            for mode in args.modes:
                for preset in args.solver_presets:
                    # Un log por combinación; el nombre codifica escenario + política + preset.
                    log_name = f"{_safe_filename(fp.stem)}__{_safe_filename(mode)}__{_safe_filename(preset)}.gurobi.log"
                    log_file = log_dir / log_name

                    try:
                        metrics = build_and_solve(
                            model_mod,
                            xlsx_path=fp,
                            sheet_name=args.sheet,
                            mode=mode,
                            w_client_pos=args.w_client_pos,
                            wms=args.wms,
                            solver_name=args.solver,
                            timelimit=args.timelimit,
                            mipgap=args.mipgap,
                            preset=preset,
                            log_file=log_file,
                            tee=args.tee
                        )
                        row = {"scenario": fp.name, "mode": mode, "preset": preset}
                        row.update(metrics)
                        writer.writerow(row)
                        print(f"[OK] {fp.name} | {mode}:{preset} | cons={metrics.get('n_constraints')} vars={metrics.get('n_vars')} | {metrics['solve_time_s']}s | "
                              f"gap={metrics['mipgap']} | best={metrics['best_bound']} | obj={metrics['objective']} | log={log_file}")
                    except Exception as e:
                        # Fila mínima para no perder el rastro del fallo en el CSV;
                        # el resto de columnas quedará vacío en el lector CSV.
                        row = {
                            "scenario": fp.name, "mode": mode, "preset": preset,
                            "status": "ERROR", "termination": str(e)
                        }
                        writer.writerow(row)
                        print(f"[ERR] {fp.name} | {mode}:{preset} | {e}")

    print(f"\nSaved results to {out_path.resolve()}")
    if args.solver.lower().startswith("gurobi"):
        print(f"Saved Gurobi logs to {log_dir.resolve()}")


if __name__ == "__main__":
    main()
