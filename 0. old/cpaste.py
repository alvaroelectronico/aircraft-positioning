# -*- coding: utf-8 -*-
"""
Modelo de posicionamiento de aeronaves - versión con BLOQUEO integrado (FIXED)
- Tiempo continuo en días: t_start[j], t_end[j] con duración D[j].
- Asignación discreta a posición: y[j,p] binaria.
- No solape dentro de la misma posición (disyunción linealizada).
- BLOQUEO de calle:
    * Para usar position4: la position3 debe estar libre en el instante de entrada/salida.
    * Para usar position5: deben estar libres position4 y position3 en entrada/salida.
- Política "cliente -> posición (soft/off)".
- Gantt y validación posterior.
Requisitos:
  pip install pyomo pandas numpy plotly openpyxl
  (y Gurobi correctamente instalado/licenciado)

Cambios respecto a la versión anterior:
- Arreglado TypeError en Set IDX_NOOV: se eliminó el inicializador como función local sin argumentos
  y se creó el índice con una lista explícita; además, se evitó redefinir el Set dos veces.
- Eliminado warning DEPRECATED de Param 'clientOf' declarando within=Any.
"""

import os
import math
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

from pyomo.environ import (
    ConcreteModel, Set, Param, Var, RangeSet, Constraint, Objective, NonNegativeReals,
    Binary, Reals, Any, minimize, value, SolverFactory
)

# =========================
# CONFIGURACIÓN
# =========================
XLSX  = "input_data.xlsx"
SHEET = "case_261"

# Fecha de inicio de planificación:
#   - None => hoy
#   - "YYYY-MM-DD" => fecha concreta (p.ej. "2025-08-14")
PLANNING_START = None

# Política "un cliente -> pocas posiciones"
#  - "off"  : sin penalización
#  - "soft" : se minimiza el nº de posiciones usadas por cliente
CLIENT_POS_POLICY = "soft"
W_CLIENT_POS = 500.0  # peso de la penalización "soft" (ajústalo si quieres forzar más)

# Objetivo compuesto
W_SUM_END    = 1.0    # peso para sumatorio de tiempos de finalización (flujo total)
W_MAKESPAN   = 0.1    # peso para makespan (Cmax) (opcional)
W_SLACK      = 0.0    # puedes añadir otros términos si quieres (no usados aquí)


# =========================
# LECTURA DE DATOS
# =========================
def read_input(xlsx_path: str, case_sheet: str, planning_start=PLANNING_START):
    """
    Espera una hoja 'case_sheet' con (nombres flexibles):
      plane, job, task, date(=early), duration, client, [late opcional]
    - Dura en días (float)
    - Si no hay 'late', se genera: late = early + duration + gran_holgura
    Además fija BASE_DATE (hoy o la que indiques) para el Gantt/report.
    """
    print(f"Leyendo Excel: {os.path.basename(xlsx_path)} / {case_sheet}")
    try:
        df = pd.read_excel(xlsx_path, sheet_name=case_sheet)
    except Exception as e:
        raise ValueError(f"No encuentro la hoja '{case_sheet}': {e}")

    # Normalizar nombres
    rename_map = {
        'plane': 'plane', 'job': 'job', 'task': 'task',
        'date': 'early', 'duration': 'duration', 'client': 'client',
        'late': 'late'
    }
    df = df.rename(columns={c: rename_map.get(c, c) for c in df.columns})

    # Filtrar filas con plane y duration válidos
    df = df[pd.to_numeric(df['plane'], errors='coerce').notna()].copy()
    df = df[pd.to_numeric(df['duration'], errors='coerce').notna()].copy()
    df['plane']    = df['plane'].astype(int)
    df['duration'] = df['duration'].astype(float)

    if 'client' not in df.columns:
        df['client'] = 'NA'
    if 'early' not in df.columns:
        raise ValueError("Falta columna 'date'/'early' en la hoja.")

    df['early'] = pd.to_numeric(df['early'], errors='coerce').fillna(0).astype(float)

    # Si no hay 'job', genera uno por avión: "<plane>-1"
    if 'job' not in df.columns or df['job'].isna().all():
        df['job'] = df['plane'].astype(str) + "-1"

    # Si no hay 'late', inventarlo con gran holgura
    if 'late' not in df.columns:
        big_pad = df['duration'].max() * 10 + 100.0
        df['late'] = df['early'] + df['duration'] + big_pad
    else:
        df['late'] = pd.to_numeric(df['late'], errors='coerce')
        # Completar NaN de 'late'
        na_late = df['late'].isna()
        if na_late.any():
            big_pad = df['duration'].max() * 10 + 100.0
            df.loc[na_late, 'late'] = df.loc[na_late, 'early'] + df.loc[na_late, 'duration'] + big_pad

    # Asegurar coherencia ventana
    bad = df['early'] + df['duration'] > df['late']
    if bad.any():
        df.loc[bad, 'late'] = df.loc[bad, 'early'] + df.loc[bad, 'duration'] + 1.0

    df = df.sort_values('early').reset_index(drop=True)

    # Conjuntos y parámetros
    jobs   = df['job'].tolist()
    planes = df['plane'].tolist()
    clients = df['client'].astype(str).tolist()

    ES = dict(zip(df['job'], df['early']))
    D  = dict(zip(df['job'], df['duration']))
    LF = dict(zip(df['job'], df['late']))
    CL = dict(zip(df['job'], df['client'].astype(str)))
    PLANE_OF = dict(zip(df['job'], df['plane']))

    # Posiciones del modelo (siempre usamos estas cinco, aunque no todas se utilicen)
    positions = ['position1', 'position2', 'position3', 'position4', 'position5']

    # Horizonte en días: un poco más allá de max(LF)
    H = float(max(LF.values()) + max(D.values()) + 10.0)

    # BASE_DATE para report/Gantt
    if planning_start is None:
        base_date = pd.Timestamp(date.today())
    else:
        base_date = pd.to_datetime(planning_start)

    print(f"Slots cargados: {len(jobs)}, Ejemplo: {jobs[:3]}")
    print(f"Posiciones cargadas: {len(positions)}, Ejemplo: {positions[:3]}")
    print("Todas las ventanas temporales son consistentes (early/late con holgura).")

    data = {
        'JOBS': jobs,
        'PLANES': sorted(set(planes)),
        'CLIENTS': sorted(set(clients)),
        'POSITIONS': positions,
        'early': ES,
        'dur': D,
        'late': LF,
        'client': CL,       # job -> client
        'plane_of': PLANE_OF,  # job -> plane
        'H': H,
        'BASE_DATE': base_date
    }
    return data


# =========================
# CONSTRUCCIÓN DEL MODELO
# =========================
def build_model(d,
                client_pos_policy: str = CLIENT_POS_POLICY,
                w_client_pos: float = W_CLIENT_POS,
                w_sum_end: float = W_SUM_END,
                w_makespan: float = W_MAKESPAN):
    """
    Construye el Pyomo con:
      - asignación a posición (y[j,p]),
      - tiempos (t_start/t_end),
      - no solape por posición,
      - BLOQUEO integrado para pos4/pos5 en entrada/salida,
      - política cliente->posición (off/soft),
      - objetivo compuesto.
    """
    m = ConcreteModel()

    # Conjuntos
    m.J = Set(initialize=d['JOBS'], ordered=True)
    m.P = Set(initialize=d['POSITIONS'], ordered=True)
    m.C = Set(initialize=d['CLIENTS'], ordered=True)

    # Parámetros de tiempo
    m.ES = Param(m.J, initialize=d['early'], within=Reals)
    m.D  = Param(m.J, initialize=d['dur'], within=Reals)
    m.LF = Param(m.J, initialize=d['late'], within=Reals)
    m.H  = Param(initialize=float(d['H']), within=Reals)

    # Mapeos
    # Declarar dentro de Any para evitar warning (y permitir strings de cliente)
    m.clientOf = Param(m.J, initialize=d['client'], within=Any)

    # Variables de tiempo
    m.t_start = Var(m.J, within=NonNegativeReals, bounds=(0, float(d['H'])))
    m.t_end   = Var(m.J, within=NonNegativeReals, bounds=(0, float(d['H'])))

    # Asignación a posición
    m.y = Var(m.J, m.P, within=Binary)

    # Asignación única
    m.c_assign = Constraint(m.J, rule=lambda mdl, j: sum(mdl.y[j, p] for p in mdl.P) == 1)

    # Duración y ventanas
    m.c_dur    = Constraint(m.J, rule=lambda mdl, j: mdl.t_end[j] == mdl.t_start[j] + mdl.D[j])
    m.c_win_lo = Constraint(m.J, rule=lambda mdl, j: mdl.t_start[j] >= mdl.ES[j])
    m.c_win_hi = Constraint(m.J, rule=lambda mdl, j: mdl.t_end[j]   <= mdl.LF[j])

    # No solape dentro de una posición (disyunción linealizada con w[j,k,p])
    bigM = float(d['H'] + max(d['dur'].values()) + 100.0)
    m.bigM = Param(initialize=bigM)

    # Índice de no-solape: TODAS las parejas ordenadas (j,k) con j != k para cada posición p
    jobs = list(m.J)
    positions = list(m.P)
    idx_noover = [(j, k, p) for p in positions for j in jobs for k in jobs if j != k]

    m.IDX_NOOV = Set(dimen=3, initialize=idx_noover)
    m.w = Var(m.IDX_NOOV, within=Binary)  # w[j,k,p]=1 => j antes que k en p

    def c_noov1(mdl, j, k, p):
        # t_start[k] >= t_end[j] - M * (1 - w + (1 - y[j,p]) + (1 - y[k,p]))
        return mdl.t_start[k] >= mdl.t_end[j] - mdl.bigM * (1 - mdl.w[j, k, p] + (1 - mdl.y[j, p]) + (1 - mdl.y[k, p]))

    def c_noov2(mdl, j, k, p):
        # t_start[j] >= t_end[k] - M * (    w + (1 - y[j,p]) + (1 - y[k,p]))
        return mdl.t_start[j] >= mdl.t_end[k] - mdl.bigM * (    mdl.w[j, k, p] + (1 - mdl.y[j, p]) + (1 - mdl.y[k, p]))

    m.c_noov1 = Constraint(m.IDX_NOOV, rule=c_noov1)
    m.c_noov2 = Constraint(m.IDX_NOOV, rule=c_noov2)

    # ===========
    # BLOQUEO de calle (INTEGRADO)
    # ===========
    # Cadena: qué posiciones “delante” deben estar libres para entrar/salir en p_back
    FRONT_CHAIN = {
        'position4': ['position3'],
        'position5': ['position4', 'position3'],
    }
    # Filtrar por posiciones existentes
    chain = {pb: [pf for pf in pfs if pf in list(m.P)]
             for pb, pfs in FRONT_CHAIN.items()
             if pb in list(m.P)}

    # Si hay bloqueo aplicable:
    if chain:
        block_idx = [(j, k, pb, pf) for pb, pfs in chain.items() for pf in pfs for j in m.J for k in m.J if j != k]
        m.IDX_BLK = Set(dimen=4, initialize=block_idx)
        m.bEntry  = Var(m.IDX_BLK, within=Binary)  # disyunción en ENTRADA
        m.bExit   = Var(m.IDX_BLK, within=Binary)  # disyunción en SALIDA
        eps = 1e-3  # ~86.4 segundos; puedes subirlo si quieres margen mayor

        def c_blk_entry_before(mdl, j, k, pb, pf):
            # Si y[j,pb]=y[k,pf]=1 y bEntry=1 => k termina antes de t_start[j] - eps
            gating = 2 - mdl.y[j, pb] - mdl.y[k, pf]
            return mdl.t_end[k] <= mdl.t_start[j] - eps + mdl.bigM * (1 - mdl.bEntry[j, k, pb, pf] + gating)

        def c_blk_entry_after(mdl, j, k, pb, pf):
            # Si y[j,pb]=y[k,pf]=1 y bEntry=0 => k empieza después de t_start[j] + eps
            gating = 2 - mdl.y[j, pb] - mdl.y[k, pf]
            return mdl.t_start[k] >= mdl.t_start[j] + eps - mdl.bigM * (mdl.bEntry[j, k, pb, pf] + gating)

        def c_blk_exit_before(mdl, j, k, pb, pf):
            # En salida (t_end[j]), misma disyunción
            gating = 2 - mdl.y[j, pb] - mdl.y[k, pf]
            return mdl.t_end[k] <= mdl.t_end[j] - eps + mdl.bigM * (1 - mdl.bExit[j, k, pb, pf] + gating)

        def c_blk_exit_after(mdl, j, k, pb, pf):
            gating = 2 - mdl.y[j, pb] - mdl.y[k, pf]
            return mdl.t_start[k] >= mdl.t_end[j] + eps - mdl.bigM * (mdl.bExit[j, k, pb, pf] + gating)

        m.c_blk_entry_before = Constraint(m.IDX_BLK, rule=c_blk_entry_before)
        m.c_blk_entry_after  = Constraint(m.IDX_BLK, rule=c_blk_entry_after)
        m.c_blk_exit_before  = Constraint(m.IDX_BLK, rule=c_blk_exit_before)
        m.c_blk_exit_after   = Constraint(m.IDX_BLK, rule=c_blk_exit_after)

    # ===========
    # Política cliente->posición (soft/off)
    # ===========
    if client_pos_policy.lower() == "soft":
        # z[c,p] = 1 si algún trabajo del cliente c usa la posición p
        m.z = Var(m.C, m.P, within=Binary)

        # z[c,p] >= y[j,p]  para todo job j de cliente c
        cp_idx = [(j, d['client'][j], p) for j in m.J for p in m.P]
        m.IDX_CP = Set(dimen=3, initialize=cp_idx)

        def c_z_cover(mdl, j, c, p):
            return mdl.z[c, p] >= mdl.y[j, p]
        m.c_z_cover = Constraint(m.IDX_CP, rule=c_z_cover)

        # Queremos minimizar el nº de posiciones por cliente => sum z[c,p]
        z_term = sum(m.z[c, p] for c in m.C for p in m.P)
    else:
        z_term = 0.0

    # ===========
    # Makespan opcional: Cmax >= t_end[j]
    # ===========
    if w_makespan > 0:
        m.Cmax = Var(within=NonNegativeReals, bounds=(0, float(d['H'])))
        m.c_cmax = Constraint(m.J, rule=lambda mdl, j: mdl.Cmax >= mdl.t_end[j])
        makespan_term = m.Cmax
    else:
        makespan_term = 0.0

    # ===========
    # Objetivo
    # ===========
    obj = (
        w_sum_end  * sum(m.t_end[j] for j in m.J) +
        (w_client_pos * z_term if client_pos_policy.lower() == "soft" else 0.0) +
        w_makespan * makespan_term
    )
    m.OBJ = Objective(expr=obj, sense=minimize)

    return m


# =========================
# SOLUCIÓN Y REPORTE
# =========================
def solve_and_report(m, d, timelimit=1500, mipgap=0.05, out_html="schedule_enhanced.html"):
    print("\nIniciando resolución con Gurobi...\n")
    opt = SolverFactory('gurobi')
    if opt is None:
        raise RuntimeError("No se encontró el solver 'gurobi'.")

    # Parámetros del solver (ajusta a tu gusto)
    opt.options['OutputFlag'] = 1
    opt.options['TimeLimit'] = timelimit
    opt.options['MIPGap']    = mipgap
    opt.options['MIPFocus']  = 3
    opt.options['Heuristics']= 1
    opt.options['RINS']      = 10
    opt.options['Cuts']      = 2
    opt.options['Presolve']  = 2
    opt.options['VarBranch'] = 1
    opt.options['BranchDir'] = -1
    opt.options['MinRelNodes'] = 1000
    opt.options['ImproveStartGap'] = 0.5
    opt.options['NoRelHeurTime'] = 60

    results = opt.solve(m, tee=True)

    # Extraer solución
    rows = []
    for j in m.J:
        t0 = float(value(m.t_start[j])); t1 = float(value(m.t_end[j]))
        pos = None
        for p in m.P:
            if value(m.y[j, p]) > 0.5:
                pos = p
                break
        rows.append((j, pos, t0, t1, d['client'][j]))
    df_pos = pd.DataFrame(rows, columns=['job', 'position', 'start', 'finish', 'client'])

    # Validación posicional: sin solapes por posición
    ok = True
    for p, g in df_pos.groupby('position'):
        g = g.sort_values('start')
        last_end = -1e9
        for _, r in g.iterrows():
            if r['start'] < last_end - 1e-9:
                ok = False
                print(f"❌ Solape en {p}: {r['job']}")
            last_end = max(last_end, r['finish'])
    if ok:
        print("✅ Sanity check posiciones: sin solapes.")

    # Guardar CSVs básicos
    # (por compatibilidad con tu flujo previo)
    df_pos.to_csv("solution_by_position.csv", index=False)
    df_jobs = df_pos[['job', 'position', 'start', 'finish']]
    df_jobs.to_csv("solution_jobs.csv", index=False)

    # KPI simple
    kpis = {
        'sum_finish': df_pos['finish'].sum(),
        'makespan': df_pos['finish'].max(),
        'n_jobs': len(df_pos)
    }
    pd.DataFrame([kpis]).to_csv("solution_kpis.csv", index=False)
    print("CSV exportados: solution_by_position.csv, solution_jobs.csv, solution_kpis.csv")

    # Plot Gantt
    try:
        plot_gantt_by_position(df_pos, d['BASE_DATE'], out_html=out_html)
        print(f"Plot HTML exportado: {out_html}")
    except Exception as e:
        print(f"No se pudo generar el Gantt HTML: {e}")

    # Report & Validación de BLOQUEO
    generate_report(df_pos, d)
    validate_blocking(df_pos)

    return df_pos


# =========================
# PLOT GANTT
# =========================
import plotly.express as px
import plotly.io as pio

def plot_gantt_by_position(df_pos: pd.DataFrame, base_date: pd.Timestamp, out_html="schedule_enhanced.html"):
    """
    Convierte start/finish (días) en timestamps y dibuja Gantt por posición.
    Colorea por cliente (discreto).
    """
    df = df_pos.copy()
    # días -> timedelta -> fecha
    df['start_dt']  = pd.to_timedelta(df['start'],  unit='D') + base_date
    df['finish_dt'] = pd.to_timedelta(df['finish'], unit='D') + base_date

    # Orden por posición
    df['position'] = pd.Categorical(df['position'], categories=sorted(df['position'].unique()), ordered=True)
    df = df.sort_values(['position', 'start_dt'])

    # Paleta discreta por cliente
    palette = px.colors.qualitative.Dark24 + px.colors.qualitative.Set3 + px.colors.qualitative.Safe
    clients = sorted(df['client'].unique())
    color_map = {c: palette[i % len(palette)] for i, c in enumerate(clients)}

    fig = px.timeline(
        df,
        x_start='start_dt', x_end='finish_dt',
        y='position',
        color='client',
        hover_data=['job', 'client', 'position'],
        color_discrete_map=color_map
    )
    fig.update_layout(
        title="Schedule por posición",
        xaxis_title="Fecha",
        yaxis_title="Posición",
        legend_title="Cliente",
        height=500 + 20 * df['position'].nunique()
    )
    fig.update_yaxes(autorange="reversed")  # estilo Gantt

    pio.write_html(fig, file=out_html, auto_open=False, include_plotlyjs="cdn")


# =========================
# REPORT Y VALIDACIÓN
# =========================
def generate_report(df_pos: pd.DataFrame, d: dict):
    """
    Resumen por avión (1 job/avión) y detalle por trabajo, con fechas reales.
    """
    base_date = d['BASE_DATE']
    df = df_pos.copy()

    # Añadir fechas reales
    df['start_dt']  = pd.to_timedelta(df['start'],  unit='D') + base_date
    df['finish_dt'] = pd.to_timedelta(df['finish'], unit='D') + base_date

    # Mapeos auxiliares
    ES = d['early']; D = d['dur']; LF = d['late']; CL = d['client']
    plane_of = d['plane_of']

    # Resumen por "avión" (aquí hay 1 job por avión)
    resumen = []
    for _, r in df.iterrows():
        avion  = plane_of[r['job']]
        cliente = CL[r['job']]
        es  = ES[r['job']]
        lf  = LF[r['job']]
        est = D[r['job']]
        resumen.append({
            'Avión': avion,
            'Cliente': cliente,
            'ES': (base_date + pd.to_timedelta(es, unit='D')).date(),
            'Primer Inicio': r['start_dt'].date(),
            'LF': (base_date + pd.to_timedelta(lf, unit='D')).date(),
            'Fin': r['finish_dt'].date(),
            'Trabajos': r['job'],
            'Posiciones': r['position'],
            'Movimientos': 2  # 1 entrada + 1 salida (con este modelo simple)
        })
    df_res = pd.DataFrame(resumen)
    print("\n" + "=" * 150)
    print("RESUMEN POR AVIÓN")
    print("=" * 150)
    if not df_res.empty:
        print(df_res.to_string(index=False, col_space=15))

    # Detalle de todos los trabajos
    det = []
    for _, r in df.iterrows():
        j = r['job']
        det.append({
            '⚠': '✅',
            'Avión': plane_of[j],
            'Trabajo': j,
            'Posición': r['position'],
            'Fecha ES': (base_date + pd.to_timedelta(ES[j], unit='D')).date(),
            'Prevista': (base_date + pd.to_timedelta(ES[j] + D[j], unit='D')).date(),
            'Real': r['finish_dt'].date(),
            'Fecha LF': (base_date + pd.to_timedelta(LF[j], unit='D')).date(),
            'Dur Est.(d)': D[j],
            'Dur Real(d)': (r['finish'] - r['start']),
            'Retraso(d)': max(0, (r['finish'] - LF[j]))
        })
    df_det = pd.DataFrame(det)
    print("\n" + "=" * 170)
    print("DETALLE DE TODOS LOS TRABAJOS")
    print("=" * 170)
    if not df_det.empty:
        print(df_det.to_string(index=False, col_space=15))


def validate_blocking(df_pos: pd.DataFrame):
    """
    Valida BLOQUEO: en entrada/salida a posición4/5 no debe haber nadie en
    las posiciones "delante" en el instante de start/end.
    """
    chain = {
        'position4': ['position3'],
        'position5': ['position4', 'position3'],
    }
    df = df_pos.copy()
    issues = []

    # por posición, construir intervalos
    pos_intervals = {
        p: df[df['position'] == p][['job', 'start', 'finish']].to_records(index=False)
        for p in df['position'].unique()
    }

    for _, r in df.iterrows():
        pb = r['position']
        if pb not in chain:
            continue
        fronts = chain[pb]
        t_in = r['start']; t_out = r['finish']
        for pf in fronts:
            if pf not in pos_intervals:  # puede no haberse usado
                continue
            for (jobk, sk, fk) in pos_intervals[pf]:
                # ENTRADA: t_in no debe caer dentro [sk, fk]
                if sk - 1e-9 <= t_in <= fk + 1e-9:
                    issues.append(f"Bloqueo ENTRADA: {r['job']}@{pb} colisiona con {jobk}@{pf}")
                # SALIDA: t_out no debe caer dentro [sk, fk]
                if sk - 1e-9 <= t_out <= fk + 1e-9:
                    issues.append(f"Bloqueo SALIDA: {r['job']}@{pb} colisiona con {jobk}@{pf}")

    print("\n=== VALIDACIÓN POST-SOLUCIÓN ===")
    if issues:
        for s in sorted(set(issues)):
            print("❌ " + s)
        print("⚠️  Validación con incidencias")
    else:
        print("✅ Validación OK")


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    d = read_input(XLSX, SHEET, planning_start=PLANNING_START)
    model = build_model(
        d,
        client_pos_policy=CLIENT_POS_POLICY,
        w_client_pos=W_CLIENT_POS,
        w_sum_end=W_SUM_END,
        w_makespan=W_MAKESPAN
    )
    solve_and_report(model, d, timelimit=1500, mipgap=0.05, out_html="schedule_enhanced.html")
