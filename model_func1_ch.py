# -*- coding: utf-8 -*-
"""
model_func1_sh_meta_alns.py — Modelo continuo (Pyomo) + BLOQUEO E/S + política cliente→posición +
                               retraso avión→cliente + reporte largo + Gantt ordenado
                               + METAHEURÍSTICA **ALNS+SA+PR** para warm-start

Basado en la versión v2, incorpora una metaheurística más fuerte inspirada en la
literatura (ALNS con destrucción/reparación adaptativa, *simulated annealing* y
*path relinking* ligero) para obtener mejores soluciones iniciales, priorizando
"una posición por cliente" sin sacrificar factibilidad.

Requisitos: pyomo, gurobi, pandas, plotly
Entrada: input_data.xlsx / case_XXX
"""

import os, math, random
from collections import defaultdict, Counter
from datetime import date
import pandas as pd

from pyomo.environ import (
    ConcreteModel, Set, Param, Var, NonNegativeReals, Binary, Any,
    Objective, Constraint, ConstraintList, minimize, value
)
from pyomo.opt import SolverFactory, TerminationCondition

import plotly.express as px
import plotly.io as pio

# =============================
# CONFIGURACIÓN / CONSTANTES
# =============================
NO_POSITIONS = 5
POSITIONS = [f"position{i}" for i in range(1, NO_POSITIONS+1)]

BASE_INTERF = [("position3","position5"), ("position4","position5")]  # front→back "base"
INTERF_IN  = list(set(BASE_INTERF + [(b,a) for (a,b) in BASE_INTERF]))
INTERF_OUT = list(set(BASE_INTERF + [(b,a) for (a,b) in BASE_INTERF]))

# separaciones mínimas (días)
SEP_IN  = 8e-2
SEP_OUT = 8e-2

# pesos objetivo (MILP)
W_MAKESPAN     = 1.0
W_CLIENT_DELAY = 1.0
W_CLIENT_POS   = 500000.0  # penalización por usar >1 posición/cliente (soft)
W_ASSIGN       = 0.0

# Política cliente→posición
CLIENT_ONE_POS_HARD = False
TWO_PHASE_SOLVE      = False

# solver
TIME_LIMIT = 1500
MIP_GAP    = 0.05

# === METAHEURÍSTICA (ALNS+SA+PR) ===
USE_METAHEURISTICS    = True
ALNS_ITERATIONS       = 2000
ALNS_TIME_LIMIT_S     = None  # segundos; si None, no se usa
ALNS_SEED             = 12345
ALNS_K_REMOVE_BASE    = 3      # empleos a eliminar por iteración (ajustable por tamaño)
ALNS_K_REMOVE_VAR     = 2
ALNS_ELITE_SIZE       = 5
ALNS_REACTION         = 0.2    # reacción de pesos adaptativos
SA_INIT_ACCEPT_RATE   = 0.8
SA_COOLING            = 0.995
SA_MIN_T              = 1e-4

# penalizaciones proxy (warm-start)
W_PROXY_TARD     = 10_000.0
W_PROXY_GAPS     = 1.0
W_PROXY_SWITCH   = 2_000.0  # 🔥 fuerte: desalienta múltiples posiciones por cliente
W_PROXY_INTERF   = 1_000.0  # penaliza cercanía IN/OUT en posiciones en conflicto
W_PROXY_EARLY    = 0.0      # (opcional) penalizar empezar demasiado temprano

PLANNING_START = "2024-11-17"  # o None → hoy

# =============================
# UTILIDADES COMUNES (interferencias / inserciones)
# =============================

def _get_interf_sets(d):
    inter_in = set(); inter_out = set()
    for a, b in d.get('INTERF_IN', []):
        inter_in.add((a,b)); inter_in.add((b,a))
    for a, b in d.get('INTERF_OUT', []):
        inter_out.add((a,b)); inter_out.add((b,a))
    return inter_in, inter_out


def _fits_without_overlap(t, D, schedule_p, eps):
    for a, b, _ in schedule_p:
        if t + D <= a - eps:
            return True
        if b + eps <= t:
            continue
        return False
    return True


def _embed_earliest_gap(ES, D, LF, schedule_p, eps):
    t = ES
    for a, b, _ in schedule_p:
        if t + D <= a - eps:
            break
        if t < b + eps:
            t = b + eps
    if t + D <= LF + 1e-9:
        return t
    return None


def _respect_interference(t, D, p, starts_by_pos, ends_by_pos, inter_in, inter_out, eps, max_bumps=200):
    bumps = 0
    while bumps < max_bumps:
        ok = True
        for q in starts_by_pos.keys():
            if (p,q) in inter_in:
                for s in starts_by_pos[q]:
                    if abs(t - s) < eps:
                        t = s + eps; ok = False; break
            if not ok: break
        if not ok:
            bumps += 1; continue
        for q in ends_by_pos.keys():
            if (p,q) in inter_out:
                for e in ends_by_pos[q]:
                    if abs((t + D) - e) < eps:
                        t = e + eps - D; ok = False; break
            if not ok: break
        if ok: return t
        bumps += 1
    return None

# =============================
# COSTE PROXY Y MÉTRICAS
# =============================

def _client_switches(plan, d):
    c2pos = defaultdict(set)
    for j,p in plan['pos'].items():
        c2pos[d['client'][j]].add(p)
    switches = 0
    for c, poses in c2pos.items():
        if len(poses) > 1:
            switches += (len(poses) - 1)
    return switches


def _proxy_cost(plan, d, eps):
    tard = 0.0; gaps = 0.0; interf = 0.0; early_pen = 0.0
    ES, D, LF = d['early'], d['dur'], d['late']
    for j, t0 in plan['t_start'].items():
        t1 = plan['t_end'][j]
        if t1 > LF[j] + 1e-9: tard += (t1 - LF[j])
        if t0 < ES[j] - 1e-9: early_pen += (ES[j]-t0)
    for p in d['POSITIONS']:
        blocks = sorted([(plan['t_start'][j], plan['t_end'][j], j)
                         for j in plan['pos'] if plan['pos'][j]==p])
        for i in range(1,len(blocks)):
            prev_end = blocks[i-1][1]; cur_st = blocks[i][0]
            if cur_st - prev_end > eps:
                gaps += (cur_st - prev_end)
    inter_in, inter_out = _get_interf_sets(d)
    starts_by_pos = defaultdict(list); ends_by_pos = defaultdict(list)
    for j,p in plan['pos'].items():
        starts_by_pos[p].append(plan['t_start'][j]); ends_by_pos[p].append(plan['t_end'][j])
    for (a,b) in inter_in:
        for s in starts_by_pos[a]:
            if any(abs(s - s2) < SEP_IN - 1e-9 for s2 in starts_by_pos[b]):
                interf += 1
    for (a,b) in inter_out:
        for e in ends_by_pos[a]:
            if any(abs(e - e2) < SEP_OUT - 1e-9 for e2 in ends_by_pos[b]):
                interf += 1
    switches = _client_switches(plan,d)
    return (W_PROXY_TARD*tard + W_PROXY_GAPS*gaps + W_PROXY_SWITCH*switches + W_PROXY_INTERF*interf + W_PROXY_EARLY*early_pen,
            tard, switches, interf)

# =============================
# OPERADORES DESTRUCCIÓN / REPARACIÓN (ALNS)
# =============================

def _remove_jobs(plan, d, rng, k):
    J = list(d['JOBS'])
    to_remove = rng.sample(J, min(k, len(J)))
    return to_remove


def _remove_client_block(plan, d, rng):
    c = rng.choice(list({d['client'][j] for j in d['JOBS']}))
    jobs = [j for j in d['JOBS'] if d['client'][j]==c]
    main_pos = Counter(plan['pos'][j] for j in jobs).most_common(1)[0][0]
    # elimina de ese cliente los jobs fuera de su posición mayoritaria
    to_remove = [j for j in jobs if plan['pos'][j]!=main_pos]
    if not to_remove:  # si ya es consistente, al menos saca 1 job aleatorio del cliente
        to_remove = [rng.choice(jobs)]
    return to_remove


def _remove_conflict_neighborhood(plan, d, rng, eps):
    inter_in, inter_out = _get_interf_sets(d)
    starts_by_pos = defaultdict(list); ends_by_pos = defaultdict(list)
    for j,p in plan['pos'].items():
        starts_by_pos[p].append((plan['t_start'][j], j))
        ends_by_pos[p].append((plan['t_end'][j], j))
    # encuentra un conflicto y elimina a los implicados
    cand = []
    for (a,b) in inter_in:
        for tA,jA in starts_by_pos[a]:
            for tB,jB in starts_by_pos[b]:
                if abs(tA - tB) < SEP_IN - 1e-9:
                    cand.extend([jA,jB])
    for (a,b) in inter_out:
        for tA,jA in ends_by_pos[a]:
            for tB,jB in ends_by_pos[b]:
                if abs(tA - tB) < SEP_OUT - 1e-9:
                    cand.extend([jA,jB])
    if not cand:
        return _remove_jobs(plan, d, rng, k=2)
    return list(set(cand))


def _repair_regret(plan, d, removed, rng, eps):
    # Inserción por regret-2: para cada job, evaluar 2 mejores posiciones y elegir el de mayor "regret"
    ES, D, LF = d['early'], d['dur'], d['late']
    P = d['POSITIONS']
    inter_in, inter_out = _get_interf_sets(d)
    # construir estructuras
    schedule = {p: sorted([(plan['t_start'][j], plan['t_end'][j], j)
                           for j in plan['pos'] if plan['pos'][j]==p], key=lambda x:x[0]) for p in P}
    starts_by_pos = {p: [s for (s,_,_) in schedule[p]] for p in P}
    ends_by_pos   = {p: [e for (_,e,_) in schedule[p]] for p in P}

    for j in removed:
        cand_list = []
        for p in P:
            t0 = _embed_earliest_gap(ES[j], D[j], LF[j], schedule[p], eps)
            if t0 is None: continue
            t_adj = _respect_interference(t0, D[j], p, starts_by_pos, ends_by_pos, inter_in, inter_out, eps)
            if t_adj is None: continue
            if not _fits_without_overlap(t_adj, D[j], schedule[p], eps):
                continue
            tard_here = max(0.0, (t_adj + D[j]) - LF[j])
            cand_list.append((tard_here, t_adj, p))
        if not cand_list:
            # si no hay hueco respetando LF, permitir atraso mínimo
            for p in P:
                t0 = schedule[p][-1][1] + eps if schedule[p] else ES[j]
                t_adj = _respect_interference(t0, D[j], p, starts_by_pos, ends_by_pos, inter_in, inter_out, eps)
                if t_adj is None: continue
                cand_list.append((max(0.0, (t_adj+D[j])-LF[j]), t_adj, p))
        cand_list.sort(key=lambda x:(x[0], x[1]+D[j]))
        if not cand_list:
            # inserción forzada: ponerlo al final de una posición aleatoria
            p = rng.choice(P); t_adj = schedule[p][-1][1] + eps if schedule[p] else ES[j]
            choice = (max(0.0,(t_adj+D[j])-LF[j]), t_adj, p)
        else:
            if len(cand_list)==1:
                choice = cand_list[0]
            else:
                regret = cand_list[1][0] - cand_list[0][0]
                # poco uso del regret numérico aquí; nos quedamos con mejor opción (tardanza mínima)
                choice = cand_list[0]
        tard, t, p = choice
        plan['pos'][j] = p; plan['t_start'][j] = t; plan['t_end'][j] = t + D[j]
        schedule[p].append((t,t+D[j],j)); schedule[p].sort(key=lambda x:x[0])
        starts_by_pos[p] = [s for (s,_,_) in schedule[p]]
        ends_by_pos[p]   = [e for (_,e,_) in schedule[p]]
    return plan

# =============================
# ALNS + SA + PR
# =============================

def _deepcopy_plan(plan):
    return {k: v.copy() for k,v in plan.items()}


def _path_relink(current, elite, d, eps, rng):
    # Relaja hacia la asignación de posiciones del elite (sin forzar tiempos)
    curr = _deepcopy_plan(current)
    target_pos = elite['pos']
    diff = [j for j in d['JOBS'] if curr['pos'][j] != target_pos[j]]
    rng.shuffle(diff)
    # aplicar cambios en pequeños lotes y reparar
    while diff:
        step = diff[:max(1,len(diff)//3)]; diff = diff[len(step):]
        for j in step:
            curr['pos'][j] = target_pos[j]
        curr = _repair_regret(curr, d, removed=[], rng=rng, eps=eps)  # reparación trivial para ajustar tiempos
    return curr


def _initial_construction(d, rng, eps):
    # earliest-gap por cliente, priorizando posición dominante del cliente
    ES, D, LF = d['early'], d['dur'], d['late']
    P = d['POSITIONS']
    inter_in, inter_out = _get_interf_sets(d)

    jobs = list(d['JOBS'])
    jobs.sort(key=lambda j:(LF[j], ES[j] + rng.random()*0.1))  # EDD con ruido

    plan = {'pos':{}, 't_start':{}, 't_end':{}}
    schedule = {p: [] for p in P}
    starts_by_pos = {p: [] for p in P}
    ends_by_pos   = {p: [] for p in P}
    c_pref = {}

    for j in jobs:
        c = d['client'][j]
        pos_order = P[:]
        if c in c_pref:
            pref = c_pref[c]; pos_order.remove(pref); pos_order = [pref] + pos_order
        best = None
        for p in pos_order:
            t0 = _embed_earliest_gap(ES[j], D[j], LF[j], schedule[p], eps)
            if t0 is None: continue
            t_adj = _respect_interference(t0, D[j], p, starts_by_pos, ends_by_pos, inter_in, inter_out, eps)
            if t_adj is None: continue
            if not _fits_without_overlap(t_adj, D[j], schedule[p], eps):
                continue
            cand = (max(0.0,(t_adj+D[j])-LF[j]), t_adj, p)
            if (best is None) or cand < best:
                best = cand
        if best is None:
            p = rng.choice(P)
            t_adj = schedule[p][-1][1] + eps if schedule[p] else ES[j]
        else:
            _, t_adj, p = best
        plan['pos'][j]=p; plan['t_start'][j]=t_adj; plan['t_end'][j]=t_adj+D[j]
        schedule[p].append((t_adj,t_adj+D[j],j)); schedule[p].sort(key=lambda x:x[0])
        starts_by_pos[p] = [s for (s,_,_) in schedule[p]]; ends_by_pos[p] = [e for (_,e,_) in schedule[p]]
        # actualizar preferencia del cliente
        counts = Counter(plan['pos'][jj] for jj in plan['pos'] if d['client'][jj]==c)
        c_pref[c] = counts.most_common(1)[0][0]
    return plan


def run_alns(d, eps=1e-3, iters=ALNS_ITERATIONS, seed=ALNS_SEED, time_limit_s=ALNS_TIME_LIMIT_S):
    rng = random.Random(seed)
    plan = _initial_construction(d, rng, eps)
    best = _deepcopy_plan(plan); best_cost, best_tard, _, _ = _proxy_cost(best, d, eps)

    # SA temperature from initial acceptance rate heuristic
    neigh = _remove_jobs(plan,d,rng,3)
    tmp = _repair_regret(_deepcopy_plan(plan), d, neigh, rng, eps)
    c0,_t,_,_ = _proxy_cost(plan, d, eps)
    c1,_t,_,_ = _proxy_cost(tmp, d, eps)
    delta = abs(c1-c0) if abs(c1-c0)>1e-9 else 1.0
    T = -delta / math.log(SA_INIT_ACCEPT_RATE)

    destroy_ops = ['rand','client','conflict']
    scores = {op:1.0 for op in destroy_ops}
    weights = {op:1.0 for op in destroy_ops}

    elite = []  # lista de (plan,cost)

    import time
    t_start = time.time()
    iter_ = 0
    while iter_ < iters:
        if time_limit_s and (time.time()-t_start > time_limit_s):
            break
        iter_ += 1
        # selección adaptativa de operador
        total_w = sum(weights.values()); pick = rng.random()*total_w
        acc=0; op_sel='rand'
        for op,w in weights.items():
            acc+=w
            if pick<=acc: op_sel=op; break

        k = ALNS_K_REMOVE_BASE + rng.randint(0,ALNS_K_REMOVE_VAR)
        if op_sel=='rand':
            removed = _remove_jobs(plan, d, rng, k)
        elif op_sel=='client':
            removed = _remove_client_block(plan, d, rng)
        else:
            removed = _remove_conflict_neighborhood(plan, d, rng, eps)

        cand = _deepcopy_plan(plan)
        for j in removed:
            # vaciar j
            if j in cand['pos']:
                del cand['pos'][j]; del cand['t_start'][j]; del cand['t_end'][j]
        cand = _repair_regret(cand, d, removed, rng, eps)
        c_new, tard_new, sw_new, interf_new = _proxy_cost(cand, d, eps)
        c_old, _, _, _ = _proxy_cost(plan, d, eps)
        delta = c_new - c_old
        accept = delta <= 0 or rng.random() < math.exp(-delta/max(T,SA_MIN_T))

        if accept:
            plan = cand
            # actualizar scores
            gain = max(0.0, (c_old - c_new))
            scores[op_sel] = (1-ALNS_REACTION)*scores[op_sel] + ALNS_REACTION*(1 + gain/(1+abs(c_old)))
            # actualizar pesos normalizados
            base = sum(scores.values()); weights = {op: max(0.05, s/base) for op,s in scores.items()}
            # elite
            if (len(elite)<ALNS_ELITE_SIZE) or (c_new < max(cost for _,cost in elite)):
                elite.append(( _deepcopy_plan(plan), c_new))
                elite = sorted(elite, key=lambda x:x[1])[:ALNS_ELITE_SIZE]
                # path-relinking hacia el mejor
                if elite:
                    plan_pr = _path_relink(plan, elite[0][0], d, eps, rng)
                    c_pr,_,_,_ = _proxy_cost(plan_pr, d, eps)
                    if c_pr < c_new:
                        plan = plan_pr; c_new = c_pr
        # enfriamiento
        T = max(SA_MIN_T, T*SA_COOLING)
        # actualizar best
        if c_new < best_cost - 1e-9:
            best = _deepcopy_plan(plan); best_cost = c_new; best_tard = tard_new
    return best, best_cost, best_tard

# =============================
# PYOMO: build_model idéntico a v2 (con hard/soft cliente→posición)
# =============================

def build_model(d, client_pos_policy='soft', w_client_pos=W_CLIENT_POS, w_makespan=W_MAKESPAN, w_client_delay=W_CLIENT_DELAY, w_assign=W_ASSIGN):
    m = ConcreteModel()

    J = sorted(d['JOBS']); P = list(d['POSITIONS'])
    C = sorted(set(d['client'][j] for j in J))
    R = sorted(set(d['plane_of'][j] for j in J))

    m.J = Set(initialize=J, ordered=True)
    m.P = Set(initialize=P, ordered=True)
    m.C = Set(initialize=C, ordered=True)
    m.R = Set(initialize=R, ordered=True)

    m.ES = Param(m.J, initialize=d['early'])
    m.D  = Param(m.J, initialize=d['dur'])
    m.LF = Param(m.J, initialize=d['late'])
    H    = float(d['H'])
    m.H  = Param(initialize=H)
    m.clientOf = Param(m.J, initialize=d['client'], within=Any)
    m.planeOf  = Param(m.J, initialize=d['plane_of'])

    plane_LF = {}
    for r in R:
        jobs_r = [j for j in J if d['plane_of'][j]==r]
        if d.get('plane_late') and r in d['plane_late']:
            plane_LF[r] = float(d['plane_late'][r])
        else:
            plane_LF[r] = max(d['late'][j] for j in jobs_r) if jobs_r else 0.0
    m.LF_plane = Param(m.R, initialize=plane_LF)

    AOC = {}
    for r in R:
        clients_r = {d['client'][j] for j in J if d['plane_of'][j]==r}
        for c in C:
            AOC[(c,r)] = 1 if c in clients_r else 0
    m.AOC = Param(m.C, m.R, initialize=AOC)

    m.E_in  = Set(initialize=list(set(d.get('INTERF_IN', []))))
    m.E_out = Set(initialize=list(set(d.get('INTERF_OUT', []))))
    m.epsIN  = Param(initialize=SEP_IN)
    m.epsOUT = Param(initialize=SEP_OUT)

    bigM = H + max(float(d['dur'][j]) for j in J) + 10.0
    m.bigM = Param(initialize=bigM)

    m.t_start = Var(m.J, within=NonNegativeReals, bounds=lambda mdl,j:(0,float(mdl.H)))
    m.t_end   = Var(m.J, within=NonNegativeReals, bounds=lambda mdl,j:(0,float(mdl.H)))
    m.y       = Var(m.J, m.P, within=Binary)

    m.z = Var(m.J, m.J, m.P, within=Binary)

    if len(list(m.E_in))>0:
        m.bIN  = Var(m.J, m.J, m.P, m.P, within=Binary)
    if len(list(m.E_out))>0:
        m.bOUT = Var(m.J, m.J, m.P, m.P, within=Binary)

    m.T_max = Var(within=NonNegativeReals, bounds=(0,float(m.H)))

    m.vPlaneDelay = Var(m.R, within=NonNegativeReals)
    m.vClientDelay = Var(m.C, within=NonNegativeReals)

    if client_pos_policy == 'soft' or CLIENT_ONE_POS_HARD:
        m.x = Var(m.C, m.P, within=Binary)
        if not CLIENT_ONE_POS_HARD:
            m.vExtraPos = Var(m.C, within=NonNegativeReals)

    m.c_assign = Constraint(m.J, rule=lambda mdl,j: sum(mdl.y[j,p] for p in mdl.P)==1)
    m.c_dur    = Constraint(m.J, rule=lambda mdl,j: mdl.t_end[j]==mdl.t_start[j]+mdl.D[j])
    m.c_win_lo = Constraint(m.J, rule=lambda mdl,j: mdl.t_start[j] >= mdl.ES[j])
    m.c_win_hi = Constraint(m.J, rule=lambda mdl,j: mdl.t_end[j]   <= mdl.LF[j])

    def _c_z_le_yj(mdl,j,k,p):
        if j==k: return Constraint.Skip
        return mdl.z[j,k,p] <= mdl.y[j,p]
    def _c_z_le_yk(mdl,j,k,p):
        if j==k: return Constraint.Skip
        return mdl.z[j,k,p] <= mdl.y[k,p]
    def _c_z_mutual(mdl,j,k,p):
        if j==k: return Constraint.Skip
        return mdl.z[j,k,p] + mdl.z[k,j,p] <= 1
    def _c_z_cover(mdl,j,k,p):
        if j==k: return Constraint.Skip
        return mdl.z[j,k,p] + mdl.z[k,j,p] >= mdl.y[j,p] + mdl.y[k,p] - 1
    m.c_z_le_yj = Constraint(m.J, m.J, m.P, rule=_c_z_le_yj)
    m.c_z_le_yk = Constraint(m.J, m.J, m.P, rule=_c_z_le_yk)
    m.c_z_mutual = Constraint(m.J, m.J, m.P, rule=_c_z_mutual)
    m.c_z_cover  = Constraint(m.J, m.J, m.P, rule=_c_z_cover)

    def _c_prec_time1(mdl,j,k,p):
        if j==k: return Constraint.Skip
        return mdl.t_start[k] >= mdl.t_end[j] - mdl.bigM*(1 - mdl.z[j,k,p])
    def _c_prec_time2(mdl,j,k,p):
        if j==k: return Constraint.Skip
        return mdl.t_start[j] >= mdl.t_end[k] - mdl.bigM*(1 - mdl.z[k,j,p])
    m.c_prec_time1 = Constraint(m.J, m.J, m.P, rule=_c_prec_time1)
    m.c_prec_time2 = Constraint(m.J, m.J, m.P, rule=_c_prec_time2)

    for j in J:
        m.add_component(f"c_Tmax_{j}", Constraint(expr=m.T_max >= m.t_end[j]))

    m.c_plane_delay_lb = ConstraintList()
    for r in R:
        for j in J:
            if d['plane_of'][j]==r:
                m.c_plane_delay_lb.add(m.vPlaneDelay[r] >= m.t_end[j] - m.LF_plane[r])
    m.c_client_delay = Constraint(m.C, rule=lambda mdl,c: mdl.vClientDelay[c] == sum(mdl.AOC[c,r]*mdl.vPlaneDelay[r] for r in mdl.R))

    if len(list(m.E_in))>0:
        def _c_block_in_before(mdl, j,k,p_b,p_f):
            if (p_f,p_b) not in mdl.E_in or j==k: return Constraint.Skip
            return mdl.t_end[k] <= mdl.t_start[j] - mdl.epsIN + mdl.bigM*(1 - mdl.bIN[j,k,p_b,p_f] + 2 - mdl.y[j,p_b] - mdl.y[k,p_f])
        def _c_block_in_after(mdl, j,k,p_b,p_f):
            if (p_f,p_b) not in mdl.E_in or j==k: return Constraint.Skip
            return mdl.t_start[k] >= mdl.t_start[j] + mdl.epsIN - mdl.bigM*(    mdl.bIN[j,k,p_b,p_f] + 2 - mdl.y[j,p_b] - mdl.y[k,p_f])
        m.c_block_in_before = Constraint(m.J, m.J, m.P, m.P, rule=_c_block_in_before)
        m.c_block_in_after  = Constraint(m.J, m.J, m.P, m.P, rule=_c_block_in_after)

    if len(list(m.E_out))>0:
        def _c_block_out_before(mdl, j,k,p_b,p_f):
            if (p_f,p_b) not in mdl.E_out or j==k: return Constraint.Skip
            return mdl.t_end[k] <= mdl.t_end[j] - mdl.epsOUT + mdl.bigM*(1 - mdl.bOUT[j,k,p_b,p_f] + 2 - mdl.y[j,p_b] - mdl.y[k,p_f])
        def _c_block_out_after(mdl, j,k,p_b,p_f):
            if (p_f,p_b) not in mdl.E_out or j==k: return Constraint.Skip
            return mdl.t_start[k] >= mdl.t_end[j] + mdl.epsOUT - mdl.bigM*(    mdl.bOUT[j,k,p_b,p_f] + 2 - mdl.y[j,p_b] - mdl.y[k,p_f])
        m.c_block_out_before = Constraint(m.J, m.J, m.P, m.P, rule=_c_block_out_before)
        m.c_block_out_after  = Constraint(m.J, m.J, m.P, m.P, rule=_c_block_out_after)

    if (client_pos_policy=='soft' or CLIENT_ONE_POS_HARD):
        m.c_link_y_to_x = Constraint(m.J, m.P, rule=lambda mdl,j,p: mdl.y[j,p] <= mdl.x[mdl.clientOf[j], p])
        def _x_le_sum_y(mdl, c, p):
            return mdl.x[c,p] <= sum(mdl.y[j,p] for j in mdl.J if mdl.clientOf[j]==c)
        m.c_x_le_sumy = Constraint(m.C, m.P, rule=_x_le_sum_y)
        if CLIENT_ONE_POS_HARD:
            m.c_onepos_hard = Constraint(m.C, rule=lambda mdl,c: sum(mdl.x[c,p] for p in mdl.P) <= 1)
        else:
            m.c_extra_pos   = Constraint(m.C, rule=lambda mdl,c: mdl.vExtraPos[c] >= sum(mdl.x[c,p] for p in mdl.P) - 1)

    obj_terms = []
    if W_MAKESPAN>0: obj_terms.append(W_MAKESPAN * m.T_max)
    if W_CLIENT_DELAY>0: obj_terms.append(W_CLIENT_DELAY * sum(m.vClientDelay[c] for c in m.C))
    if (not CLIENT_ONE_POS_HARD) and hasattr(m,'vExtraPos') and W_CLIENT_POS>0:
        obj_terms.append(W_CLIENT_POS * sum(m.vExtraPos[c] for c in m.C))
    m.obj = Objective(expr=sum(obj_terms), sense=minimize)

    return m

# =============================
# UTILIDADES I/O Y PLOTS (como v2)
# =============================

def _argmax_position(m, j):
    best_p, best_v = None, -1.0
    for p in m.P:
        v = float(value(m.y[j,p]));
        if v > best_v: best_v, best_p = v, p
    return best_p


def plot_gantt_by_position(df_pos, base_date, out_html="schedule_alns.html", color_by="client"):
    if df_pos.empty:
        print("❕ Gantt: no hay tareas que pintar.")
        return
    pos_order = sorted(df_pos['p'].unique(), key=lambda x: str(x))
    df_plot = df_pos.copy()
    df_plot['Position'] = pd.Categorical(df_plot['p'], categories=pos_order, ordered=True)
    df_plot['Start'] = df_plot['start_dt']; df_plot['Finish'] = df_plot['finish_dt']
    df_plot['Plane'] = df_plot['plane'].astype(str); df_plot['Client'] = df_plot['client'].astype(str)

    palette = px.colors.qualitative.Dark24 + px.colors.qualitative.Set3 + px.colors.qualitative.Safe
    clients = df_plot['Client'].unique().tolist()
    color_map = {c: palette[i % len(palette)] for i,c in enumerate(sorted(clients))}

    fig = px.timeline(
        df_plot, x_start="Start", x_end="Finish", y="Position",
        color="Client" if color_by=="client" else "Plane",
        hover_data={"job":True,"plane":True,"client":True,"Position":True,"Start":True,"Finish":True},
        color_discrete_map=color_map if color_by=="client" else None
    )
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(title="Schedule por Posición (ALNS warm-start)", legend_title="Cliente" if color_by=="client" else "Avión",
                      xaxis_title="Fecha", yaxis_title="Posición", bargap=0.15,
                      height=max(450, 80*len(pos_order)), template="plotly_white")
    pio.write_html(fig, file=out_html, auto_open=False, include_plotlyjs="cdn")
    print(f"Plot HTML exportado: {out_html}")


def generate_report(df_planes, model_instance):
    base_date = pd.to_datetime(data.get('BASE_DATE', date.today()))
    J = list(model_instance.J.data())
    clientOf = {j: model_instance.clientOf[j] for j in J}
    dur_map  = {j: float(model_instance.D[j])  for j in J}
    ES_map   = {j: float(model_instance.ES[j]) for j in J}
    LF_map   = {j: float(model_instance.LF[j]) for j in J}

    df = df_planes.copy()
    start_col = next((c for c in ['start_slot','start','t_start','start_time'] if c in df.columns), None)
    end_col   = next((c for c in ['finish_slot','finish','t_end','end_time'] if c in df.columns), None)
    pos_col   = next((c for c in ['p','position','pos'] if c in df.columns), None)
    if start_col is None or end_col is None:
        raise ValueError("generate_report: no encuentro columnas de inicio/fin (start*/finish*).")

    if 'plane' not in df.columns:
        if 'job' in df.columns:
            df['plane'] = df['job'].apply(lambda j: int(str(j).split('-')[0]))
        else:
            raise ValueError("generate_report: no encuentro 'plane' ni 'job'.")

    if pos_col is None:
        pos_col = 'p'; df[pos_col] = 'position?'
    if 'job' not in df.columns:
        df['job'] = df['plane'].apply(lambda r: f"{int(r)}-1")
    if 'type' not in df.columns:
        df['type'] = 'work'

    base_date = pd.to_datetime(base_date)
    def _to_datetime(series):
        if pd.api.types.is_datetime64_any_dtype(series):
            return pd.to_datetime(series)
        ser_num = pd.to_numeric(series, errors='coerce')
        if ser_num.notna().all():
            return base_date + pd.to_timedelta(ser_num, unit='D')
        return pd.to_datetime(series, errors='coerce')

    df['start_dt']  = _to_datetime(df[start_col])
    df['finish_dt'] = _to_datetime(df[end_col])

    det = df[df['type'].eq('work')][[ 'plane','job', pos_col, 'start_dt','finish_dt']].copy()
    det.rename(columns={pos_col:'p'}, inplace=True)
    det['Dur Est.(d)']  = det['job'].map(lambda j: dur_map.get(j, float('nan')))
    det['ES']  = det['job'].map(lambda j: (base_date + pd.to_timedelta(ES_map[j], unit='D')).date())
    det['LF']  = det['job'].map(lambda j: (base_date + pd.to_timedelta(LF_map[j], unit='D')).date())
    det['Real'] = det['finish_dt'].dt.date
    det['Retraso(d)']  = (pd.to_datetime(det['Real']) - pd.to_datetime(det['LF'])).dt.days.clip(lower=0)

    print("\n" + "="*150)
    print("RESUMEN POR AVIÓN")
    print("="*150)
    df_res = det.groupby('plane').agg(Cliente=('job', lambda s: clientOf[s.iloc[0]]),
                                      ES=('ES','min'), LF=('LF','max'),
                                      Fin=('Real','max'),
                                      Posiciones=('p', lambda s: ", ".join(sorted(set(map(str,s))))),
                                      Trabajos=('job', lambda s: ", ".join(sorted(set(s)))),
                                      Movimientos=('job','count')).reset_index().sort_values(['Cliente','plane'])
    print(df_res.to_string(index=False, col_space=15) if not df_res.empty else "(vacío)")

    print("\n" + "="*170)
    print("DETALLE DE TODOS LOS TRABAJOS")
    print("="*170)
    det_print = det.copy()
    det_print['⚠'] = det_print['Retraso(d)'].apply(lambda d: '❌' if d>0 else '✅')
    det_print = det_print[['⚠','plane','job','p','ES','Real','LF','Dur Est.(d)','Retraso(d)']]
    det_print.columns = ['⚠','Avión','Trabajo','Posición','Fecha ES','Real','Fecha LF','Dur Est.(d)','Retraso(d)']
    print(det_print.to_string(index=False, col_space=15))

# =============================
# LECTURA INPUT Y SOLUCIÓN
# =============================

def read_input(xlsx_path: str, case_sheet: str, planning_start=PLANNING_START):
    print(f"Leyendo Excel: {os.path.basename(xlsx_path)} / {case_sheet}")
    df_case = pd.read_excel(xlsx_path, sheet_name=case_sheet)

    rename_map = {
        'plane':'plane','task':'task','job':'job','date':'early',
        'duration':'duration','movable':'movable','flexible':'flexible','client':'client'
    }
    df_case = df_case.rename(columns={c: rename_map.get(c, c) for c in df_case.columns})

    df_case = df_case[pd.to_numeric(df_case['plane'], errors='coerce').notna()]
    df_case = df_case[pd.to_numeric(df_case['duration'], errors='coerce').notna()]
    df_case['plane'] = df_case['plane'].astype(int)
    df_case['duration'] = df_case['duration'].astype(float)
    if 'client' not in df_case: df_case['client'] = 'NA'
    if 'early' not in df_case:  raise ValueError("Falta 'date'/'early' en la hoja del case.")
    df_case['early'] = pd.to_numeric(df_case['early'], errors='coerce').fillna(0).astype(float)

    plane_late = {}
    try:
        xls = pd.ExcelFile(xlsx_path)
        if 'Planes2' in xls.sheet_names:
            _df = pd.read_excel(xlsx_path, sheet_name='Planes2')
            if {'plane','late_finish'}.issubset(_df.columns):
                for _, r in _df[['plane','late_finish']].dropna().iterrows():
                    plane_late[int(r['plane'])] = float(r['late_finish'])
        if not plane_late and 'Planes' in xls.sheet_names:
            _df = pd.read_excel(xlsx_path, sheet_name='Planes')
            if {'plane','late_finish'}.issubset(_df.columns):
                for _, r in _df[['plane','late_finish']].dropna().iterrows():
                    plane_late[int(r['plane'])] = float(r['late_finish'])
    except Exception:
        pass

    big_pad = df_case['duration'].max()*10 + 100.0
    df_case['late'] = df_case.apply(
        lambda r: plane_late.get(int(r['plane']), float(r['early'])+float(r['duration'])+big_pad), axis=1
    )
    bad = (df_case['early'] + df_case['duration'] > df_case['late'])
    if bad.any():
        df_case.loc[bad,'late'] = df_case.loc[bad,'early'] + df_case.loc[bad,'duration'] + 1.0

    df_case = df_case.sort_values('early').reset_index(drop=True)

    jobs   = [f"{int(r.plane)}-1" for _, r in df_case.iterrows()]
    planes = [int(r.plane) for _, r in df_case.iterrows()]
    es     = {jobs[i]: float(df_case.loc[i,'early'])    for i in range(len(jobs))}
    dur    = {jobs[i]: float(df_case.loc[i,'duration']) for i in range(len(jobs))}
    lf     = {jobs[i]: float(df_case.loc[i,'late'])     for i in range(len(jobs))}
    cli    = {jobs[i]: str(df_case.loc[i,'client'])     for i in range(len(jobs))}
    plane_of = {jobs[i]: planes[i] for i in range(len(jobs))}

    H = max(lf.values()) + max(dur.values()) + 10.0
    if planning_start is None:
        base_date = pd.Timestamp(date.today())
    else:
        try: base_date = pd.to_datetime(planning_start)
        except Exception: base_date = pd.Timestamp(date.today())

    print(f"Slots cargados: {len(jobs)}, Ejemplo: {[f'slot{i}' for i in range(min(3,len(jobs)))]}")
    print(f"Posiciones cargadas: {len(POSITIONS)}, Ejemplo: {POSITIONS[:3]}")
    print(f"Interferencias IN: {len(INTERF_IN)} pares | OUT: {len(INTERF_OUT)} pares")
    print("Todas las ventanas temporales son consistentes (early/late con holgura).")

    global data
    data = {
        'JOBS': jobs,
        'PLANES': sorted(set(planes)),
        'POSITIONS': POSITIONS,
        'plane_of': plane_of,
        'client': cli,
        'early': es, 'dur': dur, 'late': lf,
        'plane_late': plane_late,
        'H': H,
        'BASE_DATE': base_date,
        'INTERF_IN': INTERF_IN,
        'INTERF_OUT': INTERF_OUT
    }
    return data

# =============================
# SOLVER + EXPORTS
# =============================

def apply_warm_start_from_plan(m, plan):
    if plan is None: return
    for j,v in plan['t_start'].items():
        try: m.t_start[j].value = float(v)
        except Exception: pass
    for j,v in plan['t_end'].items():
        try: m.t_end[j].value = float(v)
        except Exception: pass
    for j in m.J:
        pj = plan['pos'].get(j, None)
        for p in m.P:
            try: m.y[j,p].value = 1.0 if (pj is not None and p==pj) else 0.0
            except Exception: pass
    if hasattr(m,'x'):
        for c in m.C:
            for p in m.P:
                any_job = any((value(m.clientOf[j])==c and (plan['pos'].get(j,None)==p)) for j in m.J)
                try: m.x[c,p].value = 1.0 if any_job else 0.0
                except Exception: pass


def _argmax_position(m, j):
    best_p, best_v = None, -1.0
    for p in m.P:
        v = float(value(m.y[j,p]));
        if v > best_v: best_v, best_p = v, p
    return best_p


def solve_and_report(model, out_html="schedule_alns.html", timelimit=TIME_LIMIT, mipgap=MIP_GAP):
    base_date = pd.Timestamp(date.today()) if data.get('BASE_DATE') is None else pd.to_datetime(data['BASE_DATE'])
    opt = SolverFactory("gurobi")
    opt.options.update({
        "TimeLimit": timelimit, "MIPGap": mipgap,
        "Heuristics": 1, "RINS": 10, "MIPFocus": 3, "Cuts": 2, "Presolve": 2,
        "ImproveStartGap": 0.5, "NoRelHeurTime": 60, "VarBranch": 1, "BranchDir": -1,
        "MinRelNodes": 1000, "OutputFlag": 1, "LogToConsole": 1, "DisplayInterval": 1,
    })

    print("\n[FASE FINAL] Optimización con Gurobi")
    results = opt.solve(model, tee=True)
    print(f"Terminación: {results.solver.termination_condition}")

    rows = []
    for j in model.J:
        p = _argmax_position(model, j)
        t0 = float(value(model.t_start[j])); t1 = float(value(model.t_end[j]))
        try: plane = data['plane_of'].get(j, int(str(j).split('-')[0]))
        except Exception: plane = str(j)
        client = str(value(model.clientOf[j])) if hasattr(model,'clientOf') else str(data['client'].get(j,'NA'))
        rows.append({
            "job": str(j), "plane": plane, "client": client, "p": p,
            "start": t0, "finish": t1,
            "start_dt": base_date + pd.to_timedelta(t0, unit="D"),
            "finish_dt": base_date + pd.to_timedelta(t1, unit="D"),
            "type": "work",
        })
    df_pos = pd.DataFrame(rows).sort_values(["p","start","finish"]).reset_index(drop=True)

    df_pos.to_csv("solution_by_position.csv", index=False)
    print("CSV exportado: solution_by_position.csv")

    plot_gantt_by_position(df_pos, base_date, out_html=out_html)
    generate_report(df_pos, model_instance=model)

# =============================
# MAIN
# =============================
if __name__ == "__main__":
    XLSX, SHEET = "input_data.xlsx", "case_261"
    d = read_input(XLSX, SHEET, planning_start=PLANNING_START)

    # === ALNS para warm-start ===
    if USE_METAHEURISTICS:
        print("\n>> ALNS+SA+PR: construyendo warm-start...")
        plan, cost, tard = run_alns(d, eps=1e-3, iters=ALNS_ITERATIONS, seed=ALNS_SEED, time_limit_s=ALNS_TIME_LIMIT_S)
        print(f"   - Mejor plan ALNS: proxy_cost={cost:.1f}, tardanza_total={tard:.3f} días, switches={_client_switches(plan,d)}")
    else:
        plan = None

    # Construir modelo y aplicar warm-start
    model = build_model(d, client_pos_policy='soft')
    if plan is not None:
        apply_warm_start_from_plan(model, plan)

    solve_and_report(model)
