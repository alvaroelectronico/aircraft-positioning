def solve_and_report(model, xlsx_path, case_sheet, timelimit=1500, mipgap=GAP):
    print("Iniciando resolución con Gurobi.")
    opt = SolverFactory('gurobi')
    base_opts = {
        "TimeLimit": timelimit, "MIPGap": mipgap, "Heuristics": 1, "RINS": 10,
        "MIPFocus": 3, "Cuts": 2, "Presolve": 2, "ImproveStartGap": 0.5,
        "NoRelHeurTime": 60, "VarBranch": 1, "BranchDir": -1, "MinRelNodes": 1000,
        "OutputFlag": 1, "LogToConsole": 1, "DisplayInterval": 1,
    }
    opt.options.update(base_opts)

    res = opt.solve(model, tee=True, load_solutions=True)
    if res.solver.termination_condition not in (TerminationCondition.optimal, TerminationCondition.feasible):
        print("WARNING:", res.solver.status, res.solver.termination_condition)
        model.write("case_conflict.mps")
        return None

    BASE = data['BASE_DATE']
    rows, movimientos = [], []
    for j in model.J:
        pos = next((p for p in model.P if value(model.y[j, p]) > 0.5), None)
        if pos is None:
            continue
        st   = float(value(model.t_start[j]))
        en   = float(value(model.t_end[j]))
        tin  = float(value(model.t_in[j, pos]))
        tout = float(value(model.t_out[j, pos]))
        plane  = int(value(model.planeOf[j]))
        client = str(value(model.clientOf[j]))
        rows.append({'plane': plane, 'job': j, 'p': pos, 'type': 'work', 'start': st, 'finish': en, 'client': client})
        movimientos.append((plane, pos, 'IN',  tin))
        movimientos.append((plane, pos, 'OUT', tout))

    df_pos = pd.DataFrame(rows).sort_values(['p', 'start', 'plane']).reset_index(drop=True)
    df_pos['start_dt']  = df_pos['start'].map(lambda d: BASE + timedelta(days=float(d)))
    df_pos['finish_dt'] = df_pos['finish'].map(lambda d: BASE + timedelta(days=float(d)))
    ...
    return df_pos


# Comprobación inicial de no solape por posición (orden temporal simple)
ok = True
for p, grp in df_pos.groupby('p'):
    g = grp.sort_values('start')
    prev = -1e-9
    for _, r in g.iterrows():
        if r['start'] < prev - 1e-9:
            print(f"❌ Solape en {p}: {r['job']}")
            ok = False
        prev = r['finish']
print("✅ Sanity check posiciones:", "sin solapes." if ok else "⚠ solapes")

# CSV de salida del modelo
df_pos.to_csv("solution_by_position.csv", index=False)
df_pos[['plane', 'job', 'p', 'start', 'finish', 'client']].to_csv("solution_jobs.csv", index=False)
kpis = []
for p, grp in df_pos.groupby('p'):
    kpis.append({'position': p, 'jobs': len(grp),
                 'start_min': grp['start'].min(), 'finish_max': grp['finish'].max()})
pd.DataFrame(kpis).to_csv("solution_kpis.csv", index=False)

# Variables principales
m.t_start = Var(m.J, within=NonNegativeReals, bounds=(0, H))
m.t_end = Var(m.J, within=NonNegativeReals, bounds=(0, H))
m.y = Var(m.J, m.P, within=Binary)

# Asignación única y duración/ventanas
m.c_assign = Constraint(m.J, rule=lambda mdl, j: sum(mdl.y[j, p] for p in mdl.P) == 1)
m.c_dur = Constraint(m.J, rule=lambda mdl, j: mdl.t_end[j] == mdl.t_start[j] + mdl.D[j])
m.c_win_lo = Constraint(m.J, rule=lambda mdl, j: mdl.t_start[j] >= mdl.ES[j])
m.c_win_hi = Constraint(m.J, rule=lambda mdl, j: mdl.t_end[j] <= mdl.LF[j])

# Ventanas por avión y precedencias
m.c_plane_es = Constraint(m.J, rule=lambda mdl, j: mdl.t_start[j] >= mdl.ES_plane[int(value(mdl.planeOf[j]))])
m.c_plane_lf = Constraint(m.J, rule=lambda mdl, j: mdl.t_end[j] <= mdl.LF_plane[int(value(mdl.planeOf[j]))])
m.c_pred = Constraint(m.PRED, rule=lambda mdl, j1, j2: mdl.t_start[j2] >= mdl.t_end[j1])


# No SOLAPE por posición
    jobs = list(m.J); positions = list(m.P)
    idx_noover = [(j, k, p) for p in positions for j in jobs for k in jobs if j != k]
    m.IDX_NOOV = Set(dimen=3, initialize=idx_noover)
    m.w = Var(m.IDX_NOOV, within=Binary)  # w[j,k,p]=1 ⇒ j antes que k en p

    def c_noov1(mdl, j, k, p):
        return mdl.t_start[k] >= mdl.t_end[j] - mdl.bigM * (1 - mdl.w[j, k, p] + (1 - mdl.y[j, p]) + (1 - mdl.y[k, p]))
    def c_noov2(mdl, j, k, p):
        return mdl.t_start[j] >= mdl.t_end[k] - mdl.bigM * ( mdl.w[j, k, p] + (1 - mdl.y[j, p]) + (1 - mdl.y[k, p]))

    m.c_noov1 = Constraint(m.IDX_NOOV, rule=c_noov1)
    m.c_noov2 = Constraint(m.IDX_NOOV, rule=c_noov2)

# Makespan
m.Tmax = Var(within=NonNegativeReals, bounds=(0, H))
m.c_mks = Constraint(m.J, rule=lambda mdl, j: mdl.Tmax >= mdl.t_end[j])

# Objetivo compuesto
obj_terms = []
obj_terms.append(W_JOBINSLOT * sum(m.v01JobInSlot[s, p, j] for s in m.S for p in m.P for j in m.J))
obj_terms.append(W_ALPHA * sum(m.v01Alpha[s, p, r] for s in m.S for p in m.P for r in m.R))
obj_terms.append(W_SWITCH * sum(m.v01SwitchPlanes[s, p] for s in m.S for p in m.P))
obj_terms.append(W_PRESENCE * sum(m.vPresence[s, p, r] for s in m.S for p in m.P for r in m.R))
obj_terms.append(W_CLIENT_DELAY * sum(m.vClientDelay[c] for c in m.C))
obj_terms.append(W_IDLE * sum(m.vIdle[s, r] for s in m.S for r in m.R))

if mode in {"soft", "medium", "hard"} and (m.viol_C is not None):
    obj_terms.append(w_client_pos * sum(m.viol_C[c] for c in m.C))
if mode in {"medium", "hard"}:
    obj_terms.append(W_CLIENT_SWITCH_LOCAL * sum(m.client_change[j, k, p] for (j, k, p) in m.CHGPAIRS))
if mode == "hard":
    obj_terms.append(W_POS_MIX_LOCAL * sum(m.u_pos[p] for p in m.P))

if wms and wms > 0:
    obj_terms.append(wms * m.Tmax)

m.OBJ = Objective(expr=sum(obj_terms), sense=minimize)
return m


BASE_DATE: 2024-11-17  (origen calendario)
Slots cargados: 200, Posiciones cargadas: 5, Ejemplo: ['position1', 'position2', 'position3']
Iniciando resolución con Gurobi.
Read LP format model from file C:\Users\FX516\AppData\Local\Temp\tmp3g116ktg.pyomo.lp
Reading time = 0.86 seconds
x1: 240438 rows, 177334 columns, 722296 nonzeros
Set parameter TimeLimit to value 1500
Set parameter MIPGap to value 0.05
Set parameter Heuristics to value 1
Set parameter RINS to value 10
Set parameter MIPFocus to value 3
Set parameter Cuts to value 2
Set parameter Presolve to value 2
Set parameter ImproveStartGap to value 0.5
Set parameter NoRelHeurTime to value 60
Set parameter VarBranch to value 1
Set parameter BranchDir to value -1
Set parameter MinRelNodes to value 1000
Set parameter OutputFlag to value 1
Set parameter LogToConsole to value 1
Set parameter DisplayInterval to value 1
Gurobi Optimizer version 12.0.2 build v12.0.2rc0 (win64 - Windows 11+.0 (26200.2))

CPU model: 11th Gen Intel(R) Core(TM) i7-11370H @ 3.30GHz, instruction set [SSE2|AVX|AVX2|AVX512]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Non-default parameters:
TimeLimit  1500
MIPGap  0.05
BranchDir  -1
Heuristics  1
MinRelNodes  1000
MIPFocus  3
NoRelHeurTime  60
RINS  10
VarBranch  1
Cuts  2
DisplayInterval  1
Presolve  2
ImproveStartGap  0.5

Optimize a model with 240438 rows, 177334 columns and 722296 nonzeros
Model fingerprint: 0xe26c4ad4
Variable types: 354 continuous, 176980 integer (176980 binary)
Coefficient statistics:
  Matrix range     [4e-02, 2e+03]
  Objective range  [8e-01, 2e+07]
  Bounds range     [1e+00, 2e+03]
  RHS range        [1e+00, 6e+03]
Presolve removed 62171 rows and 57571 columns (presolve time = 1s)...
Presolve removed 65166 rows and 57811 columns (presolve time = 2s)...
Presolve removed 65166 rows and 57811 columns (presolve time = 3s)...
Presolve removed 68345 rows and 58516 columns (presolve time = 34s)...
Presolve removed 68345 rows and 58516 columns (presolve time = 38s)...
Presolve removed 68345 rows and 58516 columns (presolve time = 69s)...
Presolve removed 68345 rows and 58516 columns (presolve time = 70s)...
Presolve removed 68535 rows and 58887 columns (presolve time = 71s)...
Presolve removed 68538 rows and 58887 columns (presolve time = 72s)...
Presolve removed 68538 rows and 58887 columns (presolve time = 93s)...
Presolve removed 68538 rows and 58887 columns (presolve time = 94s)...
Presolve removed 68538 rows and 58887 columns (presolve time = 113s)...
Presolve removed 68538 rows and 58887 columns (presolve time = 114s)...
Presolve removed 68538 rows and 58887 columns (presolve time = 128s)...
Presolve removed 68538 rows and 58887 columns (presolve time = 129s)...
Presolve removed 68538 rows and 58887 columns (presolve time = 143s)...
Presolve removed 68538 rows and 58887 columns (presolve time = 144s)...
Presolve removed 68538 rows and 58887 columns
Presolve time: 143.79s
Presolved: 171900 rows, 118447 columns, 546943 nonzeros
Variable types: 22 continuous, 118425 integer (118425 binary)
Found heuristic solution: objective 8.806257e+07
Starting NoRel heuristic
Found heuristic solution: objective 8.801190e+07
Elapsed time for NoRel heuristic: 2s (best bound 3394)
Found heuristic solution: objective 8.703576e+07
Found heuristic solution: objective 8.701211e+07
Found heuristic solution: objective 8.701205e+07
Elapsed time for NoRel heuristic: 3s (best bound 3394)
Found heuristic solution: objective 8.701203e+07
Found heuristic solution: objective 8.701203e+07
Found heuristic solution: objective 8.701201e+07
Found heuristic solution: objective 8.701200e+07
Found heuristic solution: objective 8.701195e+07
Found heuristic solution: objective 8.701194e+07
Found heuristic solution: objective 8.701190e+07
Found heuristic solution: objective 8.701190e+07
Elapsed time for NoRel heuristic: 4s (best bound 3394)
Found heuristic solution: objective 8.701190e+07
Elapsed time for NoRel heuristic: 5s (best bound 4380)
Elapsed time for NoRel heuristic: 10s (best bound 4380)
Elapsed time for NoRel heuristic: 12s (best bound 5378)
Elapsed time for NoRel heuristic: 15s (best bound 6378)
Elapsed time for NoRel heuristic: 18s (best bound 7378)
Elapsed time for NoRel heuristic: 21s (best bound 8378)
Elapsed time for NoRel heuristic: 24s (best bound 9378)
Elapsed time for NoRel heuristic: 28s (best bound 10378)
Found heuristic solution: objective 6.301190e+07
Elapsed time for NoRel heuristic: 37s (best bound 11378)
Elapsed time for NoRel heuristic: 39s (best bound 11896)
Elapsed time for NoRel heuristic: 41s (best bound 2.00119e+07)
Elapsed time for NoRel heuristic: 45s (best bound 2.00119e+07)
Elapsed time for NoRel heuristic: 49s (best bound 2.00119e+07)
Elapsed time for NoRel heuristic: 54s (best bound 2.00119e+07)
Elapsed time for NoRel heuristic: 58s (best bound 2.00119e+07)
Elapsed time for NoRel heuristic: 63s (best bound 2.00119e+07)
NoRel heuristic complete
Root relaxation presolve removed 2385 rows and 2191 columns (presolve time = 1s)...
Root relaxation presolve removed 2385 rows and 2191 columns
Root relaxation presolved: 169515 rows, 116256 columns, 491237 nonzeros

Deterministic concurrent LP optimizer: primal simplex, dual simplex, and barrier
Showing barrier log only...

Root barrier log...

Ordering time: 0.14s

Barrier statistics:
 Dense cols : 74
 AA' NZ     : 5.881e+05
 Factor NZ  : 1.918e+06 (roughly 80 MB of memory)
 Factor Ops : 1.276e+08 (less than 1 second per iteration)
 Threads    : 2

                  Objective                Residual
Iter       Primal          Dual         Primal    Dual     Compl     Time
   0   4.46609222e+10 -2.54009403e+12  3.78e+02 1.81e+06  2.29e+09   211s

Barrier performed 0 iterations in 210.64 seconds (85.33 work units)
Barrier solve interrupted - model solved by another algorithm

Concurrent spin time: 0.11s (can be avoided by choosing Method=3)

Solved with dual simplex

Root simplex log...

Iteration    Objective       Primal Inf.    Dual Inf.      Time
    8031    2.0011896e+07   0.000000e+00   0.000000e+00    211s

Use crossover to convert LP symmetric solution to basic solution...

Root crossover log...

       0 DPushes remaining with DInf 0.0000000e+00               211s

      63 PPushes remaining with PInf 0.0000000e+00               211s
       0 PPushes remaining with PInf 0.0000000e+00               211s

  Push phase complete: Pinf 0.0000000e+00, Dinf 0.0000000e+00    211s


Root simplex log...

Iteration    Objective       Primal Inf.    Dual Inf.      Time
    8097    2.0011896e+07   0.000000e+00   0.000000e+00    211s
    8097    2.0011896e+07   0.000000e+00   0.000000e+00    211s

Root relaxation: objective 2.001190e+07, 8097 iterations, 3.28 seconds (3.47 work units)
Total elapsed time = 211.60s (DegenMoves)

    Nodes    |    Current Node    |     Objective Bounds      |     Work
 Expl Unexpl |  Obj  Depth IntInf | Incumbent    BestBd   Gap | It/Node Time

     0     0 2.0012e+07    0   28 6.3012e+07 2.0012e+07  68.2%     -  214s
H    0     0                    4.601190e+07 2.0012e+07  56.5%     -  214s
H    0     0                    4.201190e+07 2.0012e+07  52.4%     -  214s
H    0     0                    2.201190e+07 2.0012e+07  9.09%     -  214s

Resetting heuristic parameters to focus on improving solution
(using Heuristics=0.5 and RINS=10)...

     0     0 2.0012e+07    0   64 2.2012e+07 2.0012e+07  9.09%     -  215s
     0     0 2.0012e+07    0    8 2.2012e+07 2.0012e+07  9.09%     -  215s

Cutting planes:
  Learned: 53
  Gomory: 1
  Cover: 1
  Implied bound: 38
  Flow cover: 29
  Zero half: 2
  RLT: 1
  Relax-and-lift: 16

Explored 1 nodes (10021 simplex iterations) in 215.49 seconds (87.95 work units)
Thread count was 8 (of 8 available processors)

Solution count 10: 2.20119e+07 2.20119e+07 4.20119e+07 ... 8.7012e+07

Optimal solution found (tolerance 5.00e-02)
Best objective 2.201189600000e+07, best bound 2.201189600000e+07, gap 0.0000%
✅ Sanity check posiciones: sin solapes.
CSV exportados: solution_by_position.csv, solution_jobs.csv, solution_kpis.csv
Plot HTML exportado: schedule_enhanced.html
======================================================================================================================================================
RESUMEN POR AVIÓN
======================================================================================================================================================
       Avión     Clientes     ES_plane     LF_plane Primer Inicio          Fin     Trabajos   Posiciones  Movimientos(E/S)
          14            3   2024-12-02   2026-03-02    2025-03-03   2025-12-21         14-1    position5                 2
          20            3   2026-02-16   2027-05-17    2026-04-12   2027-04-10         20-1    position5                 2
          50            1   2026-04-20   2026-09-28    2026-04-20   2026-07-19         50-1    position4                 2
          52            2   2026-06-22   2026-11-30    2026-08-31   2026-11-30         52-1    position3                 2
          54            2   2025-04-21   2025-10-13    2025-07-13   2025-09-13         54-1    position3                 2
          76            2   2025-09-22   2026-03-09    2025-09-29   2025-12-21         76-1    position3                 2
          78            1   2025-11-17   2026-05-04    2026-01-26   2026-04-12         78-1    position4                 2
          80            5   2026-06-01   2027-09-06    2026-08-10   2027-09-06         80-1    position2                 2
          82            5   2025-07-03   2026-01-12    2025-07-04   2025-11-15         82-1    position2                 2
          84            2   2026-02-16   2026-07-06    2026-02-16   2026-04-12         84-1    position3                 2
         114            5   2025-12-08   2026-06-15    2025-12-21   2026-04-19        114-1    position2                 2
         144            2   2025-01-06   2025-07-26    2025-07-04   2025-07-13        144-1    position3                 2
         154            1   2025-01-13   2025-07-07    2025-03-04   2025-05-13        154-1    position4                 2
         174            1   2027-02-01   2027-10-25    2027-04-10   2027-10-23        174-1    position4                 2
         204            3   2025-07-03   2025-07-04    2025-07-03   2025-07-04        204-1    position2                 2
         208            4   2025-06-30   2025-11-27    2025-06-30   2025-11-06        208-1    position1                 2
         212            4   2025-12-15   2026-05-14    2025-12-15   2026-04-23        212-1    position1                 2
         218            4   2026-06-01   2026-10-29    2026-06-01   2026-10-08        218-1    position1                 2
         228            4   2026-11-16   2027-04-09    2026-11-16   2027-03-19        228-1    position1                 2
         232            4   2027-04-26   2027-09-24    2027-04-26   2027-09-03        232-1    position1                 2
         234            4   2027-10-11   2028-03-03    2027-11-01   2028-03-03        234-1    position1                 2
         238            4   2028-03-20   2028-07-21    2028-03-20   2028-06-30        238-1    position1                 2
         252            4   2024-11-18   2025-01-26    2024-11-18   2025-01-05        252-1    position1                 2
         258            1   2025-06-23   2025-12-08    2025-06-23   2025-09-14        258-1    position4                 2
         266            4   2028-09-04   2029-01-12    2028-09-04   2028-12-22        266-1    position1                 2
        2282            4   2025-02-17   2025-06-13    2025-02-17   2025-05-23       2282-1    position1                 2
==========================================================================================================================================================================
DETALLE DE TODOS LOS TRABAJOS
==========================================================================================================================================================================
         ✔      Avión    Trabajo    Cliente   Posición   Fecha ES   Fecha LF Inicio real   Fin real  Dur Est.(d)  Dur Real(d)  Retraso(d)  Holgura a ES(d)  Holgura a LF(d)
        OK         14       14-1          3  position5 2024-12-02 2026-03-02  2025-03-03 2025-12-21        293.0        293.0         0.0            91.92            70.08
        OK         20       20-1          3  position5 2026-02-16 2027-05-17  2026-04-12 2027-04-10        363.0        363.0         0.0            55.16            36.84
        OK         50       50-1          1  position4 2026-04-20 2026-09-28  2026-04-20 2026-07-19         90.0         90.0         0.0             0.00            71.00
        OK         52       52-1          2  position3 2026-06-22 2026-11-30  2026-08-31 2026-11-30         91.0         91.0         0.0            70.00             0.00
        OK         54       54-1          2  position3 2025-04-21 2025-10-13  2025-07-13 2025-09-13         62.0         62.0         0.0            83.92            29.08
        OK         76       76-1          2  position3 2025-09-22 2026-03-09  2025-09-29 2025-12-21         83.0         83.0         0.0             7.84            77.16
        OK         78       78-1          1  position4 2025-11-17 2026-05-04  2026-01-26 2026-04-12         76.0         76.0         0.0            70.08            21.92
        OK         80       80-1          5  position2 2026-06-01 2027-09-06  2026-08-10 2027-09-06        392.0        392.0         0.0            70.00             0.00
        OK         82       82-1          5  position2 2025-07-03 2026-01-12  2025-07-04 2025-11-15        134.0        134.0         0.0             1.00            58.00
        OK         84       84-1          2  position3 2026-02-16 2026-07-06  2026-02-16 2026-04-12         55.0         55.0         0.0             0.00            85.00
        OK        114      114-1          5  position2 2025-12-08 2026-06-15  2025-12-21 2026-04-19        119.0        119.0         0.0            13.92            56.08
        OK        144      144-1          2  position3 2025-01-06 2025-07-26  2025-07-04 2025-07-13          9.0          9.0         0.0           179.92            12.08
        OK        154      154-1          1  position4 2025-01-13 2025-07-07  2025-03-04 2025-05-13         70.0         70.0         0.0            50.00            55.00
        OK        174      174-1          1  position4 2027-02-01 2027-10-25  2027-04-10 2027-10-23        196.0        196.0         0.0            68.24             1.76
        OK        204      204-1          3  position2 2025-07-03 2025-07-04  2025-07-03 2025-07-04          1.0          1.0         0.0             0.00             0.00
        OK        208      208-1          4  position1 2025-06-30 2025-11-27  2025-06-30 2025-11-06        129.0        129.0         0.0             0.00            21.00
        OK        212      212-1          4  position1 2025-12-15 2026-05-14  2025-12-15 2026-04-23        129.0        129.0         0.0             0.00            21.00
        OK        218      218-1          4  position1 2026-06-01 2026-10-29  2026-06-01 2026-10-08        129.0        129.0         0.0             0.00            21.00
        OK        228      228-1          4  position1 2026-11-16 2027-04-09  2026-11-16 2027-03-19        123.0        123.0         0.0             0.00            21.00
        OK        232      232-1          4  position1 2027-04-26 2027-09-24  2027-04-26 2027-09-03        130.0        130.0         0.0             0.00            21.00
        OK        234      234-1          4  position1 2027-10-11 2028-03-03  2027-11-01 2028-03-03        123.0        123.0         0.0            21.00             0.00
        OK        238      238-1          4  position1 2028-03-20 2028-07-21  2028-03-20 2028-06-30        102.0        102.0         0.0             0.00            21.00
        OK        252      252-1          4  position1 2024-11-18 2025-01-26  2024-11-18 2025-01-05         48.0         48.0         0.0             0.00            21.00
        OK        258      258-1          1  position4 2025-06-23 2025-12-08  2025-06-23 2025-09-14         83.0         83.0         0.0             0.00            85.00
        OK        266      266-1          4  position1 2028-09-04 2029-01-12  2028-09-04 2028-12-22        109.0        109.0         0.0             0.00            21.00
        OK       2282     2282-1          4  position1 2025-02-17 2025-06-13  2025-02-17 2025-05-23         95.0         95.0         0.0             0.00            21.00
Report CSVs: report_resumen_por_avion.csv, report_detalle_trabajos.csv
Validador: 'INTERF_IN'

==================== FACT CHECK: PRIMAL FEASIBILITY ====================

--- Resumen ---
Restricciones revisadas: 240,438
• Violaciones de restricciones: 0 | peor = 0.000e+00
• Violaciones de variables:     0 | peor = 0.000e+00
• Incidencias derivadas:        0

Process finished with exit code 0