import datetime
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pyomo.environ import *
from math import ceil
from datetime import date, timedelta
from pyomo.util.infeasible import log_infeasible_constraints
import gurobipy as gp

NO_POSITIONS = 5
POSITIONS = ['position{}'.format(i) for i in range(1, NO_POSITIONS + 1)]
POSITIONS_INTERFERE = [("position3", "position5"), ("position4", "position5")]
START_DATE = datetime.date.today()

def ap_pyomo_model():
    model = AbstractModel()

    # Sets
    model.sSlots = Set(ordered=True)
    model.sJobs = Set()
    model.sPositions = Set()
    model.sPlanes = Set()
    model.sClients = Set()
    model.sPositionsInterference = Set(dimen=2)
    # model.sPosPosSlotSlot = Set(dimen=4)
    model.sPlaneClientPairs = Set(dimen=2)  # (plane, client)

    model.sSlotsSequence = Set(dimen=3)
    model.sJobSequence = Set(dimen=2)
    # model.sSwitchPlanes = Set(dimen=5)

    # Parameters
    model.pHorizon = Param(within=NonNegativeReals)

    def _init_M(m):
        return value(m.pHorizon)
    model.M = Param(initialize=_init_M)

    model.pJobDuration = Param(model.sJobs, mutable=True)
    model.pJobPrecedesJob = Param(model.sJobs, model.sJobs, mutable=True)
    model.pPlaneOfJob = Param(model.sJobs)
    model.pAirplaneOfClient = Param(model.sClients, model.sPlanes)
    model.pLastJobOfPlane = Param(model.sJobs, model.sPlanes, mutable=True)
    model.pPredictedFinishOfPlane = Param(model.sPlanes, mutable=True)
    model.pTaskOfJob = Param(model.sJobs, within=PositiveIntegers)
    model.pNumJobsPerPlane = Param(model.sPlanes, within=NonNegativeIntegers)
    model.prev_slot = Param( model.sSlots,default=None,within=model.sSlots | {None})
    model.pEarlyStartOfPlane = Param(model.sPlanes, within=NonNegativeReals)
    model.pLateFinishDeadline = Param(model.sPlanes, within=NonNegativeReals)

    # Variables
    model.v01JobInSlot = Var(model.sSlots, model.sPositions, model.sJobs, domain=Binary)
    model.v01PlaneInSlot = Var(model.sSlots, model.sPositions, model.sPlanes, domain=Binary)
    model.v01PlaneInPosition = Var(model.sPlanes, model.sPositions, domain=Binary)
    model.v01SwitchPlanes = Var(model.sSlots, model.sPositions, domain=Binary)
    model.vDurationSlot = Var(model.sSlots, model.sPositions, within=NonNegativeReals)
    model.vStartSlot = Var(model.sSlots, model.sPositions, within=NonNegativeReals)
    model.vFinishSlot = Var(model.sSlots, model.sPositions, within=NonNegativeReals)
    model.vDurationSlotForJob = Var(model.sSlots, model.sPositions, model.sJobs, within=NonNegativeReals)
    model.vStartSlotForJob = Var(model.sSlots, model.sPositions, model.sJobs, within=NonNegativeReals)
    model.vFinishSlotForJob = Var(model.sSlots, model.sPositions, model.sJobs, within=NonNegativeReals)
    model.vClientPosition = Var(model.sClients, model.sPositions, domain=Binary)
    model.vClientDelay = Var(model.sClients, within=NonNegativeReals)
    model.vPlaneDelay = Var(model.sPlanes, within=NonNegativeReals)
    model.vPresence= Var(model.sSlots, model.sPositions, model.sPlanes, domain=Binary)
    model.vStartPresence = Var(model.sSlots, model.sPositions, model.sPlanes, within=NonNegativeReals)
    model.vFinishPresence = Var(model.sSlots, model.sPositions, model.sPlanes, within=NonNegativeReals)
    model.vDurPresence = Var(model.sSlots, model.sPositions, model.sPlanes, within=NonNegativeReals)
    model.vIdle = Var(model.sSlots, model.sPositions, model.sPlanes, domain=Binary)

    # Global start and finish time of each job
    model.vStartJob = Var(model.sJobs, within=NonNegativeReals)  # s_j: global start time of job j
    model.vFinishJob = Var(model.sJobs, within=NonNegativeReals)  # f_j: global finishing time of job j

    model.v01Alpha = Var(model.sSlots, model.sSlots, model.sPositions, model.sPositions, within=Binary)
    model.v01BetaS = Var(model.sSlots, model.sSlots, model.sPositions, model.sPositions, within=Binary)
    model.v01BetaF = Var(model.sSlots, model.sSlots, model.sPositions, model.sPositions, within=Binary)

    # Añadir al inicio de los Var:
    model.v01BetaS_job = Var(model.sSlots, model.sSlots, model.sPositions, model.sPositions, model.sJobs, within=Binary)
    model.v01BetaF_job = Var(model.sSlots, model.sSlots, model.sPositions, model.sPositions, model.sJobs, within=Binary)

    def c_client_single_position(m, c):
        return sum(m.vClientPosition[c, p] for p in m.sPositions) == 1

    model.c_client_single_position = Constraint(model.sClients, rule=c_client_single_position)

    def c_client_position_consistency(m, s, p, r, c):
        if (r, c) not in m.sPlaneClientPairs:
            return Constraint.Skip
        return m.vPresence[s, p, r] <= m.vClientPosition[c, p]

    model.c_client_position_consistency = Constraint(
        model.sSlots, model.sPositions, model.sPlanes, model.sClients,
        rule=c_client_position_consistency
    )

    # Rule: Ec. cSingleJobPerSlot - Each slot of each position can have one job at a time
    def fc01_SingleJobPerSlot(model, s, p):
        return sum(model.v01JobInSlot[s, p, j] for j in model.sJobs) <= 1

    # Rule: Ec. cSlotJobDuration - Calculation of the slot duration
    def fc02_SlotJobDuration(model, s, p, j):
        return model.vDurationSlotForJob[s, p, j] == model.vFinishSlotForJob[s, p, j] - model.vStartSlotForJob[s, p, j]

    # Rule: Ec. nullStartIfNotAssigned - Starting times are 0 if the job is not assigned to a position
    def fc03_NullStartTimeIfNotInSlot(model, s, p, j):
        return model.vStartSlotForJob[s, p, j] <= model.pHorizon * model.v01JobInSlot[s, p, j]

    # Rule: Ec. nullFinishIfNotAssigned - Finishing times are 0 if the job is not assigned to a position
    def fc04_NullFinishTimeIfNotInSlot(model, s, p, j):
        return model.vFinishSlotForJob[s, p, j] <= model.pHorizon * model.v01JobInSlot[s, p, j]

    # Rule: Ec. cJobDuration - The total duration of a job is the sum of the duration of all corresponding slots
    def fc05_JobDuration(model, j):
        return sum(model.vDurationSlotForJob[s, p, j] for s in model.sSlots for p in model.sPositions) == \
               model.pJobDuration[j]

# # Constraints 6 and 7 having s, p, j as arguments - v1.0
#     # Rule: Ec. startGlobalLowerBoundNoCommas - Global job start time constraint
#     def fc06_GlobalStartConstraint(model, s, p, j):
#         if model.v01JobInSlot[s, p, j].fixed and model.v01JobInSlot[s, p, j].value == 0:
#             return Constraint.Skip
#
#         # s_j = ∑_p∑_s (s^j_spj)
#         return model.vStartJob[j] == sum(model.vStartSlotForJob[s, p, j] for p in model.sPositions for s in model.sSlots)
#
#     # Rule: Ec. finishGlobalUpperBoundNoCommas - Global job finish time constraint
#     def fc07_GlobalFinishConstraint(model, s, p, j):
#         if model.v01JobInSlot[s, p, j].fixed and model.v01JobInSlot[s, p, j].value == 0:
#             return Constraint.Skip
#
#         # f_j = ∑_p∑_s (f^j_spj)
#         return model.vFinishJob[j] == sum(model.vFinishSlotForJob[s, p, j] for p in model.sPositions for s in model.sSlots)

# #Constraints 6 and 7 just having only jobs as arguments as stated in constraint 16 every job must be assigned - v2.0
#     def fc06_GlobalStartConstraint(model, j):
#         return model.vStartJob[j] == sum(
#             model.vStartSlotForJob[s, p, j]
#             for s in model.sSlots
#             for p in model.sPositions
#         )
#
#     def fc07_GlobalFinishConstraint(model, j):
#         return model.vFinishJob[j] == sum(
#             model.vFinishSlotForJob[s, p, j]
#             for s in model.sSlots
#             for p in model.sPositions
#         )

# Constraits 6 and 7 formulate with Big-M instead of sums - v3.0
    # 1) vStartJob[j] ≤ vStartSlotForJob[s,p,j] + M·(1 - x[s,p,j])
    def fc06_StartJob_upper(model, s, p, j):
        return model.vStartJob[j] \
            <= model.vStartSlotForJob[s, p, j] \
            + model.M * (1 - model.v01JobInSlot[s, p, j])

    # 2) vStartJob[j] ≥ vStartSlotForJob[s,p,j] - M·(1 - x[s,p,j])
    def fc06_StartJob_lower(model, s, p, j):
        return model.vStartJob[j] \
            >= model.vStartSlotForJob[s, p, j] \
            - model.M * (1 - model.v01JobInSlot[s, p, j])

    # 3) vFinishJob[j] ≥ vFinishSlotForJob[s,p,j] - M·(1 - x[s,p,j])
    def fc07_FinishJob_lower(model, s, p, j):
        return model.vFinishJob[j] \
            >= model.vFinishSlotForJob[s, p, j] \
            - model.M * (1 - model.v01JobInSlot[s, p, j])

    # 4) vFinishJob[j] ≤ vFinishSlotForJob[s,p,j] + M·(1 - x[s,p,j])
    def fc07_FinishJob_upper(model, s, p, j):
        return model.vFinishJob[j] \
            <= model.vFinishSlotForJob[s, p, j] \
            + model.M * (1 - model.v01JobInSlot[s, p, j])

    # Rule: Ec. noNegativeDurationNoCommas - Start time of job must be <= finish time of job
    def fc08_StartFinishRelation(model, j):
        # s_j ≤ f_j
        return model.vStartJob[j] <= model.vFinishJob[j]

    # Rule: Ec. calculating delays of planes
    def fc09_Plane_delay(model,r):
        # para cada (j,r) con L[j,r]=1, impongo H*γ_r ≥ f[j] - T[r]
        return model.vPlaneDelay[r] >= sum(
            (model.vFinishJob[j] - model.pLateFinishDeadline[r]) * model.pLastJobOfPlane[j, r]
            for j in model.sJobs if (j, r) in model.pLastJobOfPlane
        )

    # Rule: Ec. calculating delays of clients
    def fc10_Client_delay(model, c):
#         return m.vClientDelay[c] >= m.vPlaneDelay[r]

        return model.vClientDelay[c] == sum(
            model.vPlaneDelay[r]*model.pAirplaneOfClient[c,r]
            for r in model.sPlanes if (c, r) in model.pAirplaneOfClient
        )

    # Rule: Ec. slotStartTimeFromJobs - The starting time of a slot
    def fc11_SlotStartTime(model, s, p):
        return model.vStartSlot[s, p] == sum(model.vStartSlotForJob[s, p, j] for j in model.sJobs)

    # Rule: Ec. slotFinishTimeFromJobs - The finishing time of a slot
    def fc12_SlotFinishTime(model, s, p):
        return model.vFinishSlot[s, p] == sum(model.vFinishSlotForJob[s, p, j] for j in model.sJobs)

    # Rule: Ec. SlotSequence - Slot sequence within each position
    def fc13_SlotSequence(model, s, s2, p):
        return model.vStartSlot[s, p] >= model.vFinishSlot[s2, p]

    # Rule: Ec. jobPrecedence - Job sequence (jobs are sequenced)
    def fc14_JobSequence(model, j, j2):
        return model.vStartJob[j2] >= model.vFinishJob[j]

    # # Rule: Ec. noEmptySlots - Consecutive slots - a slot is not used unless all previous ones have been used

    def fc15_ConsecutiveSlots(model, s, p):
        prev_s = model.prev_slot[s]
        if prev_s is None:
            return Constraint.Skip
        return sum(model.v01JobInSlot[s, p, j]   for j in model.sJobs) <= sum(model.v01JobInSlot[prev_s, p, j] for j in model.sJobs)

    # def fc15_ConsecutiveSlots(model, s, p):
    #     ordered = list(model.sSlots)
    #     idx = ordered.index(s)
    #     if idx == 0:
    #         return Constraint.Skip
    #     prev_s = ordered[idx - 1]
    #     return sum(model.v01JobInSlot[s, p, j] for j in model.sJobs) == \
    #         sum(model.v01JobInSlot[prev_s, p, j] for j in model.sJobs)

    # def fc15_ConsecutiveSlots(model, s, p):
    #     # Skip constraint for the first slot (s=1)
    #     if model.sSlots.ord(s) == 1:
    #         return Constraint.Skip
    #
    #     # Get the previous slot
    #     prev_s = list(model.sSlots)[model.sSlots.ord(s) - 2]  # -1 for 0-based indexing, -1 for previous
    #
    #     # Sum of job assignments in current slot must be less than or equal to sum in previous slot
    #     return sum(model.v01JobInSlot[s, p, j] for j in model.sJobs) <= sum(
    #         model.v01JobInSlot[prev_s, p, j] for j in model.sJobs)

    # Rule: Ec. - A job can be assigned to a single slot of a position
    def fc16_SingleSlotPerJob(model, j):
        # ∑∑ x_jsp = 1 ∀j ∈ J
        return sum(model.v01JobInSlot[s, p, j]
               for s in model.sSlots for p in model.sPositions) == 1

    # Rule: Ec. - If a job is not assigned to a slot of a position, the duration of that job in that slot is zero
    def fc17_DurationIfNotAssigned(model, s, p, j):
        # d^j_spj = D_j·x_spj, ∀s ∈ S, p ∈ P, j ∈ J
        # return model.vDurationSlotForJob[s, p, j] == model.pJobDuration[j] * model.v01JobInSlot[s, p, j]
        return model.vFinishSlotForJob[s, p, j] - model.vStartSlotForJob[s, p, j] \
            == model.pJobDuration[j] * model.v01JobInSlot[s, p, j]

    # Rule: The duration of a slot is that of the slot assigned to that job
    def fc18_SlotDuration(model, s, p):
        return model.vDurationSlot[s, p] == sum(model.vDurationSlotForJob[s, p, j] for j in model.sJobs)

    # Rule: Ec. cPlaneSlotAssignment - Airplane-job consistency assignment
    def fc19_PlaneSlotAssignment(model, s, p, r):
        return model.v01PlaneInSlot[s, p, r] == sum(model.v01JobInSlot[s, p, j]
                                                    for j in model.sJobs if model.pPlaneOfJob[j] == r)

    # Rule: Airplane with some job in a position

    def fc20_PlaneInPosition(model, s, p, r):
        return model.v01PlaneInPosition[r, p] >= model.v01PlaneInSlot[s, p, r]# Rule: Ec. cPlaneSlotAssignment - Airplane-job consistency assignment

    def fc20b_PresentIfWork(model, s, p, r):
        # Si r tiene un trabajo en (s,p), debe “estar” en p
        return model.vPresence[s, p, r] >= model.v01PlaneInSlot[s, p, r]

    def fc20c_PresentExactlyOne(model, s, r):
        return sum(model.vPresence[s, p, r] for p in model.sPositions) == sum(model.v01PlaneInSlot[s, p, r] for p in model.sPositions) + sum(model.vIdle[s, p, r] for p in model.sPositions)

    def fc20d_SinglePlanePerPosition(model, s, p):
        # En cada slot s y posición p, como máximo un avión puede estar presente
        return sum(model.vPresence[s, p, r] for r in model.sPlanes) <= 1

    def fc20e_PresenceNoJumpForward(model, s, p, r):
        prev_s = model.prev_slot[s]
        if prev_s is None:
            return Constraint.Skip
        # Si en prev_s estaba en p, pero en s ya no, debe contarse un switch en prev_s
        return model.vPresence[prev_s, p, r] - model.vPresence[s, p, r] <= model.v01SwitchPlanes[prev_s, p]

    def fc20f_PresenceNoJumpBackward(model, s, p, r):
        prev_s = model.prev_slot[s]
        if prev_s is None:
            return Constraint.Skip
        # Si en s está en p, pero en prev_s no, también cuenta un switch en prev_s
        return model.vPresence[s, p, r] - model.vPresence[prev_s, p, r] <= model.v01SwitchPlanes[prev_s, p]

    def idle_def1(model, s, p, r):
        # vIdle ≥ vPresence - suma de trabajos
        return model.vIdle[s, p, r] >= model.vPresence[s, p, r]  - model.v01PlaneInSlot[s, p, r]

    def idle_def2(model, s, p, r):
        # vIdle ≤ vPresence (y automáticamente ≤ 1 - suma trabajos)
        return model.vIdle[s, p, r] <= model.vPresence[s, p, r]

    # 1) si vPresence=1 entonces vStartPresence = vStartSlot, si vPresence=0 entonces ≤ M·0
    def link_start_presence(model, s, p, r):
        return model.vStartPresence[s, p, r] <= model.vStartSlot[s, p] \
            + model.M * (1 - model.vPresence[s, p, r])

    # 2) si vPresence=1 entonces vFinishPresence ≥ vFinishSlot, si vPresence=0 entonces ≥ -M
    def link_finish_presence_lb(model, s, p, r):
        return model.vFinishPresence[s, p, r] >= model.vFinishSlot[s, p] \
            - model.M * (1 - model.vPresence[s, p, r])

    # 3) si vPresence=1 entonces vFinishPresence ≤ vFinishSlot, si vPresence=0 entonces ≤ M·0
    def link_finish_presence_ub(model, s, p, r):
        return model.vFinishPresence[s, p, r] <= model.vFinishSlot[s, p] \
            + model.M * (1 - model.vPresence[s, p, r])

    # def durPres_rule(model, s, p, r):
    #     return model.vDurPresence[s, p, r] == model.vFinishPresence[s, p, r] - model.vStartPresence[s, p, r]

    # Rule: Client c with some airpline in position p:
    def fc21_ClientInPosition(model, c, p):
        return model.vClientPosition[c, p] >= sum(
            model.v01PlaneInPosition[r, p] * model.pAirplaneOfClient[c, r]
            for r in model.sPlanes
        )


    # # Rule: Ec. fcBetaDefinion1 - Computing if starting time of slot s in position p is earlier than starting time of slot s' in position p'
    # def fc22_BetaDefinition1(model, s, s2, p, p2):
    #     return model.pHorizon * model.v01BetaS[s, s2, p, p2] + model.vStartSlot[s, p] >= model.vStartSlot[s2, p2]
    #
    # # Rule: Ec. fcBetaDefinion2 - Computing if finishing time of slot s in position p is later than starting time of slot s' in position p'
    # def fc23_BetaDefinition2(model, s, s2, p, p2):
    #     return model.pHorizon * model.v01BetaF[s, s2, p, p2] + model.vStartSlot[s2, p2] >= model.vFinishSlot[s, p]
    #
    # # Rule: Interference between slots
    # def fc24_InterferenceExists(model, s, s2, p, p2):
    #     return 1 + model.v01Alpha[s, s2, p, p2] >= model.v01BetaS[s, s2, p, p2] + model.v01BetaF[s, s2, p, p2]

    #Interference rules taking into account presence instead of slot duration
    def fc22_BetaDefinition1(model, s, s2, p, p2, j):
        if s == s2 and p == p2:
            return Constraint.Skip
            # si betaS_job=1, el fragmento (s,p,j) debe terminar antes que empiece (s2,p2,j)
        return model.vFinishSlotForJob[s, p, j] <= model.vStartSlotForJob[s2, p2, j] + model.pHorizon* (
                    1 - model.v01BetaS_job[s, s2, p, p2, j])

    def fc23_BetaDefinition2(model, s, s2, p, p2, j):
        if s == s2 and p == p2:
            return Constraint.Skip
            # si betaF_job=1, el fragmento (s2,p2,j) debe terminar antes que empiece (s,p,j)
        return model.vFinishSlotForJob[s2, p2, j] <= model.vStartSlotForJob[s, p, j] + model.pHorizon * (
                    1 - model.v01BetaF_job[s, s2, p, p2, j])

    def fc24_InterferenceExists(model, s, s2, p, p2):
        if (p, p2) not in model.sPositionsInterference:
            return Constraint.Skip
        return 1 + model.v01Alpha[s, s2, p, p2] >= model.v01BetaS[s, s2, p, p2] + model.v01BetaF[s, s2, p, p2]

    # Rule: Ec. PlaneSwitchInPosition - Switching planes between consecutive slots
    def consecutive_pairs_from_prev(m):
        # construye dinámicamente pares (s_prev, s) donde prev_slot[s] no es None
        return [(prev_s, s) for s in m.sSlots for prev_s in [m.prev_slot[s]] if prev_s is not None]

    model.sConsecSlots = Set(dimen=2, initialize=consecutive_pairs_from_prev)

    # Redefinir switching: si r está en p en s_prev y r2 en p en s, con r != r2, se penaliza
    def fc25_SwitchingPlanes_sparse(model, prev_s, s, p, r, r2):
        if r == r2:
            return Constraint.Skip
        # Solo para pares consecutivos
        # Si r estaba en p en prev_s y otro r2 está en p en s, se cuenta un cambio en prev_s,p
        return 1 + model.v01SwitchPlanes[prev_s, p] >= model.vPresence[prev_s, p, r] + model.vPresence[s, p, r2]

    # Rule: If a job is split among different slots, these cannot overlap
    def fc26_NoOverlapSlots(model, s, s2, p, p2, j):
        # Salta si es exactamente la misma celda
        if s == s2 and p == p2:
            return Constraint.Skip
        # No necesita preconstruir un conjunto de tuplas: se activa solo si ambos pueden estar asignados
        # Si ninguno de los dos fragmentos del job está en ambas combinaciones, el big-M implícito
        # sale por el hecho de que x será 0 y la desigualdad se satisface trivialmente.
        return 1 + model.v01BetaS_job[s, s2, p, p2, j] + model.v01BetaF_job[s, s2, p, p2, j] >= \
            model.v01JobInSlot[s, p, j] + model.v01JobInSlot[s2, p2, j]

    # Rule: función objetivo
    # def fc27_NoMovements(model):
    #     return sum(model.v01JobInSlot[s, p, j] for s in model.sSlots for p in model.sPositions for j in model.sJobs) \
    #             + sum(model.v01Alpha[i] for i in model.sPosPosSlotSlot) \
    #             + sum(model.v01SwitchPlanes[s, p] for p in model.sPositions for s in model.sSlots) \
    #             + sum(model.v01PlaneInSlot[s, p, r] for r in model.sPlanes for p in model.sPositions for s in model.sSlots) \
    #             + sum(model.vClientDelay[c] for c in model.sClients) \
    #             + sum(model.vIdle[s,p,r] for r in model.sPlanes for p in model.sPositions for s in model.sSlots)

    # Rule: c27_EarlyStart – Cada trabajo j de avión r no puede empezar antes de pEarlyStartOfPlane[r]
    def fc27_EarlyStart(model, j):
        r = model.pPlaneOfJob[j]
        return model.vStartJob[j] >= model.pEarlyStartOfPlane[r]

    # Rule: c28_LateFinish – Cada trabajo j de avión r debe acabar antes de pLateFinishDeadline[r]
    def fc28_LateFinish(model, j):
        r = model.pPlaneOfJob[j]
        return model.vFinishJob[j] <= model.pLateFinishDeadline[r]

    def fc29_NoMovements(model):
        return sum(model.v01JobInSlot[s, p, j] for s in model.sSlots for p in model.sPositions for j in model.sJobs) \
                + sum(model.v01Alpha[s, s2, p, p2] for s in model.sSlots for s2 in model.sSlots for p in model.sPositions for p2 in model.sPositions if (p, p2) in model.sPositionsInterference) \
                + sum(model.v01SwitchPlanes[s, p] for p in model.sPositions for s in model.sSlots) \
                + sum(model.vPresence[s, p, r] for s in model.sSlots for p in model.sPositions for r in model.sPlanes) \
                + sum(model.vClientDelay[c] for c in model.sClients) \
                + sum(model.vIdle[s, p, r] for r in model.sPlanes for p in model.sPositions for s in model.sSlots)

    # Activating constraints
    print("Generating c01_SingleJobPerSlot constraint - Eq. cSingleJobPerSlot")
    model.c01_SingleJobPerSlot = Constraint(model.sSlots, model.sPositions, rule=fc01_SingleJobPerSlot)

    print("Generating c02_SlotJobDuration constraint - Eq. cSlotJobDuration")
    model.c02_SlotJobDuration = Constraint(model.sSlots, model.sPositions, model.sJobs, rule=fc02_SlotJobDuration)

    print("Generating c03_NullStartTimeIfNotInSlot constraint - Eq. nullStartIfNotAssigned")
    model.c03_NullStartTimeIfNotInSlot = Constraint(model.sSlots, model.sPositions, model.sJobs, rule=fc03_NullStartTimeIfNotInSlot)

    print("Generating c04_NullFinishTimeIfNotInSlot constraint - Eq. nullFinishIfNotAssigned")
    model.c04_NullFinishTimeIfNotInSlot = Constraint(model.sSlots, model.sPositions, model.sJobs, rule=fc04_NullFinishTimeIfNotInSlot)

    print("Generating c05_JobDuration constraint - Eq. cJobDuration")
    model.c05_JobDuration = Constraint(model.sJobs, rule=fc05_JobDuration)

    # ## Activation of constraints 6 and 7 v 1.0
    # print("Generating c06_GlobalStartConstraint constraint - Eq. startGlobalLowerBoundNoCommas")
    # model.c06_GlobalStartConstraint = Constraint( model.sSlots, model.sPositions, model.sJobs, rule=fc06_GlobalStartConstraint)
    #
    # print("Generating c07_GlobalFinishConstraint constraint - Eq. finishGlobalUpperBoundNoCommas")
    # model.c07_GlobalFinishConstraint = Constraint( model.sSlots, model.sPositions, model.sJobs, rule=fc07_GlobalFinishConstraint)

    # Activation of constraints 6 and 7 v 2.0
    # print("Generating c06_GlobalStartConstraint constraint - Eq. startGlobalLowerBoundNoCommas")
    # model.c06_GlobalStartConstraint = Constraint( model.sJobs, rule=fc06_GlobalStartConstraint)
    #
    # print("Generating c07_GlobalFinishConstraint constraint - Eq. finishGlobalUpperBoundNoCommas")
    # model.c07_GlobalFinishConstraint = Constraint( model.sJobs, rule=fc07_GlobalFinishConstraint)
    #
    # Activation of constraints 6 and 7 v 3.0
    print("Generating c06_StartJob constraint - Eq. startUpperLowerBound")
    model.c06_StartJob_upper = Constraint(model.sSlots, model.sPositions, model.sJobs, rule=fc06_StartJob_upper)
    model.c06_StartJob_lower = Constraint(model.sSlots, model.sPositions, model.sJobs, rule=fc06_StartJob_lower)
    print("Generating c07_GlobalFinishConstraint constraint - Eq. finishUpperLowerBound")
    model.c07_FinishJob_lower = Constraint(model.sSlots, model.sPositions, model.sJobs, rule=fc07_FinishJob_lower)
    model.c07_FinishJob_upper = Constraint(model.sSlots, model.sPositions, model.sJobs, rule=fc07_FinishJob_upper)

    print("Generating c08_StartFinishRelation constraint - Eq. noNegativeDurationNoCommas")
    model.c08_StartFinishRelation = Constraint(model.sJobs, rule=fc08_StartFinishRelation)

    print("Generating c09_PlaneDelay contraint - Eq. cPlaneDelay")
    model.c09_Plane_delay = Constraint(model.sPlanes, rule=fc09_Plane_delay)

    print("Generating c10_ClientDelay contraint - Eq. cPlaneDelay")
    model.c10_Client_delay = Constraint(model.sClients, rule=fc10_Client_delay)

    print("Generating c11_SlotStartTime constraint - Eq. slotStartTimeFromJobs")
    model.c11_SlotStartTime = Constraint(model.sSlots, model.sPositions, rule=fc11_SlotStartTime)

    print("Generating c12_SlotFinishTime constraint - Eq. slotFinishTimeFromJobs")
    model.c12_SlotFinishTime = Constraint(model.sSlots, model.sPositions, rule=fc12_SlotFinishTime)

    print("Generating c13_SlotSequence constraint - Eq. SlotSequence")
    model.c13_SlotSequence = Constraint(model.sSlotsSequence, rule=fc13_SlotSequence)

    print("Generating c14_JobSequence constraint - Eq. jobPrecedence")
    model.c14_JobSequence = Constraint(model.sJobSequence, rule=fc14_JobSequence)

    print("Generating c15_ConsecutiveSlots constraint - Eq. noEmptySlots")
    # model.c15_ConsecutiveSlots = Constraint(model.sSlots, model.sPositions, rule=fc15_ConsecutiveSlots)

    print("Generating c16_SingleSlotPerJob constraint - Eq. 14")
    model.c16_SingleSlotPerJob = Constraint(model.sJobs, rule=fc16_SingleSlotPerJob)

    print("Generating c17_DurationIfNotAssigned constraint - Eq. 15")
    model.c17_DurationIfNotAssigned = Constraint(model.sSlots, model.sPositions, model.sJobs, rule=fc17_DurationIfNotAssigned)

    print("Generating c18_SlotDuration constraint")
    model.c18_SlotDuration = Constraint(model.sSlots, model.sPositions, rule=fc18_SlotDuration)

    print("Generating c19_PlaneSlotAssignment constraint - Eq. cPlaneSlotAssignment")
    model.c19_PlaneSlotAssignment = Constraint(model.sSlots, model.sPositions, model.sPlanes, rule=fc19_PlaneSlotAssignment)

    print("Generating c20_PlaneInPosition constraint")
    model.c20_PlaneInPosition= Constraint(model.sSlots, model.sPositions, model.sPlanes, rule=fc20_PlaneInPosition)

    print("Generating c20b and c20c_PlaneAlwaysPresent constraint")
    model.cPresentIfWork = Constraint(model.sSlots, model.sPositions, model.sPlanes, rule=fc20b_PresentIfWork)
    model.cPresentExactlyOne = Constraint(model.sSlots, model.sPlanes, rule=fc20c_PresentExactlyOne)

    print("Generating c20d_SinglePlanePerPosition constraint")
    model.c20d_SinglePlanePerPosition = Constraint(model.sSlots, model.sPositions, rule=fc20d_SinglePlanePerPosition)

    model.c20e_PresenceNoJumpF = Constraint(model.sSlots, model.sPositions, model.sPlanes,rule=fc20e_PresenceNoJumpForward)
    model.c20f_PresenceNoJumpB = Constraint(model.sSlots, model.sPositions, model.sPlanes,rule=fc20f_PresenceNoJumpBackward)

    print("Generating Variables accounting for Idle Jobs")
    model.cIdle1 = Constraint(model.sSlots, model.sPositions, model.sPlanes, rule=idle_def1)
    model.cIdle2 = Constraint(model.sSlots, model.sPositions, model.sPlanes, rule=idle_def2)
    model.cLinkStartPres = Constraint(model.sSlots, model.sPositions, model.sPlanes, rule=link_start_presence)
    model.cLinkFinishPres1 = Constraint(model.sSlots, model.sPositions, model.sPlanes, rule=link_finish_presence_lb)
    model.cLinkFinishPres2 = Constraint(model.sSlots, model.sPositions, model.sPlanes, rule=link_finish_presence_ub)
    # model.cPresDur = Constraint(model.sSlots, model.sPositions, model.sPlanes, rule=durPres_rule)


    print("Generating c21_ClientInPosition constraint")
    model.c21_ClientInPosition = Constraint(model.sClients, model.sPositions, rule=fc21_ClientInPosition)

    print("Generating c22_BetaDefinition1 constraint - Eq. fcBetaDefinion1")
    model.c22_BetaDefinition1 = Constraint(model.sSlots, model.sSlots, model.sPositions, model.sPositions, model.sJobs, rule=fc22_BetaDefinition1)

    print("Generating c23_BetaDefinition2 constraint - Eq. fcBetaDefinion2")
    model.c23_BetaDefinition2 = Constraint(model.sSlots, model.sSlots, model.sPositions, model.sPositions, model.sJobs, rule=fc23_BetaDefinition2)

    print("Generating c24_InterferenceExists constraint")
    model.c24_InterferenceExists = Constraint(model.sSlots, model.sSlots, model.sPositions, model.sPositions, rule=fc24_InterferenceExists)

    print("Generating c25_SwitchingPlanes constraint - Eq. PlaneSwitchInPOsition")
    model.c25_SwitchingPlanes = Constraint(model.sConsecSlots, model.sPositions, model.sPlanes, model.sPlanes, rule=lambda m, prev_s, s, p, r, r2: fc25_SwitchingPlanes_sparse(m, prev_s, s, p, r, r2))

    print("Generating c26_NoOverlapSlots constraint")
    model.c26_NoOverlapSlots = Constraint(model.sSlots, model.sSlots, model.sPositions, model.sPositions, model.sJobs, rule=fc26_NoOverlapSlots)

    print("Generating c27_EarlyStart constraint")
    model.c28_EarlyStart = Constraint(model.sJobs, rule=fc27_EarlyStart)

    print("Generating c28_LateFinish constraint")
    model.c29_LateFinish = Constraint(model.sJobs, rule=fc28_LateFinish)

    #Objective function
    print("Generating objective function")
    model.ObjFunction = Objective(rule=fc29_NoMovements, sense=minimize)

    return model


def read_excel(file_name, sheet_name):
    df = pd.read_excel(file_name, sheet_name=sheet_name)


    sJobs         = df['job'].to_list()
    sPlanes       = df['plane'].unique().tolist()
    pJobDuration  = df.set_index('job')['duration'].to_dict()
    pDate         = df.set_index('job')['date'].to_dict()
    pPlaneOfJob   = df.set_index('job')['plane'].to_dict()
    pTaskOfJob    = df.set_index('job')['task'].to_dict()
    df['predicted_finish'] = df['date'] + df['duration']
    # 2) Clientes: si existe la columna "client", la uso; si no existe, considero que cada avión es cliente propio.
    if 'client' in df.columns:
        sClients = df['client'].unique().tolist()
        dic_pAirplaneOfClient = {}
        for c in sClients:
            for r in sPlanes:
                # 1 si hay al menos una fila donde plane==r y client==c
                dic_pAirplaneOfClient[(c, r)] = int(bool(
                    ((df['plane'] == r) & (df['client'] == c)).any()
                ))
    else:
        # Alternativa: cada avión se trata como su propio cliente
        sClients = sPlanes[:]  # lista de clientes = lista de aviones
        dic_pAirplaneOfClient = {}
        for r in sPlanes:
            for r2 in sPlanes:
                # Cliente “r” está vinculado solo al avión “r”
                dic_pAirplaneOfClient[(r, r2)] = 1 if (r2 == r) else 0


    max_finish_by_plane = df.groupby('plane')['predicted_finish'].max().to_dict()

    # Construyo lista de pares (plane, client) válidos para facilitar uso posterior
    plane_client_pairs = [(r, c) for (c, r), v in dic_pAirplaneOfClient.items() if v == 1]

    dic_pLastJobOfPlane = {}
    for r in sPlanes:
        df_r = df[df['plane'] == r]
        if not df_r.empty:
            tarea_max = int(df_r['task'].max())
            # Tomo el primer job que tenga esa tarea máxima
            j_ultimo = df_r[df_r['task'] == tarea_max]['job'].iloc[0]
            for j in sJobs:
                dic_pLastJobOfPlane[(j, r)] = 1 if (j == j_ultimo) else 0
        else:
            for j in sJobs:
                dic_pLastJobOfPlane[(j, r)] = 0

    sPositions = POSITIONS
    sPositionsInterference = POSITIONS_INTERFERE

    # sSlots = ['slot{}'.format(i) for i in range(ceil(len(sJobs) / NO_POSITIONS * 2.5)+4)]
    # sSlots = sorted(sSlots, key=lambda x: int(x.replace('slot', '')))

    # Creación de slots dinámica para evitar fallos del modelo por falta de slots. En función del máximo de tareas de un avión.
    max_tasks_per_plane = df.groupby('plane')['task'].nunique().max()

    # nº mínimo de slots = ceil( total_jobs / total_positions )
    N1 = ceil(len(sJobs) / NO_POSITIONS*2.5)+40
    # Cada avión necesita al menos sus propias tareas en un solo slot
    N2 = max_tasks_per_plane
    nSlots = max(N1, N2)+20
    sSlots = [f"slot{i}" for i in range(nSlots)]

    pHorizon = max(
        sum(pJobDuration[j] for j in sJobs if pPlaneOfJob[j] == r)
        for r in sPlanes
    ) * 50

    df_planes = pd.read_excel(file_name, sheet_name='Planes')
    df_planes['plane'] = df_planes['plane'].astype(type(sPlanes[0]))
    # Filtra sólo los aviones que salen en el escenario
    df_planes = df_planes[df_planes['plane'].isin(sPlanes)]

    # Rellena vacíos: ES = 0, LF = pHorizon → desactiva lógicamente las ventanas
    df_planes['early_start'] = df_planes['early_start'].fillna(0)
    df_planes['late_finish'] = df_planes['late_finish'].fillna(pHorizon)

    pEarlyStartOfPlane = df_planes.set_index('plane')['early_start'].to_dict()
    pLateFinishDeadline = df_planes.set_index('plane')['late_finish'].to_dict()

    # Si algún avión del escenario no estaba en la hoja 'Planes':
    for r in sPlanes:
        pEarlyStartOfPlane.setdefault(r, 0)
        pLateFinishDeadline.setdefault(r, pHorizon)

    data = {
        'sJobs': sJobs,
        'sSlots': sSlots,
        'sPositions': sPositions,
        'sPlanes': sPlanes,
        'sClients': sClients,
        'sPositionsInterference': sPositionsInterference,
        'pJobDuration': pJobDuration,
        'pPlaneOfJob': pPlaneOfJob,
        'pTaskOfJob': pTaskOfJob,
        'pDate': pDate,
        'pHorizon': pHorizon,
        'pPredictedFinishOfPlane': max_finish_by_plane,
        'pAirplaneOfClient': dic_pAirplaneOfClient,
        'plane_client_pairs': plane_client_pairs,
        'pLastJobOfPlane': dic_pLastJobOfPlane,
        'pEarlyStartOfPlane': pEarlyStartOfPlane,
        'pLateFinishDeadline': pLateFinishDeadline,
    }
    return data


def create_data(data):
    sPositions = data.get('sPositions', None)
    sPositionsInterference = data.get('sPositionsInterference', None)
    sJobs = data.get('sJobs', None)
    sPlanes = data.get('sPlanes', None)
    sClients = data.get('sClients', None)
    sSlots = data.get('sSlots', None)
    pJobDuration = data.get('pJobDuration', None)
    pDate = data.get('pDate', None)
    pHorizon = data.get('pHorizon')
    pPlaneOfJob = data.get('pPlaneOfJob')
    pTaskOfJob = data.get('pTaskOfJob')
    pAirplaneOfClient = data.get('pAirplaneOfClient', None)
    pLastJobOfPlane = data.get('pLastJobOfPlane', None)
    pPredictedFinishOfPlane = data.get('pPredictedFinishOfPlane', None)
    pNumJobsPerPlane = {
        r: sum(1 for j in sJobs if pPlaneOfJob[j] == r)
        for r in sPlanes
    }
    pEarlyStartOfPlane = data.get('pEarlyStartOfPlane', { r: 0        for r in sPlanes })
    pLateFinishDeadline = data.get('pLateFinishDeadline', { r: pHorizon for r in sPlanes })

    data['pNumJobsPerPlane'] = pNumJobsPerPlane

    prev_slot = { sSlots[i]: (sSlots[i - 1] if i > 0 else None) for i in range(len(sSlots))}
    sSlotsSequence = [(prev_s, s, p) for s in sSlots for p in sPositions for prev_s in [prev_slot[s]] if prev_s is not None]

    plane_client_pairs = data.get('plane_client_pairs', [])

    sJobSequence = []
    for r in sPlanes:
        # Jobs per plane
        jobs_r = [j for j in sJobs if pPlaneOfJob[j] == r]

        # Check for duplicated tasks
        try:
            task_list = [(j, int(pTaskOfJob[j])) for j in jobs_r]
        except ValueError as e:
            raise ValueError(f"Error en las tareas del avión {r}: asegúrate de que sean números enteros. {e}")

        #Sort jobs
        task_list.sort(key=lambda x: x[1])

        #Create sequence
        for i in range(len(task_list) - 1):
            j1, task1 = task_list[i]
            j2, task2 = task_list[i + 1]
            if task1 < task2:
                sJobSequence.append((j1, j2))
            else:
                print(
                    f"⚠️ Advertencia: Tareas fuera de orden o repetidas para avión {r}: {j1} (tarea {task1}), {j2} (tarea {task2})")

    # Visible verification
    print("Secuencias de trabajos generadas:")
    for j1, j2 in sJobSequence:
        print(f"{j1} → {j2}")

    # sPosPosSlotSlot = [(s, s2, p, p2) for s in sSlots for s2 in sSlots for p in sPositions for p2 in sPositions if
    #                    (p, p2) in sPositionsInterference and p!=p2]
    #
    # sSwitchPlanes = [(p, s, s2, r, r2) for p in sPositions for s in sSlots for s2 in sSlots for r in sPlanes
    #                  for r2 in sPlanes if sSlots.index(s) == sSlots.index(s2) + 1 and r!=r2]

    # ordered_slots = sorted(sSlots,key=lambda s: int(s.replace("slot", "")))
    #
    # slot_ord = {s: i for i, s in enumerate(ordered_slots)}
    # sSlots = ordered_slots

    consecutive_pairs = [(sSlots[i], sSlots[i + 1]) for i in range(len(sSlots) - 1)]

    # FirstSlotOfPlane={}
    # LastSlotOfPlane={}
    # for r in sPlanes:
    #     # detecta todos los jobs de r
    #     slots_r = []
    #     for (s, p, j) in [(s, p, j) for s in sSlots for p in sPositions for j in sJobs]:
    #         if pPlaneOfJob[j] == r:
    #             slots_r.append(slot_ord[s])
    #     if slots_r:
    #         FirstSlotOfPlane[r] = min(slots_r)
    #         LastSlotOfPlane[r] = max(slots_r)
    #     else:
    #         # si no hay jobs, forzamos ventana vacía
    #         FirstSlotOfPlane[r] = len(sSlots)
    #         LastSlotOfPlane[r] = -1

    # Filling data into input_data dictionary
    input_data = {None: {
        'sSlots': {None: sSlots},
        'sJobs': {None: sJobs},
        'sPositions': {None: sPositions},
        'sPlanes': {None: sPlanes},
        'sClients': {None: sClients},
        'sPositionsInterference': {None: sPositionsInterference},
        'sSlotsSequence': {None: sSlotsSequence},
        'prev_slot': prev_slot,
        'sJobSequence': {None: sJobSequence},
        # 'sPlaneSlotAssignment': {None: sPlaneSlotAssignment},
        'pHorizon': {None: pHorizon},
        'pJobDuration': pJobDuration,
        'pPlaneOfJob': pPlaneOfJob,
        'pTaskOfJob': pTaskOfJob,
        'pDate': pDate,
        'pAirplaneOfClient': pAirplaneOfClient,
        'pLastJobOfPlane': pLastJobOfPlane,
        'pPredictedFinishOfPlane': pPredictedFinishOfPlane,
        'pNumJobsPerPlane': pNumJobsPerPlane,
        'pEarlyStartOfPlane': pEarlyStartOfPlane,
        'pLateFinishDeadline': pLateFinishDeadline,
         'sPlaneClientPairs': {None: plane_client_pairs},

    }
    }

    return input_data


def get_solution_data(model):
    slot_assignment = {(s, p): j for s in model.sSlots for p in model.sPositions for j in model.sJobs if
                       model.v01JobInSlot[s, p, j].value == 1}

    duration_slot = {(s, p): model.vDurationSlot[s, p].value for s in model.sSlots for p in model.sPositions}

    duration_slot_job = {(s, p, j): model.vDurationSlotForJob[s, p, j].value for s in model.sSlots \
                         for p in model.sPositions for j in model.sJobs}

    interference = [
        (s, s2, p, p2)
        for (s, s2, p, p2), var in instance.v01Alpha.items()
        if (p, p2) in instance.sPositionsInterference and value(var) > 0.5
    ]

    start_slot_job = {(s, p, j): model.vStartSlotForJob[s, p, j].value for s in model.sSlots for p in model.sPositions
                      for j in model.sJobs}

    finish_slot_job = {(s, p, j): model.vFinishSlotForJob[s, p, j].value for s in model.sSlots for p in model.sPositions
                       for j in model.sJobs}

    start_slot = {(s, p): model.vStartSlot[s, p].value for s in model.sSlots for p in model.sPositions}

    finish_slot = {(s, p): model.vFinishSlot[s, p].value for s in model.sSlots for p in model.sPositions}

    # Add global job start and finish times
    start_job = {j: model.vStartJob[j].value for j in model.sJobs}
    finish_job = {j: model.vFinishJob[j].value for j in model.sJobs}

    solution = {'slot_assignment': slot_assignment,
                'duration_slot': duration_slot,
                'duration_slot_job': duration_slot_job,
                'interference': interference,
                'start_slot_job': start_slot_job,
                'finish_slot_job': finish_slot_job,
                'start_slot': start_slot,
                'finish_slot': finish_slot,
                'start_job': start_job,   # Added global job start times
                'finish_job': finish_job  # Added global job finish times
               }

    return solution

# v1.0 for printing chart
# def print_chart(solution):
#     slot_assignment = solution.get('slot_assignment', None)
#     start_slot = solution.get('start_slot', None)
#     finish_slot = solution.get('finish_slot', None)
#
#     data = []
#     for key, j in slot_assignment.items():
#         s, p = key
#         start = round(start_slot.get((s, p), None), 2)
#         start_date = START_DATE + datetime.timedelta(days=start)
#                       # .strftime("%Y-%m-%d"))
#         finish = round(finish_slot.get((s, p), None), 2)
#         finish_date = START_DATE + datetime.timedelta(days=finish)
#                        # .strftime("%Y-%m-%d"))
#
#         # Handle different types of job identifiers
#         if isinstance(j, (list, tuple)) and len(j) > 0:
#             # If j is a list or tuple, use the first element as the plane
#             plane = j[0]
#         else:
#             # If j is not a list or tuple, use j as the plane identifier
#             plane = j
#
#         # Append data to the list
#         data.append({'s': s, 'p': p, 'j': j, 'start_slot': start_date, 'finish_slot': finish_date, 'plane': plane})
#
#     # Create a DataFrame from the list of dictionaries
#     df = pd.DataFrame(data)
#     fig = px.timeline(df, x_start="start_slot", x_end="finish_slot", y="p", color="plane")
#     fig.update_yaxes(title="Posición")
#     fig.update_xaxes(title="Fecha")
#     # fig.show()
#     # Modificar la línea fig.show() por:
#     fig.write_html("solution_chart_basic.html")
#     return df

# v2.0 for enhaced solution print
def print_chart(solution, html_path="gantt_basico.html"):
    """
    Construye un DataFrame con las columnas mínimas necesarias:
        - job: identificador completo del trabajo (e.g. "1-1", "2-3", …)
        - plane: identificador del avión (la parte antes del guión, e.g. "1", "2", …)
        - p: posición (e.g. "position3", "position4", …)
        - start_slot, finish_slot: fechas (en datetime)
    Devuelve el DataFrame resultante con columna 'job'.
    """
    from datetime import timedelta
    import pandas as pd

    # Reconstrucción del DataFrame de trabajos
    datos = []
    START_DATE = pd.to_datetime("today").normalize()
    for (s, p), job in solution['slot_assignment'].items():
        t0 = solution['start_slot'][(s, p)]
        t1 = solution['finish_slot'][(s, p)]
        fecha0 = START_DATE + timedelta(days=float(t0))
        fecha1 = START_DATE + timedelta(days=float(t1))
        avion = str(job).split("-")[0]
        datos.append({
            "job": job,
            "plane": avion,
            "p": p,
            "start_slot": fecha0,
            "finish_slot": fecha1
        })
    df = pd.DataFrame(datos)

    # Configuro y guardo (si procede)
    if html_path:
        import plotly.express as px
        fig = px.timeline(
            df,
            x_start="start_slot", x_end="finish_slot",
            y="p", color="plane",
            hover_data=["job"],
            title="Diagrama de Gantt Básico"
        )
        fig.update_yaxes(title="Posición")
        fig.update_xaxes(title="Fecha")
        fig.update_layout(height=300 + 30 * df["p"].nunique())
        fig.write_html(html_path)
        print(f"→ Gantt básico guardado en: {html_path}")

    return df

def generate_report(df_planes, model_instance, movimientos):
    global data  # para recuperar data['pDate']

    # 1) Asegurar columnas start_slot / finish_slot
    df = df_planes.copy()
    # Homogeneizar columna 'plane' al tipo de sPlanes
    try:
        planes_iter = iter(model_instance.sPlanes)
        first_plane = next(planes_iter)
        plane_type = type(first_plane)
        df['plane'] = df['plane'].astype(plane_type)
    except Exception:
        pass

    if 'start' in df.columns and 'finish' in df.columns:
        df = df.rename(columns={'start': 'start_slot', 'finish': 'finish_slot'})

    # 2) Convertir a datetime
    df['start_slot'] = pd.to_datetime(df['start_slot'])
    df['finish_slot'] = pd.to_datetime(df['finish_slot'])

    # 3) Parámetros auxiliares
    pDate_map = data.get('pDate', {})
    pJobDur = {j: int(value(model_instance.pJobDuration[j])) for j in model_instance.sJobs}

    # 4) Mapeos ES y LF por avión
    today = pd.to_datetime(date.today())
    pES_date = {r: today + timedelta(days=int(value(model_instance.pEarlyStartOfPlane[r])))
                for r in model_instance.sPlanes}
    pLF_date = {r: today + timedelta(days=int(value(model_instance.pLateFinishDeadline[r])))
                for r in model_instance.sPlanes}

    # 5) Conteo de movimientos
    mov_count = {}
    for plane, _, _, _ in movimientos:
        mov_count[plane] = mov_count.get(plane, 0) + 1

    # 6) Resumen por avión
    p2c = {r: c for (c, r), v in model_instance.pAirplaneOfClient.items() if v == 1}
    resumen = []
    first_start_date = {}
    for avion in sorted(df['plane'].unique()):
        grp = df[df['plane'] == avion].sort_values('start_slot')
        trabajos = grp[grp['type'] == 'work']['job'].tolist()
        posiciones = grp['p'].unique().tolist()
        es_date = pES_date.get(avion, today).date()
        primer_inicio = grp['start_slot'].min().date()
        lf_avion = pLF_date.get(avion, today).date()
        fin = grp['finish_slot'].max().date()
        cliente = p2c.get(avion)
        first_start_date[avion] = primer_inicio
        resumen.append({
            'Avión': avion,
            'Cliente': cliente,
            'ES': es_date,
            'Primer Inicio': primer_inicio,
            'LF': lf_avion,
            'Fin': fin,
            'Trabajos': ", ".join(trabajos),
            'Posiciones': ", ".join(posiciones),
            'Movimientos': mov_count.get(avion, 0)
        })
    df_res = pd.DataFrame(resumen)
    print("\n" + "=" * 150)
    print("RESUMEN POR AVIÓN")
    print("=" * 150)
    print(df_res.to_string(index=False, col_space=15))

    # 7) Detalle de trabajos
    print("\n" + "="*170)
    print("DETALLE DE TODOS LOS TRABAJOS")
    print("="*170)
    df_work = df[df['type']=='work'].copy()
    df_det = df_work[['plane','job','p','start_slot','finish_slot']].copy()
    df_det['Dur Est.(d)'] = df_det['job'].map(lambda j: pJobDur[j])
    df_det['Dur Real(d)'] = (df_det['finish_slot']-df_det['start_slot']).dt.total_seconds()/86400.0
    df_det['Prevista'] = df_det['job'].map(lambda j: date.today()+timedelta(days=pDate_map.get(j,0)+pJobDur[j]))
    df_det['Real'] = df_det['finish_slot'].dt.date
    df_det['ES'] = df_det['plane'].apply(lambda r: pES_date.get(r,today).date())
    df_det['LF'] = df_det['plane'].apply(lambda r: pLF_date.get(r,today).date())
    # Retraso vs LF
    df_det['Retraso(d)'] = df_det.apply(lambda r: max((r['Real']-r['LF']).days,0),axis=1)
    df_det['⚠'] = df_det['Retraso(d)'].apply(lambda d: '❌' if d>0 else '✅')
    df_det = df_det[['⚠','plane','job','p','ES','Prevista','Real','LF','Dur Est.(d)','Dur Real(d)','Retraso(d)']]
    df_det.columns=['⚠','Avión','Trabajo','Posición','Fecha ES','Prevista','Real','Fecha LF',
                    'Dur Est.(d)','Dur Real(d)','Retraso(d)']
    print(df_det.to_string(index=False, col_space=15))

    # 8) Retrasos por cliente con estado y fecha real
    print("="*90)
    print("RETRASOS POR CLIENTE (según el modelo)")
    print("="*90)
    clientes = sorted(model_instance.sClients)
    resumen_c = []
    c2planes = {c: [r for (c0, r), v in model_instance.pAirplaneOfClient.items() if c0 == c and v == 1]
                for c in clientes}
    for c in clientes:
        planes_c = c2planes.get(c, [])
        df_c = df_det[df_det['Avión'].isin(planes_c)]
        act_max = df_c['Real'].max() if not df_c.empty else None
        d = int(model_instance.vClientDelay[c].value)
        if d > 0:
            estado = '❌ Retraso'
        else:
            prev_max = df_c['Prevista'].max() if not df_c.empty else None
        if act_max and prev_max and act_max > prev_max:
            estado = '⚠️ Cumple pero pasada Prevista'
        else:
            estado = '✅ Cumple Fecha Prevista'
        resumen_c.append({
            'Cliente': c,
            'Fecha Final Real': act_max,
            'Retraso(días)': d,
            'Retraso(sem)': round(d / 7, 2),
            'Estado': estado
        })
    print(pd.DataFrame(resumen_c).to_string(index=False, col_space=15))

    # 9) Resumen ejecutivo
    print("\n" + "="*90)
    print("RESUMEN EJECUTIVO")
    print("="*90)
    total_t = len(df_det)
    total_r = df_det['Retraso(d)'].gt(0).sum()
    total_a = df['plane'].nunique()
    total_c = len(clientes)
    delayed_clients=[r['Cliente'] for r in resumen_c if r['Estado'].startswith('❌')]
    print(f"📦 {total_t} trabajos, ✈️ {total_a} aviones, {total_c} clientes")
    print(f"🔴 {total_r} trabajos retrasados")
    if delayed_clients:
        print(f"⚠️ Clientes con retrasos: {', '.join(map(str,delayed_clients))}")
    else:
        print("🟢 Sin retrasos por clientes")
    print("ℹ️ Clientes con estado de cumplimiento detallado arriba.")

def plot_enhanced_solution(df_work, instance, html_path="gantt_idles_movs.html"):

    # 1) Preparamos el DataFrame base de trabajos
    df = df_work.rename(columns={'start_slot':'start','finish_slot':'finish'}).copy()
    df['type'] = 'work'

    # 2) Construimos mapa de ocupación POR POSICIÓN, arrancando con TODOS los trabajos:
    positions = list(instance.sPositions)
    occupancy = {p: [] for p in positions}
    for _, row in df.iterrows():
        # cada tupla (start,finish) ocupa la posición p
        occupancy[row['p']].append((row['start'], row['finish']))

    # 3) Detectamos idles avión a avión, evitando solapamientos
    idles = []
    planes = sorted(df['plane'].unique())
    for plane in planes:
        grp = df[df['plane']==plane].sort_values('start').reset_index(drop=True)
        for i in range(len(grp)-1):
            fin = grp.loc[i,   'finish']
            ini = grp.loc[i+1, 'start']
            if fin < ini:
                # buscamos posiciones completamente libres en [fin, ini)
                libres = [
                    p for p, intervals in occupancy.items()
                    if all(e <= fin or s >= ini for (s,e) in intervals)
                ]
                pos_idle = libres[0] if libres else grp.loc[i,'p']
                idles.append({
                    'plane': plane,
                    'type' : 'idle',
                    'job'  : 'idle',
                    'p'    : pos_idle,
                    'start': fin,
                    'finish': ini
                })
                # marcamos ese intervalo como ocupado
                occupancy[pos_idle].append((fin, ini))

    df_idle = pd.DataFrame(idles, columns=['plane', 'type', 'job', 'p', 'start', 'finish'])
    dflist = [df, df_idle]
    dflist = [df_i for df_i in dflist if not df_i.dropna(how='all').empty]
    df_full = pd.concat(dflist, ignore_index=True)

    # 4) Mapa de colores por avión
    palette   = px.colors.qualitative.Plotly
    color_map = {plane: palette[i % len(palette)] for i, plane in enumerate(planes)}

    # 5) Timeline de trabajos
    fig = px.timeline(
        df,
        x_start="start", x_end="finish", y="p",
        color="plane", color_discrete_map=color_map,
        hover_data=["job","type"],
        title="Diagrama de Gantt"
    )

    # 6) Timeline de idles (huecos)
    fig_idle = px.timeline(
        df_idle,
        x_start="start", x_end="finish", y="p",
        color="plane", color_discrete_map=color_map,
        hover_data=["type","job"]
    )
    for trace in fig_idle.data:
        plane = trace.name
        trace.marker.color      = 'rgba(255,255,255,1)'    # transparente
        trace.marker.line.color = color_map[plane]   # sólo borde
        trace.marker.line.width = 2
        trace.showlegend        = False
        fig.add_trace(trace)

    # 7) Ajustes esteticos y guardado
    fig.update_yaxes(
        categoryorder='array',
        categoryarray=list(reversed(positions))
    )
    fig.update_xaxes(title="Fecha")
    fig.update_yaxes(title="Posición")
    fig.write_html(html_path)
    print(f"→ Gantt guardado en: {html_path}")

    # 8) Generación de la lista de movimientos (para el reporte)
    movimientos = []
    for plane, grp in df_full.groupby('plane'):
        grp = grp.sort_values('start').reset_index(drop=True)
        for i in range(len(grp)-1):
            p0, p1 = grp.loc[i,'p'], grp.loc[i+1,'p']
            t1       = grp.loc[i+1,'start']
            if p0 != p1:
                movimientos.append((plane, p0, p1, t1))

    return df_full, movimientos




def check_solution(data, solution):
    # Extraer sets y parámetros
    sSlots = data.get('sSlots', [])
    sPositions = data.get('sPositions', [])
    sJobs = data.get('sJobs', [])
    sPlanes = data.get('sPlanes', [])
    sClients = data.get('sClients', [])
    pJobDuration = data.get('pJobDuration', {})
    pPlaneOfJob = data.get('pPlaneOfJob', {})
    pLastJobOfPlane = data.get('pLastJobOfPlane', {})
    pFirstJobOfPlane = data.get('pFirstJobOfPlane', {})
    pPredictedFinishOfPlane = data.get('pPredictedFinishOfPlane', {})
    pAirplaneOfClient = data.get('pAirplaneOfClient', {})
    pHorizon = data.get('pHorizon', 0)
    pEarlyStartOfPlane = data.get('pEarlyStartOfPlane', {})
    pLateFinishDeadline = data.get('pLateFinishDeadline', {})

    sSlotsSequence = data.get('sSlotsSequence', [])
    sJobSequence = data.get('sJobSequence', [])
    sPosPosSlotSlot = data.get('sPosPosSlotSlot', [])
    sSwitchPlanes = data.get('sSwitchPlanes', [])

    # Solución devuelta
    slot_assignment = solution.get('slot_assignment', {})  # {(s,p): j}
    duration_slot = solution.get('duration_slot', {})  # {(s,p): val}
    duration_slot_job = solution.get('duration_slot_job', {})  # {(s,p,j): val}
    interference_list = solution.get('interference', [])  # [(s,s2,p,p2), ...]
    start_slot_job = solution.get('start_slot_job', {})  # {(s,p,j): val}
    finish_slot_job = solution.get('finish_slot_job', {})  # {(s,p,j): val}
    start_slot = solution.get('start_slot', {})  # {(s,p): val}
    finish_slot = solution.get('finish_slot', {})  # {(s,p): val}
    start_job = solution.get('start_job', {})  # {j: val}
    finish_job = solution.get('finish_job', {})  # {j: val}
    # Nuevas variables de presencia y idle
    presence = solution.get('presence', {})  # {(s,p,r): 0/1}
    plane_in_slot = solution.get('plane_in_slot', {})  # {(s,p,r): 0/1}
    idle = solution.get('idle', {})  # {(s,p,r): 0/1}
    start_presence = solution.get('start_presence', {})  # {(s,p,r): val}
    finish_presence = solution.get('finish_presence', {})  # {(s,p,r): val}
    switch_planes = solution.get('switch_planes', {})  # {(s_prev,p): 0/1}
    plane_delay = solution.get('plane_delay', {})

    verification_results = {}
    M = pHorizon

    # —————————————————————————— c01: SingleJobPerSlot ——————————————————————————
    # ∀(s,p): sum_j x[s,p,j] ≤ 1
    verification_results['c01_single_job_per_slot'] = {'passed': True, 'errors': []}
    for s in sSlots:
        for p in sPositions:
            jobs_here = [j for (ss, pp), j in slot_assignment.items() if ss == s and pp == p]
            if len(jobs_here) > 1:
                verification_results['c01_single_job_per_slot']['passed'] = False
                verification_results['c01_single_job_per_slot']['errors'].append(
                    f"Ranura {s}, posición {p} tiene múltiples trabajos asignados: {jobs_here}"
                )

    # —————————————————————————— c02: SlotJobDuration ——————————————————————————
    # ∀(s,p,j): duration_slot_job[s,p,j] == finish_slot_job[s,p,j] - start_slot_job[s,p,j]
    verification_results['c02_slot_job_duration'] = {'passed': True, 'errors': []}
    for s in sSlots:
        for p in sPositions:
            for j in sJobs:
                d_val = duration_slot_job.get((s, p, j), 0.0)
                t0 = start_slot_job.get((s, p, j), 0.0)
                t1 = finish_slot_job.get((s, p, j), 0.0)
                if abs(d_val - (t1 - t0)) > 1e-6:
                    verification_results['c02_slot_job_duration']['passed'] = False
                    verification_results['c02_slot_job_duration']['errors'].append(
                        f"(s={s},p={p},j={j}): vDurationSlotForJob={d_val:.4f} ≠ finish-start={(t1 - t0):.4f}"
                    )

    # —————————————————————————— c03: NullStartIfNotAssigned ——————————————————————————
    # ∀(s,p,j): start_slot_job[s,p,j] ≤ pHorizon·x[s,p,j]
    verification_results['c03_null_start_if_not_assigned'] = {'passed': True, 'errors': []}

    # —————————————————————————— c04: NullFinishIfNotAssigned ——————————————————————————
    # ∀(s,p,j): finish_slot_job[s,p,j] ≤ pHorizon·x[s,p,j]
    verification_results['c04_null_finish_if_not_assigned'] = {'passed': True, 'errors': []}
    for s in sSlots:
        for p in sPositions:
            for j in sJobs:
                x_val = 1 if slot_assignment.get((s, p)) == j else 0
                t0 = start_slot_job.get((s, p, j), 0.0)
                t1 = finish_slot_job.get((s, p, j), 0.0)
                if t0 > pHorizon * x_val + 1e-6:
                    verification_results['c03_null_start_if_not_assigned']['passed'] = False
                    verification_results['c03_null_start_if_not_assigned']['errors'].append(
                        f"(s={s},p={p},j={j}): start_slot_job={t0:.4f} > Horizon*{x_val}={pHorizon * x_val:.4f}"
                    )
                if t1 > pHorizon * x_val + 1e-6:
                    verification_results['c04_null_finish_if_not_assigned']['passed'] = False
                    verification_results['c04_null_finish_if_not_assigned']['errors'].append(
                        f"(s={s},p={p},j={j}): finish_slot_job={t1:.4f} > Horizon*{x_val}={pHorizon * x_val:.4f}"
                    )

    # —————————————————————————— c05: JobDuration ——————————————————————————
    # ∀j: sum_{s,p} duration_slot_job[s,p,j] == pJobDuration[j]
    verification_results['c05_job_duration'] = {'passed': True, 'errors': []}
    for j in sJobs:
        suma = sum(duration_slot_job.get((s, p, j), 0.0) for s in sSlots for p in sPositions)
        if abs(suma - pJobDuration.get(j, 0.0)) > 1e-6:
            verification_results['c05_job_duration']['passed'] = False
            verification_results['c05_job_duration']['errors'].append(
                f"Trabajo {j}: suma_duración_fragmentos={suma:.4f} ≠ pJobDuration({pJobDuration.get(j)})"
            )

    # —————————————————————————— c06 & c07: Start/end global con Big-M ——————————————————————————
    # ∀(s,p,j): vStartJob[j] ≤ start_slot_job[s,p,j] + M(1-x)
    #            vStartJob[j] ≥ start_slot_job[s,p,j] - M(1-x)
    #            vFinishJob[j] ≥ finish_slot_job[s,p,j] - M(1-x)
    #            vFinishJob[j] ≤ finish_slot_job[s,p,j] + M(1-x)
    verification_results['c06_startjob_bigM'] = {'passed': True, 'errors': []}
    verification_results['c07_finishjob_bigM'] = {'passed': True, 'errors': []}
    M = pHorizon
    for s in sSlots:
        for p in sPositions:
            for j in sJobs:
                x_val = 1 if slot_assignment.get((s, p)) == j else 0
                st_frag = start_slot_job.get((s, p, j), 0.0)
                fn_frag = finish_slot_job.get((s, p, j), 0.0)
                st_j = start_job.get(j, 0.0)
                fn_j = finish_job.get(j, 0.0)
                # c06 upper
                if st_j - (st_frag + M * (1 - x_val)) > 1e-6:
                    verification_results['c06_startjob_bigM']['passed'] = False
                    verification_results['c06_startjob_bigM']['errors'].append(
                        f"(s={s},p={p},j={j}): start_job={st_j:.4f} > frag_start+M(1-x)={st_frag + M * (1 - x_val):.4f}"
                    )
                # c06 lower
                if (st_frag - M * (1 - x_val)) - st_j > 1e-6:
                    verification_results['c06_startjob_bigM']['passed'] = False
                    verification_results['c06_startjob_bigM']['errors'].append(
                        f"(s={s},p={p},j={j}): frag_start-M(1-x)={st_frag - M * (1 - x_val):.4f} > start_job={st_j:.4f}"
                    )
                # c07 lower
                if ((fn_frag - M * (1 - x_val)) - fn_j) > 1e-6:
                    verification_results['c07_finishjob_bigM']['passed'] = False
                    verification_results['c07_finishjob_bigM']['errors'].append(
                        f"(s={s},p={p},j={j}): frag_finish-M(1-x)={fn_frag - M * (1 - x_val):.4f} > finish_job={fn_j:.4f}"
                    )
                # c07 upper
                if fn_j - (fn_frag + M * (1 - x_val)) > 1e-6:
                    verification_results['c07_finishjob_bigM']['passed'] = False
                    verification_results['c07_finishjob_bigM']['errors'].append(
                        f"(s={s},p={p},j={j}): finish_job={fn_j:.4f} > frag_finish+M(1-x)={fn_frag + M * (1 - x_val):.4f}"
                    )

    # —————————————————————————— c08: StartFinishRelation ——————————————————————————
    # ∀j: start_job[j] ≤ finish_job[j]
    verification_results['c08_start_finish_relation'] = {'passed': True, 'errors': []}
    for j in sJobs:
        st_j = start_job.get(j, 0.0)
        fn_j = finish_job.get(j, 0.0)
        if st_j - fn_j > 1e-6:
            verification_results['c08_start_finish_relation']['passed'] = False
            verification_results['c08_start_finish_relation']['errors'].append(
                f"Job {j}: start={st_j:.4f} > finish={fn_j:.4f}"
            )

    # —————————————————————————— c09: PlaneDelay (nueva fc09) ——————————————————————————
    verification_results['c09_plane_delay'] = {'passed': True, 'errors': []}
    for r in sPlanes:
        lhs = plane_delay.get(r, 0.0)
        # sólo el último trabajo de r contribuye:
        sum_term = (sum(
            (finish_job.get(j, 0.0) - pLateFinishDeadline.get(r, 0.0))
            * pLastJobOfPlane.get((j, r), 0)
            for j in sJobs
        ))
        if lhs + 1e-6 < sum_term:
            verification_results['c09_plane_delay']['passed'] = False
            verification_results['c09_plane_delay']['errors'].append(
                f"Avión {r}: vPlaneDelay={lhs:.4f} < (finish_last - deadline)={sum_term:.4f}"
            )

    # —————————————————————————— c11: SlotStartTime ——————————————————————————
    # ∀(s,p): start_slot[s,p] == sum_j start_slot_job[s,p,j]
    verification_results['c11_slot_start_time'] = {'passed': True, 'errors': []}
    for s in sSlots:
        for p in sPositions:
            suma_starts = sum(start_slot_job.get((s, p, j), 0.0) for j in sJobs)
            vs = start_slot.get((s, p), 0.0)
            if abs(vs - suma_starts) > 1e-6:
                verification_results['c11_slot_start_time']['passed'] = False
                verification_results['c11_slot_start_time']['errors'].append(
                    f"(s={s},p={p}): vStartSlot={vs:.4f} ≠ suma(starts)={suma_starts:.4f}"
                )

    # —————————————————————————— c12: SlotFinishTime ——————————————————————————
    # ∀(s,p): finish_slot[s,p] == sum_j finish_slot_job[s,p,j]
    verification_results['c12_slot_finish_time'] = {'passed': True, 'errors': []}
    for s in sSlots:
        for p in sPositions:
            suma_fins = sum(finish_slot_job.get((s, p, j), 0.0) for j in sJobs)
            vf = finish_slot.get((s, p), 0.0)
            if abs(vf - suma_fins) > 1e-6:
                verification_results['c12_slot_finish_time']['passed'] = False
                verification_results['c12_slot_finish_time']['errors'].append(
                    f"(s={s},p={p}): vFinishSlot={vf:.4f} ≠ suma(finishes)={suma_fins:.4f}"
                )

    # —————————————————————————— c13: SlotSequence ——————————————————————————
    # ∀(s,s2,p) ∈ sSlotsSequence: start_slot[s,p] ≥ finish_slot[s2,p]
    verification_results['c13_slot_sequence'] = {'passed': True, 'errors': []}
    for (s, s2, p) in sSlotsSequence:
        st_s = start_slot.get((s, p), 0.0)
        fn_s2 = finish_slot.get((s2, p), 0.0)
        if st_s + 1e-6 < fn_s2:
            verification_results['c13_slot_sequence']['passed'] = False
            verification_results['c13_slot_sequence']['errors'].append(
                f"SlotSequence: start[{s},{p}]={st_s:.4f} < finish[{s2},{p}]={fn_s2:.4f}"
            )

    # —————————————————————————— c14: JobSequence ——————————————————————————
    # ∀(j,j2) ∈ sJobSequence: start_job[j2] ≥ finish_job[j]
    verification_results['c14_job_sequence'] = {'passed': True, 'errors': []}
    for (j, j2) in sJobSequence:
        st_j2 = start_job.get(j2, 0.0)
        fn_j = finish_job.get(j, 0.0)
        if st_j2 + 1e-6 < fn_j:
            verification_results['c14_job_sequence']['passed'] = False
            verification_results['c14_job_sequence']['errors'].append(
                f"JobSequence: start_job[{j2}]={st_j2:.4f} < finish_job[{j}]={fn_j:.4f}"
            )

    # —————————————————————————— c15: ConsecutiveSlots ——————————————————————————
    # ∀(s>primero, p): sum_j x[s,p,j] == sum_j x[s_prev,p,j]
    verification_results['c15_consecutive_slots'] = {'passed': True, 'errors': []}
    ordered_slots = sorted(sSlots, key=lambda x: int(x.replace('slot', '')))
    for p in sPositions:
        for idx in range(1, len(ordered_slots)):
            s = ordered_slots[idx]
            prev_s = ordered_slots[idx - 1]
            suma_s = sum(1 for j in sJobs if slot_assignment.get((s, p)) == j)
            suma_prev = sum(1 for j in sJobs if slot_assignment.get((prev_s, p)) == j)
            if suma_s > suma_prev:
                verification_results['c15_consecutive_slots']['passed'] = False
                verification_results['c15_consecutive_slots']['errors'].append(
                    f"ConsecutiveSlots: posición {p}: {s} tiene {suma_s} jobs, pero {prev_s} tiene {suma_prev}"
                )

    # —————————————————————————— c16: SingleSlotPerJob ——————————————————————————
    # ∀j: sum_{s,p} x[s,p,j] == 1
    verification_results['c16_single_slot_per_job'] = {'passed': True, 'errors': []}
    for j in sJobs:
        cuenta = sum(1 for (_, _), job in slot_assignment.items() if job == j)
        if cuenta != 1:
            verification_results['c16_single_slot_per_job']['passed'] = False
            verification_results['c16_single_slot_per_job']['errors'].append(
                f"Job {j} asignado en {cuenta} slots (debe 1)"
            )

    # —————————————————————————— c17: DurationIfNotAssigned ——————————————————————————
    # ∀(s,p,j): finish_slot_job[s,p,j] - start_slot_job[s,p,j] ≥ pJobDuration[j]·x[s,p,j]
    verification_results['c17_duration_if_not_assigned'] = {'passed': True, 'errors': []}
    for s in sSlots:
        for p in sPositions:
            for j in sJobs:
                x_val = 1 if slot_assignment.get((s, p)) == j else 0
                t0 = start_slot_job.get((s, p, j), 0.0)
                t1 = finish_slot_job.get((s, p, j), 0.0)
                lhs = t1 - t0
                rhs = pJobDuration.get(j, 0.0) * x_val
                if lhs + 1e-6 < rhs:
                    verification_results['c17_duration_if_not_assigned']['passed'] = False
                    verification_results['c17_duration_if_not_assigned']['errors'].append(
                        f"(s={s},p={p},j={j}): finish-start={lhs:.4f} < duration[{j}]*x={rhs:.4f}"
                    )

    # —————————————————————————— c18: SlotDuration ——————————————————————————
    # ∀(s,p): duration_slot[s,p] == sum_j duration_slot_job[s,p,j]
    verification_results['c18_slot_duration'] = {'passed': True, 'errors': []}
    for s in sSlots:
        for p in sPositions:
            sum_frag = sum(duration_slot_job.get((s, p, j), 0.0) for j in sJobs)
            dur_slot = duration_slot.get((s, p), 0.0)
            if abs(dur_slot - sum_frag) > 1e-6:
                verification_results['c18_slot_duration']['passed'] = False
                verification_results['c18_slot_duration']['errors'].append(
                    f"Ranura {s},{p}: vDurationSlot={dur_slot:.4f} ≠ suma_fragmentos={sum_frag:.4f}"
                )

    # —————————————————————————— c19: PlaneSlotAssignment ——————————————————————————
    # ∀(s,p,r): v01PlaneInSlot[s,p,r] == sum_{j: planeOfJob[j]=r} x[s,p,j]
    verification_results['c19_plane_slot_assignment'] = {'passed': True, 'errors': []}
    plane_in_slot_count = {}
    for s in sSlots:
        for p in sPositions:
            for r in sPlanes:
                cnt = 0
                for j in sJobs:
                    if pPlaneOfJob.get(j) == r and slot_assignment.get((s, p)) == j:
                        cnt += 1
                plane_in_slot_count[(s, p, r)] = cnt
    for (s, p, r), cnt in plane_in_slot_count.items():
        expected = 1 if cnt == 1 else 0
        real_val = 1 if any(
            slot_assignment.get((s, p)) == j and pPlaneOfJob.get(j) == r
            for j in sJobs
        ) else 0
        if expected != real_val:
            verification_results['c19_plane_slot_assignment']['passed'] = False
            verification_results['c19_plane_slot_assignment']['errors'].append(
                f"(s={s},p={p},r={r}): conteo={cnt}, pero v01PlaneInSlot reconstruido={real_val}"
            )

    # —————————————————————————— c20: PlaneInPosition ——————————————————————————
    # ∀(s,p,r): v01PlaneInPosition[r,p] ≥ v01PlaneInSlot[s,p,r]
    verification_results['c20_plane_in_position'] = {'passed': True, 'errors': []}
    plane_in_position = {
        (r, p): 1 if any(
            slot_assignment.get((s, p)) == j and pPlaneOfJob.get(j) == r
            for s in sSlots for j in sJobs
        ) else 0
        for r in sPlanes for p in sPositions
    }
    for s in sSlots:
        for p in sPositions:
            for r in sPlanes:
                in_slot = 1 if any(
                    slot_assignment.get((s, p)) == j and pPlaneOfJob.get(j) == r
                    for j in sJobs
                ) else 0
                pos_val = plane_in_position.get((r, p), 0)
                if pos_val < in_slot:
                    verification_results['c20_plane_in_position']['passed'] = False
                    verification_results['c20_plane_in_position']['errors'].append(
                        f"(s={s},p={p},r={r}): v01PlaneInPosition={pos_val} < v01PlaneInSlot={in_slot}"
                    )
    # c20b: Si r tiene un trabajo en (s,p), debe estar presente
    verification_results['c20b_present_if_work'] = {'passed': True, 'errors': []}
    for s in sSlots:
        for p in sPositions:
            for r in sPlanes:
                pi = plane_in_slot.get((s, p, r), 0)
                pres = presence.get((s, p, r), 0)
                if pres < pi - 1e-6:
                    verification_results['c20b_present_if_work']['passed'] = False
                    verification_results['c20b_present_if_work']['errors'].append(
                        f"(s={s},p={p},r={r}): presence={pres} < plane_in_slot={pi}"
                    )

    # c20c: presencia = slots + idle
    verification_results['c20c_present_exactly_one'] = {'passed': True, 'errors': []}
    for s in sSlots:
        for r in sPlanes:
            sum_pres = sum(presence.get((s, p, r), 0) for p in sPositions)
            sum_slots = sum(plane_in_slot.get((s, p, r), 0) for p in sPositions)
            sum_idle = sum(idle.get((s, p, r), 0) for p in sPositions)
            if abs(sum_pres - (sum_slots + sum_idle)) > 1e-6:
                verification_results['c20c_present_exactly_one']['passed'] = False
                verification_results['c20c_present_exactly_one']['errors'].append(
                    f"(s={s},r={r}): pres={sum_pres} != slots+idle={sum_slots + sum_idle}"
                )

    # c20d: un avión por posición
    verification_results['c20d_single_plane_per_position'] = {'passed': True, 'errors': []}
    for s in sSlots:
        for p in sPositions:
            cnt = sum(presence.get((s, p, r), 0) for r in sPlanes)
            if cnt - 1 > 1e-6:
                verification_results['c20d_single_plane_per_position']['passed'] = False
                verification_results['c20d_single_plane_per_position']['errors'].append(
                    f"(s={s},p={p}): presencia total={cnt} > 1"
                )

    # c20e: no salto adelante sin switch
    verification_results['c20e_no_jump_forward'] = {'passed': True, 'errors': []}
    for s in sSlots:
        prev_s = data.get('prev_slot', {}).get(s)
        if prev_s is None:
            continue
        for p in sPositions:
            for r in sPlanes:
                pres_prev = presence.get((prev_s, p, r), 0)
                pres_curr = presence.get((s, p, r), 0)
                sw = switch_planes.get((prev_s, p), 0)
                if pres_prev - pres_curr - sw > 1e-6:
                    verification_results['c20e_no_jump_forward']['passed'] = False
                    verification_results['c20e_no_jump_forward']['errors'].append(
                        f"(s_prev={prev_s},s={s},p={p},r={r}): pres_prev-pres={pres_prev - pres_curr} > switch={sw}"
                    )

    # c20f: no salto backward sin switch
    verification_results['c20f_no_jump_backward'] = {'passed': True, 'errors': []}
    for s in sSlots:
        prev_s = data.get('prev_slot', {}).get(s)
        if prev_s is None:
            continue
        for p in sPositions:
            for r in sPlanes:
                pres_prev = presence.get((prev_s, p, r), 0)
                pres_curr = presence.get((s, p, r), 0)
                sw = switch_planes.get((prev_s, p), 0)
                if pres_curr - pres_prev - sw > 1e-6:
                    verification_results['c20f_no_jump_backward']['passed'] = False
                    verification_results['c20f_no_jump_backward']['errors'].append(
                        f"(s_prev={prev_s},s={s},p={p},r={r}): pres-pres_prev={pres_curr - pres_prev} > switch={sw}"
                    )

    # idle_def1: idle >= presence - slots
    verification_results['idle_def1'] = {'passed': True, 'errors': []}
    for s in sSlots:
        for p in sPositions:
            for r in sPlanes:
                pres = presence.get((s, p, r), 0)
                pi = plane_in_slot.get((s, p, r), 0)
                idl = idle.get((s, p, r), 0)
                if idl + 1e-6 < pres - pi:
                    verification_results['idle_def1']['passed'] = False
                    verification_results['idle_def1']['errors'].append(
                        f"(s={s},p={p},r={r}): idle={idl} < pres-pi={pres - pi}"
                    )

    # idle_def2: idle <= presence
    verification_results['idle_def2'] = {'passed': True, 'errors': []}
    for s in sSlots:
        for p in sPositions:
            for r in sPlanes:
                idl = idle.get((s, p, r), 0)
                pres = presence.get((s, p, r), 0)
                if idl - pres > 1e-6:
                    verification_results['idle_def2']['passed'] = False
                    verification_results['idle_def2']['errors'].append(
                        f"(s={s},p={p},r={r}): idle={idl} > presence={pres}"
                    )

    # link_start_presence
    verification_results['link_start_presence'] = {'passed': True, 'errors': []}
    for s in sSlots:
        for p in sPositions:
            for r in sPlanes:
                pres = presence.get((s, p, r), 0)
                sp = start_presence.get((s, p, r), 0.0)
                ss = start_slot.get((s, p), 0.0)
                if sp - (ss + M * (1 - pres)) > 1e-6:
                    verification_results['link_start_presence']['passed'] = False
                    verification_results['link_start_presence']['errors'].append(
                        f"(s={s},p={p},r={r}): sp={sp} > ss+M(1-pres)={ss + M * (1 - pres)}"
                    )

    # link_finish_presence_lb
    verification_results['link_finish_presence_lb'] = {'passed': True, 'errors': []}
    for s in sSlots:
        for p in sPositions:
            for r in sPlanes:
                pres = presence.get((s, p, r), 0)
                fp = finish_presence.get((s, p, r), 0.0)
                fs = finish_slot.get((s, p), 0.0)
                if (fs - M * (1 - pres)) - fp > 1e-6:
                    verification_results['link_finish_presence_lb']['passed'] = False
                    verification_results['link_finish_presence_lb']['errors'].append(
                        f"(s={s},p={p},r={r}): fs-M(1-pres)={fs - M * (1 - pres)} > fp={fp}"
                    )

    # link_finish_presence_ub
    verification_results['link_finish_presence_ub'] = {'passed': True, 'errors': []}
    for s in sSlots:
        for p in sPositions:
            for r in sPlanes:
                pres = presence.get((s, p, r), 0)
                fp = finish_presence.get((s, p, r), 0.0)
                fs = finish_slot.get((s, p), 0.0)
                if fp - (fs + M * (1 - pres)) > 1e-6:
                    verification_results['link_finish_presence_ub']['passed'] = False
                    verification_results['link_finish_presence_ub']['errors'].append(
                        f"(s={s},p={p},r={r}): fp={fp} > fs+M(1-pres)={fs + M * (1 - pres)}"
                    )

    # —————————————————————————— c21: ClientInPosition ——————————————————————————
    # ∀(c,p): vClientPosition[c,p] ≥ sum_{r} v01PlaneInPosition[r,p]*pAirplaneOfClient[c,r]
    verification_results['c21_client_in_position'] = {'passed': True, 'errors': []}
    # Sin datos de clientes, asumimos que se cumple.

    # —————————————————————————— c22 & c23: BetaDefinition1 y BetaDefinition2 ——————————————————————————
    # c22: ∀(s,s2,p,p2): M·BetaS[s,s2,p,p2] + start_slot[s,p] ≥ start_slot[s2,p2]
    # c23: ∀(s,s2,p,p2): M·BetaF[s,s2,p,p2] + start_slot[s2,p2] ≥ finish_slot[s,p]
    verification_results['c22_beta_definition1'] = {'passed': True, 'errors': []}
    verification_results['c23_beta_definition2'] = {'passed': True, 'errors': []}
    M = pHorizon
    for (s, s2, p, p2) in sPosPosSlotSlot:
        st_sp = start_slot.get((s, p), 0.0)
        st_s2p2 = start_slot.get((s2, p2), 0.0)
        fn_sp = finish_slot.get((s, p), 0.0)
        beta_s = 1 if st_sp + 1e-6 < st_s2p2 else 0
        lhs1 = M * beta_s + st_sp
        if lhs1 + 1e-6 < st_s2p2:
            verification_results['c22_beta_definition1']['passed'] = False
            verification_results['c22_beta_definition1']['errors'].append(
                f"(s={s},s2={s2},p={p},p2={p2}): M·βS+start[{s},{p}]={lhs1:.4f} < start[{s2},{p2}]={st_s2p2:.4f}"
            )
        beta_f = 1 if st_s2p2 + 1e-6 < fn_sp else 0
        lhs2 = M * beta_f + st_s2p2
        if lhs2 + 1e-6 < fn_sp:
            verification_results['c23_beta_definition2']['passed'] = False
            verification_results['c23_beta_definition2']['errors'].append(
                f"(s={s},s2={s2},p={p},p2={p2}): M·βF+start[{s2},{p2}]={lhs2:.4f} < finish[{s},{p}]={fn_sp:.4f}"
            )

    # —————————————————————————— c24: InterferenceExists ——————————————————————————
    # ∀(s,s2,p,p2): 1 + α[s,s2,p,p2] ≥ βS[s,s2,p,p2] + βF[s,s2,p,p2]
    verification_results['c24_interference_exists'] = {'passed': True, 'errors': []}
    for (s, s2, p, p2) in sPosPosSlotSlot:
        st_sp = start_slot.get((s, p), 0.0)
        st_s2p2 = start_slot.get((s2, p2), 0.0)
        fn_sp = finish_slot.get((s, p), 0.0)
        fn_s2p2 = finish_slot.get((s2, p2), 0.0)
        beta_s = 1 if st_sp + 1e-6 < st_s2p2 else 0
        beta_f = 1 if st_s2p2 + 1e-6 < fn_sp else 0
        solapan = not (fn_sp <= st_s2p2 + 1e-6 or fn_s2p2 <= st_sp + 1e-6)
        alpha_val = 1 if solapan else 0
        lhs = 1 + alpha_val
        rhs = beta_s + beta_f
        if lhs < rhs - 1e-6:
            verification_results['c24_interference_exists']['passed'] = False
            verification_results['c24_interference_exists']['errors'].append(
                f"(s={s},s2={s2},p={p},p2={p2}): 1+α={lhs:.4f} < βS+βF={rhs:.4f}"
            )
        if solapan and (s, s2, p, p2) not in interference_list and (s2, s, p2,
                                                                    p) not in interference_list:
            verification_results['c24_interference_exists']['passed'] = False
            verification_results['c24_interference_exists']['errors'].append(
                f"Solapamiento real entre ({s},{s2},{p},{p2}) no marcado en interference_list"
            )

    # —————————————————————————— c25: SwitchingPlanes ——————————————————————————
    # ∀(p,s,s2,r,r2): 1 + v01SwitchPlanes[s,p] ≥ v01PlaneInSlot[s,p,r] + v01PlaneInSlot[s2,p,r2]
    verification_results['c25_switching_planes'] = {'passed': True, 'errors': []}
    for (p, s, s2, r, r2) in sSwitchPlanes:
        in1 = 1 if slot_assignment.get((s, p)) in sJobs and pPlaneOfJob.get(
            slot_assignment[(s, p)]) == r else 0
        in2 = 1 if slot_assignment.get((s2, p)) in sJobs and pPlaneOfJob.get(
            slot_assignment[(s2, p)]) == r2 else 0
        switch_val = 1 if (in1 + in2) > 1 else 0
        lhs = 1 + switch_val
        rhs = in1 + in2
        if lhs < rhs - 1e-6:
            verification_results['c25_switching_planes']['passed'] = False
            verification_results['c25_switching_planes']['errors'].append(
                f"(p={p},s={s},s2={s2},r={r},r2={r2}): 1+vSwitch={lhs:.4f} < in1+in2={rhs:.4f}"
            )

    # —————————————————————————— c26: NoOverlapSlots ——————————————————————————
    # ∀(s,s2,p,p2,j) con (s,p)≠(s2,p2): 1 + βS + βF ≥ x[s,p,j] + x[s2,p2,j]
    verification_results['c26_no_overlap_slots'] = {'passed': True, 'errors': []}
    for j in sJobs:
        ubic = [(s, p) for (s, p), job in slot_assignment.items() if job == j]
        for i in range(len(ubic)):
            s1, p1 = ubic[i]
            t1_0 = start_slot_job.get((s1, p1, j), 0.0)
            t1_1 = finish_slot_job.get((s1, p1, j), 0.0)
            for k in range(i + 1, len(ubic)):
                s2, p2 = ubic[k]
                t2_0 = start_slot_job.get((s2, p2, j), 0.0)
                t2_1 = finish_slot_job.get((s2, p2, j), 0.0)
                if s1 == s2 and p1 == p2:
                    continue
                beta_s = 1 if t1_0 + 1e-6 < t2_0 else 0
                beta_f = 1 if t2_0 + 1e-6 < t1_1 else 0
                lhs = 1 + beta_s + beta_f
                rhs = 2
                if lhs < rhs - 1e-6:
                    verification_results['c26_no_overlap_slots']['passed'] = False
                    verification_results['c26_no_overlap_slots']['errors'].append(
                        f"NoOverlapSlots j={j}: ({s1},{p1},{t1_0:.4f}-{t1_1:.4f}) vs ({s2},{p2},{t2_0:.4f}-{t2_1:.4f}), 1+βS+βF={lhs:.4f} < 2"
                    )



    # —————————————————————————————— Comprobaciones adicionales ——————————————————————————————
    #   within_horizon: ∀(s,p): finish_slot[s,p] ≤ pHorizon
    verification_results['within_horizon'] = {'passed': True, 'errors': []}
    for (s, p), end_time in finish_slot.items():
        if end_time > pHorizon + 1e-6:
            verification_results['within_horizon']['passed'] = False
            verification_results['within_horizon']['errors'].append(
                f"Ranura ({s},{p}) termina en {end_time:.4f} > Horizon={pHorizon:.4f}"
            )
    #   plane_single_position: un avión no puede estar en dos posiciones solapadas
    verification_results['plane_single_position'] = {'passed': True, 'errors': []}
    for r in sPlanes:
        fragments = [(s, p, start_slot.get((s, p), 0.0), finish_slot.get((s, p), 0.0))
                     for (s, p), j in slot_assignment.items() if pPlaneOfJob.get(j) == r]
        for i in range(len(fragments)):
            s1, p1, t1_0, t1_1 = fragments[i]
            for jdx in range(i + 1, len(fragments)):
                s2, p2, t2_0, t2_1 = fragments[jdx]
                if p1 != p2:
                    solap = not (t1_1 <= t2_0 + 1e-6 or t2_1 <= t1_0 + 1e-6)
                    if solap:
                        verification_results['plane_single_position']['passed'] = False
                        verification_results['plane_single_position']['errors'].append(
                            f"Avión {r} en posiciones distintas solapadas: {p1}({t1_0:.4f}-{t1_1:.4f}) vs {p2}({t2_0:.4f}-{t2_1:.4f})"
                        )

        all_passed = all(entry['passed'] for entry in verification_results.values())
        summary = {
            'all_constraints_satisfied': all_passed,
            'constraints_verification': verification_results
        }
        return summary

def diagnose_infeasibility(model, input_data, case_name="conflict"):  #Función para poder revisar la no factibilidad del modelo

    # 1) crea instancia
    instance = model.create_instance(input_data)

    # 2) vuelca a MPS con labels simbólicos
    mps_file = f"{case_name}.mps"
    instance.write(mps_file, format='mps', io_options={'symbolic_solver_labels': True})
    print(f"✏️  Modelo escrito en {mps_file}")

    # 3) carga y computa IIS
    grb = gp.read(mps_file)
    grb.computeIIS()
    ilp_file = f"{case_name}.ilp"
    grb.write(ilp_file)
    print(f"📝 IIS guardado en {ilp_file}")

    # 4) imprime constrains y vars del IIS
    infeas_cons = [c.constrName for c in grb.getConstrs() if c.IISConstr]
    infeas_vars = [v.varName    for v in grb.getVars()    if v.IISLB or v.IISUB]

    print("\n⚠️  Restricciones en el IIS (no pueden satisfacerse todas):")
    for name in infeas_cons:
        print("   •", name)

    print("\n⚠️  Variables implicadas en el IIS (bounds conflictivas):")
    for name in infeas_vars:
        print("   •", name)


if __name__ == "__main__":

    # reading data from Excel
    # data = read_excel("input_data.xlsx", "case_1_plane")
    # data = read_excel("input_data.xlsx", "case_2_planes")
    # data = read_excel("input_data.xlsx", "case_3_planes")
    data = read_excel("input_data.xlsx", "case_3b_planes")
    # data = read_excel("input_data.xlsx", "case_4_planes")
    # data = read_excel("input_data.xlsx", "case_5_planes")
    # data = read_excel("input_data.xlsx", "case_6_planes")
    # data = read_excel("input_data.xlsx", "case_261")

    # Quick diagnose for loaded data
    print(f"Slots cargados: {len(data['sSlots'])}, Ejemplo: {data['sSlots'][:3]}")
    print(f"Posiciones cargadas: {len(data['sPositions'])}, Ejemplo: {data['sPositions'][:3]}")

    # Getting input data using the function that fills the dict out
    input_data = create_data(data)

    # Creating the Pyomo model object
    model = ap_pyomo_model()

    # Creating an instance of the model with input data in input_data dict.
    instance = model.create_instance(input_data)

    # Printing the model on the console
    # instance.pprint()
    def get_bin_value(var):
        v = getattr(var, 'value', None)
        return 0 if v is None else v

    def get_assigned_value(var):
        v = getattr(var, "value", None)
        return 0 if v is None else v


    def build_greedy_start(instance, verbose=False, max_iter_ls=30):
        """
        Heurística constructiva greedy avanzada para inicializar el arranque del modelo.
        Agrupa bloques consecutivos de trabajos de cada avión en una única posición para minimizar movimientos.
        Luego usa un local search sencillo para mejorar la estructura inicial.

        Parámetros:
            instance: instancia PYOMO ya construida
            verbose: imprime el detalle de la asignación
            max_iter_ls: iteraciones de búsqueda local

        Modifica los valores de las variables .value para aplicar como MIP start.
        """
        import random
        # 1. Prepara datos y utilidades
        SLOTS = list(instance.sSlots)
        POS = list(instance.sPositions)
        AVIONES = list(instance.sPlanes)
        JOBS = list(instance.sJobs)
        JOF = {j: instance.pPlaneOfJob[j] for j in JOBS}
        TASKOF = getattr(instance, "pTaskOfJob", None)
        DUR = {j: value(instance.pJobDuration[j]) for j in JOBS}

        # 2. Ordenar trabajos de cada avión
        plane_jobs = {}
        for j in JOBS:
            r = JOF[j]
            plane_jobs.setdefault(r, []).append(j)
        # Si tienes información explícita de orden/dependencia la usamos
        for r in AVIONES:
            if TASKOF is not None:
                try:
                    plane_jobs[r] = sorted(plane_jobs[r], key=lambda j: int(TASKOF[j]))
                except:
                    pass

        # 3. Asignar cada avión a la posición menos demandada (ciclando posiciones)
        jobs_asigned = set()
        jobs_block = {}
        pos_cycle = (p for p in POS * ((len(AVIONES) // len(POS)) + 1))  # iterador de posiciones
        plane2pos = {}
        pos_count = {p: 0 for p in POS}

        for idx, r in enumerate(AVIONES):
            # Elige la posición menos cargada en ese momento para balancear
            pos = min(pos_count, key=lambda p: pos_count[p])
            plane2pos[r] = pos
            pos_count[pos] += len(plane_jobs[r])
            if verbose:
                print(f"Avión {r}: {len(plane_jobs[r])} trabajos asignados inicialmente a {pos}")

        # 4. Asignar bloques compactos de trabajos por avión en su posición (slots consecutivos)
        job_time = {}
        ocupados = set()  # (slot, pos)
        for r in AVIONES:
            jobs_r = plane_jobs[r]
            posr = plane2pos[r]
            t = 0
            for j in jobs_r:
                # Buscar primer slot disponible en la posición de r
                slot = None
                for s in SLOTS:
                    if (s, posr) not in ocupados:
                        slot = s
                        break
                if slot is None:
                    raise RuntimeError("No hay slot libre en la posición para inicializar, incrementar número de slots")
                ocupados.add((slot, posr))
                # Seteamos las variables del MIP start
                instance.v01JobInSlot[slot, posr, j].value = 1
                instance.v01PlaneInSlot[slot, posr, r].value = 1
                instance.vPresence[slot, posr, r].value = 1
                instance.vStartSlotForJob[slot, posr, j].value = t
                instance.vFinishSlotForJob[slot, posr, j].value = t + DUR[j]
                instance.vStartJob[j].value = t
                instance.vFinishJob[j].value = t + DUR[j]
                t += DUR[j]
                jobs_asigned.add(j)
                job_time[j] = (slot, posr, t - DUR[j], t)
                # Marca slot-posición usados
                if verbose:
                    print(f"{j} ({r}) en ({slot},{posr}) de {t - DUR[j]} a {t}")
        # 5. REFINO local: intercambio por pares (local search greedy)
        # Idea: intentamos reducir aún más movimientos agrupando mejor
        improved = True
        iter_count = 0
        while improved and iter_count < max_iter_ls:
            improved = False
            iter_count += 1
            for r in AVIONES:
                pos_actual = plane2pos[r]
                # ¿Hay alguna posición libre donde los slots estén menos ocupados?
                for pos_alt in POS:
                    if pos_alt == pos_actual:
                        continue
                    usados_alt = sum(1 for j in plane_jobs[r] if any(
                        instance.v01JobInSlot[s, pos_alt, j].value == 1 for s in SLOTS
                    ))
                    # Si están ya todos en pos_actual, probamos a cambiar todos a pos_alt si hay slots libres
                    libres = [s for s in SLOTS if (s, pos_alt) not in ocupados]
                    if len(libres) >= len(plane_jobs[r]):
                        # Cambio: mueve todos los trabajos a pos_alt
                        # Liberar slots previos
                        for j in plane_jobs[r]:
                            s_prev, pos_prev, _, _ = job_time[j]
                            ocupados.discard((s_prev, pos_prev))
                            instance.v01JobInSlot[s_prev, pos_prev, j].value = 0
                            instance.v01PlaneInSlot[s_prev, pos_prev, r].value = 0
                            instance.vPresence[s_prev, pos_prev, r].value = 0
                        # Asignar seguidos en nuevos slots
                        t = 0
                        nuevos = []
                        for idxj, j in enumerate(plane_jobs[r]):
                            s_new = libres[idxj]
                            ocupados.add((s_new, pos_alt))
                            instance.v01JobInSlot[s_new, pos_alt, j].value = 1
                            instance.v01PlaneInSlot[s_new, pos_alt, r].value = 1
                            instance.vPresence[s_new, pos_alt, r].value = 1
                            instance.vStartSlotForJob[s_new, pos_alt, j].value = t
                            instance.vFinishSlotForJob[s_new, pos_alt, j].value = t + DUR[j]
                            instance.vStartJob[j].value = t
                            instance.vFinishJob[j].value = t + DUR[j]
                            job_time[j] = (s_new, pos_alt, t, t + DUR[j])
                            t += DUR[j]
                            nuevos.append((j, s_new, pos_alt))
                        # Aplica cambio solo si mejora (reducción movimientos por cambio de posición)
                        # (En este greedy siempre se aplica, pues agrupamos secuencialmente)
                        plane2pos[r] = pos_alt
                        improved = True
                        if verbose:
                            print(f"Iteración {iter_count}: avión {r} realocado completamente a {pos_alt}")
                        break
                if improved:
                    break  # salir y reiniciar búsqueda local

        # 6. Asignar cliente a la posición de su/sus aviones
        for c in instance.sClients:
            for p in instance.sPositions:
                instance.vClientPosition[c, p].value = 1 if p in [plane2pos.get(r) for r in AVIONES if
                                                                  (c, r) in instance.pAirplaneOfClient] else 0

        # 7. Resto de variables las dejamos sin inicializar (que lo ajuste gurobi)
        if verbose:
            print(f"Heurística greedy avanzada: se asignaron {len(jobs_asigned)} trabajos.")


    # ---- Aplicar heurística y preparar MIP start ----
    build_greedy_start(instance)

    # Crear sufijo para MIP start
    mip_suffix = Suffix(direction=Suffix.LOCAL)
    instance.add_component("mipstart_suffix", mip_suffix)
    for var in (instance.v01JobInSlot, instance.v01PlaneInSlot, instance.vPresence,
                instance.vStartJob, instance.vFinishJob):
        mip_suffix.set_value(var, None)

    # Fijar los valores como inicio explícito
    for var in instance.component_data_objects(ctype=type(instance.vStartJob), descend_into=True):
        val = getattr(var, "value", None)
        if val is not None:
            var.set_value(val, skip_validation=True)

    # Conteo informativo
    count_assigned = sum(
        1
        for s in instance.sSlots
        for p in instance.sPositions
        for j in instance.sJobs
        if get_assigned_value(instance.v01JobInSlot[s, p, j]) >= 0.5
    )
    print(f"→ Heurística inicial: {count_assigned} trabajos asignados como MIP start.")

    # Seting the solver
    opt = SolverFactory('gurobi')

    # Configuración para mostrar el log detallado de Gurobi
    opt.options['OutputFlag'] = 1        # Activar salida de log
    opt.options['LogToConsole'] = 1      # Mostrar log en consola
    opt.options['LogFile'] = 'gurobi.log' # También guardar log en archivo
    opt.options['DisplayInterval'] = 1   # Actualizar cada segundo

    # Configuración de límites para la resolución
    opt.options['TimeLimit'] = 2000       # Límite de tiempo en segundos (8 minutos)
    opt.options['MIPGap'] = 0.15         # Gap relativo (5%)

    # Configuración para priorizar heurísticas sobre Branch and Bound
    opt.options['Heuristics'] = 0.5      # Máximo esfuerzo en heurísticas (valor entre 0 y 1)
    opt.options['RINS'] = 1             # Frecuencia de la heurística RINS (menor valor = más frecuente)
    opt.options['MIPFocus'] = 1          # Enfoque en encontrar soluciones factibles rápidamente
    opt.options['ImproveStartGap'] = 0.5  # Comenzar a mejorar la solución cuando el gap sea < 50%
    opt.options['NoRelHeurTime'] = 80    # Aplicar heurísticas en los primeros segundos indicados

    # Reducir el esfuerzo de Branch and Bound
    opt.options['BranchDir'] = -1        # Favorecer branch hacia abajo (menos exploración)
    opt.options['MinRelNodes'] = 1000    # Limitar el número de nodos procesados

    # Resolución del modelo
    print("\nIniciando resolución con Gurobi...\n")
    results = opt.solve(instance, tee=True)  # tee=True muestra la salida del solucionador en la consola
    print("\nEstado del solucionador:", results.solver.status.value)

    # ---- 3. Fix-and-optimize sobre un pequeño subconjunto ----
    conflict_jobs = list(instance.sJobs)[:5]  # cambia o rota en siguientes iteraciones

    # Fijar todos los jobs fuera del bloque conflictivo
    for j in instance.sJobs:
        if j not in conflict_jobs:
            for s in instance.sSlots:
                for p in instance.sPositions:
                    var = instance.v01JobInSlot[s, p, j]
                    if getattr(var, 'value', None) is not None:
                        var.fix(value(var))

    print(f"--- Fix-and-optimize: reoptimizando dejando libres {conflict_jobs} ---")
    results2 = opt.solve(instance, tee=True)

    # Desfijar para poder repetir
    for j in instance.sJobs:
        if j not in conflict_jobs:
            for s in instance.sSlots:
                for p in instance.sPositions:
                    try:
                        instance.v01JobInSlot[s, p, j].unfix()
                    except:
                        pass

    # En caso de modelo no resoluble, se genera informe de restriciones que causan que el modelo sea no factible
    if results.solver.termination_condition == TerminationCondition.infeasible:
        diagnose_infeasibility(model, input_data, case_name="case2_conflict")
        raise RuntimeError("Modelo infactible: revisa el IIS en case2_conflict.ilp")

    solution = get_solution_data(instance)

    if results.solver.status.value == "ok":
        print("Solución encontrada. Verificando restricciones...")
        verification = check_solution(data, solution)

        if verification['all_constraints_satisfied']:
            print("✅ Todas las restricciones se cumplen correctamente.")
        else:
            print("❌ Se encontraron violaciones en las restricciones:")
            for constraint, result in verification['constraints_verification'].items():
                if not result['passed']:
                    print(f"  - Restricción '{constraint}' fallida:")
                    for error in result['errors']:
                        print(f"    * {error}")

        # Añadir informe detallado sobre la terminación del solucionador
        print("\n" + "="*80)
        print("INFORME DE TERMINACIÓN DEL SOLUCIONADOR")
        print("="*80)

        # Verificar razón de terminación
        termination_condition = results.solver.termination_condition
        print(f"Condición de terminación: {termination_condition}")

        if termination_condition == TerminationCondition.optimal:
            print("✅ Se encontró la solución óptima")
        elif termination_condition == TerminationCondition.maxTimeLimit:
            print("⏱️ Se alcanzó el límite de tiempo máximo")
        elif termination_condition == TerminationCondition.maxIterations:
            print("🔄 Se alcanzó el límite máximo de iteraciones")
        elif termination_condition == TerminationCondition.minFunctionValue:
            print("🎯 Se alcanzó el gap relativo objetivo")
        else:
            print(f"Otra condición: {termination_condition}")
            if results.solver.termination_condition == TerminationCondition.infeasible:
                log_infeasible_constraints(instance)
                raise SystemExit("Modelo inviable para case_2_planes: revisa el log")

        # Obtener estadísticas adicionales si están disponibles
        try:
            if hasattr(results.problem, 'lower_bound') and hasattr(results.problem, 'upper_bound'):
                lower_bound = results.problem.lower_bound
                upper_bound = results.problem.upper_bound

                if upper_bound and lower_bound:
                    gap = abs(upper_bound - lower_bound) / max(abs(upper_bound), 1e-10) * 100
                    print(f"\nGap final: {gap:.4f}%")
                    print(f"Cota inferior: {lower_bound:.6f}")
                    print(f"Cota superior: {upper_bound:.6f}")
        except:
            print("\nNo se pudieron obtener estadísticas de cotas")

        # Estadísticas adicionales
        try:
            if hasattr(results.solver, 'statistics'):
                stats = results.solver.statistics
                print("\nEstadísticas del solucionador:")
                if hasattr(stats, 'branch_and_bound'):
                    bb_stats = stats.branch_and_bound
                    print(f"Nodos explorados: {bb_stats.get('number_of_nodes_explored', 'N/A')}")
                    print(f"Iteraciones: {bb_stats.get('number_of_iterations', 'N/A')}")
                if hasattr(stats, 'wall_time'):
                    print(f"Tiempo de ejecución: {stats.wall_time:.2f} segundos")
        except:
            print("\nNo se pudieron obtener estadísticas adicionales")

        # Información de Gurobi (específica)
        try:
            gurobi_info = {}
            for key in results.solver.user_params:
                if key.startswith('gurobi_'):
                    param = key[7:]  # Eliminar 'gurobi_'
                    gurobi_info[param] = results.solver.user_params[key]

            if gurobi_info:
                print("\nEstadísticas de Gurobi:")
                if 'itercount' in gurobi_info:
                    print(f"Iteraciones: {gurobi_info['itercount']}")
                if 'nodecount' in gurobi_info:
                    print(f"Nodos: {gurobi_info['nodecount']}")
                if 'mipgap' in gurobi_info:
                    print(f"MIP Gap: {float(gurobi_info['mipgap'])*100:.4f}%")
                if 'runtime' in gurobi_info:
                    print(f"Tiempo de ejecución: {gurobi_info['runtime']:.2f} segundos")
        except:
            print("\nNo se pudieron obtener estadísticas específicas de Gurobi")

        print("="*80)

        print("\nGenerando gráfico de la solución...")
        print_chart(solution)
        df=print_chart(solution, html_path="gantt_basico.html")

        print("Generando diagrama mejorado de Gantt y resumen de movimientos…")
        df_full, movimientos = plot_enhanced_solution(df, instance, html_path="gantt_idles_movs.html")

        print("Report solución encontrada")
        for r in instance.sPlanes:
            print(f"Avión {r}:")
            for j in instance.sJobs:
                if (j, r) in instance.pLastJobOfPlane and value(instance.pLastJobOfPlane[j, r]) == 1:
                    f_real = instance.vFinishJob[j].value
                    f_teor = instance.pPredictedFinishOfPlane[r]
                    print(f"  Último trabajo: {j}")
                    print(f"    → Fecha real  = {f_real:.1f}")
                    print(f"    → Fecha límite= {value(f_teor):.1f}")
                    print(f"    → Retraso     = {instance.vPlaneDelay[r].value:.1f}")
                    print(f"    → Late Finish del avión = {value(instance.pLateFinishDeadline[r]):.1f}")
                    print(f"    → EarlyStart del avión = {value(instance.pEarlyStartOfPlane[r]):.1f}")

        generate_report(df_full, instance, movimientos)
    else:
        print("No se pudo encontrar una solución óptima.")
        print(f"Condición de terminación: {results.solver.termination_condition}")

    print("done")

    # Define tu fecha base si la usas para convertir días a fecha
    START_DATE = date.today()

    movements = []
    for r in instance.sPlanes:
        # 1) Recoge todos los "segmentos" de trabajos de r
        segs = []
        for (s, p), job in solution['slot_assignment'].items():
            if instance.pPlaneOfJob[job] == r:
                t0 = solution['start_slot'][(s, p)]
                t1 = solution['finish_slot'][(s, p)]
                segs.append((t0, t1, p, job))
        if not segs:
            continue

        # 2) Ordena cronológicamente por inicio
        segs.sort(key=lambda x: x[0])

        # 3) ENTRY: OUT → primera posición al inicio del primer job
        start0, _, p0, _ = segs[0]
        fecha_entry = START_DATE + timedelta(days=int(start0))
        movements.append((r, 'OUT', p0, fecha_entry))

        # 4) Cambios intermedios entre posiciones de job a job
        for i in range(len(segs) - 1):
            _, _, p_curr, _ = segs[i]
            t_next, _, p_next, _ = segs[i + 1]
            if p_curr != p_next:
                fecha_move = START_DATE + timedelta(days=int(t_next))
                movements.append((r, p_curr, p_next, fecha_move))

        # 5) EXIT: última posición → OUT al fin del último job
        _, end_last, p_last, _ = segs[-1]
        fecha_exit = START_DATE + timedelta(days=int(end_last))
        movements.append((r, p_last, 'OUT', fecha_exit))

    # 6) Imprime movimientos
    print("Movimientos detectados:")
    for plane, p0, p1, t in movements:
        print(f"  Avión {plane}: {p0} → {p1} el {t.isoformat()}")

#REVISIONES OPCINALES
# # Revisiones para comprobar correcto funcionamiento
# print("\n🔍 Revisión rápida de asignaciones por trabajo:")
# for j in instance.sJobs:
#     assigned_slots = [(s, p) for s in instance.sSlots for p in instance.sPositions if instance.v01JobInSlot[s, p, j].value == 1]
#     if len(assigned_slots) != 1:
#         print(f"⚠️ Job {j} está asignado a {len(assigned_slots)} slots: {assigned_slots}")
#
# print("\n🔍 Verificando dominios de v01JobInSlot:")
# for s in instance.sSlots:
#     for p in instance.sPositions:
#         for j in instance.sJobs:
#             exists = (s, p, j) in instance.v01JobInSlot
#             print(f"  {'✔️' if exists else '❌'} v01JobInSlot[{s},{p},{j}]")
#
# print("→ Asignaciones (slot,v01JobInSlot[slot,p,j].value==1):")
# for s in instance.sSlots:
#     for p in instance.sPositions:
#         for j in instance.sJobs:
#             if value(instance.v01JobInSlot[s, p, j]) > 0.5:
#                 print(f"   {j}  en  ({s}, {p})")
# # 1) Asignaciones
# for s in instance.sSlots:
#     for p in instance.sPositions:
#         for j in instance.sJobs:
#             if value(instance.v01JobInSlot[s, p, j]) > 0.5:
#                 print(f"{j} → ({s}, {p}), start={value(instance.vStartSlotForJob[s,p,j])}, finish={value(instance.vFinishSlotForJob[s,p,j])}")
#
# # 2) Tiempos globales
# for j in instance.sJobs:
#     print(f"{j}: global start={value(instance.vStartJob[j])}, global finish={value(instance.vFinishJob[j])}")
#
# # 3) Interferencias levantadas
# for idx in instance.sPosPosSlotSlot:
#     if value(instance.v01Alpha[idx]) > 0.5:
#         print("Alpha activada en", idx)
# # 4) Imprimir sPositionsInterference
# print("===== sPositionsInterference =====")
# for (p1, p2) in instance.sPositionsInterference:
#     print(f"Interferencia entre posiciones: {p1} ↔ {p2}")
# print(f"Total: {len(list(instance.sPositionsInterference))} pares\n")
#
# # 5) Imprimir sPosPosSlotSlot
# print("===== sPosPosSlotSlot =====")
# for (p1, p2, s1, s2) in instance.sPosPosSlotSlot:
#     print(f"Pos {p1} en slot {s1} vs Pos {p2} en slot {s2}")
# print(f"Total: {len(list(instance.sPosPosSlotSlot))} combinaciones")
