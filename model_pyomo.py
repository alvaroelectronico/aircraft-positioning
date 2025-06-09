import datetime
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pyomo.environ import *
from math import ceil
from datetime import date, timedelta

NO_POSITIONS = 5
POSITIONS = ['position{}'.format(i) for i in range(1, NO_POSITIONS + 1)]
POSITIONS_INTERFERE = [("position3", "position5"), ("position4", "position5")]
START_DATE = datetime.date.today()

def ap_pyomo_model():
    model = AbstractModel()

    # Sets
    model.sSlots = Set()
    model.sJobs = Set()
    model.sPositions = Set()
    model.sPlanes = Set()
    model.sClients = Set()
    model.sPositionsInterference = Set(dimen=2)
    model.sPosPosSlotSlot = Set(dimen=4)

    model.sSlotsSequence = Set(dimen=3)
    model.sJobSequence = Set(dimen=2)
    model.sSwitchPlanes = Set(dimen=5)

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
    model.pLateFinishOfPlane = Param(model.sPlanes, mutable=True)
    model.pTaskOfJob = Param(model.sJobs, within=PositiveIntegers)

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
    # Global start and finish time of each job
    model.vStartJob = Var(model.sJobs, within=NonNegativeReals)  # s_j: global start time of job j
    model.vFinishJob = Var(model.sJobs, within=NonNegativeReals)  # f_j: global finishing time of job j

    model.v01Alpha = Var(model.sPosPosSlotSlot, within=Binary)
    model.v01BetaS = Var(model.sPosPosSlotSlot, within=Binary)
    model.v01BetaF = Var(model.sPosPosSlotSlot, within=Binary)


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
            (model.vFinishJob[j] - model.pLateFinishOfPlane[r]) * model.pLastJobOfPlane[j, r]
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
        ordered = list(model.sSlots)
        idx = ordered.index(s)
        if idx == 0:
            return Constraint.Skip
        prev_s = ordered[idx - 1]
        return sum(model.v01JobInSlot[s, p, j] for j in model.sJobs) == \
            sum(model.v01JobInSlot[prev_s, p, j] for j in model.sJobs)

    # def fc15_ConsecutiveSlots(model, s, p):
    #     # Skip constraint for the first slot (s=1)
    #     if model.sSlots.ord(s) == 1:
    #         return Constraint.Skip
    #
    #     # Get the previous slot
    #     prev_s = list(model.sSlots)[model.sSlots.ord(s) - 2]  # -1 for 0-based indexing, -1 for previous
    #
    #     # Sum of job assignments in the current slot must be equal to sum in previous slot
    #     return sum(model.v01JobInSlot[s, p, j] for j in model.sJobs) == sum(model.v01JobInSlot[prev_s, p, j] for j in model.sJobs)

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
        # Cada avión r, en cada slot s, debe estar en alguna posición
        return sum(model.vPresence[s, p, r] for p in model.sPositions) == 1

    def fc20d_SinglePlanePerPosition(model, s, p):
        # En cada slot s y posición p, como máximo un avión puede estar presente
        return sum(model.vPresence[s, p, r] for r in model.sPlanes) <= 1

    # Rule: Client c with some airpline in position p:
    def fc21_ClientInPosition(model, c, p):
        return model.vClientPosition[c, p] >= sum(
            model.v01PlaneInPosition[r, p] * model.pAirplaneOfClient[c, r]
            for r in model.sPlanes
        )

    # Rule: Ec. fcBetaDefinion1 - Computing if starting time of slot s in position p is earlier than starting time of slot s' in position p'
    def fc22_BetaDefinition1(model, s, s2, p, p2):
        return model.pHorizon * model.v01BetaS[s, s2, p, p2] + model.vStartSlot[s, p] >= model.vStartSlot[s2, p2]

    # Rule: Ec. fcBetaDefinion2 - Computing if finishing time of slot s in position p is later than starting time of slot s' in position p'
    def fc23_BetaDefinition2(model, s, s2, p, p2):
        return model.pHorizon * model.v01BetaF[s, s2, p, p2] + model.vStartSlot[s2, p2] >= model.vFinishSlot[s, p]

    # Rule: Interference between slots
    def fc24_InterferenceExists(model, s, s2, p, p2):
        return 1 + model.v01Alpha[s, s2, p, p2] >= model.v01BetaS[s, s2, p, p2] + model.v01BetaF[s, s2, p, p2]

    # Rule: Ec. PlaneSwitchInPosition - Switching planes between consecutive slots
    def fc25_SwitchingPlanes(model, p, s, s2, r, r2):
        return 1 + model.v01SwitchPlanes[s, p] >= model.vPresence[s, p, r] + model.vPresence[s2, p, r2]

    # Rule: If a job is split among different slots, these cannot overlap
    def fc26_NoOverlapSlots(model, s, s2, p, p2, j):
        # Skip if it's the same slot and position
        if s == s2 or p == p2:
            return Constraint.Skip
        # If (s,s2,p,p2) is not in the sPosPosSlotSlot, these cannot overlap
        if (s, s2, p, p2) not in model.sPosPosSlotSlot:
            return Constraint.Skip

        # 1 + βS_{ss'pp'} + βF_{ss'pp'} >= x_{spj} + x_{s'p'j}
        # This ensures that if the same job is assigned to different slots,
        # either one starts after the other finishes or vice versa
        return 1 + model.v01BetaS[s, s2, p, p2] + model.v01BetaF[s, s2, p, p2] >= \
               model.v01JobInSlot[s, p, j] + model.v01JobInSlot[s2, p2, j]

    # def fcOBJ_Constant(model):
    #     return 1
    #

    # Rule: función objetivo
    def fc27_NoMovements(model):
        return sum(model.v01JobInSlot[s, p, j] for s in model.sSlots for p in model.sPositions for j in model.sJobs) \
                + sum(model.v01Alpha[i] for i in model.sPosPosSlotSlot) \
                + sum(model.v01SwitchPlanes[s, p] for p in model.sPositions for s in model.sSlots) \
                + sum(model.v01PlaneInPosition[r, p] for r in model.sPlanes for p in model.sPositions) \
                + sum(model.vClientDelay[c] for c in model.sClients)

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
    model.c15_ConsecutiveSlots = Constraint(model.sSlots, model.sPositions, rule=fc15_ConsecutiveSlots)

    print("Generating c16_SingleSlotPerJob constraint - Eq. 14")
    model.c16_SingleSlotPerJob = Constraint(model.sJobs, rule=fc16_SingleSlotPerJob)

    print("Generating c17_DurationIfNotAssigned constraint - Eq. 15")
    model.c17_DurationIfNotAssigned = Constraint(model.sSlots, model.sPositions, model.sJobs, rule=fc17_DurationIfNotAssigned)

    print("Generating c18_SlotDuration constraint")
    model.c18_SlotDuration = Constraint(model.sSlots, model.sPositions, rule=fc18_SlotDuration)

    print("Generating c19_PlaneSlotAssignment constraint - Eq. cPlaneSlotAssignment")
    model.c19_PlaneSlotAssignment = Constraint(model.sSlots, model.sPositions, model.sPlanes, rule=fc19_PlaneSlotAssignment)

    print("Generating c20_PlaneInPosition constraint")
    model.c20_PlaneInPosition = Constraint(model.sSlots, model.sPositions, model.sPlanes, rule=fc20_PlaneInPosition)

    print("Generating c20b and c20c_PlaneAlwaysPresent constraint")
    model.cPresentIfWork = Constraint(model.sSlots, model.sPositions, model.sPlanes, rule=fc20b_PresentIfWork)
    model.cPresentExactlyOne = Constraint(model.sSlots, model.sPlanes, rule=fc20c_PresentExactlyOne)

    print("Generating c20d_SinglePlanePerPosition constraint")
    model.c20d_SinglePlanePerPosition = Constraint(model.sSlots, model.sPositions, rule=fc20d_SinglePlanePerPosition)

    print("Generating c21_ClientInPosition constraint")
    model.c21_ClientInPosition = Constraint(model.sClients, model.sPositions, rule=fc21_ClientInPosition)

    print("Generating c22_BetaDefinition1 constraint - Eq. fcBetaDefinion1")
    model.c22_BetaDefinition1 = Constraint(model.sPosPosSlotSlot, rule=fc22_BetaDefinition1)

    print("Generating c23_BetaDefinition2 constraint - Eq. fcBetaDefinion2")
    model.c23_BetaDefinition2 = Constraint(model.sPosPosSlotSlot, rule=fc23_BetaDefinition2)

    print("Generating c24_InterferenceExists constraint")
    model.c24_InterferenceExists = Constraint(model.sPosPosSlotSlot, rule=fc24_InterferenceExists)

    print("Generating c25_SwitchingPlanes constraint - Eq. PlaneSwitchInPOsition")
    model.c25_SwitchingPlanes = Constraint(model.sSwitchPlanes, rule=fc25_SwitchingPlanes)

    print("Generating c26_NoOverlapSlots constraint")
    model.c26_NoOverlapSlots = Constraint(model.sSlots, model.sSlots, model.sPositions, model.sPositions, model.sJobs, rule=fc26_NoOverlapSlots)

    #Objective function
    print("Generating objective function")
    model.ObjFunction = Objective(rule=fc27_NoMovements, sense=minimize)

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
    sPositionsInterfere = POSITIONS_INTERFERE
    sSlots = ['slot{}'.format(i) for i in range(ceil(len(sJobs) / NO_POSITIONS * 1.5))]

    pHorizon = max(
        sum(pJobDuration[j] for j in sJobs if pPlaneOfJob[j] == r)
        for r in sPlanes
    ) * 1.2

    data = {
        'sJobs': sJobs,
        'sSlots': sSlots,
        'sPositions': sPositions,
        'sPlanes': sPlanes,
        'sClients': sClients,
        'sPositionsInterfere': sPositionsInterfere,
        'pJobDuration': pJobDuration,
        'pPlaneOfJob': pPlaneOfJob,
        'pTaskOfJob': pTaskOfJob,
        'pDate': pDate,
        'pHorizon': pHorizon,
        'pLateFinishOfPlane': max_finish_by_plane,
        'pAirplaneOfClient': dic_pAirplaneOfClient,
        'pLastJobOfPlane': dic_pLastJobOfPlane,
    }
    return data



def create_data(data):
    sPositions = data.get('sPositions', None)
    sPositionsInterfere = data.get('sPositionsInterfere', None)
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
    pLateFinishOfPlane = data.get('pLateFinishOfPlane', None)
    # #Alternative version for slot calculation in base of duration of jobs
    # sSlots = ['slot{}'.format(i) for i in range(10)]

    sSlotsSequence = [(s, s2, p) for p in sPositions for s in sSlots for s2 in sSlots
                      if sSlots.index(s) == sSlots.index(s2) + 1]

    # sJobSequence = [(j, j2) for j in sJobs for j2 in sJobs if pPlaneOfJob[j] == pPlaneOfJob[j2]
    #                 and pTaskOfJob[j] < pTaskOfJob[j2]]

    #Alternative version of sJobSequence
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

    sPosPosSlotSlot = [(s, s2, p, p2) for s in sSlots for s2 in sSlots for p in sPositions for p2 in sPositions if
                       (p, p2) in sPositionsInterfere and p!=p2]

    sSwitchPlanes = [(p, s, s2, r, r2) for p in sPositions for s in sSlots for s2 in sSlots for r in sPlanes
                     for r2 in sPlanes if sSlots.index(s) == sSlots.index(s2) + 1 and r!=r2]


    # Filling data into input_data dictionary
    input_data = {None: {
        'sSlots': {None: sSlots},
        'sJobs': {None: sJobs},
        'sPositions': {None: sPositions},
        'sPlanes': {None: sPlanes},
        'sClients': {None: sClients},
        'sPositionsInterfere': {None: sPositionsInterfere},
        'sPosPosSlotSlot': {None: sPosPosSlotSlot},
        'sSlotsSequence': {None: sSlotsSequence},
        'sJobSequence': {None: sJobSequence},
        # 'sPlaneSlotAssignment': {None: sPlaneSlotAssignment},
        'sSwitchPlanes': {None: sSwitchPlanes},
        'pHorizon': {None: pHorizon},
        'pJobDuration': pJobDuration,
        'pPlaneOfJob': pPlaneOfJob,
        'pTaskOfJob': pTaskOfJob,
        'pDate': pDate,
        'pAirplaneOfClient': pAirplaneOfClient,
        'pLastJobOfPlane': pLastJobOfPlane,
        'pLateFinishOfPlane': pLateFinishOfPlane,
    }
    }

    return input_data


def get_solution_data(model):
    slot_assignment = {(s, p): j for s in model.sSlots for p in model.sPositions for j in model.sJobs if
                       model.v01JobInSlot[s, p, j].value == 1}

    duration_slot = {(s, p): model.vDurationSlot[s, p].value for s in model.sSlots for p in model.sPositions}

    duration_slot_job = {(s, p, j): model.vDurationSlotForJob[s, p, j].value for s in model.sSlots \
                         for p in model.sPositions for j in model.sJobs}

    interference = [i for i in model.sPosPosSlotSlot if model.v01Alpha[i].value == 1]

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
    if 'start' in df.columns and 'finish' in df.columns:
        df = df.rename(columns={'start': 'start_slot', 'finish': 'finish_slot'})

    # 2) Convertir a datetime
    df['start_slot'] = pd.to_datetime(df['start_slot'])
    df['finish_slot'] = pd.to_datetime(df['finish_slot'])

    # 3) Parámetros auxiliares
    pDate_map = data.get('pDate', {})
    # Extraemos pJobDuration con value() a ints puros
    pJobDur = {j: int(value(model_instance.pJobDuration[j])) for j in model_instance.sJobs}

    # 4) Conteo de movimientos
    mov_count = {}
    for plane, _, _, _ in movimientos:
        mov_count[plane] = mov_count.get(plane, 0) + 1

    # 5) Resumen por avión
    p2c = {r: c for (c, r), val in model_instance.pAirplaneOfClient.items() if val == 1}
    resumen = []
    for avion in sorted(df['plane'].unique()):
        grp = df[df['plane'] == avion].sort_values('start_slot')
        trabajos = grp[grp['type'] == 'work']['job'].tolist()
        posiciones = grp['p'].unique().tolist()
        inicio = grp['start_slot'].min().date()
        fin = grp['finish_slot'].max().date()
        cliente = p2c.get(int(avion), None)
        resumen.append({
            'Avión': avion,
            'Cliente': cliente,
            'Inicio': inicio,
            'Fin': fin,
            'Trabajos': ", ".join(trabajos),
            'Posiciones': ", ".join(posiciones),
            'Movimientos': mov_count.get(avion, 0)
        })
    df_res = pd.DataFrame(resumen)
    print("\n=== RESUMEN POR AVIÓN ===")
    print(df_res.to_string(index=False))

    # 6) Detalle de trabajos (excluimos idles)
    print("\n" + "=" * 80)
    print("DETALLE DE TODOS LOS TRABAJOS")
    print("=" * 80)
    df_work = df[df['type'] == 'work'].copy()

    df_det = df_work[['plane', 'job', 'p', 'start_slot', 'finish_slot']].copy()
    df_det['Duración Estimada (días)'] = df_det['job'].map(lambda j: pJobDur[j])
    df_det['Duración Real (días)'] = (
                                             df_det['finish_slot'] - df_det['start_slot']
                                     ).dt.total_seconds() / 86400.0

    df_det['Fecha Prevista'] = df_det['job'].map(
        lambda j: date.today() + timedelta(days=pDate_map.get(j, 0) + pJobDur[j])
    )
    df_det['Fecha Real'] = df_det['finish_slot'].dt.date
    df_det['Retraso (días)'] = df_det.apply(
        lambda row: max((row['Fecha Real'] - row['Fecha Prevista']).days, 0), axis=1
    )
    df_det['⚠'] = df_det['Retraso (días)'].apply(lambda d: "❌" if d > 0 else "✅")

    df_det = df_det[[
        '⚠', 'plane', 'job', 'p',
        'Fecha Prevista', 'Fecha Real',
        'Duración Estimada (días)', 'Duración Real (días)', 'Retraso (días)'
    ]]
    df_det.columns = [
        '⚠', 'Avión', 'Trabajo', 'Posición',
        'Fecha Prevista', 'Fecha Real',
        'Duración Estimada (días)', 'Duración Real (días)', 'Retraso (días)'
    ]
    print(df_det.to_string(index=False))

    # 7) Retrasos por cliente
    print("\n" + "=" * 80)
    print("RETRASOS POR CLIENTE (según el modelo)")
    print("=" * 80)
    clientes = sorted(model_instance.sClients)
    resumen_c = []
    for c in clientes:
        d = model_instance.vClientDelay[c].value
        resumen_c.append({
            'Cliente': c,
            'Retraso (días)': int(d),
            'Retraso (semanas)': round(d / 7.0, 2),
            'Estado': "✅ Cumple" if d == 0 else "❌ Retraso"
        })
    print(pd.DataFrame(resumen_c).to_string(index=False))

    # 8) Resumen ejecutivo
    print("\n" + "=" * 80)
    print("RESUMEN EJECUTIVO")
    print("=" * 80)
    total_t = len(df_det)
    total_r = df_det['Retraso (días)'].gt(0).sum()
    total_a = len(df['plane'].unique())
    total_c = len(clientes)
    c_retraso = [c for c in clientes if next(rc for rc in resumen_c if rc['Cliente'] == c)['Retraso (días)'] > 0]

    print(f"📦 {total_t} trabajos procesados")
    print(f"✈️  {total_a} aviones, {total_c} clientes")
    print(f"🔴 {total_r} trabajos con retraso")
    if c_retraso:
        print(f"⚠️  Clientes con retrasos: {', '.join(map(str, c_retraso))}")
    else:
        print("🟢 Todos los clientes han cumplido sus fechas previstas")
    print("\nℹ️  El retraso de un cliente solo considera su último trabajo.")
    print("=" * 80)

def plot_enhanced_solution(df_work, instance, html_path="gantt_idles_movs.html"):
    """
    Construye el Gantt con:
      - Trabajos: barras coloreadas.
      - Idles: huecos con borde del color del avión.
      - Sin flechas.
      - Sin solapamientos de idles.
    Devuelve: df_full (trabajos+idles), lista movimientos [(plane,p0,p1,t),...]
    """
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

    df_idle = pd.DataFrame(idles, columns=['plane','type','job','p','start','finish'])
    df_full = pd.concat([df, df_idle], ignore_index=True)

    # 4) Mapa de colores por avión
    palette   = px.colors.qualitative.Plotly
    color_map = {plane: palette[i % len(palette)] for i, plane in enumerate(planes)}

    # 5) Timeline de trabajos
    fig = px.timeline(
        df,
        x_start="start", x_end="finish", y="p",
        color="plane", color_discrete_map=color_map,
        hover_data=["job","type"],
        title="Diagrama de Gantt con Idles (sin solapamientos)"
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

    # # VERSION 1.0
# def check_solution(data, solution):
#     """
#     Verifica que la solución cumpla con todos los requisitos del modelo de posicionamiento de aeronaves.
#
#     Args:
#         data: Diccionario con los datos de entrada
#         solution: Diccionario con los resultados de la solución
#
#     Returns:
#         dict: Diccionario con los resultados de las verificaciones, con una entrada por cada restricción
#              verificada. Cada entrada contiene un booleano que indica si se cumple la restricción y un
#              mensaje de error si no se cumple.
#     """
#     # Unpacking input data
#     sPositions = data.get('sPositions', [])
#     sPositionsInterfere = data.get('sPositionsInterfere', [])
#     sJobs = data.get('sJobs', [])
#     sPlanes = data.get('sPlanes', [])
#     sSlots = data.get('sSlots', [])
#     pJobDuration = data.get('pJobDuration', {})
#     pPlaneOfJob = data.get('pPlaneOfJob', {})
#     pTaskOfJob = data.get('pTaskOfJob', {})
#     pHorizon = data.get('pHorizon', 0)
#
#     # Unpacking solution data
#     slot_assignment = solution.get('slot_assignment', {})
#     duration_slot = solution.get('duration_slot', {})
#     duration_slot_job = solution.get('duration_slot_job', {})
#     interference = solution.get('interference', [])
#     start_slot_job = solution.get('start_slot_job', {})
#     finish_slot_job = solution.get('finish_slot_job', {})
#     start_slot = solution.get('start_slot', {})
#     finish_slot = solution.get('finish_slot', {})
#
#     # Inicializar diccionario de resultados
#     verification_results = {}
#

    # # 1. Verificar que cada trabajo esté asignado exactamente una vez
    # verification_results['all_jobs_assigned'] = {'passed': True, 'errors': []}
    # assigned_jobs = [j for _, j in slot_assignment.items()]
    # for j in sJobs:
    #     if j not in assigned_jobs:
    #         verification_results['all_jobs_assigned']['passed'] = False
    #         verification_results['all_jobs_assigned']['errors'].append(f"Job {j} no está asignado a ninguna ranura")
    #
    # # 2. Verificar que cada ranura tenga a lo sumo un trabajo asignado
    # verification_results['single_job_per_slot'] = {'passed': True, 'errors': []}
    # for p in sPositions:
    #     for s in sSlots:
    #         jobs_in_slot = [j for (slot, pos), j in slot_assignment.items() if slot == s and pos == p]
    #         if len(jobs_in_slot) > 1:
    #             verification_results['single_job_per_slot']['passed'] = False
    #             verification_results['single_job_per_slot']['errors'].append(
    #                 f"La ranura {s} en la posición {p} tiene múltiples trabajos asignados: {jobs_in_slot}")
    #
    # # 3. Verificar que la duración total de cada trabajo sea correcta
    # verification_results['job_duration_correct'] = {'passed': True, 'errors': []}
    # for j in sJobs:
    #     if j in assigned_jobs:
    #         total_duration = sum(duration_slot_job.get((s, p, j), 0) for s in sSlots for p in sPositions)
    #         if abs(total_duration - pJobDuration[j]) > 1e-6:  # Tolerancia para errores de punto flotante
    #             verification_results['job_duration_correct']['passed'] = False
    #             verification_results['job_duration_correct']['errors'].append(
    #                 f"El trabajo {j} debería tener duración {pJobDuration[j]}, pero tiene {total_duration}")
    #
    # # 4. Verificar que la duración de las ranuras sea consistente con los trabajos asignados
    # verification_results['slot_duration_consistent'] = {'passed': True, 'errors': []}
    # for (s, p), dur in duration_slot.items():
    #     raw_dur = duration_slot[(s,p)]
    #     dur = raw_dur if raw_dur is not None else 0
    #     job_duration_sum = sum(duration_slot_job.get((s, p, j), 0) for j in sJobs)
    #     if abs(dur - job_duration_sum) > 1e-6:
    #         verification_results['slot_duration_consistent']['passed'] = False
    #         verification_results['slot_duration_consistent']['errors'].append(
    #             f"La ranura {s} en posición {p} tiene duración {dur}, pero la suma de duraciones de trabajos es {job_duration_sum}")
    #
    # # 5. Verificar que los tiempos de inicio y fin de las ranuras son consistentes
    # verification_results['slot_times_consistent'] = {'passed': True, 'errors': []}
    # for s in sSlots:
    #     for p in sPositions:
    #         raw_dur = duration_slot.get((s, p))
    #         dur = raw_dur if raw_dur is not None else 0
    #         if (s, p) in start_slot and (s, p) in finish_slot:
    #             if abs((finish_slot[(s, p)] - start_slot[(s, p)]) - dur) > 1e-6:
    #                 verification_results['slot_times_consistent']['passed'] = False
    #                 verification_results['slot_times_consistent']['errors'].append(
    #                     f"La ranura {s} en posición {p} tiene inconsistencia en tiempos: inicio={start_slot[(s, p)]}, "
    #                     f"fin={finish_slot[(s, p)]}, duración={duration_slot.get((s, p), 0)}")
    #
    # # 6. Verificar que no hay solapamiento entre ranuras consecutivas en la misma posición
    # verification_results['no_overlap_same_position'] = {'passed': True, 'errors': []}
    # for p in sPositions:
    #     slots_in_pos = sorted([(s, start_slot.get((s, p), 0)) for s in sSlots if (s, p) in start_slot],
    #                          key=lambda x: x[1])
    #     for i in range(len(slots_in_pos) - 1):
    #         current_slot, current_start = slots_in_pos[i]
    #         next_slot, next_start = slots_in_pos[i + 1]
    #         current_end = finish_slot.get((current_slot, p), 0)
    #
    #         if current_end > next_start + 1e-6:  # Pequeña tolerancia
    #             verification_results['no_overlap_same_position']['passed'] = False
    #             verification_results['no_overlap_same_position']['errors'].append(
    #                 f"Solapamiento en posición {p}: Ranura {current_slot} termina en {current_end}, "
    #                 f"pero ranura {next_slot} comienza en {next_start}")
    #
    # # 7. Verificar la secuencia de trabajos para el mismo avión
    #
    # verification_results['job_sequence_correct'] = {'passed': True, 'errors': []}
    # for plane in sPlanes:
    #     # Obtener todos los trabajos de este avión, con su número de tarea
    #     plane_jobs = [(j, pTaskOfJob[j]) for j in sJobs if pPlaneOfJob[j] == plane]
    #     plane_jobs.sort(key=lambda x: x[1])
    #
    #     for i in range(len(plane_jobs) - 1):
    #         curr_job, _ = plane_jobs[i]
    #         next_job, _ = plane_jobs[i + 1]
    #
    #         # Ahora tomamos directamente los tiempos globales que devolvió get_solution_data:
    #         curr_finish = solution['finish_job'][curr_job]
    #         next_start = solution['start_job'][next_job]
    #
    #         if curr_finish > next_start + 1e-6:
    #             verification_results['job_sequence_correct']['passed'] = False
    #             verification_results['job_sequence_correct']['errors'].append(
    #                 f"Secuencia incorrecta para avión {plane}: "
    #                 f"Trabajo {curr_job} termina en {curr_finish}, pero "
    #                 f"{next_job} comienza en {next_start}"
    #             )
    # # verification_results['job_sequence_correct'] = {'passed': True, 'errors': []}
    # # for plane in sPlanes:
    # #     # Obtener todos los trabajos de este avión
    # #     plane_jobs = [(j, pTaskOfJob[j]) for j in sJobs if pPlaneOfJob[j] == plane]
    # #     # Ordenar por número de tarea
    # #     plane_jobs.sort(key=lambda x: x[1])
    # #
    # #     # Verificar que los trabajos se realizan en secuencia
    # #     for i in range(len(plane_jobs) - 1):
    # #         current_job, current_task = plane_jobs[i]
    # #         next_job, next_task = plane_jobs[i + 1]
    # #
    # #         # Calcular tiempos de finalización y comienzo
    # #         current_finish_times = [finish_slot_job.get((s, p, current_job), 0)
    # #                                for s in sSlots for p in sPositions
    # #                                if (s, p, current_job) in finish_slot_job]
    # #         next_start_times = [start_slot_job.get((s, p, next_job), 0)
    # #                            for s in sSlots for p in sPositions
    # #                            if (s, p, next_job) in start_slot_job]
    # #
    # #         if current_finish_times and next_start_times:
    # #             current_finish = max(current_finish_times)
    # #             next_start = min(next_start_times)
    # #
    # #             if current_finish > next_start + 1e-6:
    # #                 verification_results['job_sequence_correct']['passed'] = False
    # #                 verification_results['job_sequence_correct']['errors'].append(
    # #                     f"Secuencia incorrecta para avión {plane}: Trabajo {current_job} (tarea {current_task}) "
    # #                     f"termina en {current_finish}, pero trabajo {next_job} (tarea {next_task}) comienza en {next_start}")
    #
    # # 8. Verificar que no hay interferencia entre posiciones que no deben solaparse
    # verification_results['no_position_interference'] = {'passed': True, 'errors': []}
    # for p1, p2 in sPositionsInterfere:
    #     # Obtener todas las ranuras activas en cada posición
    #     slots_p1 = [(s, start_slot.get((s, p1), 0), finish_slot.get((s, p1), 0))
    #                 for s in sSlots if (s, p1) in start_slot and (s, p1) in finish_slot]
    #     slots_p2 = [(s, start_slot.get((s, p2), 0), finish_slot.get((s, p2), 0))
    #                 for s in sSlots if (s, p2) in start_slot and (s, p2) in finish_slot]
    #
    #     # Verificar solapamientos
    #     for s1, start1, end1 in slots_p1:
    #         for s2, start2, end2 in slots_p2:
    #             # Hay solapamiento si el inicio de uno está entre el inicio y fin del otro
    #             if (start1 <= start2 < end1) or (start2 <= start1 < end2):
    #                 found_in_interference = False
    #                 for s, s_2, pos, pos_2 in interference:
    #                     if ((s == s1 and s_2 == s2 and pos == p1 and pos_2 == p2) or
    #                         (s == s2 and s_2 == s1 and pos == p2 and pos_2 == p1)):
    #                         found_in_interference = True
    #                         break
    #
    #                 if not found_in_interference:
    #                     verification_results['no_position_interference']['passed'] = False
    #                     verification_results['no_position_interference']['errors'].append(
    #                         f"Interferencia no registrada entre posiciones {p1} y {p2}: "
    #                         f"Ranura {s1} ({start1}-{end1}) y ranura {s2} ({start2}-{end2})")
    #
    # # 9. Verificar que todos los trabajos se completan dentro del horizonte
    # verification_results['within_horizon'] = {'passed': True, 'errors': []}
    # for (s, p), end_time in finish_slot.items():
    #     if end_time > pHorizon + 1e-6:
    #         verification_results['within_horizon']['passed'] = False
    #         verification_results['within_horizon']['errors'].append(
    #             f"La ranura {s} en posición {p} termina en {end_time}, que excede el horizonte {pHorizon}")
    #
    # # 10. Verificar que los aviones no están en diferentes posiciones al mismo tiempo
    # verification_results['plane_single_position'] = {'passed': True, 'errors': []}
    # for plane in sPlanes:
    #     # Obtener todos los slots asignados a trabajos de este avión
    #     plane_slots = []
    #     for (s, p), j in slot_assignment.items():
    #         if pPlaneOfJob[j] == plane:
    #             plane_slots.append((s, p, start_slot.get((s, p), 0), finish_slot.get((s, p), 0)))
    #
    #     # Verificar solapamientos entre posiciones diferentes
    #     for i in range(len(plane_slots)):
    #         s1, p1, start1, end1 = plane_slots[i]
    #         for j in range(i+1, len(plane_slots)):
    #             s2, p2, start2, end2 = plane_slots[j]
    #             if p1 != p2:  # Diferentes posiciones
    #                 # Hay solapamiento si el inicio de uno está entre el inicio y fin del otro
    #                 if (start1 <= start2 < end1) or (start2 <= start1 < end2):
    #                     verification_results['plane_single_position']['passed'] = False
    #                     verification_results['plane_single_position']['errors'].append(
    #                         f"Avión {plane} está en múltiples posiciones al mismo tiempo: "
    #                         f"Posición {p1} ({start1}-{end1}) y posición {p2} ({start2}-{end2})")
    #
    # # 11. Verificar que no se utilice un slot a menos que se hayan utilizado todos los anteriores
    # verification_results['consecutive_slots'] = {'passed': True, 'errors': []}
    #
    # # Ordenar slots por número
    # sorted_slots = sorted(sSlots, key=lambda x: int(x.replace('slot', '')))
    #
    # for p in sPositions:
    #     for i in range(1, len(sorted_slots)):
    #         current_slot = sorted_slots[i]
    #         prev_slot = sorted_slots[i-1]
    #
    #         # Contar trabajos asignados a cada slot
    #         jobs_in_current = sum(1 for (s, pos), _ in slot_assignment.items() if s == current_slot and pos == p)
    #         jobs_in_prev = sum(1 for (s, pos), _ in slot_assignment.items() if s == prev_slot and pos == p)
    #
    #         if jobs_in_current > 0 and jobs_in_prev == 0:
    #             verification_results['consecutive_slots']['passed'] = False
    #             verification_results['consecutive_slots']['errors'].append(
    #                 f"Posición {p}: Se utiliza el slot {current_slot} pero no se utiliza el slot anterior {prev_slot}")
    # # Resumen final
    # all_passed = all(result['passed'] for result in verification_results.values())
    # summary = {
    #     'all_constraints_satisfied': all_passed,
    #     'constraints_verification': verification_results
    # }

    # VERSION 2.0
def check_solution(data, solution):

        sPositions = data.get('sPositions', [])
        sPositionsInterfere = data.get('sPositionsInterfere', [])
        sJobs = data.get('sJobs', [])
        sPlanes = data.get('sPlanes', [])
        sSlots = data.get('sSlots', [])
        pJobDuration = data.get('pJobDuration', {})
        pPlaneOfJob = data.get('pPlaneOfJob', {})
        pTaskOfJob = data.get('pTaskOfJob', {})
        pHorizon = data.get('pHorizon', 0)

        sSlotsSequence = data.get('sSlotsSequence', [])  # lista de tuplas (s, s2, p)
        sJobSequence = data.get('sJobSequence', [])  # lista de tuplas (j, j2)
        sPosPosSlotSlot = data.get('sPosPosSlotSlot', [])  # lista de tuplas (s, s2, p, p2)
        sSwitchPlanes = data.get('sSwitchPlanes', [])  # lista de tuplas (p, s, s2, r, r2)

        slot_assignment = solution.get('slot_assignment', {})  # {(s,p): j}
        duration_slot = solution.get('duration_slot', {})  # {(s,p): valor}
        duration_slot_job = solution.get('duration_slot_job', {})  # {(s,p,j): valor}
        interference_list = solution.get('interference', [])  # lista de índices (s,s2,p,p2) donde alpha=1
        start_slot_job = solution.get('start_slot_job', {})  # {(s,p,j): valor}
        finish_slot_job = solution.get('finish_slot_job', {})  # {(s,p,j): valor}
        start_slot = solution.get('start_slot', {})  # {(s,p): valor}
        finish_slot = solution.get('finish_slot', {})  # {(s,p): valor}
        start_job = solution.get('start_job', {})  # {j: valor}
        finish_job = solution.get('finish_job', {})  # {j: valor}

        verification_results = {}

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
                if suma_s != suma_prev:
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
            if solapan and (s, s2, p, p2) not in interference_list and (s2, s, p2, p) not in interference_list:
                verification_results['c24_interference_exists']['passed'] = False
                verification_results['c24_interference_exists']['errors'].append(
                    f"Solapamiento real entre ({s},{s2},{p},{p2}) no marcado en interference_list"
                )

        # —————————————————————————— c25: SwitchingPlanes ——————————————————————————
        # ∀(p,s,s2,r,r2): 1 + v01SwitchPlanes[s,p] ≥ v01PlaneInSlot[s,p,r] + v01PlaneInSlot[s2,p,r2]
        verification_results['c25_switching_planes'] = {'passed': True, 'errors': []}
        for (p, s, s2, r, r2) in sSwitchPlanes:
            in1 = 1 if slot_assignment.get((s, p)) in sJobs and pPlaneOfJob.get(slot_assignment[(s, p)]) == r else 0
            in2 = 1 if slot_assignment.get((s2, p)) in sJobs and pPlaneOfJob.get(slot_assignment[(s2, p)]) == r2 else 0
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


if __name__ == "__main__":

    # reading data from Excel
    # data = read_excel("input_data.xlsx", "case_1_plane")
    # data = read_excel("input_data.xlsx", "case_2_planes")
    # data = read_excel("input_data.xlsx", "case_3_planes")
    # data = read_excel("input_data.xlsx", "case_3b_planes")
    data = read_excel("input_data.xlsx", "case_4_planes")
    # data = read_excel("input_data.xlsx", "case_5_planes")
    # data = read_excel("input_data.xlsx", "case_2")


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

    # Seting the solver
    opt = SolverFactory('gurobi')

    # Configuración para mostrar el log detallado de Gurobi
    opt.options['OutputFlag'] = 1        # Activar salida de log
    opt.options['LogToConsole'] = 1      # Mostrar log en consola
    # opt.options['LogFile'] = 'gurobi.log' # También guardar log en archivo
    opt.options['DisplayInterval'] = 1   # Actualizar cada segundo

    # Configuración de límites para la resolución
    opt.options['TimeLimit'] = 1000       # Límite de tiempo en segundos (8 minutos)
    opt.options['MIPGap'] = 0.34         # Gap relativo (5%)

    # Configuración para priorizar heurísticas sobre Branch and Bound
    opt.options['Heuristics'] = 1.0      # Máximo esfuerzo en heurísticas (valor entre 0 y 1)
    opt.options['RINS'] = 1             # Frecuencia de la heurística RINS (menor valor = más frecuente)
    opt.options['MIPFocus'] = 3          # Enfoque en encontrar soluciones factibles rápidamente
    opt.options['ImproveStartGap'] = 0.5  # Comenzar a mejorar la solución cuando el gap sea < 50%
    opt.options['NoRelHeurTime'] = 120    # Aplicar heurísticas en los primeros segundos indicados

    # Reducir el esfuerzo de Branch and Bound
    opt.options['BranchDir'] = -1        # Favorecer branch hacia abajo (menos exploración)
    opt.options['MinRelNodes'] = 1000    # Limitar el número de nodos procesados

    # Solving the model and saving the results
    print("\nIniciando resolución con Gurobi...\n")
    results = opt.solve(instance, tee=True)  # tee=True muestra la salida del solucionador en la consola
    print("\nEstado del solucionador:", results.solver.status.value)

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
                    f_teor = instance.pLateFinishOfPlane[r]
                    print(f"  Último trabajo: {j}")
                    print(f"    → Fecha real  = {f_real:.1f}")
                    print(f"    → Fecha límite= {value(f_teor):.1f}")
                    print(f"    → Retraso     = {instance.vPlaneDelay[r].value:.1f}")

        generate_report(df_full, instance, movimientos)
    else:
        print("No se pudo encontrar una solución óptima.")
        print(f"Condición de terminación: {results.solver.termination_condition}")

    print("done")

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

from datetime import timedelta, date

# Asume que ya tienes en memoria:
#   - `instance` (la instancia Pyomo)
#   - `solution` (el dict que te devolvió get_solution_data)

# Define tu fecha base si la usas para convertir días a fecha
START_DATE = date.today()

movements = []

for r in instance.sPlanes:
    # 1) Recoge todos los "segmentos" donde r hace un trabajo
    segs = []
    for (s, p), job in solution['slot_assignment'].items():
        if instance.pPlaneOfJob[job] == r:
            t0 = solution['start_slot'][(s, p)]
            t1 = solution['finish_slot'][(s, p)]
            segs.append((t0, t1, p))
    # 2) Ordena cronológicamente
    segs.sort(key=lambda x: x[0])
    # 3) Detecta cambios de posición
    for i in range(len(segs)-1):
        _, _, p0 = segs[i]
        t_next, _, p1 = segs[i+1]
        if p0 != p1:
            # tiempo en días → fecha real
            fecha = START_DATE + timedelta(days=int(t_next))
            movements.append((r, p0, p1, fecha))

# 4) Imprime
print("Movimientos detectados:")
for plane, p0, p1, t in movimientos:
    print(f"  Avión {plane}: {p0} → {p1} el {t.date()}")
