import datetime
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pyomo.environ import *
from math import ceil
from datetime import date, timedelta
import random
from pyomo.util.infeasible import log_infeasible_constraints

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
    model.sPosPosSlotSlot = Set(dimen=4)

    model.prev_slot = Param( model.sSlots, default=None, within=model.sSlots | {None})

    model.sSlotsSequence = Set(dimen=3)
    model.sJobSequence = Set(dimen=2)
    model.sSwitchPlanes = Set(dimen=5)

    # Parameters
    model.pHorizon = Param(within=NonNegativeReals)

    # def _init_M(m):
    #     return value(m.pHorizon)
    #
    # model.M = Param(initialize=_init_M)

    model.pBigM = Param(model.sJobs, within=NonNegativeReals)
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
            + model.pBigM[j] * (1 - model.v01JobInSlot[s, p, j])

    # 2) vStartJob[j] ≥ vStartSlotForJob[s,p,j] - M·(1 - x[s,p,j])
    def fc06_StartJob_lower(model, s, p, j):
        return model.vStartJob[j] \
            >= model.vStartSlotForJob[s, p, j] \
            - model.pBigM[j] * (1 - model.v01JobInSlot[s, p, j])

    # 3) vFinishJob[j] ≥ vFinishSlotForJob[s,p,j] - M·(1 - x[s,p,j])
    def fc07_FinishJob_lower(model, s, p, j):
        return model.vFinishJob[j] \
            >= model.vFinishSlotForJob[s, p, j] \
            - model.pBigM[j] * (1 - model.v01JobInSlot[s, p, j])

    # 4) vFinishJob[j] ≤ vFinishSlotForJob[s,p,j] + M·(1 - x[s,p,j])
    def fc07_FinishJob_upper(model, s, p, j):
        return model.vFinishJob[j] \
            <= model.vFinishSlotForJob[s, p, j] \
            + model.pBigM[j] * (1 - model.v01JobInSlot[s, p, j])

    # Rule: Ec. noNegativeDurationNoCommas - Start time of job must be <= finish time of job
    def fc08_StartFinishRelation(model, j):
        # s_j ≤ f_j
        return model.vStartJob[j] <= model.vFinishJob[j]

    # Rule: Ec. calculating delays of planes
    def fc09_Plane_delay(model,r):
        # para cada (j,r) con L[j,r]=1, impongo H*γ_r ≥ f[j] - T[r]
        return model.vPlaneDelay[r] >= sum(
            (model.vFinishJob[j] - model.pPredictedFinishOfPlane[r]) * model.pLastJobOfPlane[j, r]
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
    def fc13_SlotSequence(model, prev_s, curr_s, p):
        return model.vStartSlot[curr_s, p] >= model.vFinishSlot[prev_s, p]

    # Rule: Ec. jobPrecedence - Job sequence (jobs are sequenced)
    def fc14_JobSequence(model, j, j2):
        return model.vStartJob[j2] >= model.vFinishJob[j]

    # # Rule: Ec. noEmptySlots - Consecutive slots - a slot is not used unless all previous ones have been used

    # def fc15_ConsecutiveSlots(model, s, p): #Versión más ligera
    #     prev_s = model.prev_slot[s]
    #     if prev_s is None:
    #         return Constraint.Skip
    #     return sum(model.v01JobInSlot[s, p, j] for j in model.sJobs) == \
    #         sum(model.v01JobInSlot[prev_s, p, j] for j in model.sJobs)

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
    # sSlots = ['slot{}'.format(i) for i in range(ceil(len(sJobs) / NO_POSITIONS*1.5)+2)]

    # <<< nuevo: tantos slots como el máximo de tareas de un avión <<<
    # (asegura que tu heurístico solo use slots que el modelo reconozca)
    max_tasks_per_plane = df.groupby('plane')['task'].nunique().max()
    # numerar slot0, slot1, …, slot(max_tasks_per_plane-1)
    sSlots = [f"slot{i}" for i in range(max_tasks_per_plane)]
    # no hace falta ningún sorted extra, vienen en orden 0,1,2…

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
        'pPredictedFinishOfPlane': max_finish_by_plane,
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
    pLateFinishOfPlane = data.get('pPredictedFinishOfPlane', None)
    # #Alternative version for slot calculation in base of duration of jobs
    # sSlots = ['slot{}'.format(i) for i in range(10)]

    # sSlotsSequence = [(s, s2, p) for p in sPositions for s in sSlots for s2 in sSlots
    #                   if sSlots.index(s) == sSlots.index(s2) + 1]

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

    # sPosPosSlotSlot = [(s, s2, p, p2) for s in sSlots for s2 in sSlots for p in sPositions for p2 in sPositions if
    #                    (p, p2) in sPositionsInterfere and p!=p2]
    #
    # sSwitchPlanes = [(p, s, s2, r, r2) for p in sPositions for s in sSlots for s2 in sSlots for r in sPlanes
    #                  for r2 in sPlanes if sSlots.index(s) == sSlots.index(s2) + 1 and r!=r2]

    consecutive_pairs = [(sSlots[i - 1], sSlots[i])
        for i in range(1, len(sSlots))
    ]

    # 1) Construye el mapping slot→slot_anterior
    prev_slot = {
        sSlots[i]: (sSlots[i - 1] if i > 0 else None)
        for i in range(len(sSlots))
    }

    # 2) Ahora ya puedes generar tu sSlotsSequence sin indexaciones:
    sSlotsSequence = [
        (prev_s, curr_s, p)
        for (prev_s, curr_s) in consecutive_pairs
        for p in sPositions
    ]

    sPosPosSlotSlot = [(s, s2, p, p2) for s in sSlots for s2 in sSlots for p in sPositions for p2 in sPositions if (p, p2) in sPositionsInterfere and p != p2]

    sSwitchPlanes = [(p, s, s2, r, r2) for p in sPositions for (s, s2) in consecutive_pairs for r in sPlanes for r2 in sPlanes if r != r2]

    # Filling data into input_data dictionary
    input_data = {None: {
        'sSlots': {None: sSlots},
        'sJobs': {None: sJobs},
        'sPositions': {None: sPositions},
        'sPlanes': {None: sPlanes},
        'sClients': {None: sClients},
        'sPositionsInterfere': {None: sPositionsInterfere},
        'sPosPosSlotSlot': {None: sPosPosSlotSlot},
        'prev_slot': prev_slot,
        'sSlotsSequence': {None: sSlotsSequence},
        'sJobSequence': {None: sJobSequence},
        # 'sPlaneSlotAssignment': {None: sPlaneSlotAssignment},
        'sSwitchPlanes': {None: sSwitchPlanes},
        'pHorizon': {None: pHorizon},
        'pBigM': {j: pHorizon for j in sJobs},
        'pJobDuration': pJobDuration,
        'pPlaneOfJob': pPlaneOfJob,
        'pTaskOfJob': pTaskOfJob,
        'pDate': pDate,
        'pAirplaneOfClient': pAirplaneOfClient,
        'pLastJobOfPlane': pLastJobOfPlane,
        'pPredictedFinishOfPlane': pLateFinishOfPlane,
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


def generate_report(data, df_planes, model_instance, movimientos):
    # global data  # para recuperar data['pDate']

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

# ------------------ HEURÍSTICA GRASP ------------------
def is_feasible_assignment(job, slot, position, solution, data):
    if any((s == slot and p == position) for (s,p) in solution['job_assignments'].values()):
        return False
    for j2,(s2,p2) in solution['job_assignments'].items():
        if s2 == slot and ((position,p2) in data['sPositionsInterfere'] or (p2,position) in data['sPositionsInterfere']):
            return False
    return True


def calculate_earliest_start(job, slot, position, solution, data):
    es = 0
    for j2,ft2 in solution['finish_times'].items():
        if (data['pPlaneOfJob'][j2]==data['pPlaneOfJob'][job]
            and data['pTaskOfJob'][j2]<data['pTaskOfJob'][job]):
            es = max(es, ft2)
    for j2,(s2,p2) in solution['job_assignments'].items():
        if s2==slot and p2==position:
            es = max(es, solution['finish_times'][j2])
    return es

def greedy_initial_solution(data):
    """
    Construye una solución factible SIN dejar huecos en los slots.
    Para cada posición p, siempre ocupa el slot “siguiente libre” (next_slot[p]),
    de forma que nunca puedas saltarte un slot y violar la restricción de continuidad.
    """
    # --- Preparar estructuras ---
    sol = {
        'job_assignments': {},  # job -> (slot, position)
        'start_times': {},      # job -> t_start
        'finish_times': {},     # job -> t_end
        'plane_positions': {}   # plane -> last position usada
    }

    slots_sorted = data['sSlots']  # [slot1, slot2, slot3, …]
    pos_sorted = sorted(data['sPositions'])
    slot_index = {s: i for i, s in enumerate(slots_sorted)}

    # next_slot[p] = primer slot libre en p
    next_slot = {p: slots_sorted[0] for p in pos_sorted}

    # 2) Orden de trabajos (por ejemplo, por fecha temprana)
    jobs = sorted(data['sJobs'], key=lambda j: data['pDate'][j])

    # 3) Asignar cada job
    for j in jobs:
        placed = False

        # calcula t0 mínimo por precedencias
        t0 = data['pDate'][j]
        for j2, ft2 in sol['finish_times'].items():
            if (data['pPlaneOfJob'][j2] == data['pPlaneOfJob'][j]
               and data['pTaskOfJob'][j2] < data['pTaskOfJob'][j]):
                t0 = max(t0, ft2)

        # probar cada posición en orden
        for p in pos_sorted:
            s = next_slot[p]
            if s is None:
                continue  # ya no quedan slots en esta posición

            # 1) Comprobación básica de factibilidad (precedencias, bloqueos, ventanas…)
            if not is_feasible_assignment(j, s, p, sol, data):
                continue

            # 2) Si pasa, calculamos tiempos y asignamos
            st = calculate_earliest_start(j, s, p, sol, data)
            ft = st + data['pJobDuration'][j]

            sol['job_assignments'][j]   = (s, p)
            sol['start_times'][j]       = st
            sol['finish_times'][j]      = ft
            sol['plane_positions'][data['pPlaneOfJob'][j]] = p
            placed = True

            # 3) Avanzar next_slot para esta posición
            idx = slot_index[s] + 1
            next_slot[p] = slots_sorted[idx] if idx < len(slots_sorted) else None

            break  # paso al siguiente job

        if not placed:
            raise ValueError(f"No hay hueco factible para el trabajo {j} en ninguna posición.")

    return sol



# 2) FUNCIÓN DE VALIDACIÓN DE LA SOLUCIÓN GREEDY (VERSIÓN CORREGIDA)
def validate_greedy_solution(model, input_data, greedy_sol, data, timelimit=10):
    instance = model.create_instance(input_data)

    # 1) Fijo los binarios x[s,p,j]
    for s in data['sSlots']:
        for p in data['sPositions']:
            for j in data['sJobs']:
                v = instance.v01JobInSlot[s, p, j]
                if greedy_sol['job_assignments'][j] == (s, p):
                    v.fix(1)
                else:
                    v.fix(0)

    # 2) Fijo los tiempos a nivel job (start/finish en slot y global)
    for j, (s, p) in greedy_sol['job_assignments'].items():
        st = greedy_sol['start_times'][j]
        ft = greedy_sol['finish_times'][j]
        instance.vStartSlotForJob[s, p, j].fix(st)
        instance.vFinishSlotForJob[s, p, j].fix(ft)
        instance.vStartJob[j].fix(st)
        instance.vFinishJob[j].fix(ft)

    # 3) **Agrego**: Fijo los agregados por slot/position
    #    ∑_j vStartSlotForJob[s,p,j] y ∑_j vFinishSlotForJob[s,p,j]
    slot_st = { (s,p): 0 for s in data['sSlots'] for p in data['sPositions'] }
    slot_ft = { (s,p): 0 for s in data['sSlots'] for p in data['sPositions'] }
    for j,(s,p) in greedy_sol['job_assignments'].items():
        slot_st[(s,p)] += greedy_sol['start_times'][j]
        slot_ft[(s,p)] += greedy_sol['finish_times'][j]

    for (s,p), st in slot_st.items():
        instance.vStartSlot[s,p].fix(st)
    for (s,p), ft in slot_ft.items():
        instance.vFinishSlot[s,p].fix(ft)

    # 4) Resolver sólo factibilidad
    solver = SolverFactory('gurobi')
    solver.options['TimeLimit']      = timelimit
    solver.options['MIPFocus']       = 1
    solver.options['FeasibilityTol'] = 1e-6

    result = solver.solve(instance, load_solutions=False, tee=True)
    term = result.solver.termination_condition.name
    if term in ('optimal', 'feasible'):
        print("✅ La solución greedy es totalmente factible.")
        return True
    else:
        print("❌ La solución greedy NO satisface todas las restricciones.")
        return False




def grasp_initial_solutions(data, alpha=0.3, num_solutions=10):
    sols = []
    for _ in range(num_solutions):
        sol = {'job_assignments':{}, 'start_times':{}, 'finish_times':{}, 'plane_positions':{}}
        jobs = sorted(data['sJobs'], key=lambda j: data['pJobDuration'][j], reverse=True)
        unassigned = jobs[:]
        while unassigned:
            candidates = []
            for j in unassigned:
                for s in data['sSlots']:
                    for p in data['sPositions']:
                        if is_feasible_assignment(j,s,p,sol,data):
                            st = calculate_earliest_start(j,s,p,sol,data)
                            ft = st + data['pJobDuration'][j]
                            candidates.append((j,s,p,st,ft))
            if not candidates:
                break
            candidates.sort(key=lambda x: x[4])
            # Manejo de caso con un solo candidato
            if len(candidates) > 1:
                best = candidates[0]
                worst = candidates[-1]
            else:
                best = worst = candidates[0]
            threshold = best[4] + alpha*(worst[4]-best[4])
            rcl = [c for c in candidates if c[4] <= threshold]
            j,s,p,st,ft = random.choice(rcl)
            sol['job_assignments'][j] = (s,p)
            sol['start_times'][j] = st
            sol['finish_times'][j] = ft
            sol['plane_positions'][data['pPlaneOfJob'][j]] = p
            unassigned.remove(j)
        sols.append(sol)
    return sols

# # ------------------ BÚSQUEDA LOCAL (VNS) ------------------
# def swap_two(sol):
#     s = sol.copy()
#     a,b = random.sample(list(sol['job_assignments'].keys()),2)
#     s['job_assignments'][a], s['job_assignments'][b] = s['job_assignments'][b], s['job_assignments'][a]
#     return s
#
# def swap_three(sol):
#     s = sol.copy()
#     keys = random.sample(list(sol['job_assignments'].keys()),3)
#     vals = [sol['job_assignments'][k] for k in keys]
#     # rotate assignments
#     s['job_assignments'][keys[0]] = vals[1]
#     s['job_assignments'][keys[1]] = vals[2]
#     s['job_assignments'][keys[2]] = vals[0]
#     return s
#
# def swap_four(sol):
#     # elige 4 trabajos al azar y rota sus posiciones
#     s = sol.copy()
#     keys = random.sample(list(sol['job_assignments'].keys()), 4)
#     vals = [sol['job_assignments'][k] for k in keys]
#     # rota: 0→1, 1→2, 2→3, 3→0
#     s['job_assignments'][keys[0]] = vals[1]
#     s['job_assignments'][keys[1]] = vals[2]
#     s['job_assignments'][keys[2]] = vals[3]
#     s['job_assignments'][keys[3]] = vals[0]
#     return s
#
#
# def relocate(sol):
#     s = sol.copy()
#     j = random.choice(list(sol['job_assignments'].keys()))
#     new_pos = random.choice(list(sol['plane_positions'].values()))
#     s['job_assignments'][j] = (sol['job_assignments'][j][0], new_pos)
#     return s
#
#
# def vns_search(data, init_sol, neighborhoods, max_no_improve=10):
#     current = init_sol
#     current_obj = sum(data['pJobDuration'][j] for j in current['finish_times'])
#     no_imp = 0
#     while no_imp < max_no_improve:
#         improved = False
#         for neigh in neighborhoods:
#             cand = neigh(current)
#             new_sol = {'job_assignments':{}, 'start_times':{}, 'finish_times':{}, 'plane_positions':{}}
#             for j,(s,p) in cand['job_assignments'].items():
#                 st = calculate_earliest_start(j,s,p,new_sol,data)
#                 ft = st + data['pJobDuration'][j]
#                 new_sol['job_assignments'][j] = (s,p)
#                 new_sol['start_times'][j] = st
#                 new_sol['finish_times'][j] = ft
#                 new_sol['plane_positions'][data['pPlaneOfJob'][j]] = p
#             obj = sum(new_sol['finish_times'].values())
#             if obj < current_obj:
#                 current, current_obj = new_sol, obj
#                 improved = True
#                 break
#         if not improved:
#             no_imp += 1
#     return current

# ------------------ OPERADORES DESTROY (ALNS) ------------------
def remove_block(sol, data, block_size=0.2):
    # elimina asignaciones en un bloque de slots consecutivos
    sol2 = {'job_assignments': sol['job_assignments'].copy()}
    slots = list(data['sSlots'])
    start = random.randint(0, len(slots)-1)
    length = max(1, int(block_size * len(slots)))
    block = slots[start:start+length]
    to_remove = [j for j,(s,p) in sol2['job_assignments'].items() if s in block]
    for j in to_remove:
        sol2['job_assignments'].pop(j)
    # tiempos quedan vacíos, serán recalculados en repair
    sol2['start_times'] = {}
    sol2['finish_times'] = {}
    sol2['plane_positions'] = {}
    return sol2


def remove_plane(sol, data):
    # elimina todas las asignaciones de un avión
    sol2 = {'job_assignments': sol['job_assignments'].copy()}
    plane = random.choice(list(data['sPlanes']))
    to_remove = [j for j in sol2['job_assignments'] if data['pPlaneOfJob'][j]==plane]
    for j in to_remove:
        sol2['job_assignments'].pop(j)
    sol2['start_times'], sol2['finish_times'], sol2['plane_positions'] = {},{},{}
    return sol2


def random_remove(sol, data, fraction=0.2):
    # elimina un porcentaje aleatorio de trabajos
    sol2 = {'job_assignments': sol['job_assignments'].copy()}
    k = max(1, int(fraction * len(sol2['job_assignments'])))
    to_remove = random.sample(list(sol2['job_assignments'].keys()), k)
    for j in to_remove:
        sol2['job_assignments'].pop(j)
    sol2['start_times'], sol2['finish_times'], sol2['plane_positions'] = {},{},{}
    return sol2

# ------------------ OPERADOR REPAIR (HEURÍSTICO VORAZ) ------------------
def repair_greedy(sol_partial, data):
    sol = {'job_assignments': sol_partial['job_assignments'].copy(),
           'start_times': sol_partial.get('start_times',{}).copy(),
           'finish_times': sol_partial.get('finish_times',{}).copy(),
           'plane_positions': sol_partial.get('plane_positions',{}).copy()}
    unassigned = [j for j in data['sJobs'] if j not in sol['job_assignments']]
    # reasigna secuencialmente con greedy makespan
    while unassigned:
        best_choice = None
        best_makespan = float('inf')
        for j in unassigned:
            for s in data['sSlots']:
                for p in data['sPositions']:
                    if is_feasible_assignment(j,s,p,sol,data):
                        st = calculate_earliest_start(j,s,p,sol,data)
                        ft = st + data['pJobDuration'][j]
                        makespan = max(sol['finish_times'].values(), default=0)
                        makespan = max(makespan, ft)
                        if makespan < best_makespan:
                            best_makespan = makespan
                            best_choice = (j,s,p,st,ft)
        if not best_choice:
            break
        j,s,p,st,ft = best_choice
        sol['job_assignments'][j] = (s,p)
        sol['start_times'][j] = st
        sol['finish_times'][j] = ft
        sol['plane_positions'][data['pPlaneOfJob'][j]] = p
        unassigned.remove(j)
    return sol

# ------------------ FUNCIÓN OBJECTIVE ------------------
def makespan(sol):
    return max(sol['finish_times'].values()) if sol['finish_times'] else float('inf')

# ------------------ FLUJO ALNS ------------------

def test_feasibility(candidate, data, time_limit=5):
    """Test de factibilidad rápida: fija binarios y resuelve MIP solo para tiempos."""
    inst2 = ap_pyomo_model().create_instance(create_data(data))
    # liberar y fijar binarios según candidate
    for idx in inst2.v01JobInSlot:
        inst2.v01JobInSlot[idx].unfix()
    for j,(s,p) in candidate['job_assignments'].items():
        inst2.v01JobInSlot[s,p,j].fix(1)
    # solver de factibilidad
    optf = SolverFactory('gurobi')
    optf.options.update({'TimeLimit': time_limit,
                         'MIPFocus': 3,
                         'Heuristics': 0.0,
                         'OutputFlag': 0})
    resf = optf.solve(inst2, tee=False)
    return resf.solver.termination_condition in (TerminationCondition.optimal, TerminationCondition.feasible)

import gurobipy as gp

def diagnose_infeasibility_with_fixings(model, input_data, greedy_sol, data, case_name="case_2_planes"):
    """
    1) Crea la instancia Pyomo y fija los x[s,p,j] según greedy_sol.
    2) Escribe todo en case_name.mps.
    3) Lee case_name.mps con gurobipy, ejecuta computeIIS(), escribe case_name.ilp.
    4) Lista en consola las restricciones y variables del IIS.
    """
    # 1) Instancia Pyomo con fixings
    inst = model.create_instance(input_data)
    for s in data['sSlots']:
        for p in data['sPositions']:
            for j in data['sJobs']:
                v = inst.v01JobInSlot[s, p, j]
                if greedy_sol['job_assignments'].get(j) == (s, p):
                    v.fix(1)
                else:
                    v.fix(0)

    # 2) Volcar a MPS
    mps_file = f"{case_name}.mps"
    inst.write(mps_file, format='mps', io_options={'symbolic_solver_labels': True})
    print(f"✏️  Modelo (con fixings) escrito en {mps_file}")

    # 3) Cargar en gurobipy y generar IIS
    grb = gp.read(mps_file)
    grb.computeIIS()
    ilp_file = f"{case_name}.ilp"
    grb.write(ilp_file)
    print(f"📝 IIS guardado en {ilp_file}")

    # 4) Mostrar en pantalla las constr. y vars del IIS
    infeas_cons = [c.constrName for c in grb.getConstrs() if c.IISConstr]
    infeas_vars = [v.varName    for v in grb.getVars()    if v.IISLB  or v.IISUB]
    print("\n⚠️  Constrains en el IIS (no pueden satisfacerse todas):")
    for name in infeas_cons:
        print("   •", name)
    print("\n⚠️  Variables implicadas en el IIS (bounds conflictivas):")
    for name in infeas_vars:
        print("   •", name)

    return infeas_cons, infeas_vars


def main():
    data = read_excel("input_data.xlsx", "case_4_planes")
    input_data = create_data(data)
    model = ap_pyomo_model()
    instance = model.create_instance(input_data)

    greedy_sol = greedy_initial_solution(data)
    import pprint
    pprint.pprint(greedy_sol)
    if not validate_greedy_solution(model, input_data, greedy_sol, data, timelimit=10):
        diagnose_infeasibility_with_fixings(model, input_data, greedy_sol, data, case_name="case2_case2")
        raise RuntimeError("La solución greedy inicial no es factible. Revisa restricciones.")


    # 1) GRASP
    random_seeds = grasp_initial_solutions(data, alpha=0.2, num_solutions=10)
    seeds = random_seeds +[greedy_sol]
    current = min(seeds, key=makespan)
    best = current.copy()
    # # 2) VNS
    # neighborhoods = [swap_two, swap_three, swap_four, relocate]
    # local = vns_search(data, best, neighborhoods, max_no_improve=20)
    # # 3) LNS
    # current = local
    # for _ in range(10):
    #     current = fix_and_optimize(instance, current, data, block_size=0.4, time_limit=15)
    # # 4) MIP start: solo binarios
    # for idx in instance.v01JobInSlot:
    #     instance.v01JobInSlot[idx].unfix()
    # for j, (s, p) in current['job_assignments'].items():
    #     instance.v01JobInSlot[s, p, j].value = 1

    # operadores ALNS: lista de (destroy, repair)
    operators = [
        (lambda sol: remove_block(sol, data, 0.2), repair_greedy),
        (lambda sol: remove_plane(sol, data), repair_greedy),
        (lambda sol: random_remove(sol, data, 0.2), repair_greedy)
    ]
    scores = {i: 1.0 for i in range(len(operators))}

    max_iter = 100
    for it in range(max_iter):
        idx = random.choices(list(scores.keys()), weights=scores.values())[0]
        destroy, repair = operators[idx]
        partial = destroy(current)
        candidate = repair(partial, data)
        # test factibilidad rápida
        if not test_feasibility(candidate, data):
            # descartamos candidata infactible
            scores[idx] *= 0.5
            continue
        # evaluar makespan
        if makespan(candidate) < makespan(current):
            current = candidate
            scores[idx] += 1.0
            if makespan(candidate) < makespan(best):
                best = candidate
        else:
            scores[idx] *= 0.9  # penalizar

    # MIP start: fijar binarios de mejor
    for idx in instance.v01JobInSlot: instance.v01JobInSlot[idx].value = 0
    for j, (s, p) in best['job_assignments'].items(): instance.v01JobInSlot[s, p, j].value = 1

    # 5) Configurar solver y resolver
    opt = SolverFactory('gurobi')

    # Configuración para mostrar el log detallado de Gurobi
    opt.options['OutputFlag'] = 1        # Activar salida de log
    opt.options['LogToConsole'] = 1      # Mostrar log en consola
    # opt.options['LogFile'] = 'gurobi.log' # También guardar log en archivo
    opt.options['DisplayInterval'] = 1   # Actualizar cada segundo

    # Configuración de límites para la resolución
    opt.options['TimeLimit'] = 1000       # Límite de tiempo en segundos (8 minutos)
    opt.options['MIPGap'] = 0.35         # Gap relativo (5%)

    # Configuración para priorizar heurísticas sobre Branch and Bound
    opt.options['Heuristics'] = 1.0      # Máximo esfuerzo en heurísticas (valor entre 0 y 1)
    opt.options['RINS'] = 1             # Frecuencia de la heurística RINS (menor valor = más frecuente)
    opt.options['MIPFocus'] = 3          # Enfoque en encontrar soluciones factibles rápidamente
    opt.options['ImproveStartGap'] = 0.5  # Comenzar a mejorar la solución cuando el gap sea < 50%
    opt.options['NoRelHeurTime'] = 120    # Aplicar heurísticas en los primeros segundos indicados

    # Reducir el esfuerzo de Branch and Bound
    opt.options['BranchDir'] = -1        # Favorecer branch hacia abajo (menos exploración)
    opt.options['MinRelNodes'] = 1000    # Limitar el número de nodos procesados

    opt.options['Presolve'] = 2
    opt.options['Cuts'] = 2
    opt.options['Threads'] = 4

    print("Iniciando resolución con Gurobi (con GRASP+VNS+LNS)...")
    results = opt.solve(instance, tee=True)

    solution = get_solution_data(instance)
    print("Solución híbrida completa.")

    print("\nEstado del solucionador:", results.solver.status.value)

    if results.solver.termination_condition == TerminationCondition.infeasible:


        import sys
        import logging

        logger = logging.getLogger('pyomo.util.infeasible')
        logger.setLevel(logging.INFO)
        with open('infeasible.log', 'w') as f:
            sys.stdout = f
            log_infeasible_constraints(instance)
            sys.stdout = sys.__stdout__
            print("⚠️ Modelo infeasible: generando IIS en 'conflict.ilp' …")

    #
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
                    f_teor = instance.pPredictedFinishOfPlane[r]
                    print(f"  Último trabajo: {j}")
                    print(f"    → Fecha real  = {f_real:.1f}")
                    print(f"    → Fecha límite= {value(f_teor):.1f}")
                    print(f"    → Retraso     = {instance.vPlaneDelay[r].value:.1f}")

        generate_report(data, df_full, instance, movimientos)
    else:
        print("No se pudo encontrar una solución óptima.")
        print(f"Condición de terminación: {results.solver.termination_condition}")

    print("done")


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

if __name__=='__main__':
    sol = main()
    if sol is None:
        print("No se obtuvo solución híbrida.")
    else:
        print("Solución obtenida.")


