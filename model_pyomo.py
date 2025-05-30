from pyomo.environ import *
from pyomo.gdp import Disjunct
from pyomo.core.base.block import Block
import pandas as pd
import plotly.express as px
import datetime

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
    model.sPositionsInterefence = Set(dimen=2)
    model.sPosPosSlotSlot = Set(dimen=4)

    model.sSlotsSequence = Set(dimen=3)
    model.sJobSequence = Set(dimen=2)
    model.sSwitchPlanes = Set(dimen=5)

    # Parameters
    model.pHorizon = Param()
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
    model.vClientPostion = Var(model.sClients, model.sPositions, domain=Binary)
    model.vClientDelay = Var(model.sClients, within=NonNegativeReals)
    model.vPlaneDelay = Var(model.sPlanes, within=NonNegativeReals)
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

    # # Rule: Ec. cJobDuration - The total duration of a job is the sum of the duration of all corresponding slots
    # def fc05_JobDuration(model, j):
    #     return sum(model.vDurationSlotForJob[s, p, j] for s in model.sSlots for p in model.sPositions) == \
    #            model.pJobDuration[j]

# # Constraints 6 and 7 having s, p, j as arguments
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
#Constraints 6 and 7 just having only jobs as arguments as stated in constraint 16 every job must be assigned
    def fc06_GlobalStartConstraint(model, j):
        return model.vStartJob[j] == sum(
            model.vStartSlotForJob[s, p, j]
            for s in model.sSlots
            for p in model.sPositions
        )

    def fc07_GlobalFinishConstraint(model, j):
        return model.vFinishJob[j] == sum(
            model.vFinishSlotForJob[s, p, j]
            for s in model.sSlots
            for p in model.sPositions
        )

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

    # Rule: Ec. noEmptySlots - Consecutive slots - a slot is not used unless all previous ones have been used
    def fc15_ConsecutiveSlots(model, s, p):
        # Skip constraint for the first slot (s=1)
        if model.sSlots.ord(s) == 1:
            return Constraint.Skip

        # Get the previous slot
        prev_s = list(model.sSlots)[model.sSlots.ord(s) - 2]  # -1 for 0-based indexing, -1 for previous

        # Sum of job assignments in the current slot must be equal to sum in previous slot
        return sum(model.v01JobInSlot[s, p, j] for j in model.sJobs) == sum(model.v01JobInSlot[prev_s, p, j] for j in model.sJobs)

    # Rule: Ec. - A job can be assigned to a single slot of a position
    def fc16_SingleSlotPerJob(model, j):
        # ∑∑ x_jsp = 1 ∀j ∈ J
        return sum(model.v01JobInSlot[s, p, j] 
               for s in model.sSlots for p in model.sPositions) == 1

    # Rule: Ec. - If a job is not assigned to a slot of a position, the duration of that job in that slot is zero
    def fc17_DurationIfNotAssigned(model, s, p, j):
        # d^j_spj = D_j·x_spj, ∀s ∈ S, p ∈ P, j ∈ J
        return model.vDurationSlotForJob[s, p, j] == model.pJobDuration[j] * model.v01JobInSlot[s, p, j]

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

    # Rule: Client c with some airpline in position p:
    def fc21_ClientInPosition(model, c, p):
        return model.vClientPosition[c, p] >= model.v01PlaneInPosition[model.pPlaneOfClient[c], p]

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
        return 1 + model.v01SwitchPlanes[s, p] >= model.v01PlaneInSlot[s, p, r] + model.v01PlaneInSlot[s2, p, r2]

    # Rule: If a job is split among different slots, these cannot overlap
    def fc26_NoOverlapSlots(model, s, s2, p, p2, j):
        # Skip if it's the same slot and position
        if s == s2 or p == p2:
            return Constraint.Skip

        # 1 + βS_{ss'pp'} + βF_{ss'pp'} >= x_{spj} + x_{s'p'j}
        # This ensures that if the same job is assigned to different slots,
        # either one starts after the other finishes or vice versa
        return 1 + model.v01BetaS[s, s2, p, p2] + model.v01BetaF[s, s2, p, p2] >= \
               model.v01JobInSlot[s, p, j] + model.v01JobInSlot[s2, p2, j]


    # Rule: función objetivo
    def fc27_NoMovements(model):
        return sum(model.v01JobInSlot[s, p, j] for s in model.sSlots for p in model.sPositions for j in model.sJobs) \
                + sum(model.v01Alpha[i] for i in model.sPosPosSlotSlot) \
                + sum(model.v01SwitchPlanes[s, p] for p in model.sPositions for s in model.sSlots) \
                + sum(model.v01PlaneInPosition[r, p] for r in model.sPlanes for p in model.sPositions)


    # Activating constraints
    print("Generating c01_SingleJobPerSlot constraint - Eq. cSingleJobPerSlot")
    model.c01_SingleJobPerSlot = Constraint(model.sSlots, model.sPositions, rule=fc01_SingleJobPerSlot)

    print("Generating c02_SlotJobDuration constraint - Eq. cSlotJobDuration")
    model.c02_SlotJobDuration = Constraint(model.sSlots, model.sPositions, model.sJobs, rule=fc02_SlotJobDuration)

    print("Generating c03_NullStartTimeIfNotInSlot constraint - Eq. nullStartIfNotAssigned")
    model.c03_NullStartTimeIfNotInSlot = Constraint(model.sSlots, model.sPositions, model.sJobs, rule=fc03_NullStartTimeIfNotInSlot)

    print("Generating c04_NullFinishTimeIfNotInSlot constraint - Eq. nullFinishIfNotAssigned")
    model.c04_NullFinishTimeIfNotInSlot = Constraint(model.sSlots, model.sPositions, model.sJobs, rule=fc04_NullFinishTimeIfNotInSlot)

    # #print("Generating c05_JobDuration constraint - Eq. cJobDuration")
    # #model.c05_JobDuration = Constraint(model.sJobs, rule=fc05_JobDuration)

    # #Restrictions 6 and 7 having as arguments s, p and j.
    # print("Generating c06_GlobalStartConstraint constraint - Eq. startGlobalLowerBoundNoCommas")
    # model.c06_GlobalStartConstraint = Constraint( model.sSlots, model.sPositions, model.sJobs, rule=fc06_GlobalStartConstraint)
    #
    # print("Generating c07_GlobalFinishConstraint constraint - Eq. finishGlobalUpperBoundNoCommas")
    # model.c07_GlobalFinishConstraint = Constraint( model.sSlots, model.sPositions, model.sJobs, rule=fc07_GlobalFinishConstraint)

    print("Generating c06_GlobalStartConstraint constraint - Eq. startGlobalLowerBoundNoCommas")
    model.c06_GlobalStartConstraint = Constraint( model.sJobs, rule=fc06_GlobalStartConstraint)

    print("Generating c07_GlobalFinishConstraint constraint - Eq. finishGlobalUpperBoundNoCommas")
    model.c07_GlobalFinishConstraint = Constraint( model.sJobs, rule=fc07_GlobalFinishConstraint)

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

    print("Generating c21_ClientInPosition constraint")
    model.c21_ClientInPosition = Constraint(model.sClients, model.sPositions, rule=fc21_ClientInPosition)

    print("Generating c22_BetaDefinition1 constraint - Eq. fcBetaDefinion1")
    model.c22_BetaDefinition1 = Constraint(model.sPosPosSlotSlot, rule=fc22_BetaDefinition1)

    print("Generating c23_BetaDefinition2 constraint - Eq. fcBetaDefinion2")
    model.c23_BetaDefinition2 = Constraint(model.sPosPosSlotSlot, rule=fc23_BetaDefinition2)

    print("Generating c24_InterferenceExists constraint")
    model.c24_InterferenceExists = Constraint(model.sPosPosSlotSlot, rule=fc24_InterferenceExists)

    # print("Generating c25_SwitchingPlanes constraint - Eq. PlaneSwitchInPOsition")
    # model.c25_SwitchingPlanes = Constraint(model.sSwitchPlanes, rule=fc25_SwitchingPlanes)
    #
    # print("Generating c26_NoOverlapSlots constraint")
    # model.c26_NoOverlapSlots = Constraint(model.sSlots, model.sSlots, model.sPositions, model.sPositions, model.sJobs, rule=fc26_NoOverlapSlots)

    #Objective function
    print("Generating objective function")
    model.ObjFunction = Objective(rule=fc27_NoMovements, sense=minimize)

    return model


def read_excel(file_name, sheet_name):
    df = pd.read_excel(file_name, sheet_name=sheet_name)
    sJobs = df.job.to_list()
    sPositions = POSITIONS
    sPositionsInterfere = POSITIONS_INTERFERE

    sSlots = ['slot{}'.format(i) for i in range(ceil(len(sJobs) / NO_POSITIONS * 1.5))]  # TODO: remove 1.2 harcode
    sPlanes = list(df.plane.unique())
    pJobDuration = df.set_index('job')['duration'].to_dict()
    pDate = df.set_index('job')['date'].to_dict()
    pPlaneOfJob = df.set_index('job')['plane'].to_dict()

    pTaskOfJob = df.set_index('job')['task'].to_dict()

    pHorizon = max(sum(pJobDuration[j] for j in sJobs if pPlaneOfJob[j] == r) for r in sPlanes) * 1.2 # TODO: remove hardcoded 1.2

    data = {'sJobs': sJobs,
            'sSlots': sSlots,
            'sPositions': sPositions,
            'sPlanes': sPlanes,
            'sPositionsInterfere': sPositionsInterfere,
            'pJobDuration': pJobDuration,
            'pPlaneOfJob': pPlaneOfJob,
            'pTaskOfJob': pTaskOfJob,
            'pDate': pDate,
            'pHorizon': pHorizon
            }
    return data


def create_data(data):
    sPositions = data.get('sPositions', None)
    sPositionsInterfere = data.get('sPositionsInterfere', None)
    sJobs = data.get('sJobs', None)
    sPlanes = data.get('sPlanes', None)
    # sSlots = data.get('sSlots', None)
    pJobDuration = data.get('pJobDuration', None)
    pDate = data.get('pDate', None)
    pHorizon = data.get('pHorizon')
    pPlaneOfJob = data.get('pPlaneOfJob')
    pTaskOfJob = data.get('pTaskOfJob')

    #Alternative version for slot calculation in base of duration of jobs
    sSlots = ['slot{}'.format(i) for i in range(10)]

    sSlotsSequence = [(s, s2, p) for p in sPositions for s in sSlots for s2 in sSlots
                      if sSlots.index(s) == sSlots.index(s2) + 1]

    # sJobSequence = [(j, j2) for j in sJobs for j2 in sJobs if pPlaneOfJob[j] == pPlaneOfJob[j2]
    #                 and pTaskOfJob[j] < pTaskOfJob[j2]]

    #Alternative version of sJobSequence
    sJobSequence = []
    for r in sPlanes:
        # Filtrar trabajos por avión
        jobs_r = [j for j in sJobs if pPlaneOfJob[j] == r]

        # Verificar que las tareas sean números y no estén duplicadas
        try:
            task_list = [(j, int(pTaskOfJob[j])) for j in jobs_r]
        except ValueError as e:
            raise ValueError(f"Error en las tareas del avión {r}: asegúrate de que sean números enteros. {e}")

        # Ordenar los trabajos por número de tarea
        task_list.sort(key=lambda x: x[1])

        # Crear pares secuenciales
        for i in range(len(task_list) - 1):
            j1, task1 = task_list[i]
            j2, task2 = task_list[i + 1]
            if task1 < task2:
                sJobSequence.append((j1, j2))
            else:
                print(
                    f"⚠️ Advertencia: Tareas fuera de orden o repetidas para avión {r}: {j1} (tarea {task1}), {j2} (tarea {task2})")

    # Imprimir para verificación
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
        'pDate': pDate
    }}

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


def print_chart(solution):
    slot_assignment = solution.get('slot_assignment', None)
    start_slot = solution.get('start_slot', None)
    finish_slot = solution.get('finish_slot', None)

    data = []
    for key, j in slot_assignment.items():
        s, p = key
        start = round(start_slot.get((s, p), None), 2)
        start_date = (START_DATE + datetime.timedelta(days=start)).strftime("%Y-%m-%d")
        finish = round(finish_slot.get((s, p), None), 2)
        finish_date = (START_DATE + datetime.timedelta(days=finish)).strftime("%Y-%m-%d")

        # Handle different types of job identifiers
        if isinstance(j, (list, tuple)) and len(j) > 0:
            # If j is a list or tuple, use the first element as the plane
            plane = j[0]
        else:
            # If j is not a list or tuple, use j as the plane identifier
            plane = j

        # Append data to the list
        data.append({'s': s, 'p': p, 'j': j, 'start_slot': start_date, 'finish_slot': finish_date, 'plane': plane})

    # Create a DataFrame from the list of dictionaries
    df = pd.DataFrame(data)
    fig = px.timeline(df, x_start="start_slot", x_end="finish_slot", y="p", color="plane")
    fig.show()

    return 0


def check_solution(data, solution):
    """
    Verifica que la solución cumpla con todos los requisitos del modelo de posicionamiento de aeronaves.

    Args:
        data: Diccionario con los datos de entrada
        solution: Diccionario con los resultados de la solución

    Returns:
        dict: Diccionario con los resultados de las verificaciones, con una entrada por cada restricción
             verificada. Cada entrada contiene un booleano que indica si se cumple la restricción y un 
             mensaje de error si no se cumple.
    """
    # Unpacking input data
    sPositions = data.get('sPositions', [])
    sPositionsInterfere = data.get('sPositionsInterfere', [])
    sJobs = data.get('sJobs', [])
    sPlanes = data.get('sPlanes', [])
    sSlots = data.get('sSlots', [])
    pJobDuration = data.get('pJobDuration', {})
    pPlaneOfJob = data.get('pPlaneOfJob', {})
    pTaskOfJob = data.get('pTaskOfJob', {})
    pHorizon = data.get('pHorizon', 0)

    # Unpacking solution data
    slot_assignment = solution.get('slot_assignment', {})
    duration_slot = solution.get('duration_slot', {})
    duration_slot_job = solution.get('duration_slot_job', {})
    interference = solution.get('interference', [])
    start_slot_job = solution.get('start_slot_job', {})
    finish_slot_job = solution.get('finish_slot_job', {})
    start_slot = solution.get('start_slot', {})
    finish_slot = solution.get('finish_slot', {})

    # Inicializar diccionario de resultados
    verification_results = {}

    # 1. Verificar que cada trabajo esté asignado exactamente una vez
    verification_results['all_jobs_assigned'] = {'passed': True, 'errors': []}
    assigned_jobs = [j for _, j in slot_assignment.items()]
    for j in sJobs:
        if j not in assigned_jobs:
            verification_results['all_jobs_assigned']['passed'] = False
            verification_results['all_jobs_assigned']['errors'].append(f"Job {j} no está asignado a ninguna ranura")

    # 2. Verificar que cada ranura tenga a lo sumo un trabajo asignado
    verification_results['single_job_per_slot'] = {'passed': True, 'errors': []}
    for p in sPositions:
        for s in sSlots:
            jobs_in_slot = [j for (slot, pos), j in slot_assignment.items() if slot == s and pos == p]
            if len(jobs_in_slot) > 1:
                verification_results['single_job_per_slot']['passed'] = False
                verification_results['single_job_per_slot']['errors'].append(
                    f"La ranura {s} en la posición {p} tiene múltiples trabajos asignados: {jobs_in_slot}")

    # 3. Verificar que la duración total de cada trabajo sea correcta
    verification_results['job_duration_correct'] = {'passed': True, 'errors': []}
    for j in sJobs:
        if j in assigned_jobs:
            total_duration = sum(duration_slot_job.get((s, p, j), 0) for s in sSlots for p in sPositions)
            if abs(total_duration - pJobDuration[j]) > 1e-6:  # Tolerancia para errores de punto flotante
                verification_results['job_duration_correct']['passed'] = False
                verification_results['job_duration_correct']['errors'].append(
                    f"El trabajo {j} debería tener duración {pJobDuration[j]}, pero tiene {total_duration}")

    # 4. Verificar que la duración de las ranuras sea consistente con los trabajos asignados
    verification_results['slot_duration_consistent'] = {'passed': True, 'errors': []}
    for (s, p), dur in duration_slot.items():
        job_duration_sum = sum(duration_slot_job.get((s, p, j), 0) for j in sJobs)
        if abs(dur - job_duration_sum) > 1e-6:
            verification_results['slot_duration_consistent']['passed'] = False
            verification_results['slot_duration_consistent']['errors'].append(
                f"La ranura {s} en posición {p} tiene duración {dur}, pero la suma de duraciones de trabajos es {job_duration_sum}")

    # 5. Verificar que los tiempos de inicio y fin de las ranuras son consistentes
    verification_results['slot_times_consistent'] = {'passed': True, 'errors': []}
    for s in sSlots:
        for p in sPositions:
            if (s, p) in start_slot and (s, p) in finish_slot:
                if abs((finish_slot[(s, p)] - start_slot[(s, p)]) - duration_slot.get((s, p), 0)) > 1e-6:
                    verification_results['slot_times_consistent']['passed'] = False
                    verification_results['slot_times_consistent']['errors'].append(
                        f"La ranura {s} en posición {p} tiene inconsistencia en tiempos: inicio={start_slot[(s, p)]}, "
                        f"fin={finish_slot[(s, p)]}, duración={duration_slot.get((s, p), 0)}")

    # 6. Verificar que no hay solapamiento entre ranuras consecutivas en la misma posición
    verification_results['no_overlap_same_position'] = {'passed': True, 'errors': []}
    for p in sPositions:
        slots_in_pos = sorted([(s, start_slot.get((s, p), 0)) for s in sSlots if (s, p) in start_slot],
                             key=lambda x: x[1])
        for i in range(len(slots_in_pos) - 1):
            current_slot, current_start = slots_in_pos[i]
            next_slot, next_start = slots_in_pos[i + 1]
            current_end = finish_slot.get((current_slot, p), 0)

            if current_end > next_start + 1e-6:  # Pequeña tolerancia
                verification_results['no_overlap_same_position']['passed'] = False
                verification_results['no_overlap_same_position']['errors'].append(
                    f"Solapamiento en posición {p}: Ranura {current_slot} termina en {current_end}, "
                    f"pero ranura {next_slot} comienza en {next_start}")

    # 7. Verificar la secuencia de trabajos para el mismo avión
    verification_results['job_sequence_correct'] = {'passed': True, 'errors': []}
    for plane in sPlanes:
        # Obtener todos los trabajos de este avión
        plane_jobs = [(j, pTaskOfJob[j]) for j in sJobs if pPlaneOfJob[j] == plane]
        # Ordenar por número de tarea
        plane_jobs.sort(key=lambda x: x[1])

        # Verificar que los trabajos se realizan en secuencia
        for i in range(len(plane_jobs) - 1):
            current_job, current_task = plane_jobs[i]
            next_job, next_task = plane_jobs[i + 1]

            # Calcular tiempos de finalización y comienzo
            current_finish_times = [finish_slot_job.get((s, p, current_job), 0)
                                   for s in sSlots for p in sPositions
                                   if (s, p, current_job) in finish_slot_job]
            next_start_times = [start_slot_job.get((s, p, next_job), 0)
                               for s in sSlots for p in sPositions
                               if (s, p, next_job) in start_slot_job]

            if current_finish_times and next_start_times:
                current_finish = max(current_finish_times)
                next_start = min(next_start_times)

                if current_finish > next_start + 1e-6:
                    verification_results['job_sequence_correct']['passed'] = False
                    verification_results['job_sequence_correct']['errors'].append(
                        f"Secuencia incorrecta para avión {plane}: Trabajo {current_job} (tarea {current_task}) "
                        f"termina en {current_finish}, pero trabajo {next_job} (tarea {next_task}) comienza en {next_start}")

    # 8. Verificar que no hay interferencia entre posiciones que no deben solaparse
    verification_results['no_position_interference'] = {'passed': True, 'errors': []}
    for p1, p2 in sPositionsInterfere:
        # Obtener todas las ranuras activas en cada posición
        slots_p1 = [(s, start_slot.get((s, p1), 0), finish_slot.get((s, p1), 0))
                    for s in sSlots if (s, p1) in start_slot and (s, p1) in finish_slot]
        slots_p2 = [(s, start_slot.get((s, p2), 0), finish_slot.get((s, p2), 0))
                    for s in sSlots if (s, p2) in start_slot and (s, p2) in finish_slot]

        # Verificar solapamientos
        for s1, start1, end1 in slots_p1:
            for s2, start2, end2 in slots_p2:
                # Hay solapamiento si el inicio de uno está entre el inicio y fin del otro
                if (start1 <= start2 < end1) or (start2 <= start1 < end2):
                    found_in_interference = False
                    for s, s_2, pos, pos_2 in interference:
                        if ((s == s1 and s_2 == s2 and pos == p1 and pos_2 == p2) or
                            (s == s2 and s_2 == s1 and pos == p2 and pos_2 == p1)):
                            found_in_interference = True
                            break

                    if not found_in_interference:
                        verification_results['no_position_interference']['passed'] = False
                        verification_results['no_position_interference']['errors'].append(
                            f"Interferencia no registrada entre posiciones {p1} y {p2}: "
                            f"Ranura {s1} ({start1}-{end1}) y ranura {s2} ({start2}-{end2})")

    # 9. Verificar que todos los trabajos se completan dentro del horizonte
    verification_results['within_horizon'] = {'passed': True, 'errors': []}
    for (s, p), end_time in finish_slot.items():
        if end_time > pHorizon + 1e-6:
            verification_results['within_horizon']['passed'] = False
            verification_results['within_horizon']['errors'].append(
                f"La ranura {s} en posición {p} termina en {end_time}, que excede el horizonte {pHorizon}")

    # 10. Verificar que los aviones no están en diferentes posiciones al mismo tiempo
    verification_results['plane_single_position'] = {'passed': True, 'errors': []}
    for plane in sPlanes:
        # Obtener todos los slots asignados a trabajos de este avión
        plane_slots = []
        for (s, p), j in slot_assignment.items():
            if pPlaneOfJob[j] == plane:
                plane_slots.append((s, p, start_slot.get((s, p), 0), finish_slot.get((s, p), 0)))

        # Verificar solapamientos entre posiciones diferentes
        for i in range(len(plane_slots)):
            s1, p1, start1, end1 = plane_slots[i]
            for j in range(i+1, len(plane_slots)):
                s2, p2, start2, end2 = plane_slots[j]
                if p1 != p2:  # Diferentes posiciones
                    # Hay solapamiento si el inicio de uno está entre el inicio y fin del otro
                    if (start1 <= start2 < end1) or (start2 <= start1 < end2):
                        verification_results['plane_single_position']['passed'] = False
                        verification_results['plane_single_position']['errors'].append(
                            f"Avión {plane} está en múltiples posiciones al mismo tiempo: "
                            f"Posición {p1} ({start1}-{end1}) y posición {p2} ({start2}-{end2})")

    # 11. Verificar que no se utilice un slot a menos que se hayan utilizado todos los anteriores
    verification_results['consecutive_slots'] = {'passed': True, 'errors': []}

    # Ordenar slots por número
    sorted_slots = sorted(sSlots, key=lambda x: int(x.replace('slot', '')))

    for p in sPositions:
        for i in range(1, len(sorted_slots)):
            current_slot = sorted_slots[i]
            prev_slot = sorted_slots[i-1]

            # Contar trabajos asignados a cada slot
            jobs_in_current = sum(1 for (s, pos), _ in slot_assignment.items() if s == current_slot and pos == p)
            jobs_in_prev = sum(1 for (s, pos), _ in slot_assignment.items() if s == prev_slot and pos == p)

            if jobs_in_current > 0 and jobs_in_prev == 0:
                verification_results['consecutive_slots']['passed'] = False
                verification_results['consecutive_slots']['errors'].append(
                    f"Posición {p}: Se utiliza el slot {current_slot} pero no se utiliza el slot anterior {prev_slot}")

    # Resumen final
    all_passed = all(result['passed'] for result in verification_results.values())
    summary = {
        'all_constraints_satisfied': all_passed,
        'constraints_verification': verification_results
    }

    return summary


if __name__ == "__main__":

    # reading data from Excel
    data = read_excel("input_data.xlsx", "case_1_plane")
    # data = read_excel("input_data.xlsx", "case_3_planes")

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
    opt.options['TimeLimit'] = 300       # Límite de tiempo en segundos (5 minutos)
    opt.options['MIPGap'] = 0.05         # Gap relativo (5%)

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
    else:
        print("No se pudo encontrar una solución óptima.")
        print(f"Condición de terminación: {results.solver.termination_condition}")

    print("done")

print("\n🔍 Revisión rápida de asignaciones por trabajo:")
for j in instance.sJobs:
    assigned_slots = [(s, p) for s in instance.sSlots for p in instance.sPositions if instance.v01JobInSlot[s, p, j].value == 1]
    if len(assigned_slots) != 1:
        print(f"⚠️ Job {j} está asignado a {len(assigned_slots)} slots: {assigned_slots}")

print("\n🔍 Verificando dominios de v01JobInSlot:")
for s in instance.sSlots:
    for p in instance.sPositions:
        for j in instance.sJobs:
            exists = (s, p, j) in instance.v01JobInSlot
            print(f"  {'✔️' if exists else '❌'} v01JobInSlot[{s},{p},{j}]")
