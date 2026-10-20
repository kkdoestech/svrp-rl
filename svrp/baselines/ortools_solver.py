"""Google OR-Tools CVRP on the *estimated* demands and costs (the classical a-priori approach).

This is the tool you already know: it plans as if the estimates were the truth. The executor then
drives the plan on the real day and pays for recourse when reality differs.
Two metaheuristics: Guided Local Search (usually OR-Tools' strongest) and Tabu Search (the paper's TS).
"""
from __future__ import annotations

import math

import numpy as np
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

from svrp.baselines.common import planning_demand, routes_to_plan

COST_SCALE = 1000      # OR-Tools needs integers: costs in 1/1000 units
DEMAND_SCALE = 100

METAHEURISTICS = {
    "gls": routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH,
    "tabu": routing_enums_pb2.LocalSearchMetaheuristic.TABU_SEARCH,
}


def default_time_limit(n_customers: int) -> float:
    return 0.5 if n_customers <= 10 else 1.0 if n_customers <= 20 else 3.0


def ortools_cvrp(dem: np.ndarray, cost: np.ndarray, capacity: float, time_limit: float = 1.0,
                 metaheuristic: str = "gls", return_objective: bool = False):
    N1 = len(dem)
    d = planning_demand(dem, capacity)
    d_int = [int(round(x * DEMAND_SCALE)) for x in d]
    cap_int = int(round(capacity * DEMAND_SCALE))
    c_int = np.rint(cost * COST_SCALE).astype(np.int64).tolist()
    n_veh = min(N1 - 1, 2 * math.ceil(d[1:].sum() / capacity) + 1)

    manager = pywrapcp.RoutingIndexManager(N1, n_veh, 0)
    routing = pywrapcp.RoutingModel(manager)
    arc = routing.RegisterTransitMatrix(c_int)
    routing.SetArcCostEvaluatorOfAllVehicles(arc)
    dem_cb = routing.RegisterUnaryTransitVector(d_int)
    routing.AddDimensionWithVehicleCapacity(dem_cb, 0, [cap_int] * n_veh, True, "load")

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    params.local_search_metaheuristic = METAHEURISTICS[metaheuristic]
    params.time_limit.FromMilliseconds(int(time_limit * 1000))
    sol = routing.SolveWithParameters(params)
    if sol is None:
        raise RuntimeError("OR-Tools found no solution")

    routes = []
    for v in range(n_veh):
        idx, r = routing.Start(v), []
        while not routing.IsEnd(idx):
            node = manager.IndexToNode(idx)
            if node != 0:
                r.append(node)
            idx = sol.Value(routing.NextVar(idx))
        routes.append(r)
    plan = routes_to_plan(routes)
    if return_objective:
        return plan, sol.ObjectiveValue() / COST_SCALE
    return plan
