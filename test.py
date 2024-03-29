from typing import List
import ray

from ray.data._internal.logical.interfaces import (
    LogicalPlan,
    Optimizer,
    PhysicalPlan,
    Rule,
)
from ray.data._internal.logical.operators.all_to_all_operator import (
    RandomizeBlocks,
    Repartition,
)
from ray.data._internal.logical.operators.map_operator import AbstractUDFMap
from ray.data._internal.logical.operators.read_operator import Read
from ray.data._internal.logical.optimizers import LogicalOptimizer,DEFAULT_PHYSICAL_RULES,PhysicalOptimizer

from add_user_rule import add_user_provided_physical_rules
from ray.data._internal.planner.planner import Planner




def dummy_map(x):
    return x*x

    


if __name__ == "__main__":
    ds = ray.data.range(10).randomize_block_order().map_batches(dummy_map)
    plan = ds._logical_plan
    print(plan.dag)
    optimized_logical_plan = LogicalOptimizer().optimize(plan)
    physical_plan = Planner().plan(optimized_logical_plan)
    # 4개 다 실행하는 지?
    print(physical_plan.dag)
    print()
    optimized_physical_plan = PhysicalOptimizer().optimize(physical_plan)
    print(optimized_physical_plan._dag)
    print()
   # print(optimized_physical_plan._op_map)




    # inverse_order = iter([Read, AbstractUDFMap, RandomizeBlocks])
    # for node in optimized_plan.dag.post_order_iter():
    #     print(node)
    #     assert isinstance(node, next(inverse_order))

    # ds = (
    #     ray.data.range(10)
    #     .randomize_block_order()
    #     .repartition(10)
    #     .map_batches(dummy_map)
    # )
    # plan = ds._logical_plan
    # optimized_plan = LogicalOptimizer().optimize(plan)

    # inverse_order = iter([Read, RandomizeBlocks, Repartition, AbstractUDFMap])
    # for node in optimized_plan.dag.post_order_iter():
    #     assert isinstance(node, next(inverse_order))
