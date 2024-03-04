from typing import List
import ray
import numpy as np

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

from preprocess import to_tensor_transform,resize_transform,normalize_transform,random_horizontal_flip_transform


if __name__ == "__main__":

    path = r"./chest_xray/test/NORMAL"
    ds = ray.data.read_images(path) \
        .map_batches(to_tensor_transform) \
        .map_batches(resize_transform) \
        .map_batches(normalize_transform) \
        .map_batches(random_horizontal_flip_transform) 
        
    
    plan = ds._logical_plan
    print(f"Logical Plan \n -> {plan.dag}\n")
    optimized_logical_plan = LogicalOptimizer().optimize(plan)
    print(f"Optimized Logical Plan \n -> {optimized_logical_plan.dag}\n")
    physical_plan = Planner().plan(optimized_logical_plan)
    print(f"Physical Plan \n -> {physical_plan.dag}\n")
    optimized_physical_plan = PhysicalOptimizer().optimize(physical_plan)
    print(f"Optimized Physical Plan \n -> {optimized_physical_plan._dag}\n")

   
