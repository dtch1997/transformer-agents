from pathlib import Path

import voltron_evaluation as vet
from voltron import instantiate_extractor, load

# Load a frozen Voltron (V-Cond) model & configure a MAP extractor
backbone, preprocess = load("v-cond", device="cuda", freeze=True)
map_extractor_fn = instantiate_extractor(backbone)

path = Path("data/voltron")
# Create ARC Grasping Harness
grasp_evaluator = vet.GraspAffordanceHarness("v-cond", backbone, preprocess, map_extractor_fn, data=path)
grasp_evaluator.fit()
grasp_evaluator.test()
