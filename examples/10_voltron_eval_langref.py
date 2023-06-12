"""
evaluate_refer.py

Example script for loading a pretrained V-Cond model (from the `voltron` library), configuring a MAP-based extractor
factory function, and then defining/invoking the ReferDetectionHarness.
"""
from pathlib import Path

import torch
import voltron_evaluation as vet
from voltron import instantiate_extractor, load


def evaluate_refer() -> None:
    # Load Backbone (V-Cond)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    backbone, preprocess = load("v-cond", device=device)

    # Create MAP Extractor Factory (single latent =>> we only predict of a single dense vector representation)
    map_extractor_fn = instantiate_extractor(backbone, n_latents=1)

    # Create Refer Detection Harness
    path = Path("data/voltron/langref")
    refer_evaluator = vet.ReferDetectionHarness("v-cond", backbone, preprocess, map_extractor_fn, data=path)
    refer_evaluator.fit()
    refer_evaluator.test()


if __name__ == "__main__":
    evaluate_refer()
