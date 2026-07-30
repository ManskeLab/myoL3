"""Muscle segmentation on the L3 crop.

Mirrors the VeinSeg approach: instead of the nnU-Net folder/CLI layout, load the
self-contained checkpoint in Python and drive `nnUNetPredictor` via
`manual_initialization`. The checkpoint carries everything needed —
`init_args["plans"]`, `init_args["configuration"]`, `init_args["dataset_json"]`,
`trainer_name`, `inference_allowed_mirroring_axes` — so no plans.json/dataset.json
files are required on disk.

The sliding-window inference (patch size, gaussian weighting, mirroring, step
size) is identical to nnU-Net training-time inference, because it *is*
nnUNetPredictor running off the trained plans.
"""
from __future__ import annotations

import os
import tempfile

import numpy as np
import SimpleITK as sitk
import torch

from . import config

# nnUNet emits warnings if these aren't set; we don't use its filesystem layout.
os.environ.setdefault("nnUNet_raw", tempfile.gettempdir())
os.environ.setdefault("nnUNet_preprocessed", tempfile.gettempdir())
os.environ.setdefault("nnUNet_results", tempfile.gettempdir())


class MuscleSegmenter:
    def __init__(self, ckpt: str, device: str = "cpu"):
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
        from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
        from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
        from nnunetv2.utilities.label_handling.label_handling import (
            determine_num_input_channels)

        self.device = torch.device(device)
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        ia = ck["init_args"]

        plans_manager = PlansManager(ia["plans"])
        configuration_manager = plans_manager.get_configuration(ia["configuration"])
        dataset_json = ia["dataset_json"]

        num_input = determine_num_input_channels(
            plans_manager, configuration_manager, dataset_json)
        label_manager = plans_manager.get_label_manager(dataset_json)

        network = get_network_from_plans(
            configuration_manager.network_arch_class_name,
            configuration_manager.network_arch_init_kwargs,
            configuration_manager.network_arch_init_kwargs_req_import,
            num_input,
            label_manager.num_segmentation_heads,
            allow_init=True,
            deep_supervision=False,
        )
        network.load_state_dict(ck["network_weights"])

        self.predictor = nnUNetPredictor(
            tile_step_size=config.SEG_TILE_STEP,
            use_gaussian=True,
            use_mirroring=config.SEG_USE_MIRRORING,
            perform_everything_on_device=(self.device.type != "cpu"),
            device=self.device,
            verbose=False,
            verbose_preprocessing=False,
            allow_tqdm=True,          # sliding-window tile progress bar
        )
        self.predictor.manual_initialization(
            network=network,
            plans_manager=plans_manager,
            configuration_manager=configuration_manager,
            parameters=[network.state_dict()],
            dataset_json=dataset_json,
            trainer_name=ck["trainer_name"],
            inference_allowed_mirroring_axes=ck["inference_allowed_mirroring_axes"],
        )

    @torch.no_grad()
    def predict(self, ct: sitk.Image) -> sitk.Image:
        """L3-cropped CT (RAI) -> muscle label image on the SAME grid as `ct`.

        Runs nnUNetPredictor's sliding-window inference (identical patch size,
        gaussian weighting, mirroring to training) fully in-process via
        `predict_single_npy_array` — no worker subprocesses, no folder layout.
        """
        from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO

        # Read into nnU-Net's (channel, z, y, x) array + geometry props via its
        # own reader, so preprocessing matches training exactly.
        with tempfile.TemporaryDirectory() as tmp:
            in_path = os.path.join(tmp, "case_0000.nii.gz")
            sitk.WriteImage(ct, in_path, useCompression=True)
            data, props = SimpleITKIO().read_images([in_path])

        seg_arr = self.predictor.predict_single_npy_array(
            data, props, None, None, False)          # -> (z, y, x) labels

        out = sitk.GetImageFromArray(seg_arr.astype(np.uint8))
        out.CopyInformation(ct)                       # identical grid to the crop
        return out
