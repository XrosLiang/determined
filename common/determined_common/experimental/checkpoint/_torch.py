import pathlib
from typing import Any, Dict, cast

import torch

from determined.experimental._native import _local_trial_from_context
from determined.pytorch import PyTorchTrial


def load_model(ckpt_dir: pathlib.Path, metadata: Dict[str, Any], **kwargs: Any) -> torch.nn.Module:
    # We used MLflow's MLmodel checkpoint format in the past. This format
    # nested the checkpoint in data/. Currently, we have the checkpoint at the
    # top level of the checkpoint directory.
    potential_model_paths = [["model.pth"], ["data", "model.pth"]]

    for nested_path in potential_model_paths:
        maybe_model = ckpt_dir.joinpath(*nested_path)
        if maybe_model.exists():
            break

    if not maybe_model.exists():
        raise AssertionError("checkpoint at {} doesn't include a model.pth file".format(ckpt_dir))

    print("CODE DIR:", str(ckpt_dir.joinpath("code")))
    trial = _local_trial_from_context(
        ckpt_dir.joinpath("code"),
        config=metadata["experiment_config"],
        hparams=metadata["hparams"],
    )

    trial = cast(PyTorchTrial, trial)
    model = trial.build_model()
    checkpoint = torch.load(ckpt_dir.joinpath("state_dict.pth"), map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])

    return model
