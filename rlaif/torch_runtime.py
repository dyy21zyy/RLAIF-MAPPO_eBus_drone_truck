"""Shared messaging for Stage 5 commands that require the optional PyTorch runtime."""

PYTORCH_REQUIRED_MESSAGE = (
    "PyTorch is required for reward-model training/evaluation. "
    "Install torch>=2.0,<3.0 and rerun this command."
)


def is_missing_torch_error(exc: ModuleNotFoundError) -> bool:
    """Return whether an import failure is specifically the optional torch dependency."""
    return exc.name == "torch" or (exc.name is not None and exc.name.startswith("torch."))
