"""
utils.wandb_helper
==================
Centralises all wandb interaction so rest of the codebase
never imports wandb directly.
"""

from types import SimpleNamespace

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:                 # user didn't install wandb
    _WANDB_AVAILABLE = False
    wandb = None                    # type: ignore

def init_wandb(args, paths) -> "wandb.sdk.wandb_run.Run | None":
    """
    Initialise a run if wandb is installed *and* user asked for it.
    """
    if not (_WANDB_AVAILABLE and getattr(args, "use_wandb", False)):
        return None

    run = wandb.init(
        project=getattr(args, "project", "lle_morl"),
        name=args.exp_name,
        dir=str(paths.base),       # wandb/ files saved UNDER experiments/…
        config=vars(args),         # full CLI config for traceability
        reinit=True,
    )
    return run

def log_metrics(step: int, **metrics):
    if _WANDB_AVAILABLE and wandb.run is not None:
        wandb.log(metrics, step=step)

def watch_model(model):
    if _WANDB_AVAILABLE and wandb.run is not None:
        wandb.watch(model, log="gradients", log_freq=1000)

def save_artifact(path: str, type_: str, name: str):
    """
    Store arbitrary file (model.zip, plot.png, …) as wandb Artifact.
    """
    if _WANDB_AVAILABLE and wandb.run is not None:
        art = wandb.Artifact(name=name, type=type_)
        art.add_file(path)
        wandb.run.log_artifact(art)
