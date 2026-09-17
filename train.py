"""
Train the RDPD + SDAF model (YOLO11m-P2) on VisDrone.

Single entry point for this project. It:
  1. Ensures the local (vendored) ultralytics copy is imported.
  2. Builds the model from `ultralytics/cfg/models/yolo11m_p2_rdpd_sdaf.yaml`.
  3. Transfers YOLO11m pretrained weights (exact + partial-conv +
     RDPD-semantic mapping) for a warm start.
  4. Runs training with the project's fixed hyperparameters.

Usage:
    python train.py                          # default settings
    python train.py --epochs 10              # quick override
    python train.py --weights <path.pt>      # alternate pretrained weights
"""

import argparse
import json
import sys
from pathlib import Path

# Ensure the local ultralytics copy is used (not a site-packages install).
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from ultralytics import YOLO

# ── Paths ─────────────────────────────────────────────────────────────
MODEL_YAML = PROJECT_ROOT / "ultralytics" / "cfg" / "models" / "yolo11m_p2_rdpd_sdaf.yaml"
DATA_YAML = PROJECT_ROOT / "data" / "VisDrone.yaml"
DEFAULT_WEIGHTS = PROJECT_ROOT / "weights" / "yolo11m.pt"
# Fallback: the pretrained weights in the original project (shared, not copied).
FALLBACK_WEIGHTS = Path("F:/python/RDPD_SDAF_VisDrone/weights/yolo11m.pt")
EXPERIMENT_NAME = "rdpd_sdaf"

# Fixed hyperparameters (fair-comparison defaults from the original project).
TRAIN_DEFAULTS = {
    "imgsz": 640,
    "epochs": 120,
    "batch": 8,
    "device": "0",
    "workers": 0,
    "optimizer": "SGD",
    "lr0": 0.01,
    "momentum": 0.937,
    "weight_decay": 0.0005,
    "cache": False,
    "amp": True,
    "seed": 0,
    "deterministic": True,
    "close_mosaic": 0,  # 0 = do NOT auto-disable Mosaic near end of training
    "patience": 500,
    "save_period": 50,
    "plots": True,
    "val": True,
    "multi_scale": False,
}


# ── Pretrained weight transfer ────────────────────────────────────────
def load_pretrained_weights(model, weights_path: Path) -> dict:
    """Load YOLO11m pretrained weights and return a transfer report.

    Handles three cases:
      - Exact matches (same key, same shape)
      - Partial Conv matches (same output channels/kernel, fewer input channels)
      - RDPD semantic-branch mapping (original stride-2 Conv -> RDPD.semantic)
    """
    report = {
        "weights_path": str(weights_path),
        "total_target_params": 0,
        "exact_matches": 0,
        "partial_conv_matches": 0,
        "rdpd_semantic_maps": 0,
        "random_init": 0,
    }

    if not weights_path.exists():
        print(f"[WARN] Pretrained weights not found: {weights_path}")
        return report

    ckpt = torch.load(str(weights_path), map_location="cpu", weights_only=False)
    model_data = ckpt.get("model", ckpt)
    csd = model_data.float().state_dict() if hasattr(model_data, "state_dict") else model_data

    target_sd = model.state_dict()
    report["total_target_params"] = len(target_sd)

    exact_matched = {k: v for k, v in target_sd.items() if k in csd and csd[k].shape == v.shape}
    report["exact_matches"] = len(exact_matched)

    partial_matched = {}
    for k, v in target_sd.items():
        if k in exact_matched:
            continue
        if k in csd and v.ndim == 4 and csd[k].ndim == 4:
            t_shape, s_shape = v.shape, csd[k].shape
            if t_shape[0] == s_shape[0] and t_shape[2:] == s_shape[2:]:
                w = torch.zeros_like(v)
                w[:, : min(t_shape[1], s_shape[1])] = csd[k][:, : min(t_shape[1], s_shape[1])]
                partial_matched[k] = w
                report["partial_conv_matches"] += 1

    rdpd_mapped = {}
    for target_name, target_mod in model.named_modules():
        if target_mod.__class__.__name__ == "RDPD":
            rdpd_mapped.update(_map_rdpd_semantic(target_name, target_mod, csd))

    report["rdpd_semantic_maps"] = len(rdpd_mapped)

    updated_sd = {**exact_matched, **partial_matched, **rdpd_mapped}
    model.load_state_dict(updated_sd, strict=False)
    report["random_init"] = report["total_target_params"] - len(updated_sd)

    num_loaded = len(updated_sd)
    print(
        f"[WEIGHTS] Transferred {num_loaded}/{report['total_target_params']} params "
        f"({100 * num_loaded / report['total_target_params']:.1f}%)"
    )
    print(
        f"  Exact: {report['exact_matches']}, "
        f"Partial conv: {report['partial_conv_matches']}, "
        f"RDPD semantic: {report['rdpd_semantic_maps']}"
    )
    return report


def _map_rdpd_semantic(target_name: str, rdpd_module, source_sd: dict) -> dict:
    """Map YOLO11m stride-2 Conv weights to RDPD's semantic branch."""
    mapped = {}
    semantic_conv = rdpd_module.semantic.conv
    t_c2, t_c1 = semantic_conv.weight.shape[:2]
    prefix = target_name + "."

    best_key = None
    for sk in source_sd:
        if sk.endswith(".conv.weight") and source_sd[sk].ndim == 4:
            sw = source_sd[sk]
            if sw.shape[0] == t_c2 and sw.shape[2:] == semantic_conv.weight.shape[2:]:
                best_key = sk
                break
    if best_key is None:
        return mapped

    base = best_key[: -len(".conv.weight")]
    sw = source_sd[best_key]
    nw = torch.zeros_like(semantic_conv.weight)
    nw[:, : min(t_c1, sw.shape[1])] = sw[:, : min(t_c1, sw.shape[1])]
    mapped[prefix + "semantic.conv.weight"] = nw

    for bn_param in ("weight", "bias", "running_mean", "running_var"):
        sk_bn = f"{base}.bn.{bn_param}"
        if sk_bn in source_sd:
            mapped[prefix + f"semantic.bn.{bn_param}"] = source_sd[sk_bn]
    sk_bt = f"{base}.bn.num_batches_tracked"
    if sk_bt in source_sd:
        mapped[prefix + "semantic.bn.num_batches_tracked"] = source_sd[sk_bt]

    return mapped


def verify_ultralytics_import() -> None:
    """Ensure ultralytics is imported from the local vendored copy."""
    import ultralytics

    ult_path = Path(ultralytics.__file__).resolve()
    assert str(ult_path).startswith(str(PROJECT_ROOT)), (
        f"ULTRA IMPORT ERROR: ultralytics imported from {ult_path}, "
        f"expected the local copy under {PROJECT_ROOT}"
    )
    print(f"[OK] ultralytics imported from: {ult_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RDPD + SDAF on VisDrone")
    parser.add_argument("--weights", type=str, default=None, help="Pretrained weights path")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--batch", type=int, default=None, help="Override batch size")
    parser.add_argument("--device", type=str, default=None, help="Override device")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    verify_ultralytics_import()

    cfg = dict(TRAIN_DEFAULTS)
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.batch is not None:
        cfg["batch"] = args.batch
    if args.device is not None:
        cfg["device"] = args.device

    weights = Path(args.weights) if args.weights else DEFAULT_WEIGHTS
    if not weights.exists() and not args.weights and FALLBACK_WEIGHTS.exists():
        weights = FALLBACK_WEIGHTS

    model = YOLO(str(MODEL_YAML))
    load_pretrained_weights(model.model, weights)

    model.train(
        data=str(DATA_YAML),
        imgsz=cfg["imgsz"],
        epochs=cfg["epochs"],
        batch=cfg["batch"],
        device=cfg["device"],
        workers=cfg["workers"],
        optimizer=cfg["optimizer"],
        lr0=cfg["lr0"],
        momentum=cfg["momentum"],
        weight_decay=cfg["weight_decay"],
        cache=cfg["cache"],
        amp=cfg["amp"],
        seed=cfg["seed"],
        deterministic=cfg["deterministic"],
        close_mosaic=cfg["close_mosaic"],
        patience=cfg["patience"],
        save_period=cfg["save_period"],
        plots=cfg["plots"],
        val=cfg["val"],
        multi_scale=cfg["multi_scale"],
        project=str(PROJECT_ROOT / "runs" / "train"),
        name=EXPERIMENT_NAME,
        pretrained=False,
    )


if __name__ == "__main__":
    main()
