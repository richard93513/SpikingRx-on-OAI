# src/train/train_spikingrx_oai.py
# -*- coding: utf-8 -*-

import math
import random
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from data.dataset_oai_bundle import OAI_Bundle_Dataset
from models.spikingrx_model import SpikingRxModel


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def masked_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    pred   : [B, 2, H, W]
    target : [B, 2, H, W]
    mask   : [B, H, W]
    """
    mask = mask.unsqueeze(1).to(pred.dtype)  # [B,1,H,W]
    se = (pred - target) ** 2
    se = se * mask
    denom = mask.sum() * pred.size(1) + eps
    return se.sum() / denom


def move_target_to_device(target: dict, device: torch.device):
    out = {}
    for k, v in target.items():
        if torch.is_tensor(v):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def llr_hybrid_loss(
    llr_pred: torch.Tensor,
    y_llr: torch.Tensor,
    bce_fn,
    smoothl1_fn,
    clip_value: float = 12.0,
    alpha_sign: float = 0.30,
    alpha_value: float = 0.70,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    同時學：
      1) LLR sign
      2) LLR magnitude/value
    """
    y_llr_sign = (y_llr >= 0).float()
    loss_sign = bce_fn(llr_pred, y_llr_sign)

    y_llr_clip = torch.clamp(y_llr, -clip_value, clip_value)
    loss_value = smoothl1_fn(llr_pred, y_llr_clip)

    loss = alpha_sign * loss_sign + alpha_value * loss_value
    return loss, loss_sign, loss_value


def get_loss_schedule(epoch: int) -> Dict[str, float]:
    if epoch <= 5:
        return {
            "phase": "phase1_ch_bootstrap",
            "w_llr": 0.10,
            "w_eq": 0.00,
            "w_ch": 1.00,
            "enable_eq_loss": False,
        }
    elif epoch <= 12:
        return {
            "phase": "phase2_ch_eq",
            "w_llr": 0.30,
            "w_eq": 0.70,
            "w_ch": 0.50,
            "enable_eq_loss": True,
        }
    else:
        return {
            "phase": "phase3_full_serial",
            "w_llr": 1.00,
            "w_eq": 0.30,
            "w_ch": 0.15,
            "enable_eq_loss": True,
        }


def summarize_aux(aux: dict) -> str:
    parts = []

    if "spike_rate_per_stage" in aux:
        sr = aux["spike_rate_per_stage"]
        if torch.is_tensor(sr):
            sr_list = [float(x) for x in sr.flatten().tolist()]
            parts.append(
                "spike_rate_per_stage=["
                + ", ".join(f"{v:.4f}" for v in sr_list)
                + "]"
            )

    if "final_rate_mean" in aux:
        parts.append(f"final_rate_mean={float(aux['final_rate_mean']):.6f}")

    if "final_rate_std" in aux:
        parts.append(f"final_rate_std={float(aux['final_rate_std']):.6f}")

    if "ch_mean_abs" in aux:
        parts.append(f"ch_mean_abs={float(aux['ch_mean_abs']):.6f}")

    if "eq_mean_abs" in aux:
        parts.append(f"eq_mean_abs={float(aux['eq_mean_abs']):.6f}")

    if "llr_map_mean_abs" in aux:
        parts.append(f"llr_map_mean_abs={float(aux['llr_map_mean_abs']):.6f}")

    return " | ".join(parts)


def compute_losses(
    pred: dict,
    target_main: dict,
    y_llr: torch.Tensor,
    bce_fn,
    smoothl1_fn,
    w_llr: float,
    w_eq: float,
    w_ch: float,
    enable_eq_loss: bool,
    device_main: torch.device,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    llr_pred = pred["llr"]
    eq_pred = pred["eq"]
    ch_pred = pred["ch"]

    loss_llr, loss_llr_sign, loss_llr_value = llr_hybrid_loss(
        llr_pred=llr_pred,
        y_llr=y_llr,
        bce_fn=bce_fn,
        smoothl1_fn=smoothl1_fn,
        clip_value=12.0,
        alpha_sign=0.30,
        alpha_value=0.70,
    )

    loss_ch = masked_mse(
        ch_pred,
        target_main["y_ch"],
        target_main["ch_mask"],
    )

    if enable_eq_loss:
        loss_eq = masked_mse(
            eq_pred,
            target_main["y_eq"],
            target_main["eq_mask"],
        )
    else:
        loss_eq = torch.zeros((), device=device_main)

    loss = w_llr * loss_llr + w_ch * loss_ch + w_eq * loss_eq

    with torch.no_grad():
        y_llr_sign = (y_llr >= 0).float()
        pred_sign = (llr_pred >= 0).float()
        sign_correct = (pred_sign == y_llr_sign).sum().item()
        sign_count = y_llr_sign.numel()

    stat = {
        "loss": loss,
        "loss_llr": loss_llr,
        "loss_llr_sign": loss_llr_sign,
        "loss_llr_value": loss_llr_value,
        "loss_eq": loss_eq,
        "loss_ch": loss_ch,
        "sign_correct": sign_correct,
        "sign_count": sign_count,
    }
    return loss, stat


@torch.no_grad()
def evaluate(
    model,
    loader,
    device_main,
    device_llr,
    bce_fn,
    smoothl1_fn,
    w_llr,
    w_eq,
    w_ch,
    enable_eq_loss,
):
    model.eval()

    total_loss = 0.0
    total_llr_loss = 0.0
    total_llr_sign_loss = 0.0
    total_llr_value_loss = 0.0
    total_eq_loss = 0.0
    total_ch_loss = 0.0

    total_sign_correct = 0.0
    total_sign_count = 0
    n_batches = 0

    for x, target, cfg, bdir in loader:
        x = x.to(device_main)
        target_main = move_target_to_device(target, device_main)
        y_llr = target["y_llr"].to(device_llr)

        pred, aux = model(x)

        _, stat = compute_losses(
            pred=pred,
            target_main=target_main,
            y_llr=y_llr,
            bce_fn=bce_fn,
            smoothl1_fn=smoothl1_fn,
            w_llr=w_llr,
            w_eq=w_eq,
            w_ch=w_ch,
            enable_eq_loss=enable_eq_loss,
            device_main=device_main,
        )

        total_loss += float(stat["loss"].item())
        total_llr_loss += float(stat["loss_llr"].item())
        total_llr_sign_loss += float(stat["loss_llr_sign"].item())
        total_llr_value_loss += float(stat["loss_llr_value"].item())
        total_eq_loss += float(stat["loss_eq"].item())
        total_ch_loss += float(stat["loss_ch"].item())
        total_sign_correct += stat["sign_correct"]
        total_sign_count += stat["sign_count"]
        n_batches += 1

    if n_batches == 0:
        return {
            "loss": math.nan,
            "loss_llr": math.nan,
            "loss_llr_sign": math.nan,
            "loss_llr_value": math.nan,
            "loss_eq": math.nan,
            "loss_ch": math.nan,
            "sign": math.nan,
        }

    return {
        "loss": total_loss / n_batches,
        "loss_llr": total_llr_loss / n_batches,
        "loss_llr_sign": total_llr_sign_loss / n_batches,
        "loss_llr_value": total_llr_value_loss / n_batches,
        "loss_eq": total_eq_loss / n_batches,
        "loss_ch": total_ch_loss / n_batches,
        "sign": total_sign_correct / max(total_sign_count, 1),
    }


def main():
    seed_everything(42)

    repo_root = Path.home() / "SpikingRx-on-OAI"
    bundle_root = repo_root / "spx_records" / "bundle"
    ckpt_dir = repo_root / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    batch_size = 8
    num_epochs = 30
    lr = 1e-3
    weight_decay = 1e-5
    val_ratio = 0.15
    num_workers = 0
    patience = 10

    device_main = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_llr = device_main

    ds = OAI_Bundle_Dataset(
        bundle_root=str(bundle_root),
        normalize=True,
        aux_normalize_eq=True,
        aux_normalize_ch=True,
        T=1,
        require_aux_targets=True,
    )

    if len(ds) < 2:
        raise RuntimeError(f"[ERROR] dataset too small: {len(ds)}")

    x0, t0, cfg0, bdir0 = ds[0]
    print(f"[Sample] bundle = {bdir0}")
    print(f"[Sample] x.shape = {tuple(x0.shape)}")
    print(f"[Sample] y_llr.shape = {tuple(t0['y_llr'].shape)}")
    print(f"[Sample] y_eq.shape = {tuple(t0['y_eq'].shape)}")
    print(f"[Sample] y_ch.shape = {tuple(t0['y_ch'].shape)}")
    print(f"[Sample] eq_mask.sum = {float(t0['eq_mask'].sum())}")
    print(f"[Sample] ch_mask.sum = {float(t0['ch_mask'].sum())}")
    print(f"[Sample] dmrs_mask.sum = {float(t0['dmrs_mask'].sum())}")
    print(f"[Sample] data_mask.sum = {float(t0['data_mask'].sum())}")
    print(f"[Sample] x.mean/std = {float(x0.mean()):.6f} / {float(x0.std()):.6f}")
    print(f"[Sample] y_llr.mean/std = {float(t0['y_llr'].mean()):.6f} / {float(t0['y_llr'].std()):.6f}")

    eq_valid = t0["eq_mask"] > 0
    ch_valid = t0["ch_mask"] > 0
    print(
        f"[Sample] y_eq(masked).mean/std = "
        f"{float(t0['y_eq'][:, eq_valid].mean()):.6f} / "
        f"{float(t0['y_eq'][:, eq_valid].std()):.6f}"
    )
    print(
        f"[Sample] y_ch(masked).mean/std = "
        f"{float(t0['y_ch'][:, ch_valid].mean()):.6f} / "
        f"{float(t0['y_ch'][:, ch_valid].std()):.6f}"
    )

    n_total = len(ds)
    n_val = max(1, int(round(n_total * val_ratio)))
    n_train = n_total - n_val

    train_ds, val_ds = random_split(
        ds,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    print(f"[Dataset] total={n_total} train={n_train} val={n_val}")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    model = SpikingRxModel(
        in_ch=4,
        out_bits=int(cfg0["G"]),
        T=1,
        device_conv=device_main,
        device_fc=device_llr,
        llr_temperature=2.0,
    )

    bce_fn = nn.BCEWithLogitsLoss()
    smoothl1_fn = nn.SmoothL1Loss(beta=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_sign = -1.0
    best_epoch = -1
    bad_epochs = 0

    best_ckpt = ckpt_dir / "spikingrx_oai_serial_best.pt"
    last_ckpt = ckpt_dir / "spikingrx_oai_serial_last.pt"

    for epoch in range(1, num_epochs + 1):
        sched = get_loss_schedule(epoch)
        phase = str(sched["phase"])
        w_llr = float(sched["w_llr"])
        w_eq = float(sched["w_eq"])
        w_ch = float(sched["w_ch"])
        enable_eq_loss = bool(sched["enable_eq_loss"])

        model.train()

        total_loss = 0.0
        total_llr_loss = 0.0
        total_llr_sign_loss = 0.0
        total_llr_value_loss = 0.0
        total_eq_loss = 0.0
        total_ch_loss = 0.0
        total_sign_correct = 0.0
        total_sign_count = 0
        n_batches = 0
        debug_printed = False

        for x, target, cfg, bdir in train_loader:
            x = x.to(device_main)
            target_main = move_target_to_device(target, device_main)
            y_llr = target["y_llr"].to(device_llr)

            optimizer.zero_grad()

            pred, aux = model(x)

            if not debug_printed:
                with torch.no_grad():
                    ch_pred = pred["ch"]
                    eq_pred = pred["eq"]
                    llr_pred = pred["llr"]

                    ch_mask = target_main["ch_mask"] > 0.5
                    eq_mask = target_main["eq_mask"] > 0.5

                    ch_pred_valid = ch_pred.permute(0, 2, 3, 1)[ch_mask]
                    ch_tgt_valid = target_main["y_ch"].permute(0, 2, 3, 1)[ch_mask]

                    eq_pred_valid = eq_pred.permute(0, 2, 3, 1)[eq_mask]
                    eq_tgt_valid = target_main["y_eq"].permute(0, 2, 3, 1)[eq_mask]

                    print(f"[Epoch {epoch:03d}][DEBUG] {summarize_aux(aux)}")
                    print(
                        f"[Epoch {epoch:03d}][DEBUG] "
                        f"ch_pred mean/std={float(ch_pred_valid.mean()):.6f}/{float(ch_pred_valid.std()):.6f} | "
                        f"ch_tgt mean/std={float(ch_tgt_valid.mean()):.6f}/{float(ch_tgt_valid.std()):.6f}"
                    )
                    print(
                        f"[Epoch {epoch:03d}][DEBUG] "
                        f"eq_pred mean/std={float(eq_pred_valid.mean()):.6f}/{float(eq_pred_valid.std()):.6f} | "
                        f"eq_tgt mean/std={float(eq_tgt_valid.mean()):.6f}/{float(eq_tgt_valid.std()):.6f}"
                    )
                    print(
                        f"[Epoch {epoch:03d}][DEBUG] "
                        f"llr_pred mean/std={float(llr_pred.mean()):.6f}/{float(llr_pred.std()):.6f} | "
                        f"y_llr mean/std={float(y_llr.mean()):.6f}/{float(y_llr.std()):.6f}"
                    )

                debug_printed = True

            loss, stat = compute_losses(
                pred=pred,
                target_main=target_main,
                y_llr=y_llr,
                bce_fn=bce_fn,
                smoothl1_fn=smoothl1_fn,
                w_llr=w_llr,
                w_eq=w_eq,
                w_ch=w_ch,
                enable_eq_loss=enable_eq_loss,
                device_main=device_main,
            )

            loss.backward()
            optimizer.step()

            total_loss += float(stat["loss"].item())
            total_llr_loss += float(stat["loss_llr"].item())
            total_llr_sign_loss += float(stat["loss_llr_sign"].item())
            total_llr_value_loss += float(stat["loss_llr_value"].item())
            total_eq_loss += float(stat["loss_eq"].item())
            total_ch_loss += float(stat["loss_ch"].item())
            total_sign_correct += stat["sign_correct"]
            total_sign_count += stat["sign_count"]
            n_batches += 1

        train_stats = {
            "loss": total_loss / max(n_batches, 1),
            "loss_llr": total_llr_loss / max(n_batches, 1),
            "loss_llr_sign": total_llr_sign_loss / max(n_batches, 1),
            "loss_llr_value": total_llr_value_loss / max(n_batches, 1),
            "loss_eq": total_eq_loss / max(n_batches, 1),
            "loss_ch": total_ch_loss / max(n_batches, 1),
            "sign": total_sign_correct / max(total_sign_count, 1),
        }

        val_stats = evaluate(
            model=model,
            loader=val_loader,
            device_main=device_main,
            device_llr=device_llr,
            bce_fn=bce_fn,
            smoothl1_fn=smoothl1_fn,
            w_llr=w_llr,
            w_eq=w_eq,
            w_ch=w_ch,
            enable_eq_loss=enable_eq_loss,
        )

        print(
            f"[Epoch {epoch:03d}] phase={phase} "
            f"w(llr,eq,ch)=({w_llr:.2f},{w_eq:.2f},{w_ch:.2f}) "
            f"Train loss={train_stats['loss']:.6f} "
            f"(llr={train_stats['loss_llr']:.6f}, "
            f"llr_sign={train_stats['loss_llr_sign']:.6f}, "
            f"llr_val={train_stats['loss_llr_value']:.6f}, "
            f"eq={train_stats['loss_eq']:.6f}, "
            f"ch={train_stats['loss_ch']:.6f}) "
            f"sign={train_stats['sign']:.6f} | "
            f"Val loss={val_stats['loss']:.6f} "
            f"(llr={val_stats['loss_llr']:.6f}, "
            f"llr_sign={val_stats['loss_llr_sign']:.6f}, "
            f"llr_val={val_stats['loss_llr_value']:.6f}, "
            f"eq={val_stats['loss_eq']:.6f}, "
            f"ch={val_stats['loss_ch']:.6f}) "
            f"sign={val_stats['sign']:.6f}"
        )

        torch.save(
            {
                "epoch": epoch,
                "phase": phase,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_stats": train_stats,
                "val_stats": val_stats,
                "cfg0": cfg0,
                "w_llr": w_llr,
                "w_eq": w_eq,
                "w_ch": w_ch,
                "enable_eq_loss": enable_eq_loss,
                "dataset_cfg": {
                    "normalize": True,
                    "aux_normalize_eq": True,
                    "aux_normalize_ch": True,
                },
            },
            last_ckpt,
        )

        improved = val_stats["sign"] > best_val_sign
        if improved:
            best_val_sign = val_stats["sign"]
            best_epoch = epoch
            bad_epochs = 0

            torch.save(
                {
                    "epoch": epoch,
                    "phase": phase,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_stats": train_stats,
                    "val_stats": val_stats,
                    "cfg0": cfg0,
                    "w_llr": w_llr,
                    "w_eq": w_eq,
                    "w_ch": w_ch,
                    "enable_eq_loss": enable_eq_loss,
                    "dataset_cfg": {
                        "normalize": True,
                        "aux_normalize_eq": True,
                        "aux_normalize_ch": True,
                    },
                },
                best_ckpt,
            )
            print(f"[Checkpoint] saved best -> {best_ckpt}")
        else:
            bad_epochs += 1

        if bad_epochs >= patience:
            print(f"[EARLY STOP] no val-sign improvement for {patience} epochs")
            break

    print(f"[DONE] best_epoch={best_epoch}, best_val_sign={best_val_sign:.6f}")
    print(f"[DONE] best_ckpt={best_ckpt}")
    print(f"[DONE] last_ckpt={last_ckpt}")


if __name__ == "__main__":
    main()
