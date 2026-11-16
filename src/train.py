"""학습 — 공식 Depth Anything V2 metric 학습 루프 (metric_depth/train.py) 와 같은 순서 · 같은 값, 데이터와 팔 (arm) 만 다르다.

공식 그대로인 것: 손실 SiLog (λ 0.5) · 정답 MIN_DEPTH ≤ gt ≤ MAX_DEPTH 인 픽셀만 · AdamW (betas 0.9 · 0.999, weight decay 0.01) ·
  학습률 인코더 5e-6 · 나머지 (헤드) 10 배 · 스텝마다 lr × (1 − 스텝 / 전체)^0.9, warmup 없음 · 마지막 못 채운 배치는 버림.
다른 것: 배치 16 (GPU 1 장) · 예측을 정답 해상도로 올려 손실 (LiDAR 정답이 드물어 정답을 줄이지 않는다) · 시드 고정 ·
  bf16 자동 혼합 정밀도 (--amp 1, GPU 만) · epoch 끝마다 val 을 재서 한 줄 (지켜보기만 — 고르는 데 쓰지 않고 마지막 epoch 을 쓴다) · last.pt (이어 하기).
팔: base = 뒤집기만 · tone = + 무작위 톤매핑 (src/data.py).

결과 (--out): train_log.csv (50 스텝마다) · val_log.csv (epoch 마다) · done.json. 체크포인트 (--ckpt): last.pt (이어 하기용, 끝나면 지움) · final.pt (가중치만)
"""
import argparse
import csv
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

import data as D  # noqa: E402
import model as M  # noqa: E402
from common import MAX_DEPTH, MIN_DEPTH, seed_all, silog, write_json  # noqa: E402
from evaluate import eval_set  # noqa: E402

ARMS = ("base", "tone")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=ARMS, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--encoder", default="vitl")
    ap.add_argument("--weights", default="", help="상대 깊이 체크포인트 (인코더만 싣는다). 비우면 무작위 초기화 (시험)")
    ap.add_argument("--limit", type=int, default=0, help="학습 장 수 제한 (smoke)")
    ap.add_argument("--val_limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--amp", type=int, default=1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    os.makedirs(a.ckpt, exist_ok=True)
    name = os.path.basename(os.path.normpath(a.out))
    dev = torch.device(a.device)
    amp = bool(a.amp) and dev.type == "cuda"

    seed_all(a.seed)
    torch.backends.cudnn.benchmark = True                          # 입력 크기가 고정
    ds = D.Thermal("train", a.limit, aug=a.arm, seed=a.seed)
    model = M.build(a.encoder, a.weights).to(dev)
    enc = [p for n, p in model.named_parameters() if "pretrained" in n]
    head = [p for n, p in model.named_parameters() if "pretrained" not in n]
    opt = torch.optim.AdamW([{"params": enc, "lr": a.lr}, {"params": head, "lr": a.lr * 10.0}], lr=a.lr, betas=(0.9, 0.999), weight_decay=0.01)
    steps_ep = len(ds) // a.bs
    if steps_ep < 1:
        raise SystemExit(f"!!! [train {name}] 학습 사진 {len(ds)} 장이 배치 {a.bs} 보다 적음")
    total = steps_ep * a.epochs
    step, ep0 = 0, 0
    last = os.path.join(a.ckpt, "last.pt")
    if a.resume and os.path.exists(last):
        ck = torch.load(last, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        step, ep0 = ck["step"], ck["epoch"]
        torch.set_rng_state(ck["rng"])
        print(f"[train {name}] 이어 하기: epoch {ep0} · step {step}", flush=True)
    p = M.count_params(model)
    print(f"[train {name}] {a.arm} · {a.encoder} · 시드 {a.seed} · train {len(ds):,} 장 · {a.epochs} epoch × {steps_ep} 스텝 (배치 {a.bs}) · "
          f"파라미터 {p['encoder_m']} + {p['head_m']} M · amp {int(amp)}", flush=True)

    new = not (a.resume and step > 0 and os.path.exists(os.path.join(a.out, "train_log.csv")))
    fh = open(os.path.join(a.out, "train_log.csv"), "w" if new else "a", newline="")
    vh = open(os.path.join(a.out, "val_log.csv"), "w" if new else "a", newline="")
    log, vlog = csv.writer(fh), csv.writer(vh)
    if new:
        log.writerow(["epoch", "step", "loss", "lr", "sec"])
        vlog.writerow(["epoch", "loss", "val_abs_rel", "val_rmse", "val_d1", "val_n"])
    t_start = time.time()
    for ep in range(ep0, a.epochs):
        model.train()
        ds.epoch = ep
        g = torch.Generator()
        g.manual_seed(a.seed * 1000 + ep)                          # 이어 해도 같은 순서
        loader = DataLoader(ds, batch_size=a.bs, shuffle=True, generator=g, num_workers=a.workers, drop_last=True, pin_memory=dev.type == "cuda")
        t0, run, n = time.time(), 0.0, 0
        for x, gt, _ in loader:
            x, gt = x.to(dev, non_blocking=True), gt.to(dev, non_blocking=True)
            lr = a.lr * (1 - step / total) ** 0.9
            opt.param_groups[0]["lr"], opt.param_groups[1]["lr"] = lr, lr * 10.0
            pred = M.predict(model, x, gt.shape[-2:], amp)
            loss = silog(pred, gt, (gt >= MIN_DEPTH) & (gt <= MAX_DEPTH))
            if not torch.isfinite(loss):
                write_json(os.path.join(a.out, "fail.json"), {"epoch": ep, "step": step, "why": "loss 가 유한하지 않음"})
                print(f"!!! [train {name}] epoch {ep + 1} step {step}: loss 가 유한하지 않음 — 멈춤", flush=True)
                return 3
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1
            run += loss.item()
            n += 1
            if step % 50 == 0 or step == 1:
                log.writerow([ep + 1, step, round(loss.item(), 5), f"{lr:.3g}", round(time.time() - t_start, 1)])
                fh.flush()
        sec = time.time() - t0
        v, _ = eval_set(model, "val", dev, a.val_limit, a.bs * 2, min(a.workers, 4), amp)
        vlog.writerow([ep + 1, round(run / max(n, 1), 5), round(v["raw"]["abs_rel"], 5), round(v["raw"]["rmse"], 4), round(v["raw"]["d1"], 5), v["n"]])
        vh.flush()
        mem = torch.cuda.max_memory_allocated() / 2**30 if dev.type == "cuda" else 0.0
        print(f"[train {name}] epoch {ep + 1}/{a.epochs} · loss {run / max(n, 1):.4f} · val AbsRel {v['raw']['abs_rel']:.4f} RMSE {v['raw']['rmse']:.3f} δ1 {v['raw']['d1']:.4f} · "
              f"{sec / 60:.1f} 분 ({sec / max(n, 1):.2f} 초/스텝, 전체 {total} 스텝이면 약 {sec / max(n, 1) * total / 3600:.1f} 시간) · GPU 최대 {mem:.1f} GiB", flush=True)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "epoch": ep + 1, "rng": torch.get_rng_state()}, last + ".tmp")
        os.replace(last + ".tmp", last)
    fh.close()
    vh.close()
    torch.save(model.state_dict(), os.path.join(a.ckpt, "final.pt"))
    os.remove(last)
    write_json(os.path.join(a.out, "done.json"), {
        "arm": a.arm, "encoder": a.encoder, "seed": a.seed, "epochs": a.epochs, "bs": a.bs, "lr": a.lr, "amp": int(amp), "images": len(ds), "steps": step,
        "minutes": round((time.time() - t_start) / 60, 1), "gpu_peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2) if dev.type == "cuda" else None,
        "params": p})
    print(f"[train {name}] 끝 — {step} 스텝", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
