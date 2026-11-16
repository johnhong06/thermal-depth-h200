"""공용: 고정 값 (평가 규칙) · 시드 · 깊이 지표.

평가 규칙 (README '판정 기준' — 결과를 보기 전에 고정). 선행 코드 둘과 같은 것: 정답 MIN_DEPTH < gt < MAX_DEPTH 인 픽셀만 · 자르기 없음 ·
  예측을 정답 해상도로 올린 뒤 (bilinear) [MIN_DEPTH, MAX_DEPTH] 로 자름 · 사진마다 지표를 구하고 사진 평균.
  예측 세 가지를 함께 잰다:
    raw    모델 출력 그대로 — MonoTher-Depth 공개 코드 (zoedepth/utils/misc.py compute_metrics) 의 방식. 주 지표.
    median 사진마다 중앙값 맞춤 — MS2 공식 벤치마크 코드 (SupDepth4Thermal eval_metric.py, align=True) 의 방식. 벤치마크 표의 수치와 견줄 때.
    focal  학습 카메라와의 초점거리 비 (fx / FX_TRAIN) 를 곱함 — 카메라가 다른 데이터에서 raw 의 치우침이 초점거리 때문인지 보려고.
  ViViD 만: 정답의 테두리를 뺀다 (border_box) — 선행 논문이 ViViD 를 그렇게 쟀고, 그 표와 견주려는 세트라서.
"""
import json
import os
import random

import numpy as np
import torch

MIN_DEPTH = 1e-3          # m — 공식 Depth Anything V2 metric 학습 (train.py --min-depth) 과 같은 값
MAX_DEPTH = 80.0          # m — 평가 상한이자 모델 출력 상한 (Sigmoid × MAX_DEPTH)
MIN_VALID = 10            # 범위 안 정답 픽셀이 이보다 적은 사진은 건너뜀 (공식 train.py 의 평가와 같음)
# 열화상 카메라 초점거리 (px, 가로 640 기준). ms2 = calib.npy 의 K_thrL (20 시퀀스 같음) · fire = 저자 firestereo.yaml cam0 · vivid = calib_ther_to_rgb.yaml 의 K_Thermal
FX = {"ms2": 387.787, "fire": 406.332, "vivid": 437.389}
FX_TRAIN = FX["ms2"]
METRICS = ("abs_rel", "sq_rel", "rmse", "rmse_log", "d1", "d2", "d3")
VARIANTS = ("raw", "focal", "median")


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def write_json(path, obj):
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
    os.replace(path + ".tmp", path)


def depth_metrics(pred, gt):
    """1 차원 텐서 (범위 안 픽셀) → 지표 dict (float). 흔히 쓰는 정의 (Eigen 등): 사진 한 장의 픽셀 평균."""
    pred = pred.clamp(min=1e-6)
    thr = torch.maximum(pred / gt, gt / pred)
    d = pred - gt
    dl = torch.log(pred) - torch.log(gt)
    return {"abs_rel": (d.abs() / gt).mean().item(), "sq_rel": (d * d / gt).mean().item(),
            "rmse": (d * d).mean().sqrt().item(), "rmse_log": (dl * dl).mean().sqrt().item(),
            "d1": (thr < 1.25).double().mean().item(), "d2": (thr < 1.25 ** 2).double().mean().item(), "d3": (thr < 1.25 ** 3).double().mean().item()}


def variants(pred, gt, mask, fx):
    """예측 (H, W) · 정답 (H, W) · 평가 마스크 → {raw · focal · median: 지표 dict}. 마스크 픽셀이 MIN_VALID 미만이면 None."""
    if int(mask.sum()) < MIN_VALID:
        return None
    p, g = pred[mask].double(), gt[mask].double()
    s = (g.median() / p.median().clamp(min=1e-6)).item()
    out = {k: depth_metrics((p * f).clamp(MIN_DEPTH, MAX_DEPTH), g) for k, f in (("raw", 1.0), ("focal", fx / FX_TRAIN), ("median", s))}
    out["median"]["scale"] = s
    return out


def eval_mask(gt, border=False):
    m = (gt > MIN_DEPTH) & (gt < MAX_DEPTH)
    if border:
        top, bottom, left, right = border_box(gt)
        keep = torch.zeros_like(m)
        keep[top:bottom, left:right] = True
        m &= keep
    return m


def border_box(gt, tol=0.4, cut_off=100, eps=0.01, min_border=40):
    """ViViD 평가에서 빼는 테두리 — MonoTher-Depth 공개 코드가 ViViD 정답에 거는 규칙을 다시 쓴 것 (그 표와 같은 조건으로 재려고).
    (zoedepth/data/data_thermal_rgbtd.py: get_black_border(depth_gt, tolerance=0.4, cut_off=100, level_diff_threshold=0.01, min_border=40))
    변마다 min_border 번째 줄에서 시작해, 그 줄에서 값 없는 픽셀 (|gt| < eps) 의 비율이 tol 을 넘는 동안 안쪽으로 한 줄씩 — cut_off 를 넘으면 멈춤.
    → (top, bottom, left, right): gt[top:bottom, left:right] 만 평가에 쓴다."""
    h, w = gt.shape
    empty = gt.abs() < eps
    rows, cols = empty.float().mean(1).tolist(), empty.float().mean(0).tolist()
    top = min_border
    while rows[top] > tol and top < h - 1:
        top += 1
        if top > cut_off:
            break
    bottom = h - min_border
    while rows[bottom] > tol and bottom > 0:
        bottom -= 1
        if h - bottom > cut_off:
            break
    left = min_border
    while cols[left] > tol and left < w - 1:
        left += 1
        if left > cut_off:
            break
    right = w - min_border
    while cols[right] > tol and right > 0:
        right -= 1
        if w - right > cut_off:
            break
    return top, bottom, left, right


def silog(pred, gt, mask, lambd=0.5):
    """공식 Depth Anything V2 metric_depth/util/loss.py 의 SiLogLoss 와 같은 식."""
    dl = torch.log(gt[mask]) - torch.log(pred[mask])
    return torch.sqrt(dl.pow(2).mean() - lambd * dl.mean().pow(2))
