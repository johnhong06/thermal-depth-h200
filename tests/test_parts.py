"""부품 시험 (CPU, 가짜 데이터) — 지표 · 손실 · 톤매핑 · 데이터 순서 · 모델 입출력 · 평가 · 표 · 묶음 풀기.

  TD_CODE=<공식 코드 폴더> <파이썬> -I -B tests/test_parts.py
TD_CODE 는 setup/fetch_code.sh 가 받은 폴더 (tests/run_local.sh 가 준비한다). 모델은 무작위 초기화한 vits 를 쓴다 (가중치를 받지 않음).
"""
import io
import json
import os
import sys
import tarfile
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, os.path.join(HERE, "..", "setup"))
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

import common as C  # noqa: E402
import data as D  # noqa: E402


def fake_tree(root, rng):
    """작은 가짜 묶음: MS2 분할마다 첫 시퀀스에 3 장 (train 은 두 시퀀스), FIReStereo 낮 · 밤 폴더 2 장씩."""
    def put(img, dep, n, h):
        os.makedirs(img, exist_ok=True)
        os.makedirs(dep, exist_ok=True)
        for i in n:
            Image.fromarray(rng.integers(3000, 6000, (h, 640), dtype=np.uint16)).save(os.path.join(img, f"{i:06d}.png"))
            d = (rng.uniform(2, 60, (h, 640)) * 256).astype(np.uint16)
            d[rng.random((h, 640)) < 0.7] = 0                       # 정답이 드문 LiDAR 처럼
            Image.fromarray(d).save(os.path.join(dep, f"{i:06d}.png"))
    for name in D.MS2_SETS:
        seqs = [s.strip() for s in open(os.path.join(HERE, "..", "splits", "ms2", f"{name}_list.txt")) if s.strip()]
        for s in seqs[:2 if name == "train" else 1]:
            put(os.path.join(root, "sync_data", s, "thr", "img_left"), os.path.join(root, "proj_depth", s, "thr", "depth_filtered"), (7, 3, 5), 256)
    for d in ("hawkins_9_test", "frick_9_test"):
        put(os.path.join(root, "firestereo", d, "img_left"), os.path.join(root, "firestereo", d, "depth_filtered"), (2, 1), 512)


class Parts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        fake_tree(cls.tmp.name, np.random.default_rng(0))
        os.environ["TD_DATA"] = cls.tmp.name
        torch.manual_seed(0)
        import model as M
        cls.M = M
        cls.model = M.build("vits").eval()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_metrics(self):
        g = torch.rand(5000, dtype=torch.float64) * 70 + 1
        p = g * (1 + 0.2 * torch.randn(5000, dtype=torch.float64)).clamp(0.3, 3)
        m = C.depth_metrics(p, g)
        gn, pn = g.numpy(), p.numpy()
        self.assertAlmostEqual(m["abs_rel"], float(np.mean(np.abs(pn - gn) / gn)), places=10)
        self.assertAlmostEqual(m["rmse"], float(np.sqrt(np.mean((pn - gn) ** 2))), places=10)
        self.assertAlmostEqual(m["rmse_log"], float(np.sqrt(np.mean((np.log(pn) - np.log(gn)) ** 2))), places=10)
        self.assertAlmostEqual(m["d1"], float(np.mean(np.maximum(pn / gn, gn / pn) < 1.25)), places=10)

    def test_variants(self):
        gt = torch.rand(64, 64) * 60 + 2
        gt[:, :20] = 0                                             # 정답 없음
        gt[0, 30] = 95.0                                           # 상한 밖
        mask = C.eval_mask(gt)
        self.assertEqual(int(mask.sum()), 64 * 44 - 1)
        v = C.variants(gt * 0.5, gt, mask, C.FX["ms2"])
        self.assertAlmostEqual(v["raw"]["abs_rel"], 0.5, places=5)
        self.assertAlmostEqual(v["median"]["abs_rel"], 0.0, places=5)
        self.assertAlmostEqual(v["median"]["scale"], 2.0, places=5)
        self.assertAlmostEqual(v["focal"]["abs_rel"], 0.5, places=5)      # 같은 카메라면 focal = raw
        f = C.variants(gt, gt, mask, C.FX["fire"])["focal"]["abs_rel"]
        self.assertAlmostEqual(f, C.FX["fire"] / C.FX_TRAIN - 1, places=3)
        self.assertIsNone(C.variants(gt, gt, torch.zeros_like(mask), C.FX["ms2"]))
        self.assertLessEqual(C.variants(gt * 5, gt, mask, C.FX["ms2"])["raw"]["rmse"], C.MAX_DEPTH)   # 예측은 상한에서 잘림

    def test_silog(self):
        gt, pred = torch.rand(4, 32, 32) * 50 + 1, torch.rand(4, 32, 32) * 50 + 1
        mask = gt > 10
        d = torch.log(gt[mask]) - torch.log(pred[mask])            # 공식 util/loss.py 의 식
        self.assertAlmostEqual(C.silog(pred, gt, mask).item(), torch.sqrt(d.pow(2).mean() - 0.5 * d.mean().pow(2)).item(), places=6)

    def test_tone(self):
        rng = np.random.default_rng(1)
        raw = rng.integers(3000, 6000, (256, 640)).astype(np.uint16)
        x = D.tone(raw)
        self.assertEqual(x.dtype, np.float32)
        self.assertTrue(x.min() == 0.0 and x.max() == 1.0)
        self.assertAlmostEqual(float((x == 0).mean()), 0.01, delta=0.003)
        np.testing.assert_allclose(D.tone((raw.astype(np.float64) * 3 + 20000).astype(np.uint16)), x, atol=2e-3)   # 카메라의 값 범위 · 배율이 달라도 같은 입력
        order = np.argsort(raw.ravel(), kind="stable")
        for s in range(20):                                        # 무작위 톤매핑: [0, 1] 안 · 밝기 순서를 뒤집지 않음
            y = D.tone(raw, np.random.default_rng(s))
            self.assertTrue(0.0 <= y.min() and y.max() <= 1.0)
            self.assertTrue(np.all(np.diff(y.ravel()[order]) >= -1e-6))
        self.assertGreater(len({float(D.tone(raw, np.random.default_rng(s)).mean()) for s in range(20)}), 10)

    def test_data(self):
        seqs = [s.strip() for s in open(os.path.join(HERE, "..", "splits", "ms2", "train_list.txt")) if s.strip()]
        keys = [k for k, _, _ in D.frames("train")]
        self.assertEqual(keys, [f"{s}/{n:06d}.png" for s in seqs[:2] for n in (3, 5, 7)])          # 목록 순서 → 이름순
        self.assertEqual([k for k, _, _ in D.frames("fire")], [f"{d}/{n:06d}.png" for d in ("frick_9_test", "hawkins_9_test") for n in (1, 2)])
        ev = D.Thermal("train")
        x0, d0, i0 = ev[1]
        self.assertEqual((tuple(x0.shape), tuple(d0.shape), i0), ((3, 256, 640), (256, 640), 1))
        g = x0 * torch.from_numpy(D.STD) + torch.from_numpy(D.MEAN)                                  # 정규화를 되돌리면 세 채널이 같은 [0, 1] 영상
        self.assertTrue(torch.allclose(g[0], g[1], atol=1e-5) and torch.allclose(g[0], g[2], atol=1e-5) and -1e-5 <= float(g.min()) and float(g.max()) <= 1 + 1e-5)
        self.assertAlmostEqual(float(d0.max()), float(np.array(Image.open(ev.items[1][2])).max()) / 256, places=4)
        tr = D.Thermal("train", aug="base", seed=3)
        flips = 0
        for ep in range(6):
            tr.epoch = ep
            x, d, _ = tr[1]
            a, b = torch.equal(x, x0) and torch.equal(d, d0), torch.equal(x, x0.flip(-1)) and torch.equal(d, d0.flip(-1))
            self.assertTrue(a or b)                                # 사진과 정답이 함께 뒤집힌다
            flips += b
            x2, d2, _ = tr[1]
            self.assertTrue(torch.equal(x, x2))                    # 같은 (시드, epoch, 번호) 면 같은 증강
        self.assertTrue(0 < flips < 6)
        self.assertEqual(len(D.Thermal("train", limit=4)), 4)
        self.assertEqual(tuple(D.Thermal("fire")[0][0].shape), (3, 512, 640))

    def test_model(self):
        M = self.M
        self.assertEqual((M.net_size(256, 640), M.net_size(512, 640)), ((252, 644), (518, 644)))
        with torch.no_grad():
            p = M.predict(self.model, torch.randn(2, 3, 256, 640), (256, 640))
        self.assertEqual(tuple(p.shape), (2, 256, 640))
        self.assertTrue(p.dtype == torch.float32 and 0 <= float(p.min()) and float(p.max()) <= C.MAX_DEPTH)
        c = M.count_params(self.model)
        self.assertTrue(20 < c["encoder_m"] < 25 and c["head_m"] > 1)
        names = [n for n, _ in self.model.named_parameters()]
        self.assertTrue(any("pretrained" in n for n in names) and any(n.startswith("depth_head.") for n in names))

    def test_eval(self):
        from evaluate import eval_set
        with tempfile.TemporaryDirectory() as out:
            s, rows = eval_set(self.model, "fire", torch.device("cpu"), bs=2, workers=0, samples=2, out=out)
            self.assertEqual((s["n"], s["n_all"], len(rows)), (4, 4, 4))
            self.assertEqual((s["day"]["n"], s["night"]["n"]), (2, 2))
            self.assertAlmostEqual(s["raw"]["abs_rel"], float(np.mean([r["raw"]["abs_rel"] for r in rows])), places=9)
            pngs = sorted(os.listdir(os.path.join(out, "samples")))
            self.assertEqual(len(pngs), 2)
            self.assertEqual(np.array(Image.open(os.path.join(out, "samples", pngs[0]))).shape, (512, 640))
        s, _ = eval_set(self.model, "test_night", torch.device("cpu"), bs=3, workers=0)
        self.assertEqual(s["n"], 3)

    def test_report(self):
        import contextlib
        import report as R

        def met(a):
            return {v: {k: a for k in C.METRICS} for v in C.VARIANTS} | {"n": 1, "n_all": 1}
        with tempfile.TemporaryDirectory() as t:
            for arm, fire in (("base", (0.300, 0.310, 0.305)), ("tone", (0.250, 0.262, 0.255))):
                for s, f in enumerate(fire):
                    d = os.path.join(t, "runs", f"{arm}_s{s}", "eval")
                    os.makedirs(d)
                    m = {k: met(0.075 + 0.001 * s) for k in ("test_day", "test_night", "test_rainy")} | {"fire": met(f) | {"day": met(f), "night": met(f)}}
                    json.dump(m, open(os.path.join(d, "metrics.json"), "w"))
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                R.summary(os.path.join(t, "runs"), os.path.join(t, "summary"))
            out = buf.getvalue().splitlines()
            self.assertLessEqual(len(out), 40)                     # run.sh 가 남기는 끝 50 줄 안에 들어가야 한다
            self.assertTrue(all(len(x) < 400 for x in out))
            j = json.load(open(os.path.join(t, "summary", "summary.json")))["judge"]
            self.assertTrue(j["gate"]["ok"] and j["gap"]["ok"])
            self.assertTrue(j["tone"]["verdict"].startswith("개선"))  # −0.049 < −잡음 폭 0.010, MS2 는 그대로
            self.assertAlmostEqual(j["tone"]["d_fire"], -0.049333, places=5)
            os.remove(os.path.join(t, "runs", "tone_s1", "eval", "metrics.json"))
            os.remove(os.path.join(t, "runs", "tone_s2", "eval", "metrics.json"))
            with contextlib.redirect_stdout(io.StringIO()):
                R.summary(os.path.join(t, "runs"), os.path.join(t, "summary"))
            self.assertIn("시드가 하나", json.load(open(os.path.join(t, "summary", "summary.json")))["judge"]["tone"]["verdict"])

    def test_unpack(self):
        import get_data as G
        with tempfile.TemporaryDirectory() as t:
            tar = os.path.join(t, "x.tar")
            good = ["sync_data/_2021-08-06-10-59-33/thr/img_left/000001.png", "proj_depth/_2021-08-06-10-59-33/thr/depth_filtered/000001.png",
                    "firestereo/frick_2_test/img_left/02250.png"]
            bad = ["../evil.png", "sync_data/_2021-08-06-10-59-33/calib.npy", "sync_data/../../x/thr/img_left/1.png", "firestereo/frick_2_train/img_left/1.png",
                   "sync_data/_2021-08-06-10-59-33/rgb/img_left/000001.png", "/abs/sync_data/_1/thr/img_left/1.png"]
            with tarfile.open(tar, "w") as tf:
                for n in good + bad:
                    ti = tarfile.TarInfo(n)
                    ti.size = 3
                    tf.addfile(ti, io.BytesIO(b"abc"))
                ti = tarfile.TarInfo("firestereo/frick_2_test/img_left/99999.png")       # 링크는 풀지 않는다
                ti.type, ti.linkname = tarfile.SYMTYPE, "/etc/passwd"
                tf.addfile(ti)
            dest = os.path.join(t, "out")
            self.assertEqual(G.unpack(tar, dest), 3)
            found = sorted(os.path.relpath(os.path.join(d, f), dest) for d, _, fs in os.walk(dest) for f in fs)
            self.assertEqual(found, sorted(good))
            self.assertEqual(os.listdir(t).count("evil.png"), 0)


if __name__ == "__main__":
    unittest.main(verbosity=1)
