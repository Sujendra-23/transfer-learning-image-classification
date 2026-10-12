"""Latency benchmark for an exported model, at batch sizes 1, 8 and 32.

Model-level (default): times `predict_on_batch` on the loaded model, with warm-up iterations
excluded, and reports p50 / p95 batch latency, per-image latency and throughput for each batch size.

    python scripts/benchmark_latency.py --model best_model_resnet50.keras --backbone ResNet50
    python scripts/benchmark_latency.py --model ... --backbone ResNet50 --device cpu --json out.json

HTTP mode times the whole `/predict` round trip (upload, decode, resize, inference, JSON) one
image at a time against a running service; the service takes one image per request, so there is
no batching to measure there:

    python scripts/benchmark_latency.py --http http://localhost:8000/predict --image some.jpg

Inputs are random pixels (convolutional networks do the same work whatever the content), or real
images from `--images DIR`. What the numbers mean depends entirely on the hardware, the device
(`--device cpu` hides an attached GPU), and the one process they were measured in; record those
alongside any figure you quote. The script prints them.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))          # model_layers, load_model
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # the serving package

DEFAULT_BATCH_SIZES = (1, 8, 32)


def percentile(values, q):
    """Linear-interpolated percentile (the same definition numpy.percentile uses)."""
    return float(np.percentile(np.asarray(values, dtype=float), q))


def summarize(batch_size, latencies_s):
    """Turn per-iteration batch latencies (seconds) into the reported row."""
    ms = [v * 1000 for v in latencies_s]
    return {
        "batch_size": batch_size,
        "iterations": len(ms),
        "batch_p50_ms": percentile(ms, 50),
        "batch_p95_ms": percentile(ms, 95),
        "batch_mean_ms": statistics.fmean(ms),
        "per_image_p50_ms": percentile(ms, 50) / batch_size,
        "per_image_p95_ms": percentile(ms, 95) / batch_size,
        "throughput_img_per_s": batch_size * len(ms) / sum(latencies_s),
    }


def benchmark(predict, make_batch, batch_sizes=DEFAULT_BATCH_SIZES, warmup=5, iterations=30):
    """
    Time `predict(batch)` for each batch size. The first `warmup` calls per batch size are run
    and discarded (graph tracing, allocator growth, caches), then `iterations` calls are timed.
    """
    if iterations < 1 or warmup < 0:
        raise ValueError("iterations must be >= 1 and warmup >= 0")
    rows = []
    for batch_size in batch_sizes:
        batch = make_batch(batch_size)
        for _ in range(warmup):
            predict(batch)
        latencies = []
        for _ in range(iterations):
            started = time.perf_counter()
            predict(batch)
            latencies.append(time.perf_counter() - started)
        rows.append(summarize(batch_size, latencies))
    return rows


def random_batch_factory(size=(224, 224), seed=0):
    rng = np.random.default_rng(seed)

    def make(batch_size):
        return rng.uniform(0, 255, size=(batch_size, *size, 3)).astype(np.float32)

    return make


def image_batch_factory(directory, size=(224, 224)):
    """Real images, resized exactly as the service does, cycled to fill any batch size."""
    from serving.inference import preprocess

    paths = sorted(p for p in Path(directory).rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not paths:
        raise SystemExit(f"no .jpg/.jpeg/.png images under {directory}")
    pool = [preprocess(p.read_bytes(), size) for p in paths[:64]]

    def make(batch_size):
        return np.stack([pool[i % len(pool)] for i in range(batch_size)])

    return make


def http_benchmark(url, image_path, warmup=5, iterations=50):
    """Sequential POSTs of one image; returns a single summary row (batch_size 1)."""
    import httpx

    data = Path(image_path).read_bytes()
    files = lambda: {"file": (Path(image_path).name, data, "application/octet-stream")}  # noqa: E731
    with httpx.Client(timeout=60) as client:
        for _ in range(warmup):
            client.post(url, files=files()).raise_for_status()
        latencies = []
        for _ in range(iterations):
            started = time.perf_counter()
            client.post(url, files=files()).raise_for_status()
            latencies.append(time.perf_counter() - started)
    return [summarize(1, latencies)]


def environment(device):
    info = {
        "python": platform.python_version(),
        "machine": platform.machine(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "device_requested": device,
    }
    try:
        import tensorflow as tf

        info["tensorflow"] = tf.__version__
        info["visible_gpus"] = [d.name for d in tf.config.get_visible_devices("GPU")]
    except ImportError:
        pass
    return info


def print_table(rows):
    header = f"{'batch':>5} {'p50 ms':>9} {'p95 ms':>9} {'p50 ms/img':>11} {'p95 ms/img':>11} {'img/s':>9}"
    print(header)
    for r in rows:
        print(f"{r['batch_size']:>5} {r['batch_p50_ms']:>9.1f} {r['batch_p95_ms']:>9.1f} "
              f"{r['per_image_p50_ms']:>11.2f} {r['per_image_p95_ms']:>11.2f} "
              f"{r['throughput_img_per_s']:>9.1f}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", help="exported .keras file (model-level benchmark)")
    parser.add_argument("--backbone", default="ResNet50")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=list(DEFAULT_BATCH_SIZES))
    parser.add_argument("--warmup", type=int, default=5, help="untimed iterations per batch size")
    parser.add_argument("--iterations", type=int, default=30, help="timed iterations per batch size")
    parser.add_argument("--device", choices=["default", "cpu"], default="default",
                        help="'cpu' hides any GPU (e.g. Apple Metal) from TensorFlow")
    parser.add_argument("--images", help="directory of real images to use instead of random pixels")
    parser.add_argument("--http", help="URL of a running /predict endpoint (end-to-end mode)")
    parser.add_argument("--image", help="image file to POST in --http mode")
    parser.add_argument("--json", help="also write the results to this file")
    args = parser.parse_args(argv)

    if args.http:
        if not args.image:
            parser.error("--http needs --image")
        rows = http_benchmark(args.http, args.image, args.warmup, args.iterations)
        mode = "http"
    else:
        if not args.model:
            parser.error("give --model (or --http with --image)")
        if args.device == "cpu":
            import tensorflow as tf

            tf.config.set_visible_devices([], "GPU")
        from serving.app import load_predictor

        predictor = load_predictor(args.model, args.backbone)
        size = predictor.input_size
        make = image_batch_factory(args.images, size) if args.images else random_batch_factory(size)
        rows = benchmark(predictor.predict, make, args.batch_sizes, args.warmup, args.iterations)
        mode = "model"

    env = environment(args.device)
    print(f"mode={mode} warmup={args.warmup} iterations={args.iterations} env={json.dumps(env)}")
    print_table(rows)
    if args.json:
        Path(args.json).write_text(json.dumps({"mode": mode, "environment": env, "rows": rows}, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
