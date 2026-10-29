"""`talos` command line.

talos run       --config configs/talos_cifar100.yaml        # train -> export -> benchmark -> register
talos train     --config ...   (also: export, benchmark, register)
talos package   --config ... --store model_store            # copy the serving files into a model store
talos ... --set epochs=1 --set data.limit_train=512         # dotted overrides
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from talos_mlops import bundle as B
from talos_mlops.config import PipelineConfig, load_config

SERVING_FILES = (B.META, B.HEAD, B.TASK_VECTOR, B.ONNX_INT8, B.ONNX_FP32, B.BENCHMARK)


def _train(cfg: PipelineConfig, tracker) -> None:
    from talos_mlops.train import run_training

    run_training(cfg, tracker)


def _export(cfg: PipelineConfig, tracker) -> None:
    from talos_mlops.export import export_onnx

    info = export_onnx(cfg.bundle_dir, cfg.export.opset, cfg.export.quantize)
    tracker.metrics(info)


def _benchmark(cfg: PipelineConfig, tracker) -> None:
    from talos_mlops.benchmark import run_benchmark

    report = run_benchmark(cfg, cfg.bundle_dir)
    for variant, r in report["variants"].items():
        tracker.metrics({f"bench_{variant}_{k}": v for k, v in r.items()})
    tracker.artifact(cfg.bundle_dir / B.BENCHMARK, "bundle")


def _register(cfg: PipelineConfig, tracker) -> None:
    meta = B.read_meta(cfg.bundle_dir)
    onnx_path = cfg.bundle_dir / (B.ONNX_INT8 if cfg.export.quantize else B.ONNX_FP32)
    version = tracker.register_onnx(
        onnx_path,
        {"task": cfg.task, "method": meta["method"], "test_top1": str(meta["metrics"]["test_top1"])},
    )
    if version:
        B.update_meta(
            cfg.bundle_dir, registered_model={"name": cfg.registered_model_name, "version": version}
        )
        print(f"[register] {cfg.registered_model_name} v{version}")


STAGES = {"train": _train, "export": _export, "benchmark": _benchmark, "register": _register}


def run_stages(cfg: PipelineConfig, stages: list[str]) -> None:
    from talos_mlops.tracking import Tracker

    tracker = Tracker(cfg)
    # Later stages reattach to the training run so one MLflow run holds the whole lineage.
    meta_path = cfg.bundle_dir / B.META
    run_id = None
    if stages[0] != "train" and meta_path.exists():
        run_id = B.read_meta(cfg.bundle_dir).get("mlflow_run_id")
    with tracker.run(run_id) as active_id:
        for stage in stages:
            print(f"==> {stage}")
            STAGES[stage](cfg, tracker)
            if stage == "train" and active_id:
                B.update_meta(cfg.bundle_dir, mlflow_run_id=active_id)


def package(cfg: PipelineConfig, store: Path) -> Path:
    dest = store / cfg.task
    dest.mkdir(parents=True, exist_ok=True)
    for name in SERVING_FILES:
        src = cfg.bundle_dir / name
        if src.exists():
            shutil.copy2(src, dest / name)
    print(f"[package] {cfg.bundle_dir} -> {dest}")
    return dest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="talos")
    parser.add_argument("command", choices=["run", *STAGES, "package"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="config override")
    parser.add_argument("--store", default="model_store", help="model store directory (package)")
    args = parser.parse_args(argv)

    cfg = load_config(args.config, args.set)
    if args.command == "package":
        package(cfg, Path(args.store))
    elif args.command == "run":
        run_stages(cfg, ["train", "export", "benchmark", "register"])
    else:
        run_stages(cfg, [args.command])


if __name__ == "__main__":
    main()
