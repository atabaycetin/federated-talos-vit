# Federated Learning and Task Localized Sparse Fine-Tuning Project for Machine Learning and Deep Learning Class

This repository supports a research project on **Deep Learning and Federated Learning**, focusing on **centralized and federated training of Vision Transformers (ViT)** on the CIFAR-100 dataset. The project explores advanced fine-tuning and pruning methods such as **LoRA** and **TaLoS**, targeting efficient adaptation in distributed, resource-constrained scenarios.

It also contains a **production layer** (`src/talos_mlops/`) that turns the centralized TaLoS method into a served model: a config-driven training pipeline tracked in MLflow, ONNX + INT8 export, a FastAPI service with Prometheus metrics and drift detection, Docker images, CI/CD with GitHub Actions, and deployment on Azure Container Apps.

---

## Production layer

```
configs/*.yaml ──► talos run ──► artifacts/<run>/          ──► talos package ──► model_store/<task>/ ──► Docker image ──► Azure Container Apps
                   │ train       meta.json, head.npz,                                                    │ /predict /health
                   │ export      task_vector.npz,                                                        │ /tasks /drift /metrics
                   │ benchmark   model(.int8).onnx                                                       └► Prometheus (alerts: drift, p95, errors)
                   └ register ─► MLflow run + model registry
```

### What gets trained

The backbone is the public DINO ViT-S/16 (`timm/vit_small_patch16_224.dino`), so nothing about it is stored. A run produces one **task bundle**:

1. **Linear-probe warmup** (1 epoch, frozen backbone), so the Fisher scores are computed through a trained head instead of a random one.
2. **TaLoS mask**: empirical diagonal Fisher on a 5% calibration split. The `1 − sparsity` *least*-sensitive backbone weights are made trainable, as in Iurada et al. (ICLR 2025).
3. **Sparse fine-tuning**: masked weights get zero gradient and zero weight decay. The run fails if any frozen weight moves.
4. **Bundle**: the head plus the sparse task vector τ = (θ_ft − θ_0) ⊙ mask, stored as a bit-packed mask plus values in a pickle-free `.npz`. Nothing else is written: no per-epoch checkpoints, optimizer state or Fisher dumps.

`method: linear_probe` (`configs/linear_probe_cifar100.yaml`) stops after step 1. It trains in minutes and produces the same bundle format, so it works as a fallback while the TaLoS run trains.

> The research `calibrate_mask` (centralized) keeps the *highest*-Fisher weights; the federated `calibrate_mask_global` and the paper keep the *lowest*. The production pipeline follows the paper. Train and test images also used different normalization in the research loaders (ImageNet vs. CIFAR statistics). The production pipeline uses a single preprocessing function for training, export and serving.

### Quickstart

```bash
conda create -n talos-mlops python=3.12 && conda activate talos-mlops && pip install -e ".[train]"   # use ".[train,dev]" to also get ruff

talos run --config configs/talos_cifar100.yaml               # train -> export -> benchmark -> register
talos package --config configs/talos_cifar100.yaml --store model_store
docker compose up --build                                     # API on :8000, Prometheus on :9090

curl -F file=@cat.png "localhost:8000/predict?top_k=3"
mlflow ui --backend-store-uri sqlite:///mlflow.db             # runs, metrics, registered model versions
```

Any config value can be overridden from the CLI, e.g. a 2-minute smoke run:
`talos run --config configs/talos_cifar100.yaml --set run_name=smoke --set epochs=1 --set data.limit_train=2048 --set data.limit_eval=512`.

### Serving

| endpoint | |
|---|---|
| `POST /predict?task=&top_k=` | multipart image upload -> top-k labels, probabilities, model version, latency |
| `GET /health` | 200 when models are loaded, 503 (with the load error) otherwise. Always public, for probes and CI. |
| `GET /tasks` | loaded tasks with method, sparsity, τ size, test metrics |
| `GET /drift` | PSI of live inputs vs. the validation set, per feature (channel means, contrast, top-1 confidence) |
| `GET /metrics` | Prometheus: request counts by status, latency histograms, confidence, predicted classes, drift PSI, task swaps |

**Authentication.** When `API_KEY` is set, every endpoint except `/health` and the docs requires it as an `X-API-Key` header. The check runs before the upload is read, so requests without the key cost almost nothing. The deployed service keeps the key as an Azure Container Apps secret (`API_KEY=secretref:api-key`), never in the image or the repo. On `/docs`, use the **Authorize** button. Without `API_KEY`, the service is open and logs a warning; that's for local runs and the CI smoke test.

```bash
curl -H "X-API-Key: $KEY" -F file=@cat.png "https://<app>.azurecontainerapps.io/predict?task=cifar10"
```

Two backends, chosen with `BACKEND`:

- **`onnx`** (default image, no torch): one INT8 ONNX Runtime session per task.
- **`taskvector`** (`docker build --target taskvector`): **one shared backbone**. Each task contributes only its sparse τ and head, and the requested task's τ is swapped in per request. To swap, the previous task's masked positions are restored from a pristine copy and the new τ is added. This is the task-arithmetic use case TaLoS was designed for.

### Results

Research results (50 epochs, see the report) are unchanged. The production model is a separate, shorter run and is reported on its own.

**Production models**, one per task on the same DINO backbone (`configs/talos_cifar100.yaml`, `configs/talos_cifar10.yaml`): 1 linear-probe epoch, then **4** sparse epochs at 60% sparsity. 8.67M of the 21.7M backbone weights are trainable. Each trained in under an hour on an Apple M5 (MPS). Checkpoints were selected on the validation split; each test set was evaluated once.

| | val top-1 | test top-1 | test top-5 |
|---|---|---|---|
| CIFAR-100: TaLoS, 60% sparsity, 4 epochs | 88.72% | **88.46%** | 98.89% |
| CIFAR-10: TaLoS, 60% sparsity, 4 epochs | 98.04% | **97.96%** | 100.00% |

Export and CPU inference for CIFAR-100 (2,000 test images; latency at batch 1, 4 threads, Apple M5 CPU). CIFAR-10 is similar: INT8 21.1 MB, 97.85% top-1, p95 11.7 ms.

| variant | size (MB) | top-1 | agreement with fp32 | p50 (ms) | p95 (ms) |
|---|---|---|---|---|---|
| PyTorch fp32 | 82.8 | 88.75% | 100% | 15.7 | 16.8 |
| ONNX fp32 | 83.8 | 88.75% | 100% | 11.7 | 12.9 |
| ONNX INT8 (served) | 21.2 | 88.65% | 98.75% | 10.5 | 11.2 |

Each task's bundle (task vector plus head) is 35.9 MB. The backbone is the public DINO checkpoint.

### Multi-task serving

Both tasks run in one service: `POST /predict?task=cifar100` or `?task=cifar10`. Measured with Azure's per-replica limits (0.5 vCPU, 1 GiB) in Docker, alternating tasks on every request:

| backend | how tasks are held | memory | latency p50 / p95 | task switch |
|---|---|---|---|---|
| `onnx` (deployed) | one INT8 model per task | 172 MiB | 66 / 155 ms | none |
| `taskvector` | one fp32 backbone + a sparse τ and head per task | 917 MiB | 782 / 1,182 ms | 155 ms mean on 0.5 vCPU (12 ms on 4 M5 cores) |

Swapping is exact: on real test images of both tasks, outputs from the shared backbone match the separately merged models with a maximum difference of 0.0.

The `taskvector` backend scales with the number of tasks better. Each added task costs only a ~36 MB sparse τ and head, instead of a full model. But at this replica size it nearly fills the memory limit and runs fp32 PyTorch, so Azure runs the `onnx` image. The task-vector design pays off once there are many tasks, or the hardware is larger.

### Monitoring and drift

`docker compose up` runs the API alongside Prometheus. Prometheus scrapes `/metrics` and evaluates three alert rules (`monitoring/alerts.yml`): input drift (PSI > 0.2 for 15 minutes), p95 latency above 250 ms, and an error rate above 1%.

At training time, the validation set is summarized into reference histograms of channel means, contrast and top-1 confidence. The service compares a rolling window of the last 500 requests against them using the Population Stability Index (PSI). Here is a check against the running stack: 200 real CIFAR-100 test images, then 300 copies darkened to 25% brightness:

| traffic | requests correct | PSI (brightness / contrast) | PSI (confidence) | drift alert |
|---|---|---|---|---|
| 200 real test images | 89% | 0.03–0.06 | 0.05 | no |
| + 300 darkened images | 85% | 1.43–1.49 | 0.01 | **yes**: Prometheus `InputDrift` fires after its 15-minute window |

The input shift is caught even though the model's confidence barely moves. A monitor that only watched the model's outputs would miss it. The end-to-end p95 latency for this run was 39 ms (CPU, in Docker on the M5).

![Prometheus alerts page: InputDrift firing for task cifar100 at PSI 1.49, latency and error-rate alerts inactive](docs/images/prometheus-alerts.png)

### CI/CD and deployment

`.github/workflows/ci.yml` runs on every push and pull request:

```
lint (ruff) ─┐
test (pytest)┴► build image ► container smoke test ► Trivy scan ► push to GHCR ► deploy to Azure ► post-deploy health check
                              (/health, /predict,   (fails on       (main only)   (main only, protected
                               /metrics)             fixable HIGH/                `production` environment)
                                                     CRITICAL CVEs)
```

The service runs on **Azure Container Apps** (Switzerland North): 0.5 vCPU / 1 GiB per replica, scaling from **0** replicas when idle to at most 1, with no Log Analytics workspace. That keeps it within the consumption free tier. Every deploy is a new revision tagged with the commit SHA.

The trained model store is published as a GitHub Release asset, not committed. CI downloads it and bakes it into the image:

```bash
bash scripts/publish_model.sh configs/talos_cifar100.yaml configs/talos_cifar10.yaml model-v2   # packages every task, uploads, sets MODEL_RELEASE
deploy/azure/setup.sh                                          # one-time: RG, environment, app, deploy identity, GitHub environment
```

**Security and supply chain**

- **No stored cloud credentials.** GitHub Actions signs in to Azure with OIDC through a user-assigned managed identity. The federated credential only trusts this repository's `production` environment, matched by repository and owner IDs, and only `main` can deploy to that environment.
- **Least privilege.** The deploy identity has *Container Apps Contributor* on the app and its environment only: no rights over the resource group, no ability to create resources, and no access to its own credentials. The running container has no Azure identity at all. The workflow's GitHub token is read-only by default.
- **Pinned inputs.** GitHub Actions are pinned to commit SHAs, and the base image to a digest. Python dependencies are installed from `pip-compile` lock files with `--require-hashes`, resolved for Linux inside the base image (`scripts/lock_requirements.sh`). Dependabot opens weekly PRs for all three, and each PR runs the full pipeline, scan included.
- **Image hardening.** The image runs as a non-root user and contains only the INT8 model and serving code (no torch, no fp32 model). Trivy runs from its official image pinned by digest, not from `trivy-action`, whose tags were hijacked in March 2026.
- **API hardening.** Requests without the right `X-API-Key` are rejected (401) before any body is read, using a constant-time comparison. Uploads are rejected by `Content-Length` (413 over 10 MB, 411 without the header) before the body is read, image dimensions are capped against decompression bombs, and nothing is ever unpickled: model bundles are `.npz` files read with `allow_pickle=False`.

---

## Project Structure

```
DLML_FL_project/
│
├── colab_runner.ipynb      # Jupyter notebook for Colab environment setup and running experiments
├── train_centralized.py    # Centralized (single-node) training script
├── train_federated.py      # Federated learning training script
├── test.py                 # Model evaluation and checkpoint testing
├── requirements.txt        # Python dependencies
├── config/                 # Configuration files (e.g., config.yaml)
├── data/                   # Dataset storage and preprocessing scripts
├── model_editing/          # Fine-tuning and pruning modules (e.g., LoRA, TaLoS)
├── models/                 # Model architectures (e.g., Vision Transformers)
├── fl_core/                # Federated learning core logic (client/server)
├── project_utils/          # Utility functions (metrics, schedulers, logging)
├── .idea/, .gitignore      # Development and VCS files
└── README.md
```

## Key Features

- **Hybrid Training Support**: Run both centralized and federated experiments with the same architecture.
- **Fine-Tuning Strategies**:
  - *LoRA*: Low-Rank Adaptation for efficient tuning.
  - *TaLoS*: Layer-wise pruning based on sensitivity.
  - *Dense*: Classic full-model fine-tuning.
- **Federated Learning Core**: Simulated client-server setup supporting custom aggregation and gradient masking.
- **ViT with DINO**: Leverages powerful self-supervised features from DINO-pretrained ViT backbones.
- **WandB Integration**: Seamless experiment logging and sweep tracking with [Weights & Biases](https://wandb.ai/).
- **Flexible Configuration**: YAML-based experiment configuration for easy reproducibility.
- **Colab Ready**: Plug-and-play support for Google Colab users.

---

## Usage

### Google Colab

A ready-made notebook (`colab_runner.ipynb`) is provided for experiments in Google Colab:
- Mounts Google Drive for data and checkpoint storage.
- Installs requirements and runs training scripts.
- Includes guidance for using wandb and running sweeps.

### Configuration

Edit `config/config.yaml` to adjust hyperparameters such as batch size, learning rate, number of epochs, fine-tuning strategy, data splits, checkpoint paths, federated learning parameters, etc.

Example:
```yaml
batch_size: 256
val_split: 0.2
num_workers: 4
learning_rate: 0.008805
weight_decay: 0.01
momentum: 0.9
epochs: 50
finetuning_method: 'lora' # or 'dense', 'talos'
checkpoint_path: './checkpoints'
...
```

## Resources

- Project Google Drive folder: [link](https://drive.google.com/drive/folders/1JXbfCxJPu4f1d09HdaeyZtpiLiD5xQoI?usp=share_link)

---

## References

1. Hsu et al., *Measuring the Effects of Non-Identical Data Distribution for Federated Visual Classification*, 2019.  
2. Reddi et al., *Adaptive Federated Optimization*, ICLR 2021.  
3. Kairouz et al., *Advances and Open Problems in Federated Learning*, Foundations and Trends in Machine Learning, 2021.  
4. Li et al., *Federated Optimization in Heterogeneous Networks*, PMLR, 2018.  
5. Karimireddy et al., *SCAFFOLD: Stochastic Controlled Averaging for Federated Learning*, PMLR, 2020.  
6. Kairouz et al., *Advances and Open Problems in Federated Learning*, Foundations and Trends in Machine Learning, 2021.  
7. Li et al., *A Survey on Federated Learning Systems: Vision, Hype and Reality for Data Privacy and Protection*, 2019.  
8. Chen et al., *Privacy and Fairness in Federated Learning: On the Perspective of Tradeoff*, ACM Computing Surveys, 2023.  
9. Pfeiffer et al., *Federated Learning for Computationally Constrained Heterogeneous Devices: A Survey*, ACM Computing Surveys, 2023.  
10. McMahan et al., *Communication-Efficient Learning of Deep Networks from Decentralized Data*, AISTATS, 2017.  
11. Hsu et al., *Federated Visual Classification with Real-World Data Distribution*, ECCV, 2020.  
12. Qu et al., *Rethinking Architecture Design for Tackling Data Heterogeneity in Federated Learning*, CVPR, 2022.  
13. Pieri et al., *Handling Data Heterogeneity via Architectural Design for Federated Visual Recognition*, NeurIPS, 2023.  
14. Li et al., *FedBN: Federated Learning on Non-IID Features via Local Batch Normalization*, ICLR, 2021.  
15. Iurada et al., *Efficient Model Editing with Task-Localized Sparse Fine-tuning*, ICLR, 2025.  

