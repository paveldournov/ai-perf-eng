# Ray / KubeRay

← [Back to Scheduling](index.md)

---

**Source:** [ray.io](https://ray.io) · [KubeRay GitHub](https://github.com/ray-project/kuberay)

Ray is an open-source distributed computing framework that scales Python workloads from a laptop to a large cluster with minimal code changes. In the AI infrastructure stack it provides a unified runtime for distributed training (Ray Train), hyperparameter tuning (Ray Tune), and inference serving (Ray Serve), with KubeRay handling Kubernetes lifecycle management.

---

## Core Model: Tasks and Actors

Ray's primitives are deliberately simple:

**Remote functions (tasks)** — stateless, run on any available worker:
```python
@ray.remote
def my_task(x):
    return x * 2

future = my_task.remote(42)   # non-blocking
result = ray.get(future)      # block for result
```

**Actors** — stateful distributed objects with persistent state across calls:
```python
@ray.remote
class ModelServer:
    def __init__(self): self.model = load_model()
    def predict(self, x): return self.model(x)
```

The scheduler places tasks/actors on nodes based on resource requirements specified as `@ray.remote(num_gpus=1, num_cpus=4)`.

---

## Ray Cluster Architecture

```
┌─────────────────────────────────────────────┐
│                Head Node                    │
│  ┌──────────────┐  ┌─────────────────────┐ │
│  │  GCS Server  │  │  Global Scheduler   │ │
│  │ (metadata,   │  │  (placement, quota) │ │
│  │  actor reg.) │  └─────────────────────┘ │
│  └──────────────┘                           │
└─────────────────────────────────────────────┘
        │                    │
┌───────▼──────┐    ┌────────▼─────┐
│  Worker Node │    │  Worker Node │  ...
│  Raylet      │    │  Raylet      │
│  (local      │    │  (local      │
│   scheduler) │    │   scheduler) │
│  Workers     │    │  Workers     │
└──────────────┘    └──────────────┘
```

Each node runs a **Raylet** (local scheduler + object store). The **GCS** (Global Control Store) on the head node maintains actor registry, placement group state, and cluster membership. Scheduling is hierarchical: the global scheduler places tasks at the node level; each Raylet handles fine-grained scheduling locally.

**Placement groups** allow co-locating a set of tasks/actors with specific topology (e.g. all on one node, or spread across nodes with specific GPU counts), critical for multi-GPU training jobs.

---

## Ray Train: Distributed Training

Ray Train wraps common training frameworks (PyTorch, JAX, TensorFlow, Lightning) into a uniform distributed API:

```python
from ray.train.torch import TorchTrainer

trainer = TorchTrainer(
    train_loop_per_worker=my_train_fn,
    scaling_config=ScalingConfig(num_workers=8, use_gpu=True),
)
result = trainer.fit()
```

Key capabilities:
- **Data-parallel training** — automatically sets up DDP/FSDP across workers
- **Fault tolerance** — checkpoints to object store; failed workers are restarted without restarting the job
- **Elastic training** — can scale worker count up/down mid-run
- **Framework integration** — PyTorch DDP, FSDP, DeepSpeed, Megatron-LM, JAX pmap/pjit

For large-scale TPU/GPU jobs, Ray Train is often combined with model-parallel libraries (Megatron, FSDP) — Ray manages the cluster and fault tolerance while the framework handles tensor/pipeline parallelism.

---

## Ray Serve: Inference Serving

Ray Serve is the inference serving layer. It composes multiple models and preprocessing steps into a **deployment graph**:

```python
@serve.deployment(num_replicas=4, ray_actor_options={"num_gpus": 1})
class LLMEndpoint:
    def __init__(self): self.model = load_llm()
    async def __call__(self, request): return await self.model.generate(...)
```

Key scheduling features:
- **Autoscaling** — scales replica count based on queue depth and latency targets
- **Request batching** — accumulates requests into a batch before forwarding to the model (configurable `max_batch_size`, `batch_wait_timeout`)
- **Router** — HTTP/gRPC router distributes requests across replicas; pluggable routing policies
- **Model composition** — chain models as a DAG; the router handles async fan-out/fan-in

Ray Serve does **not** provide KV-cache-aware routing natively (see [llm-d](llm_d.md) for that).

---

## KubeRay: Kubernetes Integration

KubeRay is a Kubernetes operator that manages Ray cluster lifecycle via three CRDs:

| CRD | Purpose |
|-----|---------|
| `RayCluster` | Defines head + worker node groups; autoscaling bounds |
| `RayJob` | Submits a one-shot job to a managed cluster; cluster torn down on completion |
| `RayService` | Manages a long-running Ray Serve deployment; rolling updates, zero-downtime upgrades |

`RayCluster` integrates with **Kueue** via the `RayJob` workload API — Kueue handles admission and quota; KubeRay handles pod lifecycle once the job is admitted.

### Example RayCluster snippet

```yaml
apiVersion: ray.io/v1
kind: RayCluster
spec:
  headGroupSpec:
    replicas: 1
    rayStartParams: { num-cpus: "4" }
  workerGroupSpecs:
    - groupName: gpu-workers
      replicas: 8
      minReplicas: 4
      maxReplicas: 16
      rayStartParams: { num-gpus: "8" }
      template:
        spec:
          containers:
            - resources:
                limits: { nvidia.com/gpu: "8" }
```

---

## Scheduling Policies and Resource Management

| Feature | Details |
|---------|---------|
| Resource types | CPUs, GPUs, memory, custom resources (e.g. `TPU`) |
| Placement strategies | `STRICT_PACK` (same node), `PACK` (fewest nodes), `SPREAD` (most nodes), `STRICT_SPREAD` |
| Priority | Per-task priority via `scheduling_strategy` |
| Gang scheduling | Placement groups reserve all resources atomically before any task starts |
| Fractional resources | `num_gpus=0.5` allows multiple tasks to share a GPU |

---

## See Also

- [Kueue](kueue.md) — cluster-level admission control that gates RayJob submission
- [llm-d](llm_d.md) — KV-cache-aware inference routing that Ray Serve lacks natively
- [Pathways](pathways.md) — Google's equivalent single-controller runtime for TPU
- [LLM inference model](../modeling/llm_inference.md) — latency model for the serving workloads Ray Serve runs
