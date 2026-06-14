---
type: Tool
title: Kueue
description: Kubernetes-native job queueing and quota management for batch and AI workloads.
tags: [scheduling, kubernetes, batch, quota]
resource: https://github.com/kubernetes-sigs/kueue
timestamp: 2026-05-30T10:50:39-07:00
---

# Kueue

← [Back to Scheduling](index.md)

---

**Source:** [github.com/kubernetes-sigs/kueue](https://github.com/kubernetes-sigs/kueue) · API stability: v1beta2

Kueue is a Kubernetes-native job queueing and admission controller. It operates at the job level — deciding *when* a job may start (i.e. when its pods may be created) — rather than at the pod scheduling level. It sits above the Kubernetes scheduler and integrates with it rather than replacing it.

---

## Core Concepts

### ResourceFlavor

A named hardware variant. Labels a set of nodes with their resource type so Kueue can enforce placement:

```yaml
apiVersion: kueue.x-k8s.io/v1beta1
kind: ResourceFlavor
metadata:
  name: h100-80gb
spec:
  nodeLabels:
    cloud.google.com/gke-accelerator: nvidia-h100-80gb
```

### ClusterQueue

A cluster-scoped queue that owns resource quota across one or more ResourceFlavors. Defines the total accelerator budget available to a group of tenants:

```yaml
apiVersion: kueue.x-k8s.io/v1beta1
kind: ClusterQueue
metadata:
  name: ml-team-cq
spec:
  namespaceSelector: {}
  resourceGroups:
    - coveredResources: [nvidia.com/gpu]
      flavors:
        - name: h100-80gb
          resources:
            - name: nvidia.com/gpu
              nominalQuota: 64
              borrowingLimit: 32   # can borrow up to 32 more from cohort
```

### LocalQueue

A namespace-scoped queue that points to a ClusterQueue. Users submit jobs to a LocalQueue; Kueue maps them to the ClusterQueue for admission:

```yaml
apiVersion: kueue.x-k8s.io/v1beta1
kind: LocalQueue
metadata:
  name: default
  namespace: team-alpha
spec:
  clusterQueueName: ml-team-cq
```

### Workload

The internal Kueue representation of an admitted or pending job. Created automatically when a supported job type (BatchJob, RayJob, JobSet, etc.) is submitted with a `kueue.x-k8s.io/queue-name` label.

### Cohort

A named group of ClusterQueues that can lend and borrow quota from each other. Enables resource sharing between teams without a single global queue.

---

## Admission Flow

```
User submits Job  →  Kueue creates Workload  →  Workload enters LocalQueue
                                                        │
                                              ClusterQueue evaluates quota
                                                        │
                              ┌─────────────────────────┴──────────────────────┐
                              │ Quota available                                  │ Quota full
                              ▼                                                  ▼
                        Workload admitted                             Workload queued
                        Pods created                                  (pending preemption
                                                                       or quota release)
```

Jobs are held as zero-pod workloads until quota is available. No pods are created (and therefore no accelerator time is consumed) while queued.

---

## Queueing Strategies

| Strategy | Behavior |
|----------|---------|
| `StrictFIFO` | Strict ordering; a low-priority job at the head blocks higher-priority jobs behind it |
| `BestEffortFIFO` | Higher-priority jobs can skip ahead of a blocked lower-priority job |

---

## Key Features

**Fair-sharing** — distributes unused quota proportionally among competing ClusterQueues within a cohort. Prevents one team from monopolizing borrowed capacity.

**Preemption** — when a high-priority workload needs resources, Kueue can evict lower-priority admitted workloads. Configurable per-ClusterQueue:
```yaml
preemption:
  reclaimWithinCohort: LowerPriority
  withinClusterQueue: LowerPriority
```

**Topology-aware scheduling** — schedules pods within a job onto nodes that minimize inter-node communication distance (rack → block → datacenter). Critical for training jobs that use all-reduce heavily.

**Partial admission** — allows a job to start with fewer workers than requested if full quota isn't available, useful for elastic training frameworks (Ray Train, PyTorch elastic).

**AdmissionChecks** — pluggable hooks that block admission until external conditions are met (e.g. a provisioning request, a quota check in an external system).

**MultiKueue** — dispatches jobs across multiple Kubernetes clusters, with a management cluster routing workloads to worker clusters based on available quota.

---

## Supported Job Types

| Job type | Integration |
|----------|------------|
| `batch/v1 Job` | Native |
| `JobSet` | Via JobSet operator |
| `RayJob` / `RayCluster` | Via KubeRay |
| Kubeflow `TFJob`, `PyTorchJob`, `MPIJob` | Via Kubeflow Training Operator |
| `Deployment`, `StatefulSet` | Serving workloads (experimental) |
| Plain `Pod` | Native |

Jobs must carry the label `kueue.x-k8s.io/queue-name: <local-queue-name>` to be managed by Kueue.

---

## Interaction with Other Schedulers

Kueue does **not** replace the Kubernetes scheduler — it controls *whether* pods are created. Once admitted, pods are scheduled normally by kube-scheduler (or a custom scheduler like volcano). For gang scheduling guarantees (all pods land together), use placement groups in Ray or the `podGroupPolicy` in Kueue's JobSet integration.

---

## See Also

- [Ray / KubeRay](ray.md) — RayJob/RayCluster are primary Kueue workload targets for ML training
- [Pathways](pathways.md) — PathwaysJob wraps JobSet, which Kueue can gate
- [llm-d](llm_d.md) — inference serving; Kueue can manage its deployment lifecycle
