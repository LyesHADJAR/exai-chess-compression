"""The parts of the cluster diagnosis that read Kubernetes' answers."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from diagnose_cluster import event_rows, gpu_claims, notebook_facts, pod_rows  # noqa: E402

NOTEBOOK = {
    "metadata": {
        "annotations": {
            "kubeflow-resource-stopped": "culled",
            "notebooks.kubeflow.org/last-activity": "2026-11-11T15:11:42Z",
            "notebooks.kubeflow.org/last_activity_check_timestamp": "2026-11-11T16:10:44Z",
            "kubectl.kubernetes.io/last-applied-configuration": "{...}",
            "notebooks.kubeflow.org/server-type": "jupyter",
        }
    }
}


def pod(name, *, node="gpu-1", started="2026-11-11T15:00:00Z", gpu=None, terminating=False):
    limits = {"cpu": "12", **({f"nvidia.com/{gpu[0]}": gpu[1]} if gpu else {})}
    meta = {"name": name, **({"deletionTimestamp": "2026-11-11T16:10:00Z"} if terminating else {})}
    return {
        "metadata": meta,
        "spec": {"nodeName": node, "containers": [{"resources": {"limits": limits}}]},
        "status": {"phase": "Running", "startTime": started},
    }


def test_culling_annotations_are_picked_out_and_the_rest_ignored():
    keys = [k for k, _ in notebook_facts(NOTEBOOK)]
    assert "kubeflow-resource-stopped" in keys
    assert "notebooks.kubeflow.org/last-activity" in keys
    assert "kubectl.kubernetes.io/last-applied-configuration" not in keys
    assert "notebooks.kubeflow.org/server-type" not in keys


def test_a_server_with_no_annotations_gives_nothing_rather_than_failing():
    assert notebook_facts({}) == []
    assert notebook_facts({"metadata": {"annotations": None}}) == []


def test_only_gpu_resources_count_as_a_gpu_claim():
    assert gpu_claims(pod("a", gpu=("mig-3g.47gb", 1))) == {"nvidia.com/mig-3g.47gb": "1"}
    assert gpu_claims(pod("b")) == {}


def test_a_pod_stuck_terminating_is_flagged():
    rows = pod_rows({"items": [pod("old", terminating=True), pod("new")]})
    assert {r["name"]: r["terminating"] for r in rows} == {"old": True, "new": False}


def test_pods_are_listed_oldest_first_so_a_leftover_comes_before_its_replacement():
    rows = pod_rows(
        {
            "items": [
                pod("new", started="2026-11-11T16:21:00Z"),
                pod("old", started="2026-11-11T15:00:00Z"),
            ]
        }
    )
    assert [r["name"] for r in rows] == ["old", "new"]


def test_a_second_pod_claiming_the_same_gpu_is_visible():
    rows = pod_rows({"items": [pod("a", gpu=("mig-3g.47gb", 1)), pod("b", gpu=("mig-3g.47gb", 1))]})
    assert sum(bool(r["gpu"]) for r in rows) == 2


def test_events_are_the_latest_ones_in_time_order():
    events = {
        "items": [
            {"lastTimestamp": f"2026-11-11T16:{m:02d}:00Z", "reason": f"R{m}", "type": "Normal",
             "involvedObject": {"name": "leanvit-0"}, "message": "x"}
            for m in (30, 10, 20)
        ]
    }  # fmt: skip
    assert [r[2] for r in event_rows(events, limit=2)] == ["R20", "R30"]


def test_event_messages_are_shortened_and_kept_on_one_line():
    long = {"items": [{"lastTimestamp": "2026-11-11T16:00:00Z", "message": "a\n" + "b" * 500}]}
    message = event_rows(long)[0][4]
    assert "\n" not in message and len(message) <= 90


def test_nothing_recorded_gives_an_empty_list():
    assert event_rows({}) == []
