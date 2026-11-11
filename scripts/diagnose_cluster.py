"""
Find out why a notebook server's GPU is full, from inside the server.

    python scripts/diagnose_cluster.py

A Kubeflow notebook pod normally carries a credential that can read its own
namespace. This uses it, read-only, to print three things that explain most
GPU and restart problems: what the idle culler did to the server, which pods
exist in the namespace and which of them claim a GPU, and what Kubernetes
recorded recently. Nothing outside your own namespace is touched, and nothing
is changed. If the server is not allowed to read something, that is reported
rather than worked around.
"""

import json
import os
import ssl
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

SERVICE = Path("/var/run/secrets/kubernetes.io/serviceaccount")
API = "https://kubernetes.default.svc"

#: Annotation words that say what the culler thought of this server.
CULLING_WORDS = ("stopped", "activity", "culling", "idle")


def notebook_facts(notebook: dict) -> list[tuple[str, str]]:
    """The annotations that show whether, and why, a server was stopped."""
    annotations = notebook.get("metadata", {}).get("annotations", {}) or {}
    return sorted(
        (key, str(value))
        for key, value in annotations.items()
        if any(word in key.lower() for word in CULLING_WORDS)
    )


def gpu_claims(pod: dict) -> dict[str, str]:
    """The GPU resources a pod asks for, by resource name."""
    claims = {}
    for container in pod.get("spec", {}).get("containers", []):
        limits = (container.get("resources") or {}).get("limits") or {}
        claims.update({k: str(v) for k, v in limits.items() if k.startswith("nvidia.com")})
    return claims


def pod_rows(pods: dict) -> list[dict]:
    """One summary per pod, with anything stuck or claiming a GPU made obvious."""
    rows = []
    for pod in pods.get("items", []):
        meta, status = pod.get("metadata", {}), pod.get("status", {})
        rows.append(
            {
                "name": meta.get("name", "?"),
                "phase": status.get("phase", "?"),
                "node": pod.get("spec", {}).get("nodeName", "?"),
                "started": status.get("startTime", "?"),
                "terminating": bool(meta.get("deletionTimestamp")),
                "gpu": gpu_claims(pod),
            }
        )
    return sorted(rows, key=lambda r: r["started"])


def event_rows(events: dict, limit: int = 20) -> list[tuple[str, str, str, str, str]]:
    """The most recent events: when, type, reason, what it was about, the message."""
    items = events.get("items", [])

    def when(event):
        return (
            event.get("lastTimestamp")
            or event.get("eventTime")
            or event.get("firstTimestamp")
            or ""
        )

    rows = [
        (
            when(e)[11:19] or "?",
            e.get("type", ""),
            e.get("reason", ""),
            e.get("involvedObject", {}).get("name", ""),
            (e.get("message") or "").replace("\n", " ")[:90],
        )
        for e in sorted(items, key=when)[-limit:]
    ]
    return rows


def notebook_name() -> str:
    """The server's name: from its URL prefix, or its pod name without the -0."""
    prefix = os.environ.get("NB_PREFIX", "")
    if prefix.strip("/"):
        return prefix.strip("/").split("/")[-1]
    return os.uname().nodename.rsplit("-", 1)[0]


def fetch(path: str, token: str, context: ssl.SSLContext) -> tuple[int, dict]:
    request = urllib.request.Request(API + path, headers={"Authorization": f"Bearer {token}"})
    try:
        with urllib.request.urlopen(request, context=context, timeout=15) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as error:
        return error.code, {}
    except OSError as error:
        return 0, {"error": str(error)}


def heading(text: str) -> None:
    print(f"\n=== {text}")


def main() -> None:
    if not (SERVICE / "token").exists():
        sys.exit("This is not running inside a Kubernetes pod, so there is nothing to read.")
    token = (SERVICE / "token").read_text()
    namespace = (SERVICE / "namespace").read_text().strip()
    context = ssl.create_default_context(cafile=str(SERVICE / "ca.crt"))
    name = notebook_name()
    print(f"namespace {namespace}, server {name}, this pod {os.uname().nodename}")

    heading("What the idle culler recorded about this server")
    status, notebook = fetch(
        f"/apis/kubeflow.org/v1/namespaces/{namespace}/notebooks/{name}", token, context
    )
    if status == 200:
        for key, value in notebook_facts(notebook) or [("(no culling annotations)", "")]:
            print(f"  {key}: {value}")
    else:
        print(f"  could not read it (HTTP {status}); this server may not be allowed to")

    heading("Pods in your namespace, and which claim a GPU")
    status, pods = fetch(f"/api/v1/namespaces/{namespace}/pods", token, context)
    if status == 200:
        for row in pod_rows(pods):
            flags = []
            if row["terminating"]:
                flags.append("STUCK TERMINATING")
            if row["gpu"]:
                flags.append(
                    "GPU " + ", ".join(f"{k.split('/')[-1]}={v}" for k, v in row["gpu"].items())
                )
            where = f"started {row['started']}  node {row['node']}"
            print(f"  {row['name']:<28}{row['phase']:<11}{where}  {' | '.join(flags)}")
    else:
        print(f"  could not read pods (HTTP {status})")

    heading("What Kubernetes recorded recently in your namespace")
    status, events = fetch(f"/api/v1/namespaces/{namespace}/events", token, context)
    if status == 200:
        for when, kind, reason, subject, message in event_rows(events) or [
            ("", "", "(none)", "", "")
        ]:
            print(f"  {when}  {kind:<8}{reason:<22}{subject:<22}{message}")
    else:
        print(f"  could not read events (HTTP {status})")

    heading("What this server sees of the GPU")
    for variable in ("NVIDIA_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
        print(f"  {variable} = {os.environ.get(variable, '(not set)')}")
    try:
        out = subprocess.run(
            ["nvidia-smi", "-L"], capture_output=True, text=True, timeout=15
        ).stdout
        print("  " + out.strip().replace("\n", "\n  "))
    except (OSError, subprocess.TimeoutExpired):
        print("  nvidia-smi is not available")


if __name__ == "__main__":
    main()
