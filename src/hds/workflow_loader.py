"""Load workflow specifications from JSON or Pegasus DAX XML."""

from __future__ import annotations

import json
from pathlib import Path
import xml.etree.ElementTree as ET

from .models import VNF, Workflow, WorkflowEdge


def load_json(path: str | Path) -> Workflow:
    """Load the documented project JSON schema."""

    source = Path(path)
    data = json.loads(source.read_text(encoding="utf-8"))
    vnfs = {
        item["id"]: VNF(
            id=item["id"],
            workload_mi=float(item["workload_mi"]),
            cpu_mips=float(item["cpu_mips"]),
            ram_mb=float(item["ram_mb"]),
            storage_mb=float(item.get("storage_mb", 64.0)),
            source_data_mb=float(item.get("source_data_mb", 0.0)),
            sink_data_mb=float(item.get("sink_data_mb", 0.0)),
        )
        for item in data["vnfs"]
    }
    edges = [
        WorkflowEdge(item["parent"], item["child"], float(item["data_mb"]))
        for item in data.get("edges", [])
    ]
    return Workflow(
        id=data.get("id", source.stem),
        family=data.get("family", "custom"),
        vnfs=vnfs,
        edges=edges,
        deadline_s=float(data["deadline_s"]),
        arrival_s=float(data.get("arrival_s", 0.0)),
    )


def load_dax(
    path: str | Path,
    deadline_s: float,
    fastest_mips: float = 16_000,
    default_cpu_mips: float = 1000,
    default_ram_mb: float = 256,
    default_storage_mb: float = 64,
) -> Workflow:
    """Load a Pegasus DAX file.

    DAX runtimes are converted to MI using ``fastest_mips``. File sizes are
    aggregated per producer-consumer dependency. Namespaces are handled
    independently of the Pegasus schema version.
    """

    source = Path(path)
    root = ET.parse(source).getroot()
    jobs: dict[str, VNF] = {}
    output_sizes: dict[str, dict[str, float]] = {}
    input_files: dict[str, set[str]] = {}

    for element in root.iter():
        if element.tag.rsplit("}", 1)[-1] != "job":
            continue
        job_id = element.attrib.get("id") or element.attrib.get("name")
        if not job_id:
            raise ValueError("DAX job without id/name")
        runtime_s = max(0.001, float(element.attrib.get("runtime", "1")))
        outputs: dict[str, float] = {}
        inputs: set[str] = set()
        for use in element:
            if use.tag.rsplit("}", 1)[-1] != "uses":
                continue
            file_name = use.attrib.get("file") or use.attrib.get("name")
            if not file_name:
                continue
            size_mb = float(use.attrib.get("size", "0")) / 1_000_000
            link = use.attrib.get("link", "input").lower()
            if link == "output":
                outputs[file_name] = max(0.0, size_mb)
            else:
                inputs.add(file_name)
        jobs[job_id] = VNF(
            job_id,
            runtime_s * fastest_mips,
            default_cpu_mips,
            default_ram_mb,
            default_storage_mb,
        )
        output_sizes[job_id] = outputs
        input_files[job_id] = inputs

    dependencies: set[tuple[str, str]] = set()
    for child in root.iter():
        if child.tag.rsplit("}", 1)[-1] != "child":
            continue
        child_id = child.attrib["ref"]
        for parent in child:
            if parent.tag.rsplit("}", 1)[-1] == "parent":
                dependencies.add((parent.attrib["ref"], child_id))

    edges: list[WorkflowEdge] = []
    for parent, child in sorted(dependencies):
        shared = set(output_sizes.get(parent, {})) & input_files.get(child, set())
        data_mb = sum(output_sizes[parent][name] for name in shared)
        edges.append(WorkflowEdge(parent, child, max(0.1, data_mb)))
    return Workflow(
        id=source.stem,
        family="dax",
        vnfs=jobs,
        edges=edges,
        deadline_s=deadline_s,
    )
