"""
Typed schema for the planner/executor DAG.

TaskNode  — one unit of work; the planner fills it, the executor consumes it.
TaskPlan  — the full DAG for a single user request.

Both are intentionally narrow: the planner must be explicit about every field
and the executor never has to rediscover what the task is.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ExecutorType(str, Enum):
    """Which execution backend handles a TaskNode."""
    AGENT = "agent"       # run_agent() tool-loop, for research/retrieval nodes
    LLM = "llm"           # single llm_call(), for synthesis/finalization nodes


@dataclass
class TaskNode:
    """
    One node in the planned DAG.

    Fields the planner must supply
    --------------------------------
    id              : stable identifier referenced by depends_on in other nodes
    title           : one-line human-readable name (used in traces)
    instructions    : full, self-contained instructions for the executor
    expected_output : description of what a correct result looks like

    Fields with defaults the planner may override
    -----------------------------------------------
    inputs          : task ids whose outputs should be forwarded as context;
                      subset of (or equal to) depends_on
    depends_on      : task ids that must finish before this node is ready;
                      topological constraint, may differ from inputs
    executor_type   : "agent" (tool loop) or "llm" (single call); default "agent"
    agent_hint      : optional routing hint for model/prompt selection
    metadata        : free-form k/v for tracing or future extensions
    """

    id: str
    title: str
    instructions: str
    expected_output: str

    inputs: list[str] = field(default_factory=list)
    depends_on: list[str] = field(default_factory=list)
    executor_type: ExecutorType = ExecutorType.AGENT
    agent_hint: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id or not self.id.strip():
            raise ValueError("TaskNode.id must be a non-empty string")
        if not self.title or not self.title.strip():
            raise ValueError("TaskNode.title must be a non-empty string")
        if not self.instructions or not self.instructions.strip():
            raise ValueError("TaskNode.instructions must be a non-empty string")
        if not self.expected_output or not self.expected_output.strip():
            raise ValueError("TaskNode.expected_output must be a non-empty string")
        if isinstance(self.executor_type, str):
            self.executor_type = ExecutorType(self.executor_type)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TaskNode":
        """Parse a TaskNode from a plain dict (e.g. parsed planner JSON)."""
        required = {"id", "title", "instructions", "expected_output"}
        missing = required - data.keys()
        if missing:
            raise ValueError(f"TaskNode missing required fields: {missing}")
        return cls(
            id=data["id"],
            title=data["title"],
            instructions=data["instructions"],
            expected_output=data["expected_output"],
            inputs=list(data.get("inputs", [])),
            depends_on=list(data.get("depends_on", [])),
            executor_type=ExecutorType(data.get("executor_type", ExecutorType.AGENT)),
            agent_hint=data.get("agent_hint"),
            metadata=dict(data.get("metadata", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "instructions": self.instructions,
            "expected_output": self.expected_output,
            "inputs": self.inputs,
            "depends_on": self.depends_on,
            "executor_type": self.executor_type.value,
            "agent_hint": self.agent_hint,
            "metadata": self.metadata,
        }


@dataclass
class TaskPlan:
    """
    The full DAG for a single user request.

    Fields the planner must supply
    --------------------------------
    goal          : verbatim or paraphrased original user request
    tasks         : all TaskNode objects; at least one required
    final_task_id : id of the node whose output is the final answer

    Optional
    ---------
    output_order  : ordered list of task ids when the final answer is a
                    concatenation of multiple nodes; if set, final_task_id
                    is still required and should be the last entry
    """

    goal: str
    tasks: list[TaskNode]
    final_task_id: str
    output_order: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.goal or not self.goal.strip():
            raise ValueError("TaskPlan.goal must be a non-empty string")
        if not self.tasks:
            raise ValueError("TaskPlan.tasks must contain at least one TaskNode")
        ids = {t.id for t in self.tasks}
        if self.final_task_id not in ids:
            raise ValueError(
                f"TaskPlan.final_task_id '{self.final_task_id}' not found in tasks"
            )
        for node in self.tasks:
            unknown = set(node.depends_on) - ids
            if unknown:
                raise ValueError(
                    f"TaskNode '{node.id}' depends_on unknown ids: {unknown}"
                )
            unknown_inputs = set(node.inputs) - ids
            if unknown_inputs:
                raise ValueError(
                    f"TaskNode '{node.id}' inputs references unknown ids: {unknown_inputs}"
                )
        if self.output_order:
            unknown_order = set(self.output_order) - ids
            if unknown_order:
                raise ValueError(
                    f"TaskPlan.output_order references unknown ids: {unknown_order}"
                )
        self._validate_no_cycles()

    # ------------------------------------------------------------------
    # DAG helpers used by the execution loop
    # ------------------------------------------------------------------

    def _validate_no_cycles(self) -> None:
        """Raise ValueError if the dependency graph contains a cycle."""
        deps = {node.id: set(node.depends_on) for node in self.tasks}
        visited: set[str] = set()
        in_stack: set[str] = set()

        def dfs(node_id: str) -> None:
            visited.add(node_id)
            in_stack.add(node_id)
            for dep in deps.get(node_id, set()):
                if dep not in visited:
                    dfs(dep)
                elif dep in in_stack:
                    raise ValueError(
                        f"TaskPlan contains a dependency cycle involving '{dep}'"
                    )
            in_stack.discard(node_id)

        for node in self.tasks:
            if node.id not in visited:
                dfs(node.id)

    def task_by_id(self, task_id: str) -> TaskNode:
        for node in self.tasks:
            if node.id == task_id:
                return node
        raise KeyError(f"No task with id '{task_id}'")

    def ready_tasks(self, completed_ids: set[str]) -> list[TaskNode]:
        """Return tasks whose dependencies are all in completed_ids."""
        return [
            node for node in self.tasks
            if node.id not in completed_ids
            and set(node.depends_on).issubset(completed_ids)
        ]

    def topological_order(self) -> list[TaskNode]:
        """
        Return all tasks in a valid topological execution order
        (dependencies before dependents).
        """
        deps = {node.id: set(node.depends_on) for node in self.tasks}
        order: list[TaskNode] = []
        remaining = {node.id: node for node in self.tasks}
        done: set[str] = set()

        while remaining:
            ready = [
                nid for nid, node in remaining.items()
                if deps[nid].issubset(done)
            ]
            if not ready:
                raise ValueError(
                    "Cannot resolve topological order — possible cycle not caught earlier"
                )
            for nid in sorted(ready):
                order.append(remaining.pop(nid))
                done.add(nid)

        return order

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TaskPlan":
        """Parse a TaskPlan from a plain dict (e.g. parsed planner JSON)."""
        required = {"goal", "tasks", "final_task_id"}
        missing = required - data.keys()
        if missing:
            raise ValueError(f"TaskPlan missing required fields: {missing}")
        tasks = [TaskNode.from_dict(t) for t in data["tasks"]]
        return cls(
            goal=data["goal"],
            tasks=tasks,
            final_task_id=data["final_task_id"],
            output_order=list(data.get("output_order", [])),
        )

    @classmethod
    def from_json(cls, text: str) -> "TaskPlan":
        """Parse a TaskPlan from a JSON string returned by the planner."""
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Planner returned invalid JSON: {exc}") from exc
        return cls.from_dict(data)

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal": self.goal,
            "tasks": [t.to_dict() for t in self.tasks],
            "final_task_id": self.final_task_id,
            "output_order": self.output_order,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)
