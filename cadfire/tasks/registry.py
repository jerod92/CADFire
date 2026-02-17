"""
Task Registry: auto-discovery system for training tasks.

Tasks register themselves via the @register_task decorator.
The training loop uses TaskRegistry.sample() to get a random task
for each episode, optionally filtered by difficulty or category.

Adding a new task is as simple as:
    @register_task
    class MyNewTask(BaseTask):
        task_name = "my_new_task"
        ...
"""

from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

import numpy as np

from cadfire.tasks.base import BaseTask


class TaskRegistry:
    """Global registry of available training tasks."""

    _tasks: Dict[str, Type[BaseTask]] = {}

    @classmethod
    def register(cls, task_class: Type[BaseTask]):
        """Register a task class."""
        name = task_class.task_name
        cls._tasks[name] = task_class

    @classmethod
    def get(cls, name: str) -> Type[BaseTask]:
        """Get a task class by name."""
        if name not in cls._tasks:
            raise KeyError(f"Task '{name}' not registered. Available: {list(cls._tasks.keys())}")
        return cls._tasks[name]

    @classmethod
    def list_tasks(cls) -> List[str]:
        """List all registered task names."""
        return sorted(cls._tasks.keys())

    @classmethod
    def list_by_category(cls, category: str) -> List[str]:
        """List tasks filtered by category."""
        return [
            name for name, tc in cls._tasks.items()
            if tc.task_category == category
        ]

    @classmethod
    def list_by_difficulty(cls, max_difficulty: float) -> List[str]:
        """List tasks up to a difficulty threshold (for curriculum learning)."""
        return [
            name for name, tc in cls._tasks.items()
            if tc.difficulty <= max_difficulty
        ]

    @classmethod
    def sample(cls, seed: int | None = None,
               category: str | None = None,
               max_difficulty: float | None = None) -> BaseTask:
        """
        Sample a random task instance.
        Optionally filter by category and/or difficulty.
        """
        candidates = list(cls._tasks.keys())

        if category:
            candidates = [n for n in candidates if cls._tasks[n].task_category == category]
        if max_difficulty is not None:
            candidates = [n for n in candidates if cls._tasks[n].difficulty <= max_difficulty]

        if not candidates:
            raise ValueError("No tasks match the given filters")

        rng = np.random.RandomState(seed)
        name = candidates[rng.randint(len(candidates))]
        return cls._tasks[name](seed=seed)

    @classmethod
    def sample_weighted(cls, weights: Dict[str, float] | None = None,
                        seed: int | None = None) -> BaseTask:
        """Sample with custom weights per task name."""
        rng = np.random.RandomState(seed)
        if weights is None:
            return cls.sample(seed=seed)

        names = list(weights.keys())
        w = np.array([weights[n] for n in names])
        w = w / w.sum()
        name = names[rng.choice(len(names), p=w)]
        return cls._tasks[name](seed=seed)

    @classmethod
    def create(cls, name: str, seed: int | None = None) -> BaseTask:
        """Create a specific task by name."""
        return cls.get(name)(seed=seed)

    @classmethod
    def discover(cls):
        """
        Auto-discover and import all task modules in the tasks package.
        This triggers the @register_task decorators.
        """
        tasks_dir = Path(__file__).parent
        for module_info in pkgutil.iter_modules([str(tasks_dir)]):
            if module_info.name not in ("base", "registry", "__init__"):
                importlib.import_module(f"cadfire.tasks.{module_info.name}")

    @classmethod
    def count(cls) -> int:
        return len(cls._tasks)


def register_task(cls: Type[BaseTask]) -> Type[BaseTask]:
    """Decorator to register a task class."""
    TaskRegistry.register(cls)
    return cls
