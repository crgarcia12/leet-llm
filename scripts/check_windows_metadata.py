#!/usr/bin/env python3

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "cpp" / "lessons" / "windows-lessons.json"


def main() -> None:
    data = json.loads(CATALOG.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1:
        raise SystemExit("unsupported Windows lesson metadata schema")
    if (data.get("platform"), data.get("accelerator")) != (
        "windows",
        "nvidia-cuda",
    ):
        raise SystemExit("Windows/NVIDIA platform metadata is missing")

    lessons = data.get("lessons", [])
    expected_ids = [f"{number:03}" for number in range(48)]
    actual_ids = [lesson.get("id") for lesson in lessons]
    if actual_ids != expected_ids:
        raise SystemExit(f"expected lesson IDs 000..047, got {actual_ids}")

    for lesson in lessons:
        lesson_id = lesson["id"]
        problem = ROOT / lesson["problem"]
        if not problem.is_dir() or not (problem / "README.md").is_file():
            raise SystemExit(f"{lesson_id}: missing problem reading at {problem}")
        if lesson_id == "000":
            if lesson.get("kind") != "orientation":
                raise SystemExit("000 must be orientation metadata")
            if any(lesson.get(key) is not None for key in (
                "cpu_oracle",
                "cuda_starter",
                "cuda_solution",
            )):
                raise SystemExit("000 is reading-only and must not claim executable artifacts")
            continue
        if lesson.get("kind") != "exercise":
            raise SystemExit(f"{lesson_id}: runnable lesson must have kind exercise")
        for key in ("cpu_oracle", "cuda_starter", "cuda_solution"):
            artifact = lesson.get(key)
            if not artifact or not (ROOT / artifact).is_file():
                raise SystemExit(f"{lesson_id}: missing {key}: {artifact}")
        cmake = ROOT / "cpp" / "problems" / f"p{lesson_id}" / "CMakeLists.txt"
        text = cmake.read_text(encoding="utf-8")
        for target in (
            f"p{lesson_id}_oracle",
            f"p{lesson_id}_cuda_starter",
            f"p{lesson_id}_cuda_solution",
        ):
            if target not in text:
                raise SystemExit(f"{lesson_id}: CMake target {target} is missing")

    print("Windows lesson metadata complete: 000 orientation + 47 CPU/CUDA exercises")


if __name__ == "__main__":
    main()
