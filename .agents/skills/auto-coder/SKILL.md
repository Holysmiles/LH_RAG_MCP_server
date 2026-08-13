---
name: auto-coder
description: 'Autonomously implements the next DEV_SPEC.md task: syncs the spec, selects a pending task, changes code and tests, runs up to three repair cycles, updates task status, and optionally creates a scoped commit. Use for "auto code", "自动开发", "自动写代码", "auto dev", "一键开发", "autopilot", or other requests to continue spec-driven development.'
---

# Auto Coder

Complete one task per cycle:

`sync spec -> select task -> implement -> verify -> update progress -> optionally commit`

Accept an optional task ID, such as `auto code B2`. If the request includes `--no-commit`, finish without asking to commit.

## Environment

Run Python and pytest through the `rag` Conda environment:

```powershell
conda run -n rag python <args>
conda run -n rag python -m pytest <args>
```

If the environment is missing, stop and report the blocker. Create it only when the user explicitly authorizes environment setup.

## Workflow

### 1. Sync and Select

1. Inspect `git status`; preserve unrelated and pre-existing changes.
2. Run:

   ```powershell
   conda run -n rag python .agents/skills/auto-coder/scripts/sync_spec.py
   ```

3. Read `.agents/skills/auto-coder/references/06-schedule.md`.
4. Select the requested task ID. Otherwise select the first `[~]` task, then the first `[ ]` task.
5. Stop if no pending task exists or a required predecessor makes the task impossible.

### 2. Read Only What Is Needed

Always read the selected task's goal, expected files, dependencies, and acceptance criteria from `06-schedule.md`. Read other references only when relevant:

- `01-overview.md`: project intent or scope
- `02-features.md`: user-facing behavior
- `03-tech-stack.md`: dependencies and implementation constraints
- `04-testing.md`: test strategy
- `05-architecture.md`: module boundaries and file placement
- `07-future.md`: roadmap or scope decisions

Resolve ambiguity in this order: explicit user instruction, selected task acceptance criteria, referenced architecture/testing constraints, then existing code patterns. Report contradictions that materially change behavior instead of guessing.

### 3. Implement

1. Inspect the affected code and tests before editing.
2. Define the smallest file-level plan that satisfies the acceptance criteria.
3. Implement within existing module boundaries and conventions.
4. Do not hardcode values the spec designates as configurable.
5. Add or update tests for changed behavior.
6. Document public contracts and non-obvious rationale; prefer clear code over redundant docstrings or line-by-line comments.

### 4. Verify and Repair

1. Run the narrowest relevant tests first, then the closest broader suite needed to detect regressions.
2. If a task-caused failure occurs, diagnose, fix, and rerun. Allow at most three fix-and-test cycles.
3. Do not spend repair cycles changing code for unrelated failures or unavailable external services.
4. If verification still fails, stop without marking the task complete. Report the failing command, root cause, attempted fixes, and remaining blocker.

### 5. Persist

Only after the acceptance criteria are satisfied and required tests pass:

1. Mark the selected task complete in `DEV_SPEC.md`.
2. Run:

   ```powershell
   conda run -n rag python .agents/skills/auto-coder/scripts/sync_spec.py --force
   ```

3. Summarize the task, changed files, and test results.
4. If `--no-commit` was supplied, leave changes uncommitted and stop.
5. Otherwise ask once whether to:
   - `commit`: stage only task-owned files and create one task-scoped commit
   - `skip`: leave changes uncommitted
   - `next`: create the scoped commit, then start the next cycle

Never stage unrelated changes. Never mark a blocked or failing task complete.
