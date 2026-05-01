# Implementation Plan: Scoped Cleanup

## Overview
Enhance the `gme cleanup` command to support scoped deletion of intermediate artifacts, specifically allowing users to clean up images, shards, or both.

## Architecture Decisions
- **Scope Enum**: Define `CleanupScope(str, Enum)` for `all`, `shards`, and `images`.
- **Default Behavior**: Keep `shards` as the default scope for safety and backward compatibility.
- **Reporting**: Update the console output to dynamically show sections based on the selected scope.
- **Image Deletion**: Cleanup for `images` will target all files in `settings.images_dir`.

## Task List

### Phase 1: Core Logic Enhancement
- [ ] **Task 1: Define `CleanupScope` and helper for images**
  - **Description**: Add the Enum and `_eligible_images` helper to `cleanup.py`.
  - **Acceptance**: `_eligible_images` returns list of image paths and their sizes.
  - **Files**: `src/gme/cleanup.py`
- [ ] **Task 2: Refactor `run_cleanup` logic**
  - **Description**: Update `run_cleanup` to use the new scope and only process relevant artifacts.
  - **Acceptance**: Correct artifacts are identified and reported for each scope.
  - **Files**: `src/gme/cleanup.py`

### Phase 2: CLI Integration
- [ ] **Task 3: Update CLI arguments**
  - **Description**: Add `--scope` option to the `cleanup` command in `cli.py`.
  - **Acceptance**: `gme cleanup --help` shows `--scope`. Default is `shards`.
  - **Files**: `src/gme/cli.py`

### Phase 3: Verification
- [ ] **Task 4: Manual Verification**
  - **Description**: Run `gme cleanup --dry-run` with different scopes on dummy data.
  - **Acceptance**: Output matches expected behavior for each scope.
- [ ] **Task 5: Add Unit Tests**
  - **Description**: Create `tests/test_cleanup.py` to verify scoping logic.
  - **Acceptance**: Tests pass for all scopes and dry-run mode.
  - **Files**: `tests/test_cleanup.py`

## Risks and Mitigations
| Risk | Impact | Mitigation |
|------|--------|------------|
| User accidentally deletes images | High | Default scope is `shards`; require explicit `--scope images` or `all`. |
| Dry-run inaccuracy | Medium | Use the same collection logic for both dry-run and actual deletion. |

## Open Questions
- None.
