# Spec: Enhanced `gme cleanup` with Scoped Deletion

## Objective
Provide fine-grained control over which intermediate artifacts are removed from the pipeline. Users should be able to clean up just images, just batch shards (in/out), or everything.

## Tech Stack
- Python 3.11+
- Typer (CLI)
- Rich (Logging/Console)
- SQLite (State DB)

## Commands
```bash
# Preview cleanup for a specific scope
uv run gme cleanup --scope shards --dry-run
uv run gme cleanup --scope images --dry-run
uv run gme cleanup --scope all --dry-run

# Execute cleanup
uv run gme cleanup --scope images
```

## Project Structure
- `src/gme/cleanup.py`: Contains the logic for identifying and deleting artifacts.
- `src/gme/cli.py`: CLI entry point and argument parsing.

## Code Style
```python
from enum import Enum

class CleanupScope(str, Enum):
    ALL = "all"
    SHARDS = "shards"
    IMAGES = "images"

def run_cleanup(scope: CleanupScope = CleanupScope.SHARDS, dry_run: bool = False) -> None:
    # Logic to handle different scopes
    ...
```

## Testing Strategy
- Unit tests for `_eligible_shards` and `_eligible_results` (existing logic).
- New unit tests for image cleanup eligibility.
- Integration test for `run_cleanup` with different scopes in a mock environment.

## Boundaries
- **Always do:** Require explicit scope for images (default to `shards` for safety if not specified).
- **Ask first:** Deleting `state.db` or `vectors.parquet`.
- **Never do:** Delete artifacts that are not yet "eligible" (e.g., in-progress shards).

## Success Criteria
- [ ] `gme cleanup` supports `--scope` with values `all`, `shards`, `images`.
- [ ] Default scope is `all`.
- [ ] `all` scope correctly identifies and deletes both shards and images.
- [ ] `--dry-run` accurately reports what would be deleted for each scope.

## Open Questions
- Should "shards" include both `in/` and `out/` as it does now? (Yes)
- Should "images" cleanup delete ALL images, or only those that are "orphaned"? (Assuming ALL images for now).
