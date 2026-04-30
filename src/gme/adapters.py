from __future__ import annotations

import difflib
import io
import string
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from rich import print as rprint

# ── Config ───────────────────────────────────────────────────────────────────


@dataclass
class PayloadConfig:
    include: list[str] | None = None
    exclude: list[str] = field(default_factory=list)
    indexes: dict[str, str] = field(default_factory=dict)


@dataclass
class DatasetConfig:
    dataset: str
    split: str = "train"
    id_column: str | None = None
    text_template: str = ""
    image_column: str | None = None
    modality: Literal["text", "image", "multimodal"] = "multimodal"
    payload: PayloadConfig = field(default_factory=PayloadConfig)

    @classmethod
    def from_yaml(cls, path: Path | str) -> DatasetConfig:
        import yaml

        with open(path) as f:
            data = yaml.safe_load(f) or {}

        if "dataset" not in data:
            raise ValueError(f"{path}: missing required key 'dataset'")

        p = data.get("payload") or {}
        return cls(
            dataset=data["dataset"],
            split=data.get("split", "train"),
            id_column=data.get("id_column"),
            text_template=data.get("text_template", ""),
            image_column=data.get("image_column"),
            modality=data.get("modality", "multimodal"),
            payload=PayloadConfig(
                include=p.get("include"),
                exclude=p.get("exclude") or [],
                indexes=p.get("indexes") or {},
            ),
        )

    def apply_overrides(
        self,
        dataset: str | None = None,
        split: str | None = None,
        id_column: str | None = None,
        text_template: str | None = None,
        image_column: str | None = None,
        modality: str | None = None,
    ) -> None:
        if dataset is not None:
            self.dataset = dataset
        if split is not None:
            self.split = split
        if id_column is not None:
            self.id_column = id_column
        if text_template is not None:
            self.text_template = text_template
        if image_column is not None:
            self.image_column = image_column
        if modality is not None:
            self.modality = modality  # type: ignore[assignment]

    def template_columns(self) -> set[str]:
        return {fn for _, fn, _, _ in string.Formatter().parse(self.text_template) if fn}

    def excluded_from_payload(self) -> set[str]:
        cols: set[str] = set()
        if self.id_column:
            cols.add(self.id_column)
        cols.update(self.payload.exclude)
        return cols

    def payload_columns(self, all_columns: list[str]) -> list[str]:
        excluded = self.excluded_from_payload()
        if self.payload.include is not None:
            return [c for c in self.payload.include if c not in excluded]
        return [c for c in all_columns if c not in excluded]


# ── NormalizedRecord ─────────────────────────────────────────────────────────


@dataclass
class NormalizedRecord:
    record_id: str
    text: str
    image_url: str
    image_path: Path | None
    pil_failed: bool
    payload: dict


# ── HuggingFaceAdapter ───────────────────────────────────────────────────────


class HuggingFaceAdapter:
    def __init__(self, cfg: DatasetConfig, images_dir: Path, max_side_px: int = 512) -> None:
        self._cfg = cfg
        self._images_dir = images_dir
        self._max_side_px = max_side_px
        self._ds = None
        self._image_type: str | None = None  # "url" | "pil"

    def load(self) -> None:
        import datasets as hf

        cfg = self._cfg
        ds = hf.load_dataset(cfg.dataset, split=cfg.split)
        col_names = list(ds.features.keys())

        if cfg.id_column and cfg.id_column not in ds.features:
            raise ValueError(
                f"id_column {cfg.id_column!r} not found. Available columns: {col_names}"
            )

        for col in cfg.template_columns():
            if col not in ds.features:
                raise ValueError(
                    f"text_template references column {col!r} which doesn't exist. "
                    f"Available columns: {col_names}"
                )

        if cfg.modality in ("image", "multimodal") and not cfg.image_column:
            raise ValueError(f"modality {cfg.modality!r} requires image_column to be set.")

        if cfg.image_column:
            if cfg.image_column not in ds.features:
                raise ValueError(
                    f"image_column {cfg.image_column!r} not found. "
                    f"Available columns: {col_names}"
                )
            feat = ds.features[cfg.image_column]
            if isinstance(feat, hf.Image):
                self._image_type = "pil"
            elif isinstance(feat, hf.Value) and feat.dtype in ("string", "large_string"):
                self._image_type = "url"
            else:
                raise ValueError(
                    f"image_column {cfg.image_column!r} has unsupported type {feat!r}. "
                    f"Supported: string (URL) or Image (PIL)."
                )

        if cfg.payload.include is not None:
            include_set = set(cfg.payload.include)
            for col in cfg.payload.indexes:
                if col not in include_set:
                    raise ValueError(
                        f"indexes column {col!r} is not in payload.include."
                    )

        self._ds = ds

    @property
    def total(self) -> int:
        if self._ds is None:
            raise RuntimeError("Call load() first.")
        return len(self._ds)

    def iter_records(self, limit: int | None = None) -> Iterator[NormalizedRecord]:
        if self._ds is None:
            raise RuntimeError("Call load() first.")

        cfg = self._cfg
        ds = self._ds.select(range(min(limit, len(self._ds)))) if limit is not None else self._ds
        all_columns = list(ds.features.keys())
        payload_cols = cfg.payload_columns(all_columns)

        id_warned = False

        for idx, row in enumerate(ds):
            row = dict(row)

            if cfg.id_column:
                record_id = str(row[cfg.id_column])
            else:
                if not id_warned:
                    rprint(
                        "[yellow]Warning: no id_column set — using row index. "
                        "Point IDs will not survive dataset updates.[/yellow]"
                    )
                    id_warned = True
                record_id = str(idx)

            try:
                text = cfg.text_template.format_map(row) if cfg.text_template else ""
            except KeyError as e:
                rprint(
                    f"[yellow]Warning: text_template missing key {e} "
                    f"for record {record_id!r} — skipping.[/yellow]"
                )
                continue

            payload: dict = {}
            for k in payload_cols:
                v = row.get(k)
                payload[k] = v if isinstance(v, (str, int, float, bool)) or v is None else str(v)

            image_url = ""
            image_path: Path | None = None
            pil_failed = False

            if cfg.image_column:
                img_val = row.get(cfg.image_column)
                if self._image_type == "url":
                    image_url = str(img_val or "")
                elif self._image_type == "pil":
                    image_path = self._write_pil_image(img_val, record_id)
                    if image_path is None and img_val is not None:
                        pil_failed = True

            yield NormalizedRecord(
                record_id=record_id,
                text=text,
                image_url=image_url,
                image_path=image_path,
                pil_failed=pil_failed,
                payload=payload,
            )

    def _write_pil_image(self, pil_img: object, record_id: str) -> Path | None:
        if pil_img is None:
            return None
        try:
            from gme.images import _normalize_image, _sha256

            buf = io.BytesIO()
            pil_img.save(buf, format="JPEG", quality=95)  # type: ignore[union-attr]
            normalized = _normalize_image(buf.getvalue(), self._max_side_px)
            sha = _sha256(normalized)
            path = self._images_dir / f"{sha}.jpg"
            if not path.exists():
                path.write_bytes(normalized)
            return path
        except Exception as exc:
            rprint(f"[yellow]Warning: PIL image write failed for {record_id!r}: {exc}[/yellow]")
            return None


# ── gme inspect ──────────────────────────────────────────────────────────────


def run_inspect(
    dataset: str,
    split: str = "train",
    generate_config: bool = False,
    config_path: Path = Path("dataset.yaml"),
) -> None:
    import datasets as hf
    import yaml
    from rich import print as rprint
    from rich.table import Table

    rprint(f"Loading schema for [bold]{dataset}[/bold]…")
    ds = hf.load_dataset(dataset, split=split)
    total = len(ds)
    rprint(f"\nDataset: [bold]{dataset}[/bold]  ({split} split, {total:,} rows)\n")

    sample = ds.select(range(min(100, total)))
    id_candidates = {"id", "_id", "uuid", "key"}
    id_col_guess: str | None = None
    image_col_guess: str | None = None
    string_cols: list[str] = []

    table = Table(show_header=True, header_style="bold")
    table.add_column("Column")
    table.add_column("Type")
    table.add_column("Notes")

    for col_name, feat in ds.features.items():
        type_str = _feature_type_str(feat)
        notes = ""

        if isinstance(feat, hf.Image):
            notes = "image column"
            if image_col_guess is None:
                image_col_guess = col_name
        elif isinstance(feat, hf.Value) and feat.dtype in ("string", "large_string"):
            string_cols.append(col_name)
            if col_name.lower() in id_candidates:
                values = [str(sample[i][col_name]) for i in range(len(sample))]
                if len(set(values)) == len(values):
                    notes = "good ID candidate"
                    if id_col_guess is None:
                        id_col_guess = col_name
        elif not isinstance(feat, hf.Value):
            notes = "not embeddable directly"

        table.add_row(col_name, type_str, notes)

    rprint(table)

    if not generate_config:
        return

    cfg_dict: dict = {
        "dataset": dataset,
        "split": split,
    }
    if id_col_guess:
        cfg_dict["id_column"] = id_col_guess
    non_id_strings = [c for c in string_cols if c != id_col_guess]
    if non_id_strings:
        cfg_dict["text_template"] = " ".join(f"{{{c}}}" for c in non_id_strings[:2])
    if image_col_guess:
        cfg_dict["image_column"] = image_col_guess
        cfg_dict["modality"] = "multimodal" if non_id_strings else "image"
    else:
        cfg_dict["modality"] = "text"

    cfg_dict["payload"] = {
        "include": non_id_strings or None,
        "indexes": {id_col_guess: "keyword"} if id_col_guess else {},
    }

    new_content = yaml.dump(cfg_dict, default_flow_style=False, sort_keys=False)

    if config_path.exists():
        existing = config_path.read_text()
        if existing == new_content:
            rprint(f"[yellow]{config_path} already exists and is identical — no changes.[/yellow]")
            return
        diff = list(
            difflib.unified_diff(
                existing.splitlines(keepends=True),
                new_content.splitlines(keepends=True),
                fromfile=f"existing {config_path}",
                tofile="generated",
            )
        )
        rprint(f"[yellow]{config_path} already exists. Diff (existing → generated):[/yellow]")
        for line in diff:
            rprint(line, end="")
        rprint(f"\n[yellow]Edit {config_path} manually or delete it to regenerate.[/yellow]")
    else:
        config_path.write_text(new_content)
        rprint(f"[green]→ {config_path} written (review before running gme ingest).[/green]")


def _feature_type_str(feat: object) -> str:
    import datasets as hf

    if isinstance(feat, hf.Image):
        return "Image (PIL)"
    if isinstance(feat, hf.Value):
        return feat.dtype
    if isinstance(feat, hf.Sequence):
        return f"sequence<{_feature_type_str(feat.feature)}>"
    if isinstance(feat, dict):
        return "struct"
    return type(feat).__name__
