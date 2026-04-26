from pathlib import Path

from pydantic import Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Gemini
    gemini_api_key: str = Field(..., alias="GEMINI_API_KEY")
    gemini_model: str = "gemini-embedding-2"
    embedding_dim: int = Field(1536, alias="EMBEDDING_DIM")

    # Qdrant
    qdrant_url: str = Field("http://localhost:6333", alias="QDRANT_URL")
    qdrant_api_key: str = Field("", alias="QDRANT_API_KEY")
    qdrant_collection: str = Field("hm_products", alias="QDRANT_COLLECTION")

    # Paths
    data_dir: Path = Field(Path("data"), alias="DATA_DIR")

    # Batch tuning (Tier 1 @ 90%)
    records_per_shard: int = Field(40, alias="RECORDS_PER_SHARD")
    max_concurrent_jobs: int = Field(9, alias="MAX_CONCURRENT_JOBS")
    max_enqueued_tokens: int = Field(432_000, alias="MAX_ENQUEUED_TOKENS")
    tokens_per_record_estimate: int = 1200  # conservative: ~150 text + ~1024 image

    # Image download
    image_download_concurrency: int = Field(32, alias="IMAGE_DOWNLOAD_CONCURRENCY")
    image_max_side_px: int = Field(512, alias="IMAGE_MAX_SIDE_PX")

    # Dataset
    hf_dataset: str = "Qdrant/hm_ecommerce_products"
    dataset_split: str = "train"
    exclude_columns: list[str] = ["dense_embedding", "sparse_indices", "sparse_values"]

    @computed_field  # type: ignore[misc]
    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @computed_field  # type: ignore[misc]
    @property
    def images_dir(self) -> Path:
        return self.data_dir / "images"

    @computed_field  # type: ignore[misc]
    @property
    def batches_in_dir(self) -> Path:
        return self.data_dir / "batches" / "in"

    @computed_field  # type: ignore[misc]
    @property
    def batches_out_dir(self) -> Path:
        return self.data_dir / "batches" / "out"

    @computed_field  # type: ignore[misc]
    @property
    def vectors_parquet(self) -> Path:
        return self.data_dir / "vectors.parquet"

    @computed_field  # type: ignore[misc]
    @property
    def state_db(self) -> Path:
        return self.data_dir / "state.db"

    def ensure_dirs(self) -> None:
        for d in (
            self.raw_dir,
            self.images_dir,
            self.batches_in_dir,
            self.batches_out_dir,
            self.data_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
