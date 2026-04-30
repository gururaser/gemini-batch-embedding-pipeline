from qdrant_client import QdrantClient
from qdrant_client.models import (
    BinaryQuantization,
    BinaryQuantizationConfig,
    Distance,
    HnswConfigDiff,
    OptimizersConfigDiff,
    PayloadSchemaType,
    VectorParams,
)
from rich import print

from gme.adapters import DatasetConfig
from gme.config import Settings, get_settings


def get_qdrant_client(settings: Settings) -> QdrantClient:
    kwargs: dict = {"url": settings.qdrant_url}
    if settings.qdrant_api_key:
        kwargs["api_key"] = settings.qdrant_api_key
    return QdrantClient(**kwargs)


def run_qdrant_init(settings: Settings | None = None, cfg: DatasetConfig | None = None) -> None:
    if settings is None:
        settings = get_settings()
    if cfg is None:
        cfg = DatasetConfig.from_yaml("dataset.yaml")
    client = get_qdrant_client(settings)
    collection = settings.qdrant_collection

    existing = [c.name for c in client.get_collections().collections]
    if collection in existing:
        print(f"Collection '{collection}' already exists — skipping creation.")
    else:
        client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(
                size=settings.embedding_dim,
                distance=Distance.COSINE,
                on_disk=True,
                hnsw_config=HnswConfigDiff(m=16, ef_construct=128, on_disk=True),
                quantization_config=BinaryQuantization(
                    binary=BinaryQuantizationConfig(always_ram=True),
                ),
            ),
            optimizers_config=OptimizersConfigDiff(
                default_segment_number=4,
                indexing_threshold=20_000,
            ),
            on_disk_payload=True,
        )
        print(
            f"Collection '{collection}' created "
            f"(dim={settings.embedding_dim}, COSINE, on-disk, binary quantization)."
        )

    if not cfg.payload.indexes:
        print("No payload indexes configured in dataset.yaml.")
        return

    for field_name, type_str in cfg.payload.indexes.items():
        try:
            schema_type = PayloadSchemaType(type_str)
        except ValueError:
            supported = [e.value for e in PayloadSchemaType]
            raise ValueError(
                f"Unknown index type {type_str!r} for field {field_name!r}. "
                f"Supported: {supported}"
            )
        try:
            client.create_payload_index(
                collection_name=collection,
                field_name=field_name,
                field_schema=schema_type,
            )
        except Exception:
            pass  # index may already exist

    print(f"Payload indexes ensured for: {', '.join(cfg.payload.indexes)}")
