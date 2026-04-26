
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

from gme.config import Settings, get_settings

KEYWORD_INDEX_FIELDS = [
    "article_id",
    "product_type_name",
    "product_group_name",
    "colour_group_name",
    "perceived_colour_master_name",
    "index_group_name",
    "garment_group_name",
    "department_name",
    "section_name",
]


def get_qdrant_client(settings: Settings) -> QdrantClient:
    """Initialize and return a QdrantClient based on provided settings."""
    kwargs: dict = {"url": settings.qdrant_url}
    if settings.qdrant_api_key:
        kwargs["api_key"] = settings.qdrant_api_key
    return QdrantClient(**kwargs)


def run_qdrant_init(settings: Settings | None = None) -> None:
    """
    Initialize the Qdrant collection with appropriate vector and payload configurations.
    Enables binary quantization and on-disk storage for efficiency.
    """
    if settings is None:
        settings = get_settings()

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

    for field in KEYWORD_INDEX_FIELDS:
        try:
            client.create_payload_index(
                collection_name=collection,
                field_name=field,
                field_schema=PayloadSchemaType.KEYWORD,
            )
        except Exception:
            pass  # index may already exist

    print(f"Payload indexes ensured for: {', '.join(KEYWORD_INDEX_FIELDS)}")
