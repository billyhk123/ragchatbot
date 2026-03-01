"""Pathway document store – runs in a background thread inside the main service."""

import json
import logging
import os
import sys
import threading
import time
from pathlib import Path

logger = logging.getLogger(__name__)

# region agent log — diagnostic: discover installed Pathway module structure
_PATHWAY_OK = False
try:
    import pathway as pw
    logger.info("[Pathway:DIAG] pathway version=%s", getattr(pw, "__version__", "unknown"))

    import pathway.stdlib.indexing as _psi
    _idx_attrs = [x for x in dir(_psi) if not x.startswith("_")]
    logger.info("[Pathway:DIAG] pathway.stdlib.indexing attrs=%s", _idx_attrs)

    _idx_submodules = []
    for name in ["bm25", "hybrid_index", "nearest_neighbors", "data_index",
                  "full_text_document_index", "vector_document_index", "retrievers"]:
        try:
            __import__(f"pathway.stdlib.indexing.{name}")
            _idx_submodules.append(name)
        except ImportError:
            pass
    logger.info("[Pathway:DIAG] available submodules=%s", _idx_submodules)

    BruteForceKnnFactory = None
    HybridIndexFactory = None
    TantivyBM25Factory = None

    try:
        from pathway.stdlib.indexing import (
            BruteForceKnnFactory, HybridIndexFactory, TantivyBM25Factory,
        )
        logger.info("[Pathway:DIAG] import strategy: top-level OK")
    except ImportError as e1:
        logger.info("[Pathway:DIAG] top-level import failed: %s", e1)
        try:
            from pathway.stdlib.indexing.bm25 import TantivyBM25Factory
            from pathway.stdlib.indexing.hybrid_index import HybridIndexFactory
            from pathway.stdlib.indexing.nearest_neighbors import BruteForceKnnFactory
            logger.info("[Pathway:DIAG] import strategy: submodule OK")
        except ImportError as e2:
            logger.info("[Pathway:DIAG] submodule import failed: %s", e2)
            for attr in _idx_attrs:
                obj = getattr(_psi, attr, None)
                if obj and "Factory" in attr:
                    logger.info("[Pathway:DIAG] found factory: %s = %s", attr, obj)

    if BruteForceKnnFactory and HybridIndexFactory and TantivyBM25Factory:
        _PATHWAY_OK = True
    else:
        logger.error("[Pathway] Missing indexing classes — Pathway retriever will not start")
        logger.error("[Pathway] BruteForceKnnFactory=%s HybridIndexFactory=%s TantivyBM25Factory=%s",
                     BruteForceKnnFactory, HybridIndexFactory, TantivyBM25Factory)

    from google.cloud import storage
    from pathway.xpacks.llm import embedders
    from pathway.xpacks.llm.document_store import DocumentStore
    from pathway.xpacks.llm.parsers import UnstructuredParser
    from pathway.xpacks.llm.splitters import RecursiveSplitter

except Exception:
    logger.exception("[Pathway] Failed to import pathway — retriever will not start")
# endregion

GCS_BUCKET = os.environ.get("GCS_BUCKET", "ragchatbot-raw")
GCS_PREFIX = os.environ.get("GCS_PREFIX", "")
LOCAL_DIR = Path(os.environ.get("LOCAL_DATA_DIR", "/app/data/raw"))
SYNC_INTERVAL = int(os.environ.get("GCS_SYNC_INTERVAL", "60"))
SYNC_DELETE = os.environ.get("GCS_SYNC_DELETE", "false").lower() == "true"

PATHWAY_HOST = os.environ.get("PATHWAY_HOST", "0.0.0.0") or "0.0.0.0"
PATHWAY_PORT = int(os.environ.get("PATHWAY_PORT", "8081"))

EMBEDDING_MODEL = os.environ.get(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)

STATE_FILE = Path("/app/data/.gcs_state.json")


def _load_state() -> dict:
    if not STATE_FILE.exists():
        return {}
    return json.loads(STATE_FILE.read_text(encoding="utf-8"))


def _save_state(state: dict) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _sync_gcs_once() -> None:
    client = storage.Client()
    bucket = client.bucket(GCS_BUCKET)

    state = _load_state()
    seen = set()

    blobs = client.list_blobs(bucket, prefix=GCS_PREFIX)
    for blob in blobs:
        name = blob.name
        if name.endswith("/"):
            continue

        seen.add(name)
        generation = str(blob.generation or "")
        if state.get(name) == generation and (LOCAL_DIR / name).exists():
            continue

        target = LOCAL_DIR / name
        target.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(target))
        state[name] = generation

    if SYNC_DELETE:
        local_files = [
            p for p in LOCAL_DIR.rglob("*") if p.is_file() and p.name != STATE_FILE.name
        ]
        for p in local_files:
            rel = str(p.relative_to(LOCAL_DIR)).replace("\\", "/")
            if rel not in seen:
                p.unlink(missing_ok=True)
                state.pop(rel, None)

    _save_state(state)


def _start_sync_loop():
    _sync_gcs_once()
    if SYNC_INTERVAL <= 0:
        return

    def loop():
        while True:
            time.sleep(SYNC_INTERVAL)
            try:
                _sync_gcs_once()
            except Exception:
                logger.exception("[Pathway] GCS sync error")

    t = threading.Thread(target=loop, name="gcs-sync", daemon=True)
    t.start()


def main():
    """Start the Pathway document store (blocking – run in a thread)."""
    if not _PATHWAY_OK:
        logger.error("[Pathway] Skipping start — required classes not available")
        return
    LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("[Pathway] Starting on %s:%s", PATHWAY_HOST, PATHWAY_PORT)
    _start_sync_loop()

    docs = pw.io.fs.read(
        path=str(LOCAL_DIR),
        format="binary",
        with_metadata=True,
        mode="streaming",
    )

    parser = UnstructuredParser(chunking_mode="single")
    splitter = RecursiveSplitter(
        chunk_size=400,
        chunk_overlap=40,
        encoding_name="cl100k_base",
    )

    embedder = embedders.SentenceTransformerEmbedder(model=EMBEDDING_MODEL)
    retriever_factory = HybridIndexFactory(
        [TantivyBM25Factory(), BruteForceKnnFactory(embedder=embedder)]
    )

    store = DocumentStore(
        docs,
        retriever_factory=retriever_factory,
        parser=parser,
        splitter=splitter,
    )

    webserver = pw.io.http.PathwayWebserver(
        host=PATHWAY_HOST, port=PATHWAY_PORT, with_cors=True
    )

    def serve(route, schema, handler, summary, description):
        queries, writer = pw.io.http.rest_connector(
            webserver=webserver,
            route=route,
            methods=("GET", "POST"),
            schema=schema,
            autocommit_duration_ms=50,
            delete_completed_queries=False,
            documentation=pw.io.http.EndpointDocumentation(
                summary=summary,
                description=description,
                method_types=("GET",),
            ),
        )
        writer(handler(queries))

    serve(
        "/v1/retrieve",
        store.RetrieveQuerySchema,
        store.retrieve_query,
        "Similarity search",
        "Return top-k chunks using hybrid (BM25 + vector) retrieval.",
    )
    serve(
        "/v1/statistics",
        store.StatisticsQuerySchema,
        store.statistics_query,
        "Index stats",
        "Get indexer statistics and last update time.",
    )
    serve(
        "/v1/inputs",
        store.InputsQuerySchema,
        store.inputs_query,
        "Indexed inputs",
        "Get metadata for all indexed source files.",
    )

    pw.run(monitoring_level=pw.MonitoringLevel.NONE)
