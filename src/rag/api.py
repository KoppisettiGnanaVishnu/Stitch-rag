
import logging
import os
import tempfile
import time
import uuid
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import (
    Depends,
    FastAPI,
    File,
    HTTPException,
    Request,
    UploadFile,
)
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from rag.config import RAGConfig
from rag.pipeline import RAGPipeline


# ----------------------------------------------------
# Configuration
# ----------------------------------------------------

MAX_FILE_BYTES = int(
    os.environ.get("STITCH_MAX_FILE_BYTES", 25 * 1024 * 1024)
)

API_TOKEN = os.environ.get("STITCH_API_TOKEN")

RATE_LIMIT = int(os.environ.get("STITCH_RATE_LIMIT", "60"))

RATE_WINDOW = 60  # seconds

security = HTTPBearer(auto_error=False)

logging.basicConfig(
    level=os.environ.get("STITCH_LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

logger = logging.getLogger("stitch.api")

# ----------------------------------------------------
# In-memory index registry
# ----------------------------------------------------

_PIPELINES: Dict[str, RAGPipeline] = {}

# Simple per-IP sliding-window rate limiter

_REQUEST_LOG: Dict[str, deque] = defaultdict(deque)


def _rate_limit(request: Request):

    client = request.client.host if request.client else "unknown"

    now = time.time()

    bucket = _REQUEST_LOG[client]

    while bucket and bucket[0] < now - RATE_WINDOW:

        bucket.popleft()

    if len(bucket) >= RATE_LIMIT:

        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Try again later."
        )

    bucket.append(now)


def _auth(
    credentials: Optional[HTTPAuthorizationCredentials] = None
):

    if API_TOKEN is None:

        return

    if (
        credentials is None
        or credentials.credentials != API_TOKEN
    ):

        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API token."
        )


class AskRequest(BaseModel):

    index_id: str

    question: str

    strategy: str = "similarity"

    history: Optional[List[Dict[str, Any]]] = None


class IndexResponse(BaseModel):

    index_id: str

    chunks: int

    sources: List[str]


class AskResponse(BaseModel):

    answer: str

    results: List[Dict[str, Any]]

    evaluation: Dict[str, Any]


def create_app() -> FastAPI:

    app = FastAPI(
        title="Stitch — Document Intelligence API",
        description=(
            "REST API for indexing documents and "
            "answering questions over them."
        ),
        version="0.2.0",
    )

    @app.get("/health")
    def health():

        return {"status": "ok", "service": "stitch"}

    @app.post("/index", response_model=IndexResponse)
    async def index_documents(
        request: Request,
        files: List[UploadFile] = File(...),
        credentials: Optional[
            HTTPAuthorizationCredentials
        ] = Depends(security),
    ):

        _rate_limit(request)

        _auth(credentials)

        if not files:

            raise HTTPException(
                status_code=400,
                detail="Please upload at least one document."
            )

        temp_dir = Path(
            tempfile.mkdtemp(prefix="stitch_")
        )

        saved_paths = []
        source_names = []

        for upload in files:

            filename = upload.filename or "document"

            content = await upload.read()

            if len(content) > MAX_FILE_BYTES:

                raise HTTPException(
                    status_code=413,
                    detail=(
                        f"{filename} exceeds the "
                        f"{MAX_FILE_BYTES} byte limit."
                    )
                )

            target = temp_dir / filename

            target.write_bytes(content)

            saved_paths.append(target)
            source_names.append(filename)

        started = time.time()

        try:

            pipeline = RAGPipeline(
                saved_paths,
                source_name=source_names,
            )

        except Exception as error:

            logger.exception("indexing failed")

            raise HTTPException(
                status_code=422,
                detail=f"Failed to index documents: {error}"
            ) from error

        index_id = uuid.uuid4().hex

        _PIPELINES[index_id] = pipeline

        sources = sorted({
            chunk["source"] for chunk in pipeline.chunks
        })

        logger.info(
            "indexed %d files -> %d chunks in %.2fs (%s)",
            len(files),
            len(pipeline.chunks),
            time.time() - started,
            index_id,
        )

        return IndexResponse(
            index_id=index_id,
            chunks=len(pipeline.chunks),
            sources=sources,
        )

    @app.post("/ask", response_model=AskResponse)
    def ask_question(
        request: Request,
        body: AskRequest,
        credentials: Optional[
            HTTPAuthorizationCredentials
        ] = Depends(security),
    ):

        _rate_limit(request)

        _auth(credentials)

        pipeline = _PIPELINES.get(body.index_id)

        if pipeline is None:

            raise HTTPException(
                status_code=404,
                detail="Unknown index_id. Call /index first."
            )

        strategy = body.strategy.lower()

        started = time.time()

        if strategy == "mmr":

            results = pipeline.retrieve_mmr(
                body.question
            )

        else:

            results = pipeline.retrieve(body.question)

        context = pipeline.create_context(results)

        answer = pipeline.generator.generate(
            body.question,
            context,
            history=body.history
        )

        evaluation = pipeline.evaluator.evaluate_answer(
            body.question,
            context,
            answer,
            results
        )

        logger.info(
            "ask %s strategy=%s in %.2fs",
            body.index_id,
            strategy,
            time.time() - started,
        )

        return AskResponse(
            answer=answer,
            results=results,
            evaluation=evaluation,
        )

    @app.delete("/index/{index_id}")
    def delete_index(
        index_id: str,
        request: Request,
        credentials: Optional[
            HTTPAuthorizationCredentials
        ] = Depends(security),
    ):

        _rate_limit(request)

        _auth(credentials)

        removed = _PIPELINES.pop(index_id, None)

        if removed is None:

            raise HTTPException(
                status_code=404,
                detail="Unknown index_id."
            )

        logger.info("deleted index %s", index_id)

        return {"deleted": index_id}

    return app


app = create_app()


def main():

    import uvicorn

    uvicorn.run(
        "rag.api:app",
        host=os.environ.get("STITCH_API_HOST", "0.0.0.0"),
        port=int(os.environ.get("STITCH_API_PORT", "8000")),
        reload=False,
    )


if __name__ == "__main__":

    main()
