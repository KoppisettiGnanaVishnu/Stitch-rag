
import os
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import (
    FastAPI,
    File,
    HTTPException,
    UploadFile,
)
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from rag.config import RAGConfig
from rag.pipeline import RAGPipeline


# ----------------------------------------------------
# In-memory index registry. Each uploaded document set
# becomes a pipeline keyed by an index_id. For a
# single-tenant deployment this is enough; a
# production system would persist pipelines or move
# them behind a job queue.
# ----------------------------------------------------

_PIPELINES: Dict[str, RAGPipeline] = {}


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
        files: List[UploadFile] = File(...)
    ):

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

            target = temp_dir / filename

            content = await upload.read()

            target.write_bytes(content)

            saved_paths.append(target)
            source_names.append(filename)

        try:

            pipeline = RAGPipeline(
                saved_paths,
                source_name=source_names,
            )

        except Exception as error:

            raise HTTPException(
                status_code=422,
                detail=f"Failed to index documents: {error}"
            ) from error

        index_id = uuid.uuid4().hex

        _PIPELINES[index_id] = pipeline

        sources = sorted({
            chunk["source"] for chunk in pipeline.chunks
        })

        return IndexResponse(
            index_id=index_id,
            chunks=len(pipeline.chunks),
            sources=sources,
        )

    @app.post("/ask", response_model=AskResponse)
    def ask_question(request: AskRequest):

        pipeline = _PIPELINES.get(request.index_id)

        if pipeline is None:

            raise HTTPException(
                status_code=404,
                detail="Unknown index_id. Call /index first."
            )

        strategy = request.strategy.lower()

        if strategy == "mmr":

            results = pipeline.retrieve_mmr(
                request.question
            )

        else:

            results = pipeline.retrieve(request.question)

        context = pipeline.create_context(results)

        answer = pipeline.generator.generate(
            request.question,
            context,
            history=request.history
        )

        evaluation = pipeline.evaluator.evaluate_answer(
            request.question,
            context,
            answer,
            results
        )

        return AskResponse(
            answer=answer,
            results=results,
            evaluation=evaluation,
        )

    @app.delete("/index/{index_id}")
    def delete_index(index_id: str):

        removed = _PIPELINES.pop(index_id, None)

        if removed is None:

            raise HTTPException(
                status_code=404,
                detail="Unknown index_id."
            )

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
