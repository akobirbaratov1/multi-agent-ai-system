"""
FastAPI Application — REST API for Multi-Agent AI System
Full OpenAPI documentation at /docs
"""

import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from api.routes import (
    chat_router,
    crm_router,
    knowledge_router,
    monitoring_router,
    operator_router,
    webhooks_router,
)
from api.schemas import HealthResponse
from core import config
from core.logging_config import configure_logging, get_logger, set_request_id
from memory.vector_store import get_vector_store
from tools.crm import get_crm
from tools.scheduler import FollowUpScheduler

configure_logging()
logger = get_logger(__name__)

WEB_DIR = config.BASE_DIR / "web"
REQUEST_ID_HEADER = "X-Request-ID"


@asynccontextmanager
async def lifespan(app: FastAPI):
    problems = config.validate()
    for problem in problems:
        logger.error("Configuration problem: %s", problem)

    if problems and config.IS_PRODUCTION:
        # Failing fast beats serving traffic with an open admin surface or a
        # wildcard CORS policy.
        raise RuntimeError(
            "Refusing to start in production with configuration problems: "
            + "; ".join(problems)
        )

    # Warm the shared stores so the first request doesn't pay index load.
    get_vector_store()
    get_crm()

    # Scheduled follow-ups previously sat in a queue nothing ever drained.
    scheduler = FollowUpScheduler()
    scheduler.start()

    logger.info(
        "Multi-Agent AI System started",
        extra={"version": config.VERSION, "env": config.APP_ENV, "demo_mode": config.DEMO_MODE},
    )
    try:
        yield
    finally:
        await scheduler.stop()
        logger.info("Multi-Agent AI System shutting down")


app = FastAPI(
    title="Multi-Agent AI System",
    description="Production-ready Multi-Agent AI with LangGraph, Claude & FAISS RAG",
    version=config.VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# A wildcard origin combined with credentials is rejected by browsers and, where
# honoured, lets any site issue authenticated cross-origin calls. Credentials are
# only enabled when the deployment names its origins explicitly.
_allow_credentials = "*" not in config.CORS_ORIGINS

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=_allow_credentials,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-API-Key"],
)


@app.middleware("http")
async def request_context(request: Request, call_next):
    """Attach a correlation ID to every request and its log records."""
    request_id = request.headers.get(REQUEST_ID_HEADER) or str(uuid.uuid4())
    set_request_id(request_id)
    request.state.request_id = request_id
    try:
        response = await call_next(request)
    finally:
        set_request_id(None)
    response.headers[REQUEST_ID_HEADER] = request_id
    return response


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """
    Return an opaque error to the caller and keep the detail in the logs.

    Route handlers previously surfaced `str(exc)` straight to the client,
    leaking internal paths, driver messages and API errors to anyone who could
    trigger a fault.
    """
    request_id = getattr(request.state, "request_id", "-")
    logger.exception("Unhandled error", extra={"path": request.url.path})
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "Internal server error",
            "request_id": request_id,
        },
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    # `exc.errors()` can carry the original exception object in `ctx`, which is
    # not JSON serializable — encode it the way FastAPI's own handler does.
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "detail": jsonable_encoder(exc.errors(), exclude={"ctx", "input", "url"}),
            "request_id": getattr(request.state, "request_id", "-"),
        },
    )


if WEB_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")

app.include_router(chat_router)
app.include_router(knowledge_router)
app.include_router(crm_router)
app.include_router(monitoring_router)
app.include_router(operator_router)
app.include_router(webhooks_router)


@app.get("/", include_in_schema=False)
async def root():
    index = WEB_DIR / "index.html"
    if not index.is_file():
        return JSONResponse({"service": "multi-agent-ai-system", "docs": "/docs"})
    return FileResponse(str(index))


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """System health check with stats."""
    return {
        "status": "healthy",
        "version": config.VERSION,
        "environment": config.APP_ENV,
        "demo_mode": config.DEMO_MODE,
        "vector_store_stats": get_vector_store().get_stats(),
        "crm_stats": get_crm().get_stats(),
    }


@app.get("/health/live", tags=["Health"], include_in_schema=False)
async def liveness():
    """Liveness probe — process is up. Never touches dependencies."""
    return {"status": "alive"}


@app.get("/health/ready", tags=["Health"])
async def readiness():
    """Readiness probe — dependencies are usable and configuration is coherent."""
    problems = config.validate()
    ready = not problems

    try:
        get_vector_store().get_stats()
    except Exception:
        logger.exception("Vector store is not readable")
        problems.append("vector store unavailable")
        ready = False

    return JSONResponse(
        status_code=status.HTTP_200_OK if ready else status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"status": "ready" if ready else "not_ready", "problems": problems},
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=config.APP_PORT,
        reload=not config.IS_PRODUCTION,
    )
