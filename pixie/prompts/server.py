"""FastAPI server for SDK."""

import argparse
import os
import colorlog
import logging
import webbrowser
import threading
import time


import dotenv
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
import httpx
from strawberry.fastapi import GraphQLRouter
import uvicorn

from pixie.prompts.file_watcher import (
    discover_and_load_modules,
    init_prompt_storage,
)
from pixie.prompts.graphql import schema


logger = logging.getLogger(__name__)

# Global logging mode
_logging_mode: str = "default"


def setup_logging(mode: str = "default"):
    """Configure logging for the entire application.

    Sets up colored logging with consistent formatting for all loggers.

    Args:
        mode: Logging mode - "default", "verbose", or "debug"
            - default: INFO for server events, WARNING+ for all modules
            - verbose: INFO+ for all modules
            - debug: DEBUG+ for all modules
    """
    global _logging_mode
    _logging_mode = mode

    # Determine log level based on mode
    if mode == "debug":
        level = logging.DEBUG
    elif mode == "verbose":
        level = logging.INFO
    else:  # default
        level = logging.INFO

    colorlog.basicConfig(
        level=level,
        format="[%(log_color)s%(levelname)-8s%(reset)s][%(asctime)s]\t%(message)s",
        datefmt="%H:%M:%S",
        log_colors={
            "DEBUG": "cyan",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "red,bg_white",
        },
        force=True,
    )

    # Configure uvicorn loggers to use the same format
    for logger_name in ["uvicorn", "uvicorn.access", "uvicorn.error"]:
        uvicorn_logger = logging.getLogger(logger_name)
        uvicorn_logger.handlers = []
        uvicorn_logger.propagate = True

    # In default mode, set most loggers to WARNING+ except specific modules
    if mode == "default":
        # Set root logger to WARNING
        logging.getLogger().setLevel(logging.WARNING)
        # Allow INFO for pixie modules
        logging.getLogger("pixie").setLevel(logging.INFO)
        # Suppress uvicorn access logs in default mode
        logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Returns:
        Configured FastAPI application instance with GraphQL router.
    """
    # Setup logging first (use global logging mode)
    setup_logging(_logging_mode)

    # Discover and load applications on every app creation (including reloads)
    discover_and_load_modules()

    env_path = os.getcwd() + "/.env"

    dotenv.load_dotenv(env_path)
    lifespan = init_prompt_storage()

    app = FastAPI(
        title="Pixie Prompts Dev Server",
        description="Server for managing prompts",
        version="0.1.0",
        lifespan=lifespan,
    )
    # Matches:
    # 1. http://localhost followed by an optional port (:8080, :3000, etc.)
    # 2. http://127.0.0.1 followed by an optional port
    # 3. https://yourdomain.com (the production domain)
    origins_regex = r"http://(localhost|127\.0\.0\.1)(:\d+)?|https://gopixie\.ai"
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=origins_regex,
        allow_credentials=True,
        allow_methods=["*"],  # Allows all methods
        allow_headers=["*"],  # Allows all headers
    )

    # Add GraphQL router with GraphiQL enabled
    graphql_app = GraphQLRouter(
        schema,
        graphiql=True,
    )

    app.include_router(graphql_app, prefix="/graphql")

    REMOTE_URL = "https://gopixie.ai"

    @app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
    async def proxy_all(request: Request, path: str):
        url = f"{REMOTE_URL}/{path}"
        if request.url.query:
            url += f"?{request.url.query}"

        logger.debug("Proxying request to: %s", url)

        async with httpx.AsyncClient(follow_redirects=True, timeout=30.0) as client:
            response = await client.request(
                method=request.method,
                url=url,
                headers={
                    k: v
                    for k, v in request.headers.items()
                    if k.lower() not in ["host"]
                },
                content=await request.body(),
            )

            # Explicitly remove compression-related headers
            headers = {
                k: v
                for k, v in response.headers.items()
                if k.lower()
                not in ["content-encoding", "content-length", "transfer-encoding"]
            }

            return Response(
                content=response.content,
                status_code=response.status_code,
                headers=headers,
            )

    return app


def start_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    reload: bool = False,
    log_mode: str = "default",
) -> None:
    """Start the SDK server.

    Args:
        host: Host to bind to
        port: Port to bind to
        reload: Enable auto-reload for development
        log_mode: Logging mode - "default", "verbose", or "debug"
        storage_directory: Directory to store prompt definitions
    """
    global _logging_mode
    _logging_mode = log_mode

    # Setup logging (will be called again in create_app for reload scenarios)
    setup_logging(log_mode)

    # Determine server URL
    server_url = f"http://{host}:{port}"
    if host == "0.0.0.0":
        server_url = f"http://127.0.0.1:{port}"

    # Log server start info
    logger.info("Starting Pixie SDK Server")
    logger.info("Server: %s", server_url)
    logger.info("GraphQL: %s/graphql", server_url)

    # Display gopixie.ai web link
    # encoded_url = quote(f"{server_url}/graphql", safe="")
    # pixie_web_url = f"https://gopixie.ai?url={encoded_url}"
    logger.info("")
    logger.info("=" * 60)
    logger.info("")
    logger.info("🎨 Pixie Web UI:")
    logger.info("")
    logger.info("   %s", server_url)
    logger.info("")
    logger.info("=" * 60)
    logger.info("")

    # Open browser after a short delay (in a separate thread)
    def open_browser():
        time.sleep(1.5)  # Wait for server to start
        webbrowser.open(server_url)

    threading.Thread(target=open_browser, daemon=True).start()

    uvicorn.run(
        "pixie.prompts.server:create_app",
        host=host,
        port=port,
        loop="asyncio",
        reload=reload,
        reload_includes=[".env"],
        factory=True,
        log_config=None,
    )


def main():
    """Start the Pixie server.

    Loads environment variables and starts the server with auto-reload enabled.
    Supports --verbose and --debug flags for enhanced logging.
    """
    parser = argparse.ArgumentParser(description="Pixie Prompts development server")
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging (INFO+ for all modules)",
    )
    parser.add_argument(
        "--debug",
        "-d",
        action="store_true",
        help="Enable debug logging (DEBUG+ for all modules)",
    )
    parser.add_argument(
        "--port",
        "-p",
        type=int,
        default=None,
        help="Port to run the server on (overrides PIXIE_SDK_PORT env var)",
    )
    args = parser.parse_args()

    # Determine logging mode
    log_mode = "default"
    if args.debug:
        log_mode = "debug"
    elif args.verbose:
        log_mode = "verbose"

    dotenv.load_dotenv(os.getcwd() + "/.env")
    port = args.port or int(os.getenv("PIXIE_SDK_PORT", "8000"))

    start_server(port=port, reload=True, log_mode=log_mode)


if __name__ == "__main__":
    main()
