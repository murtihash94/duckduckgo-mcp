from pathlib import Path
from duckduckgo_mcp_server.server import mcp
from fastapi import FastAPI
from fastapi.responses import FileResponse

STATIC_DIR = Path(__file__).parent / "static"

# Create the streamable HTTP app from the MCP server
mcp_app = mcp.streamable_http_app()

# Create FastAPI app with lifespan management
app = FastAPI(
    lifespan=lambda _: mcp.session_manager.run(),
)


@app.get("/", include_in_schema=False)
async def serve_index():
    return FileResponse(STATIC_DIR / "index.html")


# Mount the MCP app to handle MCP requests
app.mount("/", mcp_app)
