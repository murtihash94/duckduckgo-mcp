import uvicorn


def main():
    uvicorn.run(
        "duckduckgo_mcp_server.app:app",  # import path to your `app`
        host="0.0.0.0",
        port=8000,
        reload=True,  # optional
    )
