# DuckDuckGo Search MCP Server

[![smithery badge](https://smithery.ai/badge/@nickclyde/duckduckgo-mcp-server)](https://smithery.ai/server/@nickclyde/duckduckgo-mcp-server)

A Model Context Protocol (MCP) server that provides web search capabilities through DuckDuckGo, with additional features for content fetching and parsing. Now with Databricks Apps support!

<a href="https://glama.ai/mcp/servers/phcus2gcpn">
  <img width="380" height="200" src="https://glama.ai/mcp/servers/phcus2gcpn/badge" alt="DuckDuckGo Server MCP server" />
</a>

## Features

- **Web Search**: Search DuckDuckGo with advanced rate limiting and result formatting
- **Content Fetching**: Retrieve and parse webpage content with intelligent text extraction
- **Rate Limiting**: Built-in protection against rate limits for both search and content fetching
- **Error Handling**: Comprehensive error handling and logging
- **LLM-Friendly Output**: Results formatted specifically for large language model consumption
- **Databricks Apps Integration**: Deploy and run on Databricks Apps with streamable HTTP transport

## Installation

### Installing via Smithery

To install DuckDuckGo Search Server for Claude Desktop automatically via [Smithery](https://smithery.ai/server/@nickclyde/duckduckgo-mcp-server):

```bash
npx -y @smithery/cli install @nickclyde/duckduckgo-mcp-server --client claude
```

### Installing via `uv`

Install directly from PyPI using `uv`:

```bash
uv pip install duckduckgo-mcp-server
```

## Usage

### Running with Claude Desktop

1. Download [Claude Desktop](https://claude.ai/download)
2. Create or edit your Claude Desktop configuration:
   - On macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
   - On Windows: `%APPDATA%\Claude\claude_desktop_config.json`

Add the following configuration:

```json
{
    "mcpServers": {
        "ddg-search": {
            "command": "uvx",
            "args": ["duckduckgo-mcp-server"]
        }
    }
}
```

3. Restart Claude Desktop

### Development

For local development, you can use the MCP CLI:

```bash
# Run with the MCP Inspector
mcp dev server.py

# Install locally for testing with Claude Desktop
mcp install server.py
```

### Running on Databricks Apps

This MCP server can be deployed on Databricks Apps for scalable, production-ready deployment.

#### Prerequisites

- Databricks CLI installed and configured
- `uv` package manager

#### Local Development with Databricks Mode

Start the server locally with FastAPI and uvicorn:

```bash
# Install dependencies
uv sync

# Start the server with hot-reload
uvicorn duckduckgo_mcp_server.app:app --reload

# Or use the convenience command
uv run duckduckgo-mcp-server-databricks
```

The server will be available at `http://localhost:8000` with a landing page and the MCP endpoint at `http://localhost:8000/mcp/`.

#### Deploying to Databricks Apps

There are two ways to deploy on Databricks Apps:

##### Using `databricks apps` CLI

1. Configure Databricks authentication:
```bash
export DATABRICKS_CONFIG_PROFILE=<your-profile-name>
databricks auth login --profile "$DATABRICKS_CONFIG_PROFILE"
```

2. Create a Databricks app:
```bash
databricks apps create duckduckgo-mcp-server
```

3. Upload and deploy:
```bash
DATABRICKS_USERNAME=$(databricks current-user me | jq -r .userName)
databricks sync . "/Users/$DATABRICKS_USERNAME/duckduckgo-mcp-server"
databricks apps deploy duckduckgo-mcp-server --source-code-path "/Workspace/Users/$DATABRICKS_USERNAME/duckduckgo-mcp-server"
```

##### Using `databricks bundle` CLI

1. Build the wheel and deploy using bundle:
```bash
uv build --wheel
databricks bundle deploy
databricks bundle run duckduckgo-mcp-server
```

#### Connecting to the Databricks-Deployed Server

Use the Streamable HTTP transport with your app URL:

```python
from databricks.sdk import WorkspaceClient
from databricks_mcp import DatabricksOAuthClientProvider
from mcp.client.streamable_http import streamablehttp_client as connect
from mcp import ClientSession

client = WorkspaceClient()

async def main():
    app_url = "https://your.app.url.databricksapps.com/mcp/"
    async with connect(app_url, auth=DatabricksOAuthClientProvider(client)) as (
        read_stream,
        write_stream,
        _,
    ):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            # Use the search tool
            result = await session.call_tool("search", {"query": "Python programming", "max_results": 5})
            print(result)
```

**Note:** The URL must end with `/mcp/` (including the trailing slash).

## Available Tools

### 1. Search Tool

```python
async def search(query: str, max_results: int = 10) -> str
```

Performs a web search on DuckDuckGo and returns formatted results.

**Parameters:**
- `query`: Search query string
- `max_results`: Maximum number of results to return (default: 10)

**Returns:**
Formatted string containing search results with titles, URLs, and snippets.

### 2. Content Fetching Tool

```python
async def fetch_content(url: str) -> str
```

Fetches and parses content from a webpage.

**Parameters:**
- `url`: The webpage URL to fetch content from

**Returns:**
Cleaned and formatted text content from the webpage.

## Features in Detail

### Rate Limiting

- Search: Limited to 30 requests per minute
- Content Fetching: Limited to 20 requests per minute
- Automatic queue management and wait times

### Result Processing

- Removes ads and irrelevant content
- Cleans up DuckDuckGo redirect URLs
- Formats results for optimal LLM consumption
- Truncates long content appropriately

### Error Handling

- Comprehensive error catching and reporting
- Detailed logging through MCP context
- Graceful degradation on rate limits or timeouts

## Contributing

Issues and pull requests are welcome! Some areas for potential improvement:

- Additional search parameters (region, language, etc.)
- Enhanced content parsing options
- Caching layer for frequently accessed content
- Additional rate limiting strategies

## License

This project is licensed under the MIT License.