from mcp.server.fastmcp import FastMCP, Context
import httpx
from bs4 import BeautifulSoup
from typing import List, Dict, Optional, Any, Union
from dataclasses import dataclass
import urllib.parse
import sys
import traceback
import asyncio
from datetime import datetime, timedelta
import time
import re
from dotenv import load_dotenv
import os
import base64
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import aiosmtplib
from pydantic import BaseModel, EmailStr, Field
import logging

# Load environment variables and configure logging
load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    title: str
    link: str
    snippet: str
    position: int


# === EMAIL DATA MODELS ===

class SMTPConfig(BaseModel):
    """SMTP server configuration."""
    host: str = Field(..., description="SMTP server hostname")
    port: int = Field(587, description="SMTP server port")
    secure: bool = Field(False, description="Use SSL/TLS (True for port 465, False for port 587)")
    auth: Dict[str, str] = Field(..., description="Authentication credentials")
    from_email: EmailStr = Field(..., description="Sender email address")

class EmailAttachment(BaseModel):
    """Email attachment model."""
    filename: str = Field(..., description="Attachment filename")
    content: str = Field(..., description="Base64 encoded attachment content")
    encoding: str = Field("base64", description="Content encoding")

class EmailMessage(BaseModel):
    """Email message model."""
    to: Union[EmailStr, List[EmailStr]] = Field(..., description="Recipient email address(es)")
    cc: Optional[Union[EmailStr, List[EmailStr]]] = Field(None, description="CC email address(es)")
    bcc: Optional[Union[EmailStr, List[EmailStr]]] = Field(None, description="BCC email address(es)")
    subject: str = Field(..., description="Email subject")
    text: Optional[str] = Field(None, description="Plain text email body")
    html: Optional[str] = Field(None, description="HTML email body")
    attachments: Optional[List[EmailAttachment]] = Field(None, description="Email attachments")
    
    class Config:
        populate_by_name = True


class RateLimiter:
    def __init__(self, requests_per_minute: int = 30):
        self.requests_per_minute = requests_per_minute
        self.requests = []

    async def acquire(self):
        now = datetime.now()
        # Remove requests older than 1 minute
        self.requests = [
            req for req in self.requests if now - req < timedelta(minutes=1)
        ]

        if len(self.requests) >= self.requests_per_minute:
            # Wait until we can make another request
            wait_time = 60 - (now - self.requests[0]).total_seconds()
            if wait_time > 0:
                await asyncio.sleep(wait_time)

        self.requests.append(now)


class DuckDuckGoSearcher:
    BASE_URL = "https://html.duckduckgo.com/html"
    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    def __init__(self):
        self.rate_limiter = RateLimiter()

    def format_results_for_llm(self, results: List[SearchResult]) -> str:
        """Format results in a natural language style that's easier for LLMs to process"""
        if not results:
            return "No results were found for your search query. This could be due to DuckDuckGo's bot detection or the query returned no matches. Please try rephrasing your search or try again in a few minutes."

        output = []
        output.append(f"Found {len(results)} search results:\n")

        for result in results:
            output.append(f"{result.position}. {result.title}")
            output.append(f"   URL: {result.link}")
            output.append(f"   Summary: {result.snippet}")
            output.append("")  # Empty line between results

        return "\n".join(output)

    async def search(
        self, query: str, ctx: Context, max_results: int = 10
    ) -> List[SearchResult]:
        try:
            # Apply rate limiting
            await self.rate_limiter.acquire()

            # Create form data for POST request
            data = {
                "q": query,
                "b": "",
                "kl": "",
            }

            await ctx.info(f"Searching DuckDuckGo for: {query}")

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.BASE_URL, data=data, headers=self.HEADERS, timeout=30.0
                )
                response.raise_for_status()

            # Parse HTML response
            soup = BeautifulSoup(response.text, "html.parser")
            if not soup:
                await ctx.error("Failed to parse HTML response")
                return []

            results = []
            for result in soup.select(".result"):
                title_elem = result.select_one(".result__title")
                if not title_elem:
                    continue

                link_elem = title_elem.find("a")
                if not link_elem:
                    continue

                title = link_elem.get_text(strip=True)
                link = link_elem.get("href", "")

                # Skip ad results
                if "y.js" in link:
                    continue

                # Clean up DuckDuckGo redirect URLs
                if link.startswith("//duckduckgo.com/l/?uddg="):
                    link = urllib.parse.unquote(link.split("uddg=")[1].split("&")[0])

                snippet_elem = result.select_one(".result__snippet")
                snippet = snippet_elem.get_text(strip=True) if snippet_elem else ""

                results.append(
                    SearchResult(
                        title=title,
                        link=link,
                        snippet=snippet,
                        position=len(results) + 1,
                    )
                )

                if len(results) >= max_results:
                    break

            await ctx.info(f"Successfully found {len(results)} results")
            return results

        except httpx.TimeoutException:
            await ctx.error("Search request timed out")
            return []
        except httpx.HTTPError as e:
            await ctx.error(f"HTTP error occurred: {str(e)}")
            return []
        except Exception as e:
            await ctx.error(f"Unexpected error during search: {str(e)}")
            traceback.print_exc(file=sys.stderr)
            return []


class WebContentFetcher:
    def __init__(self):
        self.rate_limiter = RateLimiter(requests_per_minute=20)

    async def fetch_and_parse(self, url: str, ctx: Context) -> str:
        """Fetch and parse content from a webpage"""
        try:
            await self.rate_limiter.acquire()

            await ctx.info(f"Fetching content from: {url}")

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    url,
                    headers={
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                    },
                    follow_redirects=True,
                    timeout=30.0,
                )
                response.raise_for_status()

            # Parse the HTML
            soup = BeautifulSoup(response.text, "html.parser")

            # Remove script and style elements
            for element in soup(["script", "style", "nav", "header", "footer"]):
                element.decompose()

            # Get the text content
            text = soup.get_text()

            # Clean up the text
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = " ".join(chunk for chunk in chunks if chunk)

            # Remove extra whitespace
            text = re.sub(r"\s+", " ", text).strip()

            # Truncate if too long
            if len(text) > 8000:
                text = text[:8000] + "... [content truncated]"

            await ctx.info(
                f"Successfully fetched and parsed content ({len(text)} characters)"
            )
            return text

        except httpx.TimeoutException:
            await ctx.error(f"Request timed out for URL: {url}")
            return "Error: The request timed out while trying to fetch the webpage."
        except httpx.HTTPError as e:
            await ctx.error(f"HTTP error occurred while fetching {url}: {str(e)}")
            return f"Error: Could not access the webpage ({str(e)})"
        except Exception as e:
            await ctx.error(f"Error fetching content from {url}: {str(e)}")
            return f"Error: An unexpected error occurred while fetching the webpage ({str(e)})"


# === EMAIL CONFIGURATION SERVICE ===

def get_smtp_config_from_env() -> Optional[SMTPConfig]:
    """Get SMTP configuration from environment variables."""
    host = os.getenv('SMTP_HOST')
    user = os.getenv('SMTP_USER')
    from_email = os.getenv('SMTP_FROM')
    password = os.getenv('SMTP_PASS')
    
    if not all([host, user, from_email, password]):
        return None
    
    port = int(os.getenv('SMTP_PORT', '587'))
    secure = os.getenv('SMTP_SECURE', 'false').lower() == 'true'
    
    return SMTPConfig(
        host=host,
        port=port,
        secure=secure,
        auth={
            'user': user,
            'pass': password
        },
        from_email=from_email
    )

# === EMAIL SERVICE ===

async def create_email_message(smtp_config: SMTPConfig, email_message: EmailMessage) -> tuple:
    """Create a MIME email message from the email data."""
    msg = MIMEMultipart()
    msg['From'] = smtp_config.from_email
    msg['Subject'] = email_message.subject
    
    # Handle recipients
    recipients = []
    
    # TO recipients
    to_list = email_message.to if isinstance(email_message.to, list) else [email_message.to]
    msg['To'] = ', '.join(str(email) for email in to_list)
    recipients.extend(str(email) for email in to_list)
    
    # CC recipients
    if email_message.cc:
        cc_list = email_message.cc if isinstance(email_message.cc, list) else [email_message.cc]
        msg['Cc'] = ', '.join(str(email) for email in cc_list)
        recipients.extend(str(email) for email in cc_list)
    
    # BCC recipients (not added to headers)
    if email_message.bcc:
        bcc_list = email_message.bcc if isinstance(email_message.bcc, list) else [email_message.bcc]
        recipients.extend(str(email) for email in bcc_list)
    
    # Add body content
    if email_message.text:
        msg.attach(MIMEText(email_message.text, 'plain'))
    if email_message.html:
        msg.attach(MIMEText(email_message.html, 'html'))
    
    # Add attachments
    if email_message.attachments:
        for attachment in email_message.attachments:
            part = MIMEBase('application', 'octet-stream')
            
            # Decode base64 content
            content = base64.b64decode(attachment.content)
            part.set_payload(content)
            encoders.encode_base64(part)
            
            part.add_header(
                'Content-Disposition',
                f'attachment; filename= {attachment.filename}'
            )
            msg.attach(part)
    
    return msg, recipients

async def send_with_connection_fallback(smtp_config: SMTPConfig, msg, recipients):
    """Send email with automatic connection method fallback."""
    connection_methods = []
    
    # Determine connection methods based on configuration
    if smtp_config.port == 465 or smtp_config.secure:
        connection_methods.append(("implicit TLS", True, False))
    else:
        connection_methods.append(("STARTTLS", False, True))
        connection_methods.append(("implicit TLS fallback", True, False))
    
    # Add plain connection as last resort
    connection_methods.append(("plain connection", False, False))
    
    last_error = None
    
    for method_name, use_tls, start_tls in connection_methods:
        try:
            logger.info(f"Trying to send email using: {method_name}")
            
            if use_tls and not start_tls:
                # Implicit TLS
                await aiosmtplib.send(
                    msg,
                    hostname=smtp_config.host,
                    port=smtp_config.port,
                    username=smtp_config.auth['user'],
                    password=smtp_config.auth['pass'],
                    use_tls=True,
                    tls_context=ssl.create_default_context()
                )
            elif start_tls and not use_tls:
                # STARTTLS
                await aiosmtplib.send(
                    msg,
                    hostname=smtp_config.host,
                    port=smtp_config.port,
                    username=smtp_config.auth['user'],
                    password=smtp_config.auth['pass'],
                    start_tls=True,
                    tls_context=ssl.create_default_context()
                )
            else:
                # Plain connection
                await aiosmtplib.send(
                    msg,
                    hostname=smtp_config.host,
                    port=smtp_config.port,
                    username=smtp_config.auth['user'],
                    password=smtp_config.auth['pass']
                )
            
            logger.info(f"Email sent successfully using {method_name}")
            return
            
        except Exception as e:
            last_error = e
            logger.warning(f"Send method '{method_name}' failed: {e}")
            continue
    
    # If all methods failed, raise the last error
    raise Exception(f"All send methods failed. Last error: {last_error}")

async def send_email_smtp(smtp_config: SMTPConfig, email_message: EmailMessage) -> str:
    """Send an email using the provided SMTP configuration."""
    try:
        # Create email message
        msg, recipients = await create_email_message(smtp_config, email_message)
        
        # Send email with connection fallback
        await send_with_connection_fallback(smtp_config, msg, recipients)
        
        return f"Email sent successfully! Recipients: {', '.join(recipients)}"
        
    except Exception as e:
        raise Exception(f"Failed to send email: {str(e)}")

async def test_smtp_connection(smtp_config: SMTPConfig) -> str:
    """Test SMTP connection with provided configuration."""
    connection_methods = []
    
    try:
        # Method 1: Try based on port and secure setting
        if smtp_config.port == 465 or smtp_config.secure:
            connection_methods.append(("implicit TLS", True, False))
        else:
            connection_methods.append(("STARTTLS", False, True))
            connection_methods.append(("implicit TLS fallback", True, False))
        
        # Method 2: Always add plain connection as last resort (for testing)
        connection_methods.append(("plain connection", False, False))
        
        last_error = None
        
        for method_name, use_tls, start_tls in connection_methods:
            try:
                logger.info(f"Trying connection method: {method_name}")
                
                server = aiosmtplib.SMTP(
                    hostname=smtp_config.host,
                    port=smtp_config.port,
                    use_tls=use_tls,
                    tls_context=ssl.create_default_context() if use_tls else None
                )
                
                await server.connect()
                
                if start_tls and not use_tls:
                    await server.starttls(tls_context=ssl.create_default_context())
                
                await server.login(smtp_config.auth['user'], smtp_config.auth['pass'])
                await server.quit()
                
                return f"SMTP connection successful to {smtp_config.host}:{smtp_config.port} with user {smtp_config.auth['user']} ({method_name})"
                
            except Exception as e:
                last_error = e
                logger.warning(f"Connection method '{method_name}' failed: {e}")
                continue
        
        # If all methods failed, raise the last error
        raise Exception(f"All connection methods failed. Last error: {last_error}")
        
    except Exception as e:
        raise Exception(f"SMTP connection failed: {str(e)}")


# Initialize FastMCP server
mcp = FastMCP("DuckDuckGo Search MCP Server on Databricks Apps")
searcher = DuckDuckGoSearcher()
fetcher = WebContentFetcher()


@mcp.tool()
async def search(query: str, ctx: Context, max_results: int = 10) -> str:
    """
    Search DuckDuckGo and return formatted results.

    Args:
        query: The search query string
        max_results: Maximum number of results to return (default: 10)
        ctx: MCP context for logging
    """
    try:
        results = await searcher.search(query, ctx, max_results)
        return searcher.format_results_for_llm(results)
    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        return f"An error occurred while searching: {str(e)}"


@mcp.tool()
async def fetch_content(url: str, ctx: Context) -> str:
    """
    Fetch and parse content from a webpage URL.

    Args:
        url: The webpage URL to fetch content from
        ctx: MCP context for logging
    """
    return await fetcher.fetch_and_parse(url, ctx)


@mcp.tool()
async def send_email(
    to: str,
    subject: str,
    body: str,
    is_html: bool = False
) -> str:
    """
    Send a simple email using SMTP configuration from environment variables.

    Args:
        to: Recipient email address
        subject: Email subject
        body: Email body content
        is_html: Whether body is HTML (default: False)

    Returns:
        Success message or error message
    """
    try:
        # Get SMTP config from environment
        smtp_cfg = get_smtp_config_from_env()
        if not smtp_cfg:
            return "Error: No SMTP configuration found in environment variables. Please set SMTP_HOST, SMTP_PORT, SMTP_SECURE, SMTP_USER, SMTP_PASS, SMTP_FROM."
        
        # Create email message
        email_data = {
            "to": to,
            "subject": subject
        }
        
        if is_html:
            email_data["html"] = body
        else:
            email_data["text"] = body
        
        # Parse email message
        email_msg = EmailMessage(**email_data)
        
        # Send email
        result = await send_email_smtp(smtp_cfg, email_msg)
        return result
        
    except Exception as e:
        error_message = str(e)
        logger.error(f"Error sending email: {error_message}")
        return f"Error sending email: {error_message}"


@mcp.tool()
async def send_custom_email(
    email: Dict[str, Any],
    smtp_config: Optional[Dict[str, Any]] = None
) -> str:
    """
    Send a custom email with full configuration options.
    
    Args:
        email: Email message details including:
            - to: Recipient email address(es) (string or list)
            - cc: CC email address(es) (optional, string or list)
            - bcc: BCC email address(es) (optional, string or list)
            - subject: Email subject
            - text: Plain text email body (optional)
            - html: HTML email body (optional)
            - attachments: Email attachments (optional, list of dicts with filename, content, encoding)
        smtp_config: SMTP server configuration (optional, uses environment if not provided):
            - host: SMTP server host (e.g., smtp.gmail.com)
            - port: SMTP server port (e.g., 587)
            - secure: Use SSL/TLS (true for port 465, false for port 587)
            - auth: {user: "email", pass: "password"}
            - from_email: Sender email address
    
    Returns:
        Success message with details or error message
    """
    try:
        # Use provided SMTP config or fall back to environment config
        if smtp_config:
            smtp_cfg = SMTPConfig(**smtp_config)
        else:
            smtp_cfg = get_smtp_config_from_env()
            if not smtp_cfg:
                return "Error: No SMTP configuration found. Either provide smtp_config in the request or set SMTP environment variables (SMTP_HOST, SMTP_PORT, SMTP_SECURE, SMTP_USER, SMTP_PASS, SMTP_FROM)."
        
        # Parse email message
        email_msg = EmailMessage(**email)
        
        # Send email
        result = await send_email_smtp(smtp_cfg, email_msg)
        return result
        
    except Exception as e:
        error_message = str(e)
        logger.error(f"Error sending custom email: {error_message}")
        return f"Error sending custom email: {error_message}"


@mcp.tool()
async def test_smtp_connection_tool() -> str:
    """
    Test SMTP connection using configuration from environment variables.
    
    Returns:
        Connection test result or error message
    """
    try:
        smtp_cfg = get_smtp_config_from_env()
        if not smtp_cfg:
            return "Error: No SMTP configuration found in environment variables. Please set SMTP_HOST, SMTP_PORT, SMTP_SECURE, SMTP_USER, SMTP_PASS, SMTP_FROM."
        
        # Test connection
        result = await test_smtp_connection(smtp_cfg)
        return result
        
    except Exception as e:
        error_message = str(e)
        logger.error(f"Connection test failed: {error_message}")
        return f"Connection test failed: {error_message}"


def main():
    mcp.run()


if __name__ == "__main__":
    main()
