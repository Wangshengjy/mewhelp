"""Environment-backed configuration for the customer support service."""

from dataclasses import dataclass
from pathlib import Path
import os

from dotenv import load_dotenv
from sqlalchemy import URL


@dataclass(frozen=True, slots=True)
class Settings:
    chat_base_url: str
    chat_model: str
    chat_api_key: str
    chat_temperature: float = 0.2
    chat_context_token_budget: int = 6000
    chat_token_safety_margin: int = 500
    mysql_host: str = "127.0.0.1"
    mysql_port: int = 3306
    mysql_database: str = "mewhelp"
    mysql_user: str = "mewhelp"
    mysql_password: str = "mewhelp"
    tool_timeout_seconds: int = 10

    @property
    def database_url(self) -> URL:
        """Build a database URL without hand-encoding credentials."""
        return URL.create(
            "mysql+pymysql",
            username=self.mysql_user,
            password=self.mysql_password,
            host=self.mysql_host,
            port=self.mysql_port,
            database=self.mysql_database,
        )

    @classmethod
    def load(cls) -> "Settings":
        """Load settings from mewhelp/.env, allowing process variables to win."""
        env_file = Path(__file__).resolve().parents[1] / ".env"
        load_dotenv(env_file, override=False)

        required = {
            "CHAT_BASE_URL": os.getenv("CHAT_BASE_URL"),
            "CHAT_MODEL": os.getenv("CHAT_MODEL"),
            "CHAT_API_KEY": os.getenv("CHAT_API_KEY"),
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            raise ValueError("Missing required environment variable(s): " + ", ".join(missing))

        try:
            temperature = float(os.getenv("CHAT_TEMPERATURE", "0.2"))
            token_budget = int(os.getenv("CHAT_CONTEXT_TOKEN_BUDGET", "6000"))
            safety_margin = int(os.getenv("CHAT_TOKEN_SAFETY_MARGIN", "500"))
            mysql_port = int(os.getenv("MYSQL_PORT", "3306"))
            tool_timeout_seconds = int(os.getenv("TOOL_TIMEOUT_SECONDS", "10"))
        except ValueError as exc:
            raise ValueError("numeric settings must contain valid numbers") from exc

        if token_budget <= 0:
            raise ValueError("CHAT_CONTEXT_TOKEN_BUDGET must be greater than zero")
        if safety_margin < 0 or safety_margin >= token_budget:
            raise ValueError("CHAT_TOKEN_SAFETY_MARGIN must be non-negative and smaller than the token budget")
        if mysql_port <= 0:
            raise ValueError("MYSQL_PORT must be greater than zero")
        if tool_timeout_seconds <= 0:
            raise ValueError("TOOL_TIMEOUT_SECONDS must be greater than zero")

        return cls(
            chat_base_url=required["CHAT_BASE_URL"].rstrip("/"),
            chat_model=required["CHAT_MODEL"],
            chat_api_key=required["CHAT_API_KEY"],
            chat_temperature=temperature,
            chat_context_token_budget=token_budget,
            chat_token_safety_margin=safety_margin,
            mysql_host=os.getenv("MYSQL_HOST", "127.0.0.1"),
            mysql_port=mysql_port,
            mysql_database=os.getenv("MYSQL_DATABASE", "mewhelp"),
            mysql_user=os.getenv("MYSQL_USER", "mewhelp"),
            mysql_password=os.getenv("MYSQL_PASSWORD", "mewhelp"),
            tool_timeout_seconds=tool_timeout_seconds,
        )
