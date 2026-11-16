from importlib import import_module
from pathlib import Path

import pytest
from app import settings as settings_module


def _settings_class():
    settings_path = Path("app/settings.py")
    assert settings_path.is_file(), "app.settings has not been implemented"
    module = import_module("app.settings")
    assert hasattr(module, "Settings"), "Settings has not been implemented"
    return module.Settings


def _set_required_environment(monkeypatch):
    monkeypatch.setenv("CHAT_BASE_URL", "https://provider.example/v1")
    monkeypatch.setenv("CHAT_MODEL", "provider-model")
    monkeypatch.setenv("CHAT_API_KEY", "test-key")


def test_loads_model_settings_from_environment(monkeypatch):
    _set_required_environment(monkeypatch)
    monkeypatch.setenv("CHAT_TEMPERATURE", "0.25")
    monkeypatch.setenv("CHAT_CONTEXT_TOKEN_BUDGET", "5000")
    monkeypatch.setenv("CHAT_TOKEN_SAFETY_MARGIN", "500")

    settings = _settings_class().load()

    assert settings.chat_base_url == "https://provider.example/v1"
    assert settings.chat_model == "provider-model"
    assert settings.chat_api_key == "test-key"
    assert settings.chat_temperature == 0.25
    assert settings.chat_context_token_budget == 5000
    assert settings.chat_token_safety_margin == 500


def test_loads_mysql_defaults_and_tool_timeout(monkeypatch, tmp_path):
    _set_required_environment(monkeypatch)
    for name in (
        "MYSQL_HOST",
        "MYSQL_PORT",
        "MYSQL_DATABASE",
        "MYSQL_USER",
        "MYSQL_PASSWORD",
        "TOOL_TIMEOUT_SECONDS",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(settings_module, "__file__", str(tmp_path / "app" / "settings.py"))

    settings = _settings_class().load()

    assert settings.mysql_host == "127.0.0.1"
    assert settings.mysql_port == 3306
    assert settings.mysql_database == "mewhelp"
    assert settings.mysql_user == "mewhelp"
    assert settings.mysql_password == "mewhelp"
    assert settings.tool_timeout_seconds == 10
    assert settings.database_url.render_as_string(hide_password=False) == (
        "mysql+pymysql://mewhelp:mewhelp@127.0.0.1:3306/mewhelp"
    )


def test_mysql_environment_overrides_are_encoded_in_database_url(monkeypatch):
    _set_required_environment(monkeypatch)
    monkeypatch.setenv("MYSQL_HOST", "mysql.example")
    monkeypatch.setenv("MYSQL_PORT", "3307")
    monkeypatch.setenv("MYSQL_DATABASE", "support_data")
    monkeypatch.setenv("MYSQL_USER", "support@example")
    monkeypatch.setenv("MYSQL_PASSWORD", "p@ss/w:rd")
    monkeypatch.setenv("TOOL_TIMEOUT_SECONDS", "17")

    settings = _settings_class().load()

    assert settings.mysql_host == "mysql.example"
    assert settings.mysql_port == 3307
    assert settings.mysql_database == "support_data"
    assert settings.mysql_user == "support@example"
    assert settings.mysql_password == "p@ss/w:rd"
    assert settings.tool_timeout_seconds == 17
    assert settings.database_url.render_as_string(hide_password=False) == (
        "mysql+pymysql://support%40example:p%40ss%2Fw%3Ard@mysql.example:3307/support_data"
    )


def test_missing_api_key_names_the_missing_variable(monkeypatch, tmp_path):
    _set_required_environment(monkeypatch)
    monkeypatch.delenv("CHAT_API_KEY")
    isolated_settings_file = tmp_path / "app" / "settings.py"
    monkeypatch.setattr(settings_module, "__file__", str(isolated_settings_file))

    with pytest.raises(ValueError, match="CHAT_API_KEY"):
        _settings_class().load()


def test_loads_values_from_the_project_dotenv_file(monkeypatch, tmp_path):
    settings_file = tmp_path / "app" / "settings.py"
    dotenv_file = tmp_path / ".env"
    dotenv_file.write_text(
        "CHAT_BASE_URL=https://dotenv.example/v1\n"
        "CHAT_MODEL=dotenv-model\n"
        "CHAT_API_KEY=dotenv-test-key\n"
        "CHAT_TEMPERATURE=0.35\n"
        "CHAT_CONTEXT_TOKEN_BUDGET=4000\n"
        "CHAT_TOKEN_SAFETY_MARGIN=300\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(settings_module, "__file__", str(settings_file))
    names = [
        "CHAT_BASE_URL",
        "CHAT_MODEL",
        "CHAT_API_KEY",
        "CHAT_TEMPERATURE",
        "CHAT_CONTEXT_TOKEN_BUDGET",
        "CHAT_TOKEN_SAFETY_MARGIN",
    ]
    for name in names:
        monkeypatch.delenv(name, raising=False)

    try:
        settings = settings_module.Settings.load()
    finally:
        for name in names:
            monkeypatch.delenv(name, raising=False)

    assert settings.chat_base_url == "https://dotenv.example/v1"
    assert settings.chat_model == "dotenv-model"
    assert settings.chat_api_key == "dotenv-test-key"
    assert settings.chat_temperature == 0.35
    assert settings.chat_context_token_budget == 4000
    assert settings.chat_token_safety_margin == 300
