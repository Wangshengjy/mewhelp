import pytest
from fastapi.testclient import TestClient
from langchain_core.runnables import RunnableLambda

from app.main import get_chat_model
from app.schemas import AfterSalesExtraction


class FakeStructuredModel:
    def __init__(self, result=None, error: Exception | None = None):
        self.result = result
        self.error = error
        self.schema = None
        self.options = None

    def with_structured_output(self, schema, **kwargs):
        self.schema = schema
        self.options = kwargs

        async def invoke(_input):
            if self.error is not None:
                raise self.error
            return self.result

        return RunnableLambda(invoke)


@pytest.fixture
def extraction_client(test_app):
    result = AfterSalesExtraction(
        order_number=None,
        request_type="quality_issue",
        expected_solution=None,
    )
    model = FakeStructuredModel(result=result)
    test_app.dependency_overrides[get_chat_model] = lambda: model
    try:
        with TestClient(test_app) as client:
            yield client, model
    finally:
        test_app.dependency_overrides.clear()


def test_extraction_returns_all_fields_including_nulls(extraction_client):
    client, model = extraction_client

    response = client.post(
        "/api/after-sales/extract",
        json={"description": "商品有划痕，我该怎么办？"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "order_number": None,
        "request_type": "quality_issue",
        "expected_solution": None,
    }
    assert model.schema is AfterSalesExtraction
    assert model.options == {"method": "function_calling"}


def test_extraction_failure_is_sanitized(test_app):
    model = FakeStructuredModel(error=RuntimeError("provider secret=do-not-return"))
    test_app.dependency_overrides[get_chat_model] = lambda: model
    try:
        with TestClient(test_app) as client:
            response = client.post(
                "/api/after-sales/extract",
                json={"description": "订单 123 的商品坏了"},
            )
    finally:
        test_app.dependency_overrides.clear()

    assert response.status_code == 502
    assert "provider secret" not in response.text
    assert "do-not-return" not in response.text
