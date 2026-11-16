"""LangChain chain for structured after-sales detail extraction."""

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable

from app.prompts import extraction_prompt
from app.schemas import AfterSalesExtraction


def build_extraction_chain(model: BaseChatModel) -> Runnable:
    """Combine the approved extraction prompt and model structured-output parser."""
    return extraction_prompt | model.with_structured_output(
        AfterSalesExtraction,
        method="function_calling",
    )
