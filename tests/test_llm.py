import pytest

from src.llm import friendly_api_error


class RateLimitError(Exception):
    pass


@pytest.mark.parametrize("message, expected, fatal", [
    # Real OpenAI message from the first live run: a 429 that is really an empty account.
    ("Error code: 429 - You have no credits remaining. Add credits to continue using the API at "
     "https://platform.openai.com/settings/organization/billing/.", "no remaining quota or credit", True),
    ("Error code: 429 - {'error': {'code': 'insufficient_quota'}}", "no remaining quota or credit", True),
    ("Error code: 429 - Rate limit reached for requests", "rate limit reached", False),
])
def test_429_errors_are_told_apart(message, expected, fatal):
    text, is_fatal = friendly_api_error(RateLimitError(message), "OpenAI")
    assert expected in text and is_fatal is fatal
