from retrieval.openalex_api import OpenAlexAPI


class StubClient:
    def __init__(self):
        self.calls = []

    def get_json(self, url, *, params=None, headers=None, cache_namespace=None):
        self.calls.append({
            "url": url,
            "params": params,
            "headers": headers,
            "cache_namespace": cache_namespace,
        })
        return {"results": []}


def test_openalex_api_uses_query_param_for_api_key(monkeypatch):
    monkeypatch.setenv("OPENALEX_API_KEY", "test-key")
    client = StubClient()

    OpenAlexAPI(client=client).search("er stress", limit=3)

    assert client.calls
    assert client.calls[0]["params"]["api_key"] == "test-key"
    assert client.calls[0]["headers"] is None
