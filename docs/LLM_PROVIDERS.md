# LLM provider configuration

The RAG agent sends standard `system` and `user` messages to an HTTPS Chat
Completions API. Provider selection belongs to each `task_<n>:agent` object, so
retrieval and reranking experiments remain independent of a particular vendor.

## DeepSeek

The committed experiments use this provider configuration:

```json
{
  "type": "rag",
  "rag:provider": "deepseek",
  "rag:baseUrl": "https://api.deepseek.com",
  "rag:model": "deepseek-v4-flash",
  "rag:apiKeyEnv": "DEEPSEEK_API_KEY",
  "rag:timeoutSeconds": 120,
  "rag:maxRetries": 2,
  "rag:maxTokens": 64,
  "rag:temperature": 0,
  "rag:thinking": false,
  "rag:fallback": false
}
```

Export the credential in the same shell that launches QryEval_PLUS:

```sh
export DEEPSEEK_API_KEY="your-api-key"
qryeval doctor --config configs/examples/deepseek_grounded_rag.json
qryeval demo --config configs/examples/deepseek_grounded_rag.json --questions 1
```

`doctor` checks that the referenced environment variable exists, but it does
not send a paid model request. The one-question demo is the smallest real API
validation. The answer record in the returned Python batch includes provider,
model, request identifier, latency, token usage, success, and fallback status.
The configured `metadataPath` persists the same non-secret status as a
`.llm.json` sidecar without changing the TriviaQA prediction format.

Model identifiers are external service state. Check the provider's model list
before a long experiment and update `rag:model` when necessary.

## Other OpenAI-compatible APIs

Use `openai-compatible` for a hosted service, vLLM, LM Studio, or an OpenAI-
compatible Ollama endpoint:

```json
{
  "rag:provider": "openai-compatible",
  "rag:baseUrl": "https://provider.example/v1",
  "rag:apiPath": "/chat/completions",
  "rag:model": "provider-model-id",
  "rag:apiKeyEnv": "PROVIDER_API_KEY",
  "rag:apiKeyRequired": true,
  "rag:timeoutSeconds": 120,
  "rag:maxRetries": 2,
  "rag:maxTokens": 64,
  "rag:temperature": 0
}
```

For a local endpoint that does not require authentication, set
`rag:apiKeyRequired` to `false`. Keep it `true` for remote providers. Prefer
HTTPS for every non-local endpoint.

## Failure semantics

`rag:fallback` defaults to `false`. A missing credential, timeout, HTTP error,
malformed response, or empty answer therefore fails the pipeline visibly. This
is the required setting for quality evaluation because it prevents an API
outage from being counted as a model answer.

Setting `rag:fallback` to `true` enables the deterministic extractive fallback.
Use it only for availability demonstrations, and inspect `llm.success` and
`llm.fallback_used` before interpreting answer metrics.
