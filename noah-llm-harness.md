# How the Noah LLM Harness Works

A visual explanation of the harness in `irl-harness` (`src/Noah.Core`, `src/Noah.LLM`),
covering three things:

1. **The harness loop** — text in, text out, with tools in between.
2. **Prompt assembly** — what actually gets sent to the model each call, and how
   each prompt is the previous prompt plus whatever was appended.
3. **Per-token generation on the GPU** — how the prompt populates VRAM, flows
   through the transformer for every token, and how the KV cache grows.

> **Two loops, don't conflate them.** The *outer* loop is the harness: it makes
> one API call per model turn and resends the whole prompt each time. The *inner*
> loop is the provider's GPU: it runs one forward pass per output token and only
> feeds the newest token, because everything before it is held in the KV cache.

---

## 1. The harness loop — text in, text out

The harness is a thin streaming wrapper. The model only ever sees text (and tool
schemas); it only ever produces text (and tool-call requests). Everything else —
files, shells, compilation — happens because a tool result is turned back into
text and appended to the next prompt.

```mermaid
flowchart TD
    U["User text<br/>TUI · JSON · RPC · SDK"] --> S["Agent.SendAsync<br/>new user message"]
    S --> H["history = SessionTree.Linearize<br/>+ new user message"]
    H --> C{"estimated tokens<br/>&gt; context window?"}
    C -->|yes| CMP["Compactor<br/>summarise oldest half, else truncate"]
    CMP --> P
    C -->|no| P["ContextEngine.PrepareAsync<br/>system layer + transformed history"]
    P --> R["Build ChatRequest<br/>messages + tool schemas + options"]
    R --> L["ILLMClient.CompleteAsync<br/>POST to provider, read SSE stream"]
    L --> D["onDelta stream<br/>content · reasoning · tool-call deltas"]
    D --> O{"response contains<br/>tool calls?"}
    O -->|no| F["Final assistant text<br/>AgentResponse + persist turn to tree"]
    O -->|yes| T["tool_call hook → registry lookup<br/>ExecuteAsync → tool_result hook"]
    T --> A["append assistant message<br/>+ tool result to history"]
    A --> C
    F --> OUT["Text shown to user"]
```

Key properties (see `src/Noah.Core/Agent/Agent.cs`):

- The loop repeats until the model answers **without** tool calls, bounded by
  `MaxIterations` (`Agent.cs:261`).
- Tools **never throw to the model** — failures become readable `ToolResult`
  text the model can recover from (`Agent.cs:429-438`).
- Output is streamed token-by-token through the single `onDelta` boundary
  (`src/Noah.Sdk/LLM/ILLMClient.cs:31`); providers only translate wire formats.

---

## 2. Prompt assembly — each prompt is the last one plus new messages

Every API call rebuilds the same layered system prefix and appends the growing
conversation. The prefix is deterministic, so providers can cache it (see
`ARCHITECTURE.md:120`).

```mermaid
flowchart TB
    subgraph SYS["System layer — rebuilt every iteration, deterministic prefix"]
        B["1 · baseline system prompt<br/>or SYSTEM.md / --system-prompt"]
        E["1b · environment<br/>working directory"]
        G["2 · project instructions<br/>nearest AGENTS.md"]
        K["3 · skills catalog<br/>name + description only, loaded on demand"]
    end
    subgraph HIST["Conversation history — append-only"]
        M1["user message"]
        M2["assistant message<br/>text and/or tool_calls"]
        M3["tool result message"]
    end
    TS["Tool schemas<br/>registered ITool definitions"]
    SYS --> PREP["ContextEngine.PrepareAsync"]
    HIST --> PREP
    PREP --> REQ["ChatRequest"]
    TS --> REQ
    REQ --> WIRE["Provider wire format<br/>one leading system block<br/>+ messages array + tools array"]
    WIRE --> API["LLM API call"]
```

What "each subsequent prompt is a copy of the last with the new prompt appended"
looks like in practice (`ContextEngine.cs:89`, `ChatCompletionsClient.cs:62`):

```mermaid
sequenceDiagram
    autonumber
    participant H as Harness
    participant P as Provider / LLM
    Note over H: Same system layer every call — cacheable prefix
    H->>P: Call 1 — [system, user1]
    P-->>H: assistant1 (with a tool_call)
    Note over H: append assistant1 + tool result to history
    H->>P: Call 2 — [system, user1, assistant1, tool1]
    P-->>H: assistant2 (final text)
    Note over H: next user turn appends again — history is never rewritten
    H->>P: Call 3 — [system, user1, assistant1, tool1, assistant2, user2]
```

- History is only ever **appended to**; the model is stateless between calls.
- When history nears the window, a compactor replaces the **oldest** messages
  with a summary (`ContextEngine.cs:106`), which is the one time the prefix
  changes.
- The harness estimates tokens cheaply as `chars / 4` plus per-message overhead
  (`TokenEstimator.cs:11`).

---

## 3. Per-token generation on the GPU

Once the prompt reaches the provider, the GPU generates one token at a time.
Each token is one full pass through the network; the sequence on the GPU grows
by exactly one each step.

### 3a. One forward pass = one token

```mermaid
flowchart LR
    P["Current sequence<br/>prompt + tokens so far"] --> TOK["Tokenizer<br/>text → token IDs"]
    TOK --> EMB["Embedding lookup<br/>token ID → vector of size d_model"]
    EMB --> BLK["Transformer blocks × L<br/>self-attention + MLP"]
    BLK --> NORM["Final layer norm"]
    NORM --> LOG["Logits<br/>one score per vocabulary entry"]
    LOG --> SAMP["Sampler<br/>temperature / top-p"]
    SAMP --> NEXT["One new token"]
    NEXT --> APP["Append to sequence"]
    APP -. "decode step: only the new token is fed" .-> EMB
```

### 3b. What lives in GPU RAM

```mermaid
flowchart TB
    subgraph VRAM["GPU VRAM during generation"]
        W["Model weights<br/>loaded once, frozen<br/>size fixed by parameter count"]
        KV["KV cache<br/>one key + value vector per token, per layer<br/>grows with every generated token"]
        ACT["Activations / scratch<br/>current token only"]
        LB["Logits buffer<br/>one entry per vocabulary item"]
    end
    W --> PASS["Forward pass"]
    KV --> PASS
    ACT --> PASS
    PASS --> LB
    LB --> NEW["sampled token"]
    NEW -->|"append its K and V for every layer"| KV
```

- **Prefill** (first pass) processes the whole prompt at once and fills the KV
  cache with one entry per prompt token.
- **Decode** (every token after) feeds only the newest token; attention reads
  the cached K/V of all earlier tokens instead of recomputing them.
- That is why cost per token is roughly flat while the KV cache — and therefore
  VRAM — grows linearly with sequence length.

### 3c. The sequence grows by one token per step

| Step | What is fed to the GPU this pass | Sequence length | KV-cache rows | Sampled |
| --- | --- | --- | --- | --- |
| Prefill | all prompt tokens at once | N | N | token 1 |
| Decode 1 | token 1 only | N+1 | N+1 | token 2 |
| Decode 2 | token 2 only | N+2 | N+2 | token 3 |
| Decode 3 | token 3 only | N+3 | N+3 | token 4 |

### 3d. How this relates to the harness's growing prompt

The harness sends a **new, longer prompt on each API call** (Diagram 2). The
provider does not recompute the shared prefix from scratch: because the prefix
is byte-identical, its KV cache is reused and only the newly appended suffix is
processed (prompt/prefix caching). So the two "copy and append" behaviours line
up:

- Harness: `prompt(n+1) = prompt(n) + new messages`.
- GPU: `sequence(n+1) = sequence(n) + one token`, with the old prefix's K/V
  reused.

---

## Source map

| Concept | Location |
| --- | --- |
| Agent loop, iteration cap, tool execution, append | `src/Noah.Core/Agent/Agent.cs:242`, `:261`, `:391`, `:446` |
| Streaming boundary / `ChatRequest` | `src/Noah.Sdk/LLM/ILLMClient.cs:6`, `:31` |
| Layered system prompt | `src/Noah.Core/Context/ContextEngine.cs:48` |
| Context assembly + compaction | `src/Noah.Core/Context/ContextEngine.cs:89`, `:106` |
| Token estimate (`chars / 4`) | `src/Noah.Core/Context/TokenEstimator.cs:11` |
| Provider wire format + SSE parse | `src/Noah.LLM/Providers/ChatCompletionsClient.cs:62`, `:159` |
| Provider-neutral message model | `src/Noah.Sdk/Models/ChatMessage.cs:21` |
| Prompt caching rationale | `ARCHITECTURE.md:120` |
