# Inkling streaming tool-call salvage mod

Recovers tool calls that **leak as message content** in Inkling's streaming path.

## The bug

Deterministic trigger: **multi-turn + streaming + the assistant turn starting
directly with a tool call** (no preamble text). vLLM's streaming
DelegatingParser drops the content-kind marker that begins a turn-initial tool
call, so the whole block surfaces as content with `tool_calls` empty:

```text
<|content_invoke_tool_json|>{"name":"search_notes","args":{"query":"…"}}<|end_message|>
```

Single-turn streaming and multi-turn **non**-streaming both parse correctly, and
it is independent of `reasoning_effort` (leaks even with no reasoning emitted).
The parser's own `token_id_terminals={}` workaround (see the comment in
`vllm/parser/inkling.py`) does not cover this case. Because agent UIs like
OpenWebUI stream and are inherently multi-turn (tool → result → next call), they
hit it constantly. Reported upstream.

## What it does

The chat-completion server streams via `DelegatingParser.parse_delta` (in
`vllm/parser/abstract_parser.py`) — the legacy `ToolParser.extract_tool_calls_streaming`
interface is **not** called on this path. So this mod monkeypatches
`DelegatingParser.parse_delta`, wrapping the streamed `DeltaMessage`:

- Buffers streamed **content** and scans for complete
  `<|content_invoke_tool_json|>{json}<|end_message|>` blocks.
- Each complete block is parsed (with HTML-entity unescape, so it also covers
  the `&quot;` artifact) and re-emitted as a `tool_call` delta.
- Markers split across streamed deltas are held until complete; a leading
  `<|message_model|>name` slot is dropped (the tool name is taken from the JSON);
  stray framing markers are stripped from surfaced content.
- Already-parsed `tool_calls` and `reasoning` pass through untouched; an
  unterminated block at stream end is surfaced **raw** so nothing is silently
  dropped; a runaway buffer (>16 KB) is flushed raw as a safety valve.

Patching the shared `DelegatingParser` is safe for every other model because the
salvage only fires on content containing `<|content_invoke_tool_json|>`, which no
other model emits — the marker string is the guard. Idempotent; refuses to patch
if `DelegatingParser` is absent.

## Scope and limitations

- Streaming only (the non-streaming path already parses these correctly).
- The recovered tool call carries a synthetic id (`call_inkling_salvage_N`).
- Because the salvage runs inside `parse_delta`, the server's `tools_streamed`
  bookkeeping sees the recovered `tool_calls` and sets `finish_reason=tool_calls`
  (verified), so `finish_reason`-gating clients work too.
- Band-aid. Remove once vLLM fixes the streaming reasoning→tool handoff.

## Usage

Applied by `recipes/inkling-small-nvfp4-tuned.yaml`, or manually:

```bash
./launch-cluster.sh --apply-mod mods/inkling-tool-salvage exec vllm serve /model ...
```

Requires an Inkling-capable image (`vllm-node-inkling`).
