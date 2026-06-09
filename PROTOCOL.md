# Wire protocol

Subprotocol: `tts.v1`. Negotiate via the `Sec-WebSocket-Protocol` header.

## Connection lifecycle

```
connect (subprotocol=tts.v1)
  → server accepts
  → client sends auth frame (within auth_timeout_s)
  → server sends auth.ok
  → client sends tts.start
  → server sends tts.started, tts.queued, tts.processing (in that order)
  → client streams tts.text [, tts.flush]*
  → client sends tts.end (or tts.cancel)
  → server streams audio binary frames
  → server sends tts.done (or tts.cancelled / tts.error)
  → server closes the socket
```

One connection = one synthesis session. To do a second synthesis, open a
new connection.

## Client → server frames

All client frames are JSON text messages. Binary frames from the client
are rejected with `UNEXPECTED_BINARY`.

### auth

```json
{"type": "auth", "token": "...", "client_id": "optional-stable-id"}
```

Must be the first frame after connect. If absent within `auth_timeout_s`
the server closes the connection.

### tts.start

```json
{"type": "tts.start", "request_id": "abc", "voice": "female", "instructions": "请用广东话表达"}
```

`request_id` is required. `voice` is the vllm-omini speaker name (the
file name under `/root/.cache/vllm-omni/speakers/` without extension).
If unset, the gateway uses its configured `default_voice`.

`instructions` is optional — a voice-style / emotion / speed / language
directive (CosyVoice3's instruct2 mode). Examples:
`请用广东话表达`, `请用尽可能快地语速说一句话`, `用悲伤的语气朗读`.

For backwards compatibility, `prompt_audio_id` is accepted as an alias
for `voice` — they map to the same vllm-omini speaker file. New
clients should prefer `voice`.

### tts.text

```json
{"type": "tts.text", "text": "你好,这是一段。", "flush": false}
```

Plain text. The gateway buffers it across calls and emits sentences to
the engine when sentence boundaries (or `max_chars`) are reached. If you
want to force the buffer to flush after this push, set `flush: true`.

### tts.flush

```json
{"type": "tts.flush"}
```

Explicit boundary. Server pushes whatever's in the sentence buffer to
the engine and acks with `tts.flushed`. Note: `tts.flushed` only
confirms the buffer was forwarded — not that audio has been generated.

### tts.end

```json
{"type": "tts.end"}
```

No more text is coming. The server flushes the buffer, signals
end-of-input to the engine, and waits for synthesis to complete.

### tts.cancel

```json
{"type": "tts.cancel"}
```

Stop synthesis immediately. The server will call `engine.abort()` and,
if the connection is still alive, send `tts.cancelled` before closing.

## Server → client frames

JSON text frames are control/status. Binary frames are raw audio in the
format declared by `tts.started`.

### auth.ok

```json
{"type": "auth.ok", "client_id": "client-a"}
```

### tts.started

```json
{"type": "tts.started", "request_id": "abc", "sample_rate": 24000, "format": "pcm16", "channels": 1}
```

Sent right after `tts.start` is accepted. Use this to know the audio
format the server will stream.

### tts.queued

```json
{"type": "tts.queued", "request_id": "abc", "position": 7}
```

Sent before the session waits for a GPU slot. `position` is a snapshot
at queue-entry time — it is **not** a guarantee of completion order, and
it can drift if other sessions are cancelled. Use it for UX, not for
business logic.

### tts.processing

```json
{"type": "tts.processing", "request_id": "abc"}
```

Sent after acquiring a GPU slot, before the first audio chunk. Clients
that want to be polite (don't push text while waiting in line) should
hold their `tts.text` calls until they see this.

### Binary audio frames

Raw bytes in the format declared by `tts.started`. For `pcm16`, that's
little-endian 16-bit signed samples, mono, at the declared sample rate.
Clients should buffer and play these as they arrive.

### tts.done

```json
{
  "type": "tts.done", "request_id": "abc",
  "generated_samples": 384000, "generated_duration_ms": 16000
}
```

Sent **after** all audio frames. `generated_samples` is the count
**produced server-side**; it is not a guarantee that the client
received and played them all. If the connection closes mid-flight or
backpressure trips, the client should rely on its own count of received
bytes.

### tts.cancelled

```json
{"type": "tts.cancelled", "request_id": "abc", "reason": "client_cancelled"}
```

Best-effort acknowledgement of `tts.cancel`. May not arrive if the
connection has already been torn down.

### tts.error

```json
{"type": "tts.error", "request_id": "abc", "code": "ENGINE_ERROR", "message": "..."}
```

Terminal. After this, the connection closes. Common codes:

| code | meaning |
| ---- | ------- |
| `BAD_JSON` | message wasn't valid JSON |
| `BAD_REQUEST` | message was JSON but invalid types/fields |
| `BAD_STATE` | event sent in the wrong session state |
| `UNKNOWN_EVENT` | unrecognized `type` |
| `UNEXPECTED_BINARY` | client sent a binary frame; only text allowed |
| `UNAUTHORIZED` | bad/missing token in auth frame |
| `AUTH_TIMEOUT` | no auth frame within timeout |
| `START_TIMEOUT` | no `tts.start` within timeout after auth |
| `IDLE_TIMEOUT` | no client input for too long during STARTED |
| `SESSION_TIMEOUT` | session exceeded `max_session_s` |
| `MISSING_REQUEST_ID` | `tts.start` had no `request_id` |
| `BACKPRESSURE` | server text queue full; engine is far behind |
| `CLIENT_SLOW` | server outbound queue full; client isn't draining audio |
| `ENGINE_ERROR` | engine raised mid-synthesis |
| `TASK_ERROR` | producer task raised unexpectedly |
| `SERVER_ERROR` | uncaught endpoint exception |
| `SERVER_SHUTDOWN` | server is stopping; this session was terminated |

### tts.keepalive

```json
{"type": "tts.keepalive", "request_id": "abc", "ts": 1733600000000}
```

Server-initiated heartbeat. Keeps NAT/LB connections alive. No client
action required, but you can use it as a "the server is still healthy"
signal.

## Best-effort guarantees

- All audio binary frames precede `tts.done` / `tts.cancelled` /
  `tts.error`.
- `tts.cancelled` and `tts.error` may not arrive if the socket has
  already been torn down.
- `generated_samples` is server-side production, not client-side
  reception.
- `position` in `tts.queued` is a snapshot for UX, not a strict ordering.
