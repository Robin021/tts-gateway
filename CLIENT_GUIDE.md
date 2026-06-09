# tts-gateway · 客户端接入指南

这份文档是给**调用方**的。如果你是接入这个 TTS 服务的开发者,看这里就够了。

---

## TL;DR

服务地址(默认):`<gateway-host>:8000`

| 想做什么 | 用哪个接口 |
|---|---|
| **给已有项目加 TTS,不想动太多代码** | HTTP `POST /v1/audio/speech`(OpenAI 兼容) |
| **做对话 / 实时 / 客户能打断** | WebSocket `/ws/tts` |
| **看接口文档(Swagger UI)** | `http://<gateway-host>:8000/docs` |
| **机器可读 OpenAPI schema** | `http://<gateway-host>:8000/openapi.json` |

所有接口都需要 Bearer token: `Authorization: Bearer <你拿到的 token>`。

---

## 一. HTTP 方式(OpenAI 兼容)

### 直接 curl

```bash
curl -X POST http://<gateway-host>:8000/v1/audio/speech \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -o output.wav \
  -d '{
    "input": "你好,这是一个测试。",
    "voice": "female",
    "response_format": "wav"
  }'

# Mac 听:
afplay output.wav
```

### Python(用 OpenAI SDK)

```python
from openai import OpenAI

client = OpenAI(
    api_key="YOUR_TOKEN",
    base_url="http://<gateway-host>:8000/v1",
)

resp = client.audio.speech.create(
    model="cosyvoice3",      # 占位,实际服务端固定
    voice="female",          # 见下面"可选音色"
    input="你好,这是一个测试。",
    response_format="wav",   # 或 "pcm"
)
resp.stream_to_file("output.wav")
```

### 流式(低延迟)

`stream: true` —— 服务端一边合成一边发,客户端拿到第一字节大约 600ms。

```python
import httpx

with httpx.stream(
    "POST",
    "http://<gateway-host>:8000/v1/audio/speech",
    headers={"Authorization": "Bearer YOUR_TOKEN"},
    json={
        "input": "你好,这是流式合成。",
        "voice": "female",
        "response_format": "pcm",  # 流式建议用 pcm
        "stream": True,
    },
    timeout=None,
) as resp:
    resp.raise_for_status()
    for chunk in resp.iter_bytes():
        # chunk 是 24kHz mono pcm16 音频字节,实时播放或写盘
        play(chunk)
```

### 请求体字段

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `input` | string | ✓ | 要合成的文本(UTF-8) |
| `voice` | string | | 音色名称,见"可选音色"。不传用默认 `female` |
| `instructions` | string | | ⚠️ **当前后端不支持运行时指令**。要切换情感/方言/语速,**用预上传的 voice 变体**——见下方"通过 voice 切换风格" |
| `response_format` | `"pcm"` / `"wav"` | | 默认 `pcm` |
| `stream` | bool | | 默认 `false` |
| `model` | string | | OpenAI SDK 兼容用,服务端忽略 |
| `speed` | number | | OpenAI SDK 兼容用,服务端忽略 |
| `request_id` | string | | 客户端可选追踪 ID |

### 通过 voice 切换风格(方言/情感/语速)

⚠️ **当前后端 (cosyvoice3 + vllm-omini) 的 `instructions` 参数不可靠**:vllm-omini 在 cosyvoice3 模型路径上没接通这个字段,实测 per-request 注入会**丢字 / 乱码**。

唯一稳定的办法是**预上传 voice 变体** —— 把指令固化进 voice 的 ref_text。客户端通过 `voice` 字段切换:

```bash
# 不要这样写(不稳定):
{"input": "...", "voice": "female", "instructions": "请用广东话"}

# 这样写(稳定):
{"input": "...", "voice": "female_cantonese"}
```

实测可用变体(后端预上传):

| voice | 效果 |
|---|---|
| `female` / `male` / `linzhiling` | 标准音色,普通话 |
| `female_cantonese` | 粤语女声 |
| `female_sichuan` | 四川话女声 |
| `female_excited` / `male_excited` | 兴奋语气 |
| `female_sad` / `male_sad` | 悲伤语气 |
| `female_fast` / `female_slow` | 快/慢语速女声 |
| `male_calm` | 沉稳男声 |

调用前查最新列表:

```bash
curl http://<gateway-host>:8000/v1/audio/voices \
  -H "Authorization: Bearer YOUR_TOKEN"
```

**新增风格**: 找运维加 voice 变体(参考音频 + 指令 → 新 voice 名),返回你一个新名字。

**未来**: 当 vllm-omini 在 cosyvoice3 模型路径上正确实现 `instructions` 字段后,客户端代码里可以从 `voice="female_cantonese"` 改回 `voice="female" + instructions="请用广东话"`。

### 响应

成功 `200 OK`,`Content-Type: audio/pcm` 或 `audio/wav`,body 是音频字节。

错误:
- `401 Unauthorized` — 没传或错的 Bearer token
- `422 Unprocessable Entity` — 请求体格式不对(缺 `input` 等)
- `502 Bad Gateway` — 后端引擎报错(看 `detail` 字段)

---

## 二. WebSocket 方式(推荐,功能完整)

WebSocket 比 HTTP 多两件事:**流式输入文本**(边打字边合成)+ **中途打断**。如果你做的是对话场景,只用 WebSocket。

### 端点 + 子协议

- URL: `ws://<gateway-host>:8000/ws/tts`
- 子协议(Sec-WebSocket-Protocol): `tts.v1`

### 会话生命周期

```
client                                server
  │                                      │
  ├── connect (subprotocol=tts.v1) ─────>│
  │                                      │
  ├── {"type":"auth", ...} ────────────> │
  │ <───── {"type":"auth.ok", ...}       │
  │                                      │
  ├── {"type":"tts.start", ...} ───────> │
  │ <───── {"type":"tts.started", ...}   │
  │ <───── {"type":"tts.queued", ...}    │
  │ <───── {"type":"tts.processing", ...} │
  │                                      │
  ├── {"type":"tts.text", "text":"..."}─> │
  ├── {"type":"tts.text", "text":"..."}─> │
  ├── {"type":"tts.end"} ──────────────> │
  │                                      │
  │ <─── [binary audio chunks] ──────────│
  │ <───── {"type":"tts.done", ...}      │
  │                                      │
  │ (server closes)                      │
```

### 一段完整 JS 示例

```javascript
const TOKEN = "YOUR_TOKEN";
const ws = new WebSocket("ws://<gateway-host>:8000/ws/tts", "tts.v1");
ws.binaryType = "arraybuffer";

ws.onopen = () => {
  // 1. 认证
  ws.send(JSON.stringify({
    type: "auth",
    token: TOKEN,
    client_id: "user-123",  // 可选,日志追踪用
  }));

  // 2. 开始一次合成会话
  ws.send(JSON.stringify({
    type: "tts.start",
    request_id: "req-001",
    voice: "female",        // 见下面"可选音色"
  }));

  // 3. 推送文本(可以分多次,LLM 流式吐字时正合适)
  ws.send(JSON.stringify({ type: "tts.text", text: "你好,这是一个流式 " }));
  ws.send(JSON.stringify({ type: "tts.text", text: "TTS 测试。" }));

  // 4. 告诉服务端文本结束
  ws.send(JSON.stringify({ type: "tts.end" }));
};

ws.onmessage = (event) => {
  if (typeof event.data === "string") {
    const evt = JSON.parse(event.data);
    console.log("event:", evt);
    // 关注的事件:
    // - tts.started : 合成会话已建立
    // - tts.processing : 拿到 GPU 槽,马上要出音频
    // - tts.done : 合成完成
    // - tts.error : 出错(看 evt.code 和 evt.message)
    // - tts.cancelled : 你发过 tts.cancel,服务端确认了
  } else {
    // 二进制 = 音频数据 (默认 pcm16, mono, 24kHz)
    playPCM(event.data);
  }
};

// 中途打断(用户说"等等"):
function interrupt() {
  ws.send(JSON.stringify({ type: "tts.cancel" }));
}
```

### 客户端 → 服务端 事件

| 事件 | 字段 | 说明 |
|---|---|---|
| `auth` | `token`, `client_id?` | 必须是连接后**第一帧**;5 秒不发服务端会断你 |
| `tts.start` | `request_id`, `voice?`, `instructions?` | 一次连接一次会话(start 之后就只能 cancel/end)。`instructions` 见 HTTP 部分"指令"小节,同样语义 |
| `tts.text` | `text`, `flush?` | 多次累积。`flush:true` 强制冲刷当前句缓存 |
| `tts.flush` | (无) | 等价于上面的 `flush:true`,但不带文本 |
| `tts.end` | (无) | 不再有文本,等服务端把最后一段合成完 |
| `tts.cancel` | (无) | 立刻停止,丢弃后续音频 |

### 服务端 → 客户端 事件

| 事件 | 关键字段 | 说明 |
|---|---|---|
| `auth.ok` | `client_id` | 认证通过 |
| `tts.started` | `request_id`, `sample_rate`, `format`, `channels` | 注意音频格式从这里读 |
| `tts.queued` | `position` | 在队列中的位置(快照,仅 UX) |
| `tts.processing` | `request_id` | 已抢到 GPU 槽,马上要出音频了 |
| (binary) | — | 音频数据,格式见 `tts.started` |
| `tts.done` | `generated_samples`, `generated_duration_ms` | 合成正常结束 |
| `tts.cancelled` | `reason` | `tts.cancel` 后的回应(可能因连接已断而收不到) |
| `tts.error` | `code`, `message` | 见下面"错误码" |
| `tts.keepalive` | `ts` | 服务端心跳,客户端可忽略 |

### 音频格式

服务端在 `tts.started` 里告诉你格式,目前固定:

```json
{
  "type": "tts.started",
  "sample_rate": 24000,
  "format": "pcm16",
  "channels": 1
}
```

也就是 **24kHz 单声道、16bit signed little-endian PCM**。Web 端用 `AudioWorklet` 或 `AudioBufferSourceNode` 直接播。

### 错误码

| `code` | 含义 |
|---|---|
| `BAD_JSON` | 帧不是合法 JSON |
| `BAD_REQUEST` | 字段类型错 |
| `BAD_STATE` | 在错的状态发了帧(比如还没 auth 就 tts.start) |
| `UNKNOWN_EVENT` | 不认识的 `type` |
| `UNEXPECTED_BINARY` | 客户端发了二进制(只允许 JSON 文本) |
| `UNAUTHORIZED` | token 错 |
| `AUTH_TIMEOUT` | 5 秒没发 auth 帧 |
| `START_TIMEOUT` | auth 后 10 秒没发 tts.start |
| `IDLE_TIMEOUT` | 30 秒没发任何 tts.text(已 start 但停滞) |
| `SESSION_TIMEOUT` | 单连接超过 5 分钟 |
| `MISSING_REQUEST_ID` | tts.start 没传 request_id |
| `BACKPRESSURE` | 文本推得太快,服务端处理不过来 |
| `CLIENT_SLOW` | 你不接收音频,服务端的发送队列爆了 |
| `ENGINE_ERROR` | TTS 引擎报错 |
| `SERVER_SHUTDOWN` | 服务在重启 |

收到 `tts.error` 后服务端会主动断开连接。客户端收到错误就当本次会话失败,要重连。

---

## 三. 可选音色

调用前先看一下:

```bash
curl http://<gateway-host>:8000/v1/audio/voices \
  -H "Authorization: Bearer YOUR_TOKEN"
```

返回 `voices` 数组里的字符串,就是你能传给 `voice` 字段的合法值。

**当前预置**:

基础音色(zero_shot,纯克隆):
| voice | 备注 |
|---|---|
| `female` | 默认,标准女声(普通话) |
| `male` | 标准男声(普通话) |
| `linzhiling` | 特色音色(克隆林志玲) |

风格变体(同样的录音,但 ref_text 烧入了不同 instruction):
| voice | 效果 |
|---|---|
| `female_cantonese` | 粤语女声 |
| `female_sichuan` | 四川话女声 |
| `female_excited` / `male_excited` | 兴奋语气 |
| `female_sad` / `male_sad` | 悲伤语气 |
| `female_fast` | 快语速女声 |
| `female_slow` | 慢语速女声 |
| `male_calm` | 沉稳冷静男声 |

> **没列在表里的组合不要用**。例如 `male_cantonese`(合成失败)、`male_fast`(语音糊成一团)等是模型对该男声参考音频 + 该指令的组合学不出有效输出,运维已从 voice 列表里移除。

要新增音色或风格变体,找运维上传(参考音频 + 指令 → 新 voice 名)。

---

## 四. 性能 / 限制

| 指标 | 实测 | 备注 |
|---|---|---|
| TTFB(发请求到第一字节音频) | ~600ms | vllm-omini 自身的 prefill 延迟 |
| 中途打断响应 | ~1ms 收到 cancelled,但**约 1 秒内还有尾音**会到达 | vllm 输出粒度限制,客户端要在收到 cancelled 后丢弃后续音频 |
| 实时率 | ~0.5x(音频时长 / 实际耗时) | 比说话快 2 倍 |
| 文本长度 | 单次推荐 < 500 字 | 太长可能被截断 |
| 并发会话 | 默认上限 32(可联系运维调) | |
| 单 WS 会话最长 | 5 分钟 | 长会话拆多次 |

---

## 五. 在线交互式文档

服务跑起来后,直接浏览器开:

- **Swagger UI**:`http://<gateway-host>:8000/docs`
  可以填表单 / 直接发请求,适合快速测
- **ReDoc**:`http://<gateway-host>:8000/redoc`
  适合阅读,排版好看
- **OpenAPI JSON**:`http://<gateway-host>:8000/openapi.json`
  生成 SDK 用

WebSocket 协议 OpenAPI 不支持描述,所以 Swagger 里只看到 HTTP 部分。WebSocket 看本文档第二节。

---

## 六. 常见问题

**Q: 我没有 token 怎么办?**
A: 找运维拿。服务端的 `GATEWAY_AUTH_TOKEN` 环境变量配的就是合法 token 列表(逗号分隔),让运维生成一个给你。

**Q: 听起来变速了?**
A: 你播放时 sample_rate 设的不是 24000Hz。检查 `tts.started` 里的 `sample_rate` 字段。

**Q: 流式 HTTP 拿不到音频,要等好久?**
A: 客户端没启用流式接收。HTTPX 里要用 `client.stream(...)` + `iter_bytes()`,不能用普通 `client.post()`(那个会缓冲全部)。

**Q: cancel 之后还有声音?**
A: 这是已知限制。vllm-omini 输出粒度是"段"(每段 ~1 秒),aclose 后这一段还会来完。客户端在收到 `tts.cancelled` 之后**主动丢弃后续 binary 帧**就行。

**Q: 同一个 WS 连接能做第二次合成吗?**
A: 不行,一连接一会话。第二次合成开新 WS。连接成本相对 GPU 合成成本可以忽略。

**Q: HTTP 接口能打断吗?**
A: 不能,HTTP 是单向请求。要打断只能用 WebSocket。
