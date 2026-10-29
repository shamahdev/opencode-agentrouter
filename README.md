# @shamah/opencode-agentrouter

OpenCode V2 adaptation of [madgagarin/pi-agentrouter](https://github.com/madgagarin/pi-agentrouter), based on [`e7dc3b4` (v2.4.1)](https://github.com/madgagarin/pi-agentrouter/tree/e7dc3b4bb4272e7187ea759e4a746a2814e1210f).

See original [README](https://github.com/madgagarin/pi-agentrouter/README.md). This README documents only installation and differences introduced by the adaptation.

## How to Install

### NPM Package

Add `@shamah/opencode-agentrouter` to your plugin list in `~/.config/opencode/opencode.json`:

```json
{
  "plugins": ["@shamah/opencode-agentrouter"],
  "model": "agentrouter-openai/gpt-6.1-sol"
}
```

### Local Checkout

From this checkout:

```sh
npm install
npm run build
```

Use the checkout's absolute path in your OpenCode configuration:

```json
{
  "plugins": ["/absolute/path/to/opencode-agentrouter"],
  "model": "agentrouter-openai/gpt-6.1-sol"
}
```

### Authenticate

Use `/connect` → **AgentRouter**. Both provider families share the host integration's credentials. The upstream environment variables `AGENTROUTER_API_KEY` and `AGENT_ROUTER_API_KEY` also work.


## Configuration location

Pi paths are replaced by OpenCode/XDG paths:

| File | Default location |
|---|---|
| Plugin configuration | `~/.config/opencode/agentrouter.json` |
| Pricing cache | `~/.local/state/opencode/agentrouter/.agentrouter-models-cache.json` |
| Shared pacing timestamp | `~/.local/state/opencode/agentrouter/.agentrouter-pacing` |
| Redacted retry diagnostics | `~/.local/state/opencode/agentrouter/.agentrouter-debug.log` |

`XDG_CONFIG_HOME` and `XDG_STATE_HOME` are respected. Configuration writes use owner-only permissions, not encryption.

In addition to upstream's `apiKey` and `minIntervalMs`, this adaptation accepts:

- `baseURL`: gateway root without `/v1`; defaults to `https://agentrouter.org`.
- `maxRetries`: transport retry count; defaults to upstream's ten retries. Set `0` to disable plugin retries. OpenCode may also apply its own retry policy.


MIT attribution is preserved in [LICENSE](LICENSE) and [NOTICE](NOTICE).
