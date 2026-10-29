#!/usr/bin/env node
import { manage } from "./index.js";
const [action = "status", value] = process.argv.slice(2);
try {
  if (action === "--help" || action === "help") {
    console.log("Usage: opencode-agentrouter [status|check|pricing|sync|pacing <ms>|key <key>]\ncheck makes real model requests and consumes tokens. Prefer /connect → AgentRouter inside OpenCode for host credentials; standalone CLI uses environment/plugin config keys.");
  } else console.log(await manage(action, value));
} catch (err) {
  console.error(err instanceof Error ? err.message : String(err));
  process.exitCode = 1;
}
