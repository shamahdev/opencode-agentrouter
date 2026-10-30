// MIT © madgagarin. Faithful migration of pi-agentrouter.

import * as os from "node:os";
import { mkdir, readFile, writeFile, rename, appendFile } from "node:fs/promises";
import { join, dirname } from "node:path";
import { createHash, randomUUID } from "node:crypto";
import { Plugin, Provider, Model, Integration } from "@opencode/plugin";
import { Money } from "@opencode/schema/money";
import { Schema } from "effect";
import { createOpenAI } from "@ai-sdk/openai";
import { createAnthropic } from "@ai-sdk/anthropic";
import { createOpenAICompatible } from "@ai-sdk/openai-compatible";

export const BASE_URL = "https://agentrouter.org";
export const DEFAULT_PACING_MS = 3500;
export const DEFAULT_MAX_RETRIES = 10;
export const PROVIDERS = { openai: "agentrouter-openai", anthropic: "agentrouter-clode" } as const;
export const CANONICAL_OPENCODE_HEADER = "You are an expert coding assistant operating inside OpenCode. You help users by reading files, executing commands, editing code, and writing new files.";
export const LANGUAGE_PREAMBLE = "[Instruction: You are an expert coding assistant operating inside OpenCode. Please carefully analyze the technical context, understand the user request, follow all project instructions and coding standards, and respond thoroughly in the requested language.]";
export type Json = Record<string, any>;
export type Protocol = "responses" | "chat" | "anthropic";
export type Fetch = (input: RequestInfo | URL, init?: RequestInit) => Promise<Response>;

export const KNOWN_MODEL_SPECS: Record<string, ModelSpec> = {
  "gpt-6-astra": {
    id: "gpt-6-astra",
    name: "GPT 6 Astra",
    providerType: "openai",
    api: "openai-responses",
    input: ["text", "image"],
    contextWindow: 1048576,
    maxTokens: 131072,
    reasoning: true,
    compat: { sendSessionAffinityHeaders: false },
    cost: { input: 8.0 / 1_000_000, output: 40.0 / 1_000_000, cacheRead: 0, cacheWrite: 0 },
  },
  "gpt-6.1-sol": {
    id: "gpt-6.1-sol",
    name: "GPT 6.1 Sol",
    providerType: "openai",
    api: "openai-responses",
    input: ["text", "image"],
    contextWindow: 1050000,
    maxTokens: 128000,
    reasoning: true,
    compat: { sendSessionAffinityHeaders: false },
    cost: { input: 3.0 / 1_000_000, output: 15.0 / 1_000_000, cacheRead: 0, cacheWrite: 0 },
  },
  "gpt-5.6-sol": {
    id: "gpt-5.6-sol",
    name: "GPT 5.6 Sol",
    providerType: "openai",
    api: "openai-responses",
    input: ["text", "image"],
    contextWindow: 1048576,
    maxTokens: 131072,
    reasoning: true,
    compat: { sendSessionAffinityHeaders: false },
    cost: { input: 8.0 / 1_000_000, output: 40.0 / 1_000_000, cacheRead: 0, cacheWrite: 0 },
  },
  "deepseek-v4-flash": {
    id: "deepseek-v4-flash",
    name: "DeepSeek V4 Flash",
    providerType: "openai",
    contextWindow: 1048576,
    maxTokens: 65536,
    reasoning: true,
    compat: {
      sendSessionAffinityHeaders: false,
    },
    cost: { input: 4.0 / 1_000_000, output: 12.0 / 1_000_000, cacheRead: 2.0 / 1_000_000, cacheWrite: 0 },
  },
  "deepseek-v4f": {
    id: "deepseek-v4f",
    name: "DeepSeek V4F",
    providerType: "openai",
    contextWindow: 1048576,
    maxTokens: 65536,
    reasoning: true,
    compat: {
      sendSessionAffinityHeaders: false,
    },
    cost: { input: 4.0 / 1_000_000, output: 12.0 / 1_000_000, cacheRead: 2.0 / 1_000_000, cacheWrite: 0 },
  },
  "glm-5.3": {
    id: "glm-5.3",
    name: "GLM 5.3",
    providerType: "openai",
    contextWindow: 1048576,
    maxTokens: 131072,
    reasoning: true,
    compat: { sendSessionAffinityHeaders: false },
    cost: { input: 3.0 / 1_000_000, output: 12.0 / 1_000_000, cacheRead: 0, cacheWrite: 0 },
  },
  "glm-5.2": {
    id: "glm-5.2",
    name: "GLM 5.2",
    providerType: "openai",
    contextWindow: 1048576,
    maxTokens: 131072,
    reasoning: true,
    compat: { sendSessionAffinityHeaders: false },
    cost: { input: 3.0 / 1_000_000, output: 12.0 / 1_000_000, cacheRead: 0, cacheWrite: 0 },
  },
  "gpt-5.5": {
    id: "gpt-5.5",
    name: "GPT 5.5",
    providerType: "openai",
    contextWindow: 1048576,
    maxTokens: 131072,
    reasoning: true,
    compat: { sendSessionAffinityHeaders: false },
    cost: { input: 4.0 / 1_000_000, output: 8.0 / 1_000_000, cacheRead: 0, cacheWrite: 0 },
  },
  "claude-opus-4-8": {
    id: "claude-opus-4-8",
    name: "Claude Opus 4.8",
    providerType: "anthropic",
    input: ["text", "image"],
    contextWindow: 1048576,
    maxTokens: 65536,
    reasoning: true,
    compat: {
      forceAdaptiveThinking: true,
      allowEmptySignature: true,
      sendSessionAffinityHeaders: true,
      supportsEagerToolInputStreaming: false,
    },
    cost: { input: 8.0 / 1_000_000, output: 40.0 / 1_000_000, cacheRead: 0, cacheWrite: 0 },
  },
  "claude-opus-5-5": {
    id: "claude-opus-5-5",
    name: "Claude Opus 5.5",
    providerType: "anthropic",
    input: ["text", "image"],
    contextWindow: 1000000,
    maxTokens: 128000,
    reasoning: true,
    compat: {
      forceAdaptiveThinking: true,
      allowEmptySignature: true,
      sendSessionAffinityHeaders: true,
      supportsEagerToolInputStreaming: false,
    },
    cost: { input: 12.0 / 1_000_000, output: 12.0 / 1_000_000, cacheRead: 0, cacheWrite: 0 },
  },
  "claude-opus-5": {
    id: "claude-opus-5",
    name: "Claude Opus 5",
    providerType: "anthropic",
    input: ["text", "image"],
    contextWindow: 1048576,
    maxTokens: 65536,
    reasoning: true,
    compat: {
      forceAdaptiveThinking: true,
      allowEmptySignature: true,
      sendSessionAffinityHeaders: true,
      supportsEagerToolInputStreaming: false,
    },
    cost: { input: 6.0 / 1_000_000, output: 30.0 / 1_000_000, cacheRead: 0, cacheWrite: 0 },
  },
  "claude-opus-4-7": {
    id: "claude-opus-4-7",
    name: "Claude Opus 4.7",
    providerType: "anthropic",
    input: ["text", "image"],
    contextWindow: 1048576,
    maxTokens: 65536,
    reasoning: true,
    compat: {
      forceAdaptiveThinking: true,
      allowEmptySignature: true,
      sendSessionAffinityHeaders: true,
      supportsEagerToolInputStreaming: false,
    },
    cost: { input: 8.0 / 1_000_000, output: 40.0 / 1_000_000, cacheRead: 0, cacheWrite: 0 },
  },
  "claude-opus-4-6": {
    id: "claude-opus-4-6",
    name: "Claude Opus 4.6",
    providerType: "anthropic",
    input: ["text", "image"],
    contextWindow: 1048576,
    maxTokens: 65536,
    reasoning: true,
    compat: {
      forceAdaptiveThinking: true,
      allowEmptySignature: true,
      sendSessionAffinityHeaders: true,
      supportsEagerToolInputStreaming: false,
    },
    cost: { input: 2.0 / 1_000_000, output: 10.0 / 1_000_000, cacheRead: 0, cacheWrite: 0 },
  },
};

export interface ModelSpec {
  id: string;
  name: string;
  providerType: "openai" | "anthropic";
  api?: string;
  headers?: Record<string, string>;
  input?: ("text" | "image")[];
  contextWindow: number;
  maxTokens: number;
  reasoning: boolean;
  thinkingLevelMap?: Record<string, string | null>;
  compat?: Record<string, any>;
  cost?: {
    input: number;
    output: number;
    cacheRead: number;
    cacheWrite: number;
  };
}

export interface ApiPricingModel {
  model_name: string;
  quota_type: number;
  model_ratio: number;
  model_price: number;
  owner_by?: string;
  completion_ratio: number;
  enable_groups: string[];
  supported_endpoint_types: string[];
}

export const FLAGSHIP_MODELS: string[] = [
  "agentrouter-openai/deepseek-v4-flash",
  "agentrouter-openai/gpt-6-astra",
  "agentrouter-openai/gpt-6.1-sol",
  "agentrouter-openai/gpt-5.6-sol",
  "agentrouter-clode/claude-opus-5-5",
];

export const DECOMMISSIONED_MODELS: string[] = [
  "agentrouter-clode/claude-opus-5",
  "agentrouter-clode/claude-opus-4-8",
  "agentrouter-clode/claude-opus-4-7",
  "agentrouter-clode/claude-opus-4-6",
  "agentrouter-openai/glm-5.3",
  "agentrouter-openai/glm-5.2",
  "agentrouter-openai/gpt-5.5",
];

export const AVAILABLE_MODEL_IDS = new Set<string>([
  "gpt-6-astra",
  "gpt-6.1-sol",
  "gpt-5.6-sol",
  "claude-opus-5-5",
]);

export const DECOMMISSIONED_MODEL_IDS = new Set<string>([
  "claude-opus-5",
  "claude-opus-4-8",
  "claude-opus-4-7",
  "claude-opus-4-6",
  "glm-5.3",
  "glm-5.2",
  "gpt-5.5",
]);

export function isBinaryDump(text: string): boolean {
  if (typeof text !== "string" || text.length === 0) return false;
  if (text.includes("\x7fELF") || text.includes("ELF\x02\x01")) return true;
  if (text.includes("\x00\x00")) return true;

  const sampleLen = Math.min(text.length, 2000);
  let nonPrintable = 0;
  for (let i = 0; i < sampleLen; i++) {
    const code = text.charCodeAt(i);
    if (code === 0 || code === 0xfffd || (code < 32 && code !== 9 && code !== 10 && code !== 13)) {
      nonPrintable++;
    }
  }
  if (nonPrintable > 15) return true;

  // Unbroken string tokens (minified binary dumps, strings binary dumps without whitespace)
  if (text.length > 600) {
    const tokens = text.split(/(?:\\n|\\r|[\s\r\n\t])+/);
    for (const t of tokens) {
      if (t.length > 600) return true;
    }
  }

  // Large strings (>8KB) with extremely low whitespace density (< 1.5%)
  if (text.length > 8000) {
    let ws = 0;
    const checkLen = Math.min(text.length, 20000);
    for (let i = 0; i < checkLen; i++) {
      const code = text.charCodeAt(i);
      if (code === 32 || code === 10 || code === 13 || code === 9) ws++;
      else if (code === 92 && i + 1 < checkLen && (text.charCodeAt(i + 1) === 110 || text.charCodeAt(i + 1) === 114)) {
        ws++;
      }
    }
    if (ws / checkLen < 0.015) return true;
  }

  return false;
}

export function sanitizeToolCallArguments(args: string): string {
  if (typeof args !== "string") {
    return JSON.stringify(args ?? {});
  }
  // Homophone / false-positive Russian WAF replacements
  let sanitized = args.replace(/Ключевое/g, "Главное").replace(/ключевое/g, "главное");
  // Tool call arguments MUST be valid JSON (never replace with REDACTED_NOTE)
  try {
    JSON.parse(sanitized);
    return sanitized;
  } catch {
    return JSON.stringify({ raw: sanitized });
  }
}

export function enforceToolCallPairing(messages: any[], nativeAnthropic = false): void {
  if (!Array.isArray(messages)) return;

  for (let i = 0; i < messages.length; i++) {
    const msg = messages[i];
    if (!msg || typeof msg !== "object") continue;

    // Case 1: OpenAI tool result message (role === "tool")
    if (msg.role === "tool") {
      const toolCallId = msg.tool_call_id;
      let prevAssistant: any = null;
      for (let j = i - 1; j >= 0; j--) {
        if (messages[j]?.role === "tool") continue;
        if (messages[j]?.role === "assistant") {
          prevAssistant = messages[j];
        }
        break;
      }

      if (prevAssistant) {
        if (!Array.isArray(prevAssistant.tool_calls)) {
          prevAssistant.tool_calls = [];
        }
        for (const tc of prevAssistant.tool_calls) {
          if (tc?.function && typeof tc.function.arguments === "string") {
            tc.function.arguments = sanitizeToolCallArguments(tc.function.arguments);
          }
        }
        const hasMatchingToolCall = prevAssistant.tool_calls.some(
          (tc: any) => tc && (tc.id === toolCallId || (!toolCallId && tc.id))
        );
        const hasContentToolCalls =
          Array.isArray(prevAssistant.content) &&
          prevAssistant.content.some(
            (b: any) =>
              b &&
              typeof b === "object" &&
              (b.type === "toolCall" || b.type === "tool_use" || b.type === "function_call") &&
              (b.id === toolCallId || b.tool_call_id === toolCallId || (!toolCallId && (b.id || b.tool_call_id)))
          );
        if (!hasMatchingToolCall && !hasContentToolCalls && toolCallId) {
          prevAssistant.tool_calls.push({
            id: toolCallId,
            type: "function",
            function: {
              name: msg.name || "tool",
              arguments: "{}",
            },
          });
        }
      } else {
        // Orphaned tool message with no preceding assistant turn: convert to safe plain user message
        msg.role = "user";
        if (msg.content === REDACTED_NOTE) {
          msg.content = REDACTED_NOTE;
        } else {
          const contentStr = typeof msg.content === "string" ? msg.content : JSON.stringify(msg.content ?? "");
          msg.content = `[Tool Result ${toolCallId || ""}]: ${contentStr}`;
        }
        delete msg.tool_call_id;
        delete msg.name;
      }
    }

    // Case 2: Anthropic tool result blocks inside user message ({ type: "tool_result" })
    if (msg.role === "user" && Array.isArray(msg.content)) {
      const hasToolResult = msg.content.some((b: any) => b && typeof b === "object" && b.type === "tool_result");
      if (hasToolResult) {
        let prevAssistant: any = null;
        for (let j = i - 1; j >= 0; j--) {
          if (messages[j]?.role === "assistant") {
            prevAssistant = messages[j];
          }
          break;
        }

        for (let bIdx = 0; bIdx < msg.content.length; bIdx++) {
          const block = msg.content[bIdx];
          if (block && typeof block === "object" && block.type === "tool_result") {
            const toolUseId = block.tool_use_id;
            if (prevAssistant) {
              const inContent =
                Array.isArray(prevAssistant.content) &&
                prevAssistant.content.some(
                  (b: any) => b && typeof b === "object" && (b.type === "tool_use" || b.type === "toolCall") && b.id === toolUseId
                );
              const inToolCalls =
                Array.isArray(prevAssistant.tool_calls) &&
                prevAssistant.tool_calls.some((tc: any) => tc && tc.id === toolUseId);

              if (!inContent && !inToolCalls && toolUseId) {
                if (!Array.isArray(prevAssistant.content)) {
                  prevAssistant.content = prevAssistant.content ? [{ type: "text", text: String(prevAssistant.content) }] : [];
                }
                prevAssistant.content.push({
                  type: "tool_use",
                  id: toolUseId,
                  name: "tool",
                  input: {},
                });
                // OpenCode Anthropic SDK requires native blocks, not a parallel OpenAI tool_calls field.
                if (!nativeAnthropic) {
                  if (!Array.isArray(prevAssistant.tool_calls)) prevAssistant.tool_calls = [];
                  prevAssistant.tool_calls.push({
                    id: toolUseId,
                    type: "function",
                    function: { name: "tool", arguments: "{}" },
                  });
                }
              }
            } else {
              // Orphaned tool_result block with no preceding assistant turn: convert to plain text block
              if (block.content === REDACTED_NOTE || (Array.isArray(block.content) && block.content[0]?.text === REDACTED_NOTE)) {
                msg.content[bIdx] = {
                  type: "text",
                  text: REDACTED_NOTE,
                };
              } else {
                const contentStr = typeof block.content === "string" ? block.content : JSON.stringify(block.content ?? "");
                msg.content[bIdx] = {
                  type: "text",
                  text: `[Tool Result ${toolUseId || ""}]: ${contentStr}`,
                };
              }
            }
          }
        }
      }
    }
  }
}

export function sanitizeDeepSeekText(text: string): string {
  if (typeof text !== "string") return text;
  if (isBinaryDump(text)) return REDACTED_NOTE;
  // Replace false-positive blocked Russian word in AgentRouter upstream WAF
  return text.replace(/Ключевое/g, "Главное").replace(/ключевое/g, "главное");
}

export function cleanContent(text: string): string {
  if (typeof text !== "string") return text;
  if (isBinaryDump(text)) return "[Binary output omitted]";
  return text
    .replace(/\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])/g, "")
    .replace(/[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]/g, "")
    .replace(/[\uD800-\uDBFF](?![\uDC00-\uDFFF])|(?<![\uD800-\uDBFF])[\uDC00-\uDFFF]/g, "");
}

export function enforceCanonicalRootPrompt(systemPrompt: string | any[] | undefined): string | any[] {
  if (!systemPrompt) {
    return CANONICAL_OPENCODE_HEADER;
  }

  if (typeof systemPrompt === "string") {
    const text = systemPrompt.trim();
    const openCodeHeaderRegex = /(?:You are [^\n\r]*operating inside OpenCode[^\n\r]*\n?|You are (?:opencode|OpenCode)[^\n\r]*\n?)/i;

    if (openCodeHeaderRegex.test(text)) {
      const match = text.match(openCodeHeaderRegex);
      if (match && match.index !== undefined && match.index > 0) {
        const header = match[0].trim();
        const prefix = text.slice(0, match.index).trim();
        const rest = text.slice(match.index + match[0].length).trim();
        return `${header}\n\n${prefix}${rest ? "\n\n" + rest : ""}`;
      }
      return text;
    } else {
      return `${CANONICAL_OPENCODE_HEADER}\n\n${text}`;
    }
  }

  if (Array.isArray(systemPrompt)) {
    if (systemPrompt.length === 0) {
      return [{ type: "text", text: CANONICAL_OPENCODE_HEADER }];
    }
    const firstBlock = systemPrompt[0];
    if (firstBlock && typeof firstBlock.text === "string") {
      firstBlock.text = enforceCanonicalRootPrompt(firstBlock.text) as string;
    }
    return systemPrompt;
  }

  return systemPrompt;
}

export function isDeepSeekRequest(event: any, ctx: any): boolean {
  const modelId = (
    (event as any)?.payload?.model ||
    (event as any)?.model?.id ||
    (ctx as any)?.model?.id ||
    ""
  ).toLowerCase();
  return modelId.includes("deepseek");
}

export const WAF_BLOCK_RE = /sensitive[_ ]words?[_ ]detected|content-blocked/i;

export const SENSITIVE_WORDS_RE = /sensitive[_ ]words?[_ ]detected|content-blocked/i;

export const REDACTED_NOTE = "[Output omitted]";

export function isRecord(v: unknown): v is Record<string, unknown> {
  return typeof v === "object" && v !== null && !Array.isArray(v);
}

export function fingerprintOf(msg: Record<string, unknown>): string {
  return createHash("sha256").update(JSON.stringify(msg)).digest("hex");
}

export function firstUserAnchor(messages: unknown[]): string {
  for (const m of messages) {
    if (isRecord(m) && isHumanUserMessage(m as Record<string, unknown>)) return fingerprintOf(m);
  }
  return String(messages.length);
}

export function isTextBlock(block: Record<string, unknown>): boolean {
  return block.type === "text" || block.type === "input_text" || block.type === "output_text";
}

export function isHumanUserMessage(msg: Record<string, unknown>): boolean {
  if (msg.role !== "user") return false;
  if (typeof msg.content === "string") return true;
  if (Array.isArray(msg.content)) {
    const hasToolResult = msg.content.some((b) => isRecord(b) && b.type === "tool_result");
    const hasHumanText = msg.content.some(
      (b) =>
        isRecord(b) &&
        (b.type === "text" || b.type === "input_text") &&
        typeof b.text === "string" &&
        b.text.trim().length > 0 &&
        b.text !== REDACTED_NOTE
    );
    if (hasToolResult && !hasHumanText) return false;
    return true;
  }
  return false;
}

export function lastHumanUserIndex(messages: unknown[]): number {
  for (let i = messages.length - 1; i >= 0; i--) {
    if (isRecord(messages[i]) && isHumanUserMessage(messages[i] as Record<string, unknown>)) {
      return i;
    }
  }
  return -1;
}

export function hasRedactableText(content: unknown, msg?: Record<string, unknown>): boolean {
  if (msg) {
    if (msg.details && (typeof msg.details === "string" || Object.keys(msg.details).length > 0)) return true;
    if (typeof (msg as any).reasoning_content === "string" && (msg as any).reasoning_content.trim().length > 0) return true;
    if (typeof (msg as any).thinking === "string" && (msg as any).thinking.trim().length > 0) return true;
  }
  if (msg?.type === "function_call") return false;
  if (msg?.type === "function_call_output") return hasRedactableText((msg as any).output);
  if (typeof content === "string") return content.length > 0 && content !== REDACTED_NOTE;
  if (Array.isArray(content)) {
    for (const b of content) {
      if (!isRecord(b)) continue;
      if (b.type === "thinking" || b.type === "reasoning") return true;
      if (isTextBlock(b) && typeof b.text === "string" && b.text.length > 0 && b.text !== REDACTED_NOTE) return true;
      if (b.type === "tool_result") {
        if (typeof b.content === "string" && b.content.length > 0 && b.content !== REDACTED_NOTE) return true;
        if (Array.isArray(b.content)) {
          for (const c of b.content) {
            if (isRecord(c) && isTextBlock(c) && typeof c.text === "string" && c.text.length > 0 && c.text !== REDACTED_NOTE) {
              return true;
            }
          }
        }
      }
    }
  }
  return false;
}

export function redactBlocks(content: unknown): void {
  if (!Array.isArray(content)) return;
  for (let idx = content.length - 1; idx >= 0; idx--) {
    const block = content[idx];
    if (!isRecord(block)) continue;

    // For thinking / reasoning blocks, keep a harmless placeholder rather than deleting,
    // because thinking models (like DeepSeek) strictly require thinking blocks to be present in multi-turn history
    if (block.type === "thinking" || block.type === "reasoning") {
      if (typeof (block as any).thinking === "string") {
        (block as any).thinking = "Thinking...";
      }
      if (typeof (block as any).text === "string") {
        (block as any).text = "Thinking...";
      }
      continue;
    }

    if (isTextBlock(block) && typeof block.text === "string") {
      block.text = REDACTED_NOTE;
    } else if (block.type === "tool_result") {
      if (typeof block.content === "string") {
        block.content = REDACTED_NOTE;
      } else if (Array.isArray(block.content)) {
        for (const b of block.content) {
          if (isRecord(b) && isTextBlock(b) && typeof b.text === "string") {
            b.text = REDACTED_NOTE;
          }
        }
      }
    }
  }
}

export function isHideable(msg: Record<string, unknown>): boolean {
  return (
    msg.role === "user" ||
    msg.role === "tool" ||
    msg.role === "toolResult" ||
    msg.type === "function_call_output"
  );
}

export function redactMessageAt(messages: unknown[], i: number): void {
  const msg = messages[i];
  if (!isRecord(msg)) return;

  // Clear tool result details / raw diffs / attachments
  if ("details" in msg) delete msg.details;
  // DeepSeek and reasoning models require reasoning_content/thinking to exist in thinking mode.
  // Never delete them completely; keep a sanitized minimal placeholder.
  if ("reasoning_content" in msg) {
    (msg as any).reasoning_content = "Thinking...";
  }
  if ("thinking" in msg) {
    (msg as any).thinking = "Thinking...";
  }

  if (msg.type === "function_call_output") {
    if (typeof (msg as any).output === "string") (msg as any).output = REDACTED_NOTE;
    else redactBlocks((msg as any).output);
    return;
  }
  if (msg.role === "tool" || msg.role === "toolResult") {
    msg.content = REDACTED_NOTE;
    return;
  }
  if (msg.role === "user") {
    if (typeof msg.content === "string") {
      msg.content = REDACTED_NOTE;
    } else if (Array.isArray(msg.content)) {
      redactBlocks(msg.content);
      if (msg.content.length === 0) {
        msg.content = [{ type: "text", text: REDACTED_NOTE }];
      }
    }
  }
}

// OpenCode multiplexes credentials and sessions inside one server. State cannot be process-global.
export class PoisonRedactionState {
  redactSet = new Set<string>();
  escalatePending = false;
  isSensitiveBlock = false;
  exhausted = false;
  sessionAnchor: string | null = null;
  wafNotified = false;
  lastWafBlockTime = 0;

  resetPoisonRedactionState(): void {
    this.redactSet.clear();
    this.escalatePending = false;
    this.isSensitiveBlock = false;
    this.exhausted = false;
    this.sessionAnchor = null;
    this.wafNotified = false;
    this.lastWafBlockTime = 0;
  }

  triggerEscalation(sensitive = false): void {
    this.escalatePending = true;
    this.isSensitiveBlock = sensitive;
    if (sensitive) this.lastWafBlockTime = Date.now();
  }

  applyPoisonRedaction(payload: Record<string, unknown>): void {
    const messages = Array.isArray(payload.messages) ? payload.messages : (payload as any).input;
    if (!Array.isArray(messages) || messages.length === 0) return;

    // Prune failed assistant messages carrying error status/text in-place so dead error turns do not linger
    for (let i = messages.length - 1; i >= 0; i--) {
      const m = messages[i];
      if (!isRecord(m)) continue;
      if (m.role === "assistant") {
        if (m.stopReason === "error") {
          messages.splice(i, 1);
        } else if (typeof m.content === "string" && WAF_BLOCK_RE.test(m.content)) {
          messages.splice(i, 1);
        }
      }
    }

    const fps = messages.map((m) => (isRecord(m) ? fingerprintOf(m) : ""));

    const anchor = firstUserAnchor(messages);
    if (anchor !== this.sessionAnchor) {
      this.sessionAnchor = anchor;
      for (const fp of this.redactSet) {
        if (!fps.includes(fp)) {
          this.redactSet.delete(fp);
        }
      }
      this.exhausted = false;
      this.wafNotified = false;
    }

    if (this.redactSet.size > 0 && !fps.some((fp) => fp && this.redactSet.has(fp))) {
      this.redactSet.clear();
    }

    const lastHumanUser = lastHumanUserIndex(messages);

    for (let i = 0; i < messages.length; i++) {
      if (i === lastHumanUser) continue;
      if (!isRecord(messages[i]) || !isHideable(messages[i] as Record<string, unknown>)) continue;
      if (this.redactSet.has(fps[i])) redactMessageAt(messages, i);
    }

    if (this.escalatePending) {
      this.escalatePending = false;
      const sensitive = this.isSensitiveBlock;
      this.isSensitiveBlock = false;

      if (sensitive) {
        // Sensitive words / content-blocked detected:
        // 1. Immediately sweep and redact ANY binary dumps anywhere in history
        for (let i = 0; i < messages.length; i++) {
          if (!isRecord(messages[i])) continue;
          const m = messages[i] as Record<string, unknown>;
          let hasBinary = false;
          if (typeof m.content === "string" && isBinaryDump(m.content)) hasBinary = true;
          if (Array.isArray(m.content)) {
            for (const b of m.content) {
              if (isRecord(b)) {
                if (typeof b.text === "string" && isBinaryDump(b.text)) hasBinary = true;
                if (typeof b.content === "string" && isBinaryDump(b.content)) hasBinary = true;
              }
            }
          }
          if (i !== lastHumanUser && hasBinary && isHideable(m)) {
            this.redactSet.add(fps[i]);
            redactMessageAt(messages, i);
          }
        }

        // 2. Neutralize the most recent tool result(s) searching backwards
        let toolsRedactedCount = 0;
        for (let i = messages.length - 1; i >= 0; i--) {
          if (!isRecord(messages[i])) continue;
          const m = messages[i] as Record<string, unknown>;
          const isToolMsg =
            m.role === "tool" ||
            m.role === "toolResult" ||
            m.type === "function_call_output" ||
            (m.role === "user" && Array.isArray(m.content) && m.content.some((b) => isRecord(b) && b.type === "tool_result"));

          if (isToolMsg && !this.redactSet.has(fps[i]) && hasRedactableText(m.content, m)) {
            this.redactSet.add(fps[i]);
            redactMessageAt(messages, i);
            toolsRedactedCount++;
            if (toolsRedactedCount >= 4) break;
          }
        }
        // If no tool turns were left to redact, redact older user turns one by one (searching backwards, excluding lastHumanUser)
        if (toolsRedactedCount === 0) {
          for (let i = messages.length - 1; i >= 0; i--) {
            if (i === lastHumanUser) continue;
            if (!isRecord(messages[i])) continue;
            const m = messages[i] as Record<string, unknown>;
            if (m.role !== "user" || !isHideable(m)) continue;
            if (this.redactSet.has(fps[i])) continue;
            this.redactSet.add(fps[i]);
            if (hasRedactableText(m.content, m)) {
              redactMessageAt(messages, i);
              break;
            }
          }
        }
      } else {
        // Language ratio block: Stage 1 older user turns, Stage 2 assistant & tool
        let anyRedacted = false;
        for (let i = 0; i < messages.length; i++) {
          if (i === lastHumanUser) continue;
          if (!isRecord(messages[i])) continue;
          const m = messages[i] as Record<string, unknown>;
          if (m.role !== "user") continue;
          if (this.redactSet.has(fps[i])) continue;
          this.redactSet.add(fps[i]);
          if (hasRedactableText(m.content, m)) {
            redactMessageAt(messages, i);
            anyRedacted = true;
          }
        }
        if (!anyRedacted) {
          for (let i = 0; i < messages.length; i++) {
            if (i === lastHumanUser) continue;
            if (!isRecord(messages[i])) continue;
            const m = messages[i] as Record<string, unknown>;
            if (!isHideable(m) || m.role === "user") continue;
            if (this.redactSet.has(fps[i])) continue;
            this.redactSet.add(fps[i]);
            if (hasRedactableText(m.content, m)) {
              redactMessageAt(messages, i);
              anyRedacted = true;
            }
          }
        }
      }
    }

    // Exhausted if there are no more hideable messages left to redact other than lastHumanUser
    this.exhausted = true;
    for (let i = 0; i < messages.length; i++) {
      if (i === lastHumanUser) continue;
      if (!isRecord(messages[i]) || !isHideable(messages[i] as Record<string, unknown>)) continue;
      if (!this.redactSet.has(fps[i])) {
        this.exhausted = false;
        break;
      }
    }
  }
}

export function isCompactionPayload(payload: Record<string, unknown>): boolean {
  if (
    typeof payload.system === "string" &&
    (payload.system.includes("summarization assistant") || payload.system.includes("context summarization"))
  ) {
    return true;
  }

  const messages = Array.isArray(payload.messages) ? payload.messages : (payload as any).input;
  if (!Array.isArray(messages)) return false;

  for (const m of messages) {
    if (!m || typeof m !== "object") continue;
    const role = (m as any).role;
    const content = (m as any).content;

    if (role === "system" || role === "developer") {
      const text =
        typeof content === "string"
          ? content
          : Array.isArray(content)
          ? content.map((b: any) => b?.text || "").join(" ")
          : "";
      if (text.includes("summarization assistant") || text.includes("context summarization")) {
        return true;
      }
    }

    if (typeof content === "string") {
      if (
        (content.includes("<conversation>") && content.includes("</conversation>")) ||
        (content.includes("# Conversation") && content.includes("# Instructions")) ||
        content.includes("The messages above are a conversation to summarize") ||
        content.includes("This is the PREFIX of a turn that was too large to keep")
      ) {
        return true;
      }
    } else if (Array.isArray(content)) {
      for (const b of content) {
        if (b && typeof b === "object" && typeof b.text === "string") {
          if (
            (b.text.includes("<conversation>") && b.text.includes("</conversation>")) ||
            (b.text.includes("# Conversation") && b.text.includes("# Instructions")) ||
            b.text.includes("The messages above are a conversation to summarize") ||
            b.text.includes("This is the PREFIX of a turn that was too large to keep")
          ) {
            return true;
          }
        }
      }
    }
  }

  return false;
}

export function cleanSerializedConversation(inner: string): string {
  // Split on block boundaries: [User]:, [Assistant thinking]:, [Assistant]:, [Assistant tool calls]:, [Tool result]:
  const blockRegex = /(?:^|\n+)(?=\[(?:User|Assistant thinking|Assistant|Assistant tool calls|Tool result)\]:)/;
  const blocks = inner.split(blockRegex);
  const out: string[] = [];

  for (const rawBlock of blocks) {
    const block = rawBlock.trim();
    if (!block) continue;

    // 1. Drop thinking blocks completely — thinking is scratchpad, wastes tokens, and contains sensitive reasoning tokens
    if (block.startsWith("[Assistant thinking]:")) {
      continue;
    }

    // 2. Neutralize tool results — replace raw dumped files, scraped web pages, tokens, and diffs with clean placeholder
    if (block.startsWith("[Tool result]:")) {
      out.push("[Tool result]: [Content withheld by local policy]");
      continue;
    }

    // 3. Neutralize tool call arguments — keep tool name and signature clean without massive payload dumps
    if (block.startsWith("[Assistant tool calls]:")) {
      const stripped = block.replace(/\([\s\S]*?\)(?=;\s*|$)/g, "()");
      out.push(stripped);
      continue;
    }

    // 4. Drop assistant messages that were WAF error dumps
    if (block.startsWith("[Assistant]:")) {
      if (WAF_BLOCK_RE.test(block)) {
        continue;
      }
      out.push(block);
      continue;
    }

    out.push(block);
  }

  let res = out.join("\n\n");
  res = sanitizeDeepSeekText(res);
  return res;
}

export function sanitizeCompactionText(text: string): string {
  // Format 1: <conversation>...</conversation> (Standard compaction & branch summary)
  if (text.includes("<conversation>")) {
    text = text.replace(/<conversation>([\s\S]*?)<\/conversation>/g, (_m, inner) => {
      return `<conversation>\n${cleanSerializedConversation(inner)}\n</conversation>`;
    });
  }

  // Format 2: # Conversation ... # Instructions (Split-turn prefix summary)
  if (text.includes("# Conversation")) {
    text = text.replace(/# Conversation([\s\S]*?)(?=# Instructions|$)/g, (_m, inner) => {
      return `# Conversation\n${cleanSerializedConversation(inner)}\n\n`;
    });
  }

  // Format 3: Clean <previous-summary> if present
  if (text.includes("<previous-summary>")) {
    text = text.replace(/<previous-summary>([\s\S]*?)<\/previous-summary>/g, (_m, inner) => {
      let cleaned = inner.replace(/Error: 500: \{"message":"sensitive words detected[\s\S]*?}/gi, "");
      cleaned = cleaned.replace(/sensitive[_ ]words?[_ ]detected|content-blocked/gi, "");
      cleaned = sanitizeDeepSeekText(cleaned);
      return `<previous-summary>${cleaned}</previous-summary>`;
    });
  }

  return text;
}

export function sanitizeCompactionPayload(payload: Record<string, unknown>): boolean {
  const messages = Array.isArray(payload.messages) ? payload.messages : (payload as any).input;
  if (!Array.isArray(messages) || messages.length === 0) return false;

  let sanitized = false;
  for (const msg of messages) {
    if (!msg || typeof msg !== "object") continue;

    if (typeof msg.content === "string") {
      if (
        msg.content.includes("<conversation>") ||
        msg.content.includes("# Conversation") ||
        msg.content.includes("<previous-summary>")
      ) {
        msg.content = sanitizeCompactionText(msg.content);
        sanitized = true;
      }
    } else if (Array.isArray(msg.content)) {
      for (const block of msg.content) {
        if (block && typeof block === "object" && typeof block.text === "string") {
          if (
            block.text.includes("<conversation>") ||
            block.text.includes("# Conversation") ||
            block.text.includes("<previous-summary>")
          ) {
            block.text = sanitizeCompactionText(block.text);
            sanitized = true;
          }
        }
      }
    }
  }

  return sanitized;
}

export function normalizeMessagesForAgentRouter(messages: any[], isDeepSeek: boolean = false): void {
  if (!Array.isArray(messages)) return;
  // Reviewed correction: binary-output heuristics must never omit the current user's request.
  const latestUser = messages[lastHumanUserIndex(messages)];
  const latestContent = latestUser ? structuredClone(latestUser.content) : undefined;

  // Prune poisoned/failed assistant turns globally across all models (e.g. prior 402 quota failure, network errors, empty turns)
  for (let i = messages.length - 1; i >= 0; i--) {
    const msg = messages[i];
    if (!msg || typeof msg !== "object") continue;
    if (msg.role === "assistant") {
      const hasToolCalls = Array.isArray(msg.tool_calls) && msg.tool_calls.length > 0;
      const hasToolUseBlocks =
        Array.isArray(msg.content) &&
        msg.content.some((b: any) => b && typeof b === "object" && (b.type === "tool_use" || b.type === "tool_call"));
      const hasTools = hasToolCalls || hasToolUseBlocks;
      const isEmptyContent =
        !msg.content ||
        msg.content === "" ||
        (Array.isArray(msg.content) && (
          msg.content.length === 0 ||
          msg.content.every((b: any) => b && typeof b === "object" && (b.type === "text" || b.type === "input_text") && (!b.text || b.text.trim() === ""))
        ));

      const hasValidRc = typeof msg.reasoning_content === "string" && msg.reasoning_content.trim().length > 0;
      if (
        msg.stopReason === "error" ||
        (msg.stopReason === "aborted" && isEmptyContent) ||
        (typeof msg.content === "string" && WAF_BLOCK_RE.test(msg.content)) ||
        (isEmptyContent && !hasTools && !hasValidRc)
      ) {
        messages.splice(i, 1);
      }
    }
  }

  enforceToolCallPairing(messages);


  for (const msg of messages) {
    if (!msg || typeof msg !== "object") continue;
    if (msg.role === "developer") {
      msg.role = "system";
    }
    if (msg.role === "assistant") {
      let extractedThinking: string | undefined;

      const hasToolCalls = Array.isArray(msg.tool_calls) && msg.tool_calls.length > 0;
      const hasContentToolCalls =
        Array.isArray(msg.content) &&
        msg.content.some(
          (b: any) => b && typeof b === "object" && (b.type === "toolCall" || b.type === "tool_use" || b.type === "function_call")
        );
      const isToolTurn = hasToolCalls || hasContentToolCalls;

      let textContent = "";

      if (Array.isArray(msg.content)) {
        const thinkingParts: string[] = [];
        const nonThinkingParts: any[] = [];
        const convertedToolCalls: any[] = [];

        for (const part of msg.content) {
          if (part && typeof part === "object") {
            if (part.type === "thinking" || part.type === "reasoning") {
              const text = typeof part.thinking === "string" ? part.thinking.trim() : typeof part.text === "string" ? part.text.trim() : "";
              if (text.length > 0) thinkingParts.push(text);
            } else if (part.type === "toolCall" || part.type === "tool_use" || part.type === "function_call") {
              convertedToolCalls.push({
                id: part.id || part.tool_call_id || `call_${Math.random().toString(36).slice(2, 9)}`,
                type: "function",
                function: {
                  name: part.name || part.function?.name || "tool",
                  arguments: typeof part.arguments === "string" ? part.arguments : JSON.stringify(part.arguments ?? {}),
                },
              });
            } else {
              nonThinkingParts.push(part);
            }
          } else {
            nonThinkingParts.push(part);
          }
        }

        if (convertedToolCalls.length > 0) {
          if (!Array.isArray(msg.tool_calls) || msg.tool_calls.length === 0) {
            msg.tool_calls = convertedToolCalls;
          }
        }

        if (thinkingParts.length > 0) {
          extractedThinking = thinkingParts.join("\n");
        }

        if (nonThinkingParts.length === 1 && nonThinkingParts[0].type === "text") {
          textContent = nonThinkingParts[0].text || "";
        } else if (nonThinkingParts.length > 0) {
          textContent = nonThinkingParts
            .map((p) => (typeof p === "string" ? p : p?.text || ""))
            .filter(Boolean)
            .join("\n");
        }
      } else if (typeof msg.content === "string") {
        textContent = msg.content;
      }

      const hadRc = typeof msg.reasoning_content === "string";
      const hasValidRc = hadRc && msg.reasoning_content.trim().length > 0;
      const validRc = hasValidRc ? msg.reasoning_content.trim() : undefined;
      const validExtracted = typeof extractedThinking === "string" && extractedThinking.trim().length > 0 ? extractedThinking.trim() : undefined;
      const fallbackReasoning = isToolTurn ? "Executing tools..." : (hadRc && isDeepSeek ? "Thinking..." : undefined);
      const reasoning = validRc || validExtracted || fallbackReasoning;

      if (reasoning) {
        msg.reasoning_content = reasoning;
      } else {
        delete msg.reasoning_content;
      }

      if (textContent && textContent.includes("[Tool Call]:")) {
        textContent = textContent.replace(/\[Tool Call\]:[^\n]+(\n|$)/g, "").trim();
      }

      msg.content = textContent;
      if (isDeepSeek && (!msg.content || msg.content.trim() === "") && !isToolTurn) {
        msg.content = "[Interrupted]";
      }
    }

    if (isDeepSeek) {
      if (typeof msg.content === "string") {
        msg.content = sanitizeDeepSeekText(msg.content);
      } else if (Array.isArray(msg.content)) {
        for (const block of msg.content) {
          if (block && typeof block === "object") {
            if (typeof block.text === "string") {
              block.text = sanitizeDeepSeekText(block.text);
            }
            if (typeof block.content === "string") {
              block.content = sanitizeDeepSeekText(block.content);
            } else if (Array.isArray(block.content)) {
              for (const sub of block.content) {
                if (sub && typeof sub === "object" && typeof sub.text === "string") {
                  sub.text = sanitizeDeepSeekText(sub.text);
                }
              }
            }
            if (typeof block.thinking === "string") {
              block.thinking = sanitizeDeepSeekText(block.thinking);
            }
          }
        }
      }
      if (Array.isArray(msg.tool_calls)) {
        for (const tc of msg.tool_calls) {
          if (tc?.function && typeof tc.function.arguments === "string") {
            tc.function.arguments = sanitizeToolCallArguments(tc.function.arguments);
          }
        }
      }
      if (msg.role === "assistant") {
        const hasTools = (Array.isArray(msg.tool_calls) && msg.tool_calls.length > 0);
        if (msg.reasoning_content && typeof msg.reasoning_content === "string") {
          msg.reasoning_content = sanitizeDeepSeekText(msg.reasoning_content).trim();
          if (msg.reasoning_content.length === 0) {
            if (hasTools) {
              msg.reasoning_content = "Executing tools...";
            } else {
              msg.reasoning_content = "Thinking...";
            }
          }
        } else if (hasTools) {
          msg.reasoning_content = "Executing tools...";
        } else {
          delete msg.reasoning_content;
        }
      }
    }
  }

  enforceToolCallPairing(messages);
  if (latestUser) latestUser.content = latestContent;
}

export function frameUserTurnsForDeepSeek(messages: any[]): void {
  if (!Array.isArray(messages)) return;
  const latestUser = messages[lastHumanUserIndex(messages)];
  for (const msg of messages) {
    if (!msg || typeof msg !== "object") continue;
    if (msg.role !== "user") continue;
    if (msg === latestUser) {
      // Keep the stable preamble while preserving the user's exact text.
      const prefix = (text: string) => {
        const cleaned = text.replace(/Ключевое/g, "Главное").replace(/ключевое/g, "главное");
        return cleaned.startsWith(LANGUAGE_PREAMBLE) ? cleaned : `${LANGUAGE_PREAMBLE}\n\n${cleaned}`;
      };
      if (typeof msg.content === "string") msg.content = prefix(msg.content);
      else if (Array.isArray(msg.content)) {
        const block = msg.content.find((b: any) => b && (b.type === "text" || b.type === "input_text") && typeof b.text === "string");
        if (block) block.text = prefix(block.text);
      }
      continue;
    }

    if (typeof msg.content === "string") {
      const cleaned = sanitizeDeepSeekText(cleanContent(msg.content));
      if (cleaned === REDACTED_NOTE) {
        msg.content = REDACTED_NOTE;
        continue;
      }
      if (!cleaned.startsWith(LANGUAGE_PREAMBLE)) {
        msg.content = cleaned ? `${LANGUAGE_PREAMBLE}\n\n${cleaned}` : LANGUAGE_PREAMBLE;
      } else {
        msg.content = cleaned;
      }
    } else if (Array.isArray(msg.content)) {
      if (msg.content.length === 0) {
        msg.content.push({ type: "text", text: LANGUAGE_PREAMBLE });
      } else {
        const first = msg.content[0];
        if (first && typeof first === "object" && first.type === "tool_result") {
          continue;
        }
        let added = false;
        for (const block of msg.content) {
          if (block && typeof block === "object" && (block.type === "text" || block.type === "input_text")) {
            if (typeof block.text === "string") {
              const cleaned = sanitizeDeepSeekText(cleanContent(block.text));
              if (cleaned === REDACTED_NOTE) {
                block.text = REDACTED_NOTE;
                added = true;
                break;
              }
              if (!cleaned.startsWith(LANGUAGE_PREAMBLE)) {
                block.text = `${LANGUAGE_PREAMBLE}\n\n${cleaned}`;
              } else {
                block.text = cleaned;
              }
              added = true;
              break;
            }
          }
        }
        if (!added && msg.content.length > 0 && msg.content[0].type !== "tool_result") {
          msg.content.unshift({ type: "text", text: LANGUAGE_PREAMBLE });
        }
      }
    }
  }
}

export function extractCookieFromResponse(res: Response): string | null {
  try {
    const rawSetCookie = res.headers.get("set-cookie");
    if (rawSetCookie) {
      const match = rawSetCookie.match(/(acw_tc=[^;]+)/i);
      if (match) {
        return match[1];
      }
    }
  } catch {}
  return null;
}

export function attachCookieToHeaders(headers: unknown, cookieStr: string): void {
  if (!headers || !cookieStr) return;
  if (headers instanceof Headers) {
    const existing = headers.get("cookie");
    if (!existing) {
      headers.set("cookie", cookieStr);
    } else if (!existing.includes(cookieStr.split("=")[0])) {
      headers.set("cookie", `${existing}; ${cookieStr}`);
    }
    return;
  }
  if (Array.isArray(headers)) {
    let found = false;
    for (const pair of headers) {
      if (typeof pair[0] === "string" && pair[0].toLowerCase() === "cookie") {
        found = true;
        if (!pair[1].includes(cookieStr.split("=")[0])) {
          pair[1] = `${pair[1]}; ${cookieStr}`;
        }
        break;
      }
    }
    if (!found) {
      headers.push(["Cookie", cookieStr]);
    }
    return;
  }
  if (typeof headers === "object") {
    let found = false;
    for (const [k, v] of Object.entries(headers)) {
      if (k.toLowerCase() === "cookie") {
        found = true;
        if (typeof v === "string" && !v.includes(cookieStr.split("=")[0])) {
          (headers as any)[k] = `${v}; ${cookieStr}`;
        }
        break;
      }
    }
    if (!found) {
      (headers as any)["Cookie"] = cookieStr;
    }
  }
}

export function getHeader(headers: unknown, name: string): string | undefined {
  if (!headers) return undefined;
  const target = name.toLowerCase();
  if (headers instanceof Headers) {
    return headers.get(target) || undefined;
  }
  if (Array.isArray(headers)) {
    for (const [k, v] of headers) {
      if (typeof k === "string" && k.toLowerCase() === target) {
        return typeof v === "string" ? v : String(v);
      }
    }
    return undefined;
  }
  if (typeof headers === "object") {
    for (const [k, v] of Object.entries(headers)) {
      if (k.toLowerCase() === target) {
        return typeof v === "string" ? v : String(v);
      }
    }
  }
  return undefined;
}

export function setHeader(headers: unknown, name: string, value: string): void {
  if (!headers) return;
  const target = name.toLowerCase();
  if (headers instanceof Headers) {
    headers.set(name, value);
    return;
  }
  if (Array.isArray(headers)) {
    for (let i = 0; i < headers.length; i++) {
      if (typeof headers[i][0] === "string" && headers[i][0].toLowerCase() === target) {
        headers[i][1] = value;
        return;
      }
    }
    headers.push([name, value]);
    return;
  }
  if (typeof headers === "object") {
    for (const k of Object.keys(headers)) {
      if (k.toLowerCase() === target) {
        (headers as Record<string, any>)[k] = value;
        return;
      }
    }
    (headers as Record<string, any>)[name] = value;
  }
}

export function removeHeader(headers: unknown, name: string): void {
  if (!headers) return;
  const target = name.toLowerCase();
  if (headers instanceof Headers) {
    headers.delete(name);
    return;
  }
  if (Array.isArray(headers)) {
    for (let i = headers.length - 1; i >= 0; i--) {
      if (typeof headers[i][0] === "string" && headers[i][0].toLowerCase() === target) {
        headers.splice(i, 1);
      }
    }
    return;
  }
  if (typeof headers === "object") {
    for (const k of Object.keys(headers)) {
      if (k.toLowerCase() === target) {
        delete (headers as Record<string, any>)[k];
      }
    }
  }
}

export function wrapStreamWithBuffer(
  reader: ReadableStreamDefaultReader<Uint8Array>,
  buffered: Uint8Array[]
): ReadableStream<Uint8Array> {
  return new ReadableStream<Uint8Array>({
    async pull(controller) {
      if (buffered.length > 0) {
        controller.enqueue(buffered.shift()!);
        return;
      }
      const { value, done } = await reader.read();
      if (done) {
        controller.close();
      } else {
        controller.enqueue(value);
      }
    },
    cancel(reason) {
      return reader.cancel(reason);
    },
  });
}

export function buildModelsFromPricing(apiPricing: ApiPricingModel[] | null): {
    openaiModels: any[];
    claudeModels: any[];
    newModels: string[];
  } {
    const openaiModels: any[] = [];
    const claudeModels: any[] = [];
    const newModels: string[] = [];
    const processed = new Set<string>();

    if (apiPricing && apiPricing.length > 0) {
      for (const item of apiPricing) {
        const id = item.model_name;
        processed.add(id);
        const spec = KNOWN_MODEL_SPECS[id];

        const inCost = (item.model_ratio * 2.0) / 1_000_000;
        const outCost = (item.model_ratio * item.completion_ratio * 2.0) / 1_000_000;

        if (spec) {
          const modelObj: any = {
            id: spec.id,
            name: spec.name,
            reasoning: spec.reasoning,
            input: spec.input || ["text"],
            contextWindow: spec.contextWindow,
            maxTokens: spec.maxTokens,
            cost: { input: inCost, output: outCost, cacheRead: 0, cacheWrite: 0 },
            compat: spec.compat || { sendSessionAffinityHeaders: true },
          };
          if (spec.thinkingLevelMap) {
            modelObj.thinkingLevelMap = { ...spec.thinkingLevelMap };
          }
          if (spec.api) {
            modelObj.api = spec.api;
          }
          if (spec.headers) {
            modelObj.headers = { ...spec.headers };
          }
          if (spec.providerType === "anthropic") {
            claudeModels.push(modelObj);
          } else {
            openaiModels.push(modelObj);
          }
        } else {
          newModels.push(id);
          const isAnthropic =
            id.toLowerCase().startsWith("claude") ||
            (item.supported_endpoint_types.includes("anthropic") && !item.supported_endpoint_types.includes("openai"));
          const isVision =
            id.toLowerCase().includes("vision") ||
            id.toLowerCase().startsWith("claude") ||
            id.toLowerCase().startsWith("gpt-6");
          const modelObj = {
            id,
            name: id,
            reasoning: true,
            input: isVision ? ["text", "image"] : ["text"],
            contextWindow: 131072,
            maxTokens: 16384,
            cost: { input: inCost, output: outCost, cacheRead: 0, cacheWrite: 0 },
            compat: isAnthropic
              ? {
                  forceAdaptiveThinking: true,
                  allowEmptySignature: true,
                  sendSessionAffinityHeaders: true,
                }
              : {
                  sendSessionAffinityHeaders: true,
                },
          };
          if (isAnthropic) {
            claudeModels.push(modelObj);
          } else {
            openaiModels.push(modelObj);
          }
        }
      }
    }

    // Register any known non-decommissioned models that were not in live pricing
    // (e.g. deepseek-v4-flash when temporarily unlisted on gateway) so that settings.json
    // and enabledModels do not trigger "No models match pattern" warnings.
    for (const [id, spec] of Object.entries(KNOWN_MODEL_SPECS)) {
      if (DECOMMISSIONED_MODEL_IDS.has(id)) continue;
      if (!processed.has(id)) {
        processed.add(id);
        const modelObj: any = {
          id: spec.id,
          name: spec.name,
          reasoning: spec.reasoning,
          input: spec.input || ["text"],
          contextWindow: spec.contextWindow,
          maxTokens: spec.maxTokens,
          cost: spec.cost || { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
          compat: spec.compat || { sendSessionAffinityHeaders: true },
        };
        if (spec.thinkingLevelMap) {
          modelObj.thinkingLevelMap = { ...spec.thinkingLevelMap };
        }
        if (spec.api) {
          modelObj.api = spec.api;
        }
        if (spec.headers) {
          modelObj.headers = { ...spec.headers };
        }
        if (spec.providerType === "anthropic") {
          claudeModels.push(modelObj);
        } else {
          openaiModels.push(modelObj);
        }
      }
    }

    return { openaiModels, claudeModels, newModels };
  }

export interface AgentRouterConfig {
  apiKey?: string;
  baseURL?: string;
  minIntervalMs?: number;
  maxRetries?: number;
}
export function configDir(env: NodeJS.ProcessEnv = process.env): string {
  return join(env.XDG_CONFIG_HOME || join(os.homedir(), ".config"), "opencode");
}
export function stateDir(env: NodeJS.ProcessEnv = process.env): string {
  return join(env.XDG_STATE_HOME || join(os.homedir(), ".local", "state"), "opencode", "agentrouter");
}
export function normalizeApiKey(key?: string): string {
  return (key ?? "").trim().replace(/^["']|["']$/g, "").trim();
}
export async function readJson(file: string): Promise<unknown> {
  try { return JSON.parse(await readFile(file, "utf8")); }
  catch (err) { if ((err as NodeJS.ErrnoException).code === "ENOENT") return undefined; throw err; }
}
export async function writeJson(file: string, value: unknown): Promise<void> {
  await mkdir(dirname(file), { recursive: true });
  const temp = `${file}.${randomUUID()}.tmp`;
  await writeFile(temp, JSON.stringify(value, null, 2) + "\n", { mode: 0o600 });
  await rename(temp, file);
}
export function validateConfig(value: unknown): AgentRouterConfig {
  if (value === undefined) return {};
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error("AgentRouter config must be an object");
  const c = value as AgentRouterConfig;
  if (c.apiKey !== undefined && typeof c.apiKey !== "string") throw new Error("apiKey must be a string");
  if (c.minIntervalMs !== undefined && (!Number.isFinite(c.minIntervalMs) || c.minIntervalMs < 0)) throw new Error("minIntervalMs must be a non-negative finite number");
  if (c.maxRetries !== undefined && (!Number.isSafeInteger(c.maxRetries) || c.maxRetries < 0)) throw new Error("maxRetries must be a non-negative integer");
  if (c.baseURL !== undefined) {
    const url = new URL(c.baseURL);
    if (!["http:", "https:"].includes(url.protocol) || url.username || url.password || url.search || url.hash) throw new Error("baseURL must be an HTTP(S) origin or path without credentials, query or fragment");
  }
  return c;
}
export async function loadConfig(file = join(configDir(), "agentrouter.json")): Promise<AgentRouterConfig> {
  return validateConfig(await readJson(file));
}
export function effectiveConfig(c: AgentRouterConfig) {
  return { ...c, baseURL: (c.baseURL ?? BASE_URL).replace(/\/+$/, ""), minIntervalMs: c.minIntervalMs ?? DEFAULT_PACING_MS, maxRetries: c.maxRetries ?? DEFAULT_MAX_RETRIES };
}
export function environmentKey(env: NodeJS.ProcessEnv = process.env): string {
  return normalizeApiKey(env.AGENTROUTER_API_KEY || env.AGENT_ROUTER_API_KEY);
}
export async function loadCachedPricing(dir = stateDir()): Promise<ApiPricingModel[] | null> {
  try { return parsePricing(await readJson(join(dir, ".agentrouter-models-cache.json"))); }
  catch { return null; }
}
export async function saveCachedPricing(pricing: ApiPricingModel[], dir = stateDir()): Promise<void> {
  await writeJson(join(dir, ".agentrouter-models-cache.json"), pricing);
}
export async function saveConfig(config: AgentRouterConfig, file = join(configDir(), "agentrouter.json")): Promise<void> {
  await writeJson(file, validateConfig(config));
}

export function getOpenCodeUserAgent(): string {
  return `opencode-agentrouter (${os.platform()} ${os.release()}; ${os.arch()})`;
}

export function isAgentRouterUrl(urlStr?: string, baseURL = BASE_URL): boolean {
  if (!urlStr) return false;
  try {
    const url = new URL(urlStr);
    return url.origin === new URL(baseURL).origin ||
      ((url.protocol === "https:" || url.protocol === "http:") && ["agentrouter.org", "air-outer.com"].includes(url.hostname));
  } catch { return false; }
}

export function isAgentRouter(providerName?: string, baseURL?: string): boolean {
  return Object.values(PROVIDERS).some(id => id === providerName) || isAgentRouterUrl(baseURL);
}

export function parsePricing(value: unknown): ApiPricingModel[] | null {
  if (!Array.isArray(value)) return null;
  return value.filter((m): m is ApiPricingModel => m && typeof m.model_name === "string" &&
    Number.isFinite(m.model_ratio) && m.model_ratio >= 0 && Number.isFinite(m.completion_ratio) && m.completion_ratio >= 0 &&
    Array.isArray(m.supported_endpoint_types) && m.supported_endpoint_types.every((s: unknown) => typeof s === "string"));
}

// Convert only at the OpenCode boundary: upstream costs are per token, OpenCode costs per million.
export interface OpenCodeModelSpec {
  id: string;
  name: string;
  protocol: Protocol;
  context: number;
  output: number;
  image: boolean;
  reasoning: boolean;
  cost: { input: number; output: number; cache_read?: number; cache_write?: number };
}
export function toOpenCodeModelSpecs(pricing: ApiPricingModel[] | null): OpenCodeModelSpec[] {
  const { openaiModels, claudeModels } = buildModelsFromPricing(pricing);
  return [...openaiModels, ...claudeModels].map(model => ({
    id: model.id, name: model.name,
    protocol: claudeModels.includes(model) ? "anthropic" : model.api === "openai-responses" ? "responses" : "chat",
    context: model.contextWindow, output: model.maxTokens,
    image: model.input.includes("image"), reasoning: model.reasoning,
    cost: { input: model.cost.input * 1_000_000, output: model.cost.output * 1_000_000,
      cache_read: model.cost.cacheRead * 1_000_000, cache_write: model.cost.cacheWrite * 1_000_000 },
  }));
}
export function providerFor(model: OpenCodeModelSpec): string {
  return model.protocol === "anthropic" ? PROVIDERS.anthropic : PROVIDERS.openai;
}

export async function sleep(ms: number, signal?: AbortSignal): Promise<void> {
  signal?.throwIfAborted();
  if (ms <= 0) return;
  await new Promise<void>((resolve, reject) => {
    const done = () => { signal?.removeEventListener("abort", abort); resolve(); };
    const timer = setTimeout(done, ms);
    const abort = () => { clearTimeout(timer); reject(signal?.reason ?? new DOMException("Aborted", "AbortError")); };
    signal?.addEventListener("abort", abort, { once: true });
  });
}
export class Pacer {
  private queue: Promise<void> = Promise.resolve();
  private last = 0;
  constructor(private dir: string, private interval: () => number, private wait = sleep, private now = Date.now) {}
  async enter(signal?: AbortSignal): Promise<void> {
    const task = this.queue.then(async () => {
      signal?.throwIfAborted();
      this.last = Math.max(this.last, await getLastRequestEndTime("agentrouter", this.dir));
      const remaining = Math.max(0, this.last + this.interval() - this.now());
      await this.wait(remaining, signal);
      this.last = this.now();
      await setLastRequestEndTime(this.last, "agentrouter", this.dir);
    });
    this.queue = task.catch(() => {});
    return task;
  }
  async end(): Promise<void> {
    this.last = this.now();
    await setLastRequestEndTime(this.last, "agentrouter", this.dir);
  }
}
export type Failure = "rate" | "waf" | "cluster" | "thinking" | "schema" | "pairing" | "challenge" | "service" | "auth";
export function classifyFailure(status: number, text: string, attempt = 0): Failure | null {
  const s = text.toLowerCase();
  // Never retry authentication/authorization or exhausted quota except upstream's explicit transient token-cache error.
  if (status === 402 || status === 403) return null;
  if (status === 401) return attempt < 2 && (text.includes("无效的令牌") || (s.includes("new_api_error") && s.includes("token"))) ? "auth" : null;
  if (status === 429 || /rate[_ ]limit|tokens per minute|token rate limit|tpm|rpm/.test(s)) return "rate";
  if (WAF_BLOCK_RE.test(s)) return "waf";
  if (/different azure.*resource|use the same resource|could not be decrypted|decryption[_ ]failure|encrypted content|cross[-_ ]resource|cross resource|cross-origin/.test(s)) return "cluster";
  if (status === 400 && /reasoning_content|content\[\]\.thinking|thinking mode must be passed back/.test(s)) return "thinking";
  if (status === 422 || /failed to deserialize|unknown variant|expected one of `text`/.test(s)) return "schema";
  if (status === 400 && /tool_result.*tool_use|unexpected.*tool_use_id/.test(s)) return "pairing";
  if (status === 405 || (status === 400 && s.includes("blocked"))) return "challenge";
  if ([502, 503, 504].includes(status) || (status === 500 && s.includes("temporarily unavailable"))) return "service";
  return null;
}
export function retryDelay(failure: Failure, attempt: number, headers: Headers, text: string, pacing: number, now = Date.now(), random = Math.random): number {
  if (failure === "rate") {
    const ms = Number(headers.get("retry-after-ms"));
    const after = headers.get("retry-after");
    const seconds = after !== null ? Number(after) : NaN;
    const date = after !== null && !Number.isFinite(seconds) ? Date.parse(after) : NaN;
    const hint = text.match(/retry (?:after|in) (\d+(?:\.\d+)?) seconds?/i);
    // Server Retry-After is a minimum, not capped by the upstream's local 60s backoff cap.
    const server = ms > 0 ? ms : Number.isFinite(seconds) ? seconds * 1000 : Number.isFinite(date) ? date - now : hint ? Number(hint[1]) * 1000 : 0;
    return Math.max(server, Math.min(60000, Math.round(Math.max(pacing, 4000) * 1.6 ** attempt + random() * 2000)));
  }
  if (failure === "challenge") return Math.min(30000, Math.round(Math.max(pacing, 3000) * 1.5 ** attempt + random() * 1000));
  if (failure === "auth") return Math.round(1500 * 1.5 ** attempt + random() * 500);
  return Math.min(30000, Math.round(Math.max(pacing, 2000) * 1.4 ** attempt + random() * 800));
}
export async function inspectStreamInitialChunks(response: Response): Promise<{ response: Response; error?: string }> {
  if (!response.body) return { response };
  const reader = response.body.getReader();
  const chunks: Uint8Array[] = [];
  const decoder = new TextDecoder();
  let pending = "";
  const restore = () => ({ response: new Response(wrapStreamWithBuffer(reader, chunks), { status: response.status, statusText: response.statusText, headers: response.headers }) });
  try {
    while (true) {
      const next = await reader.read();
      if (next.done) return restore();
      chunks.push(next.value); pending += decoder.decode(next.value, { stream: true });
      // Parse complete SSE frames; keyword matching model output would cause false retries.
      let match: RegExpExecArray | null;
      while ((match = /\r?\n\r?\n/.exec(pending))) {
        const frame = pending.slice(0, match.index); pending = pending.slice(match.index + match[0].length);
        const data = frame.split(/\r?\n/).filter(l => l.startsWith("data:")).map(l => l.slice(5).trimStart()).join("\n");
        if (!data) continue;
        if (data === "[DONE]") return restore();
        let value: Json;
        try { value = JSON.parse(data); } catch { continue; }
        const type = value.type ?? frame.match(/^event:\s*(.+)$/m)?.[1];
        if (value.error || type === "error" || type === "response.failed" || value.response?.status === "failed") {
          await reader.cancel("Upstream failure before generation");
          return { response, error: JSON.stringify(value) };
        }
        if (value.choices?.some((c: Json) => c.delta?.content || c.delta?.reasoning_content || c.delta?.tool_calls?.length || c.finish_reason) ||
          type === "content_block_delta" || type === "content_block_start" || type === "message_stop" ||
          ["response.output_text.delta", "response.reasoning_summary_text.delta", "response.function_call_arguments.delta", "response.completed", "response.incomplete"].includes(type)) return restore();
      }
    }
  } catch (err) {
    await reader.cancel(err).catch(() => {});
    return { response, error: err instanceof Error ? err.message : String(err) };
  }
}
function withCompletion(response: Response, complete: () => Promise<void>): Response {
  if (!response.body) { void complete(); return response; }
  const reader = response.body.getReader();
  let ended = false;
  const end = async () => { if (!ended) { ended = true; await complete(); } };
  const stream = new ReadableStream<Uint8Array>({
    async pull(c) {
      try { const next = await reader.read(); if (next.done) { await end(); c.close(); } else c.enqueue(next.value); }
      catch (err) { await end(); c.error(err); }
    },
    async cancel(reason) { try { await reader.cancel(reason); } finally { await end(); } },
  });
  return new Response(stream, { status: response.status, statusText: response.statusText, headers: response.headers });
}
export interface TransportOptions {
  baseURL: string;
  stateDir: string;
  minIntervalMs?: number | (() => number);
  maxRetries?: number;
  fetch?: Fetch;
  wait?: typeof sleep;
  pacer?: Pacer;
  log?: (message: string) => void | Promise<void>;
}
export function installAgentRouterFetchHook(options: TransportOptions): Fetch {
  const rawFetch = options.fetch ?? globalThis.fetch.bind(globalThis);
  const wait = options.wait ?? sleep;
  const interval = options.minIntervalMs ?? DEFAULT_PACING_MS;
  const currentPacing = typeof interval === "function" ? interval : () => interval;
  const retries = options.maxRetries ?? DEFAULT_MAX_RETRIES;
  const pacer = options.pacer ?? new Pacer(options.stateDir, currentPacing, wait);
  const origin = new URL(options.baseURL).origin;
  const sessions = new Map<string, { redaction: PoisonRedactionState; cookie: string; salt: number }>();
  return async (input, init) => {
    const url = new URL(input instanceof Request ? input.url : input.toString());
    // Scope by exact origin; never leak keys/cookies or pace unrelated provider traffic.
    if (url.origin !== origin) return rawFetch(input, init);
    const request = new Request(input, init);
    const signal = request.signal;
    signal.throwIfAborted();
    const text = request.method === "GET" || request.method === "HEAD" ? "" : await request.text();
    let source: Json | undefined;
    try { const parsed = JSON.parse(text); if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) source = parsed; } catch {}
    const generation = /\/(?:chat\/completions|responses|messages)$/.test(url.pathname) && source;
    const headers = new Headers(request.headers);
    // Preserve OpenCode/SDK headers and expose the body to the host's fetch middleware.
    if (!generation || !source) return rawFetch(request, { headers, ...(text ? { body: text } : {}), signal });
    const protocol: Protocol = url.pathname.endsWith("/responses") ? "responses" : url.pathname.endsWith("/messages") ? "anthropic" : "chat";
    const session = headers.get("session_id") ?? headers.get("x-opencode-session") ?? source.prompt_cache_key ?? "default";
    const key = `${headers.get("authorization") ?? headers.get("x-api-key") ?? ""}:${session}`;
    let state = sessions.get(key);
    if (!state) { state = { redaction: new PoisonRedactionState(), cookie: "", salt: 0 }; sessions.set(key, state); }
    // In-flight policy stays stable; later calls observe pacing changes without losing session state.
    const pacing = currentPacing();
    for (let attempt = 0; ; attempt++) {
      signal.throwIfAborted();
      const body = beforeProviderRequest(source, protocol, state.redaction);
      if (state.salt) {
        if (typeof body.prompt_cache_key === "string") body.prompt_cache_key = `${body.prompt_cache_key.replace(/-v\d+$/, "")}-v${state.salt}`;
        headers.delete("cookie"); headers.delete("x-session-affinity"); headers.delete("affinity");
        if (headers.has("session_id")) headers.set("session_id", `${String(session).replace(/-r(?:l)?\d+$/, "")}-rl${state.salt}`);
      }
      if (state.cookie && !headers.has("cookie")) headers.set("cookie", state.cookie);
      if (attempt) headers.set("x-client-request-id", randomUUID());
      await pacer.enter(signal);
      let response: Response;
      try { response = await rawFetch(request, { headers, body: JSON.stringify(body), signal }); }
      catch (err) {
        signal.throwIfAborted();
        response = new Response(JSON.stringify({ error: { message: err instanceof Error ? err.message : String(err) } }), { status: 503, headers: { "Content-Type": "application/json" } });
      }
      const cookie = extractCookieFromResponse(response);
      if (cookie) state.cookie = cookie;
      let errorText = "";
      let status = response.status;
      let streamError = false;
      if (response.ok && response.headers.get("content-type")?.includes("text/event-stream")) {
        const inspected = await inspectStreamInitialChunks(response);
        response = inspected.response;
        if (inspected.error) { streamError = true; errorText = inspected.error; status = classifyFailure(400, errorText) === "rate" ? 429 : 400; }
      } else if (!response.ok) errorText = await response.clone().text();
      if (response.ok && !streamError) return withCompletion(response, () => pacer.end());
      const failure = classifyFailure(status, errorText, attempt);
      if (!failure || attempt >= retries) {
        if (streamError) response = new Response(JSON.stringify({ error: { message: errorText, type: "upstream_error" } }), { status, headers: { "Content-Type": "application/json" } });
        return withCompletion(response, () => pacer.end());
      }
      if (failure === "waf") {
        // Retry only if another history turn can be hidden. Rephrasing the latest user is up to the user.
        const before = beforeProviderRequest(source, protocol, state.redaction);
        const after = beforeProviderRequest(source, protocol, state.redaction, true);
        if (JSON.stringify(before) === JSON.stringify(after)) {
          if (streamError) response = new Response(JSON.stringify({ error: { message: errorText } }), { status, headers: { "Content-Type": "application/json" } });
          return withCompletion(response, () => pacer.end());
        }
      }
      state.cookie = ""; headers.delete("cookie"); headers.delete("x-session-affinity"); headers.delete("affinity");
      if (failure === "rate" || failure === "cluster") state.salt++;
      const delay = retryDelay(failure, attempt, response.headers, errorText, pacing);
      await response.body?.cancel().catch(() => {});
      await pacer.end();
      // Diagnostics intentionally omit request bodies, credentials and upstream error echoes.
      await options.log?.(`Gateway ${failure} (${status}); retry ${attempt + 1}/${retries} after ${delay}ms`);
      await arDebugLog(`Gateway ${failure} (${status}); retry ${attempt + 1}/${retries} after ${delay}ms`, options.stateDir);
      await wait(delay, signal);
    }
  };
}

export function cleanJsonSchemaObject(schema: unknown): void {
  if (!isRecord(schema) || Array.isArray(schema)) return;
  delete schema.$schema;
  if ((schema.type === "object" || schema.properties) && !Array.isArray(schema.required)) schema.required = [];
  if (schema.items === null) delete schema.items;
  // OpenCode tools use nested union schemas. Traverse schemas, never const/default/example data.
  for (const [key, value] of Object.entries(schema)) {
    if (["properties", "$defs", "definitions", "patternProperties", "dependentSchemas"].includes(key) && isRecord(value)) Object.values(value).forEach(cleanJsonSchemaObject);
    if (["items", "additionalProperties", "not", "if", "then", "else", "contains", "propertyNames"].includes(key)) cleanJsonSchemaObject(value);
    if (["allOf", "anyOf", "oneOf", "prefixItems"].includes(key) && Array.isArray(value)) value.forEach(cleanJsonSchemaObject);
  }
}

export function sanitizeOpenAiTools(tools: any[], preserveCacheAnnotations = false): void {
  if (!Array.isArray(tools)) return;
  for (const tool of tools) {
    if (!isRecord(tool)) continue;
    const fn = (tool.function || tool) as Json;
    cleanJsonSchemaObject(fn.parameters ?? fn.input_schema);
    // Cache annotations are siblings of schemas, not arbitrary properties inside them.
    if (!preserveCacheAnnotations) {
      delete tool.cache_control;
      if (isRecord(tool.function)) delete tool.function.cache_control;
    }
  }
}

export function sanitizeResponsesInput(input: any[]): void {
  if (!Array.isArray(input)) return;
  for (let i = input.length - 1; i >= 0; i--) {
    const item = input[i];
    if (!isRecord(item)) continue;
    if (item.type === "reasoning" || "encrypted_content" in item ||
      (typeof item.id === "string" && item.id.startsWith("rs_")) ||
      (item.role === "assistant" && item.status === "error")) {
      input.splice(i, 1); continue;
    }
    delete item.id;
    if (Array.isArray(item.content)) sanitizeResponsesInput(item.content);
  }
}

function stripPromptCacheAnnotations(messages: Json[]): void {
  for (const message of messages) {
    if (!isRecord(message)) continue;
    delete message.cache_control;
    if (!Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (!isRecord(block)) continue;
      delete block.cache_control;
      if (block.type === "tool_result" && Array.isArray(block.content)) {
        for (const part of block.content) if (isRecord(part)) delete part.cache_control;
      }
    }
  }
}

export function beforeProviderRequest(source: Json, protocol: Protocol, state?: PoisonRedactionState, escalate = false): Json {
  const payload = structuredClone(source);
  const isDeepSeek = String(payload.model ?? "").toLowerCase().includes("deepseek");
  if (isDeepSeek) delete payload.thinking;
  if (Array.isArray(payload.tools)) sanitizeOpenAiTools(payload.tools, protocol === "anthropic");
  if (protocol === "responses") {
    sanitizeResponsesInput(payload.input);
    delete payload.previous_response_id;
    delete payload.prompt_cache_retention;
    if (Array.isArray(payload.include)) payload.include = payload.include.filter((item: string) => item !== "reasoning.encrypted_content");
  }
  const messages = protocol === "responses" ? payload.input : payload.messages;
  if (Array.isArray(messages)) {
    if (isCompactionPayload(payload)) sanitizeCompactionPayload(payload);
    if (state) {
      if (escalate) state.triggerEscalation(true);
      state.applyPoisonRedaction(payload);
    }
    if (protocol === "chat") {
      normalizeMessagesForAgentRouter(messages, isDeepSeek);
      if (isDeepSeek) frameUserTurnsForDeepSeek(messages);
      if (messages[0]?.role === "system") messages[0].content = enforceCanonicalRootPrompt(messages[0].content);
      else messages.unshift({ role: "system", content: CANONICAL_OPENCODE_HEADER });
    } else if (protocol === "anthropic") {
      // The OpenCode Anthropic SDK already emits native tool/thinking blocks. Do not flatten them to Chat.
      enforceToolCallPairing(messages, true);
      const latest = lastHumanUserIndex(messages);
      messages.forEach((message: Json, index: number) => {
        if (index === latest || !Array.isArray(message.content)) return;
        for (const block of message.content) {
          if (block.type !== "tool_result") continue;
          if (typeof block.content === "string" && isBinaryDump(block.content)) block.content = REDACTED_NOTE;
          else if (Array.isArray(block.content)) for (const part of block.content) {
            if (part.type === "text" && typeof part.text === "string" && isBinaryDump(part.text)) part.text = REDACTED_NOTE;
          }
        }
      });
    }
    if (protocol !== "anthropic") stripPromptCacheAnnotations(messages);
  }
  if (protocol === "anthropic") {
    payload.system = enforceCanonicalRootPrompt(payload.system);
    for (const message of payload.messages ?? []) for (const block of Array.isArray(message.content) ? message.content : []) {
      if (block.type === "thinking" && !block.signature) block.signature = "";
    }
  }
  return payload;
}

export async function fetchLivePricing(baseURL = BASE_URL, fetcher: Fetch = fetch): Promise<ApiPricingModel[] | null> {
  try {
    const response = await fetcher(`${baseURL}/api/pricing`, { headers: { "User-Agent": getOpenCodeUserAgent() } });
    if (!response.ok) return null;
    const data = await response.json() as { data?: unknown };
    const pricing = parsePricing(data.data);
    return pricing?.length ? pricing : null;
  } catch { return null; }
}
export async function fetchTokenUsage(apiKey: string, baseURL = BASE_URL, fetcher: Fetch = fetch, date = new Date()): Promise<number | null> {
  if (!apiKey) return null;
  const y = date.getFullYear();
  const m = date.getMonth();
  const month = String(m + 1).padStart(2, "0");
  const endDay = new Date(y, m + 1, 0).getDate();
  const url = `${baseURL}/v1/dashboard/billing/usage?start_date=${y}-${month}-01&end_date=${y}-${month}-${endDay}`;
  try {
    const response = await fetcher(url, { headers: { Authorization: `Bearer ${apiKey}`, "User-Agent": getOpenCodeUserAgent() } });
    if (!response.ok) return null;
    const data = await response.json() as { total_usage?: unknown };
    return typeof data.total_usage === "number" && Number.isFinite(data.total_usage) ? data.total_usage : null;
  } catch { return null; }
}
export interface ModelProbeResult { model: string; status: "READY" | "QUOTA_EXHAUSTED" | "NOT_IN_TOKEN" | "FORBIDDEN" | "ERROR"; code: number; message?: string }
export async function probeModelQuota(modelId: string, apiKey: string, isAnthropic: boolean, isResponsesApi?: boolean, baseURL = BASE_URL, fetcher: Fetch = fetch): Promise<ModelProbeResult> {
  const model = { id: modelId, protocol: isAnthropic ? "anthropic" : (isResponsesApi ?? (KNOWN_MODEL_SPECS[modelId]?.api === "openai-responses" || modelId.startsWith("gpt-6") || modelId.includes("-sol"))) ? "responses" : "chat" };
  const headers = new Headers({ "Content-Type": "application/json", "User-Agent": getOpenCodeUserAgent() });
  let endpoint: string;
  let body: Record<string, unknown>;
  // Probe output cap of 16 tokens is ported verbatim from upstream.
  if (model.protocol === "anthropic") {
    endpoint = "messages"; headers.set("x-api-key", apiKey); headers.set("anthropic-version", "2023-06-01");
    body = { model: model.id, system: CANONICAL_OPENCODE_HEADER, messages: [{ role: "user", content: "ping" }], max_tokens: 16 };
  } else {
    headers.set("Authorization", `Bearer ${apiKey}`);
    if (model.protocol === "responses") { endpoint = "responses"; body = { model: model.id, input: [{ role: "user", content: "ping" }], max_output_tokens: 16 }; }
    else { endpoint = "chat/completions"; body = { model: model.id, messages: [{ role: "system", content: CANONICAL_OPENCODE_HEADER }, { role: "user", content: "ping" }], max_tokens: 16 }; }
  }
  try {
    const response = await fetcher(`${baseURL}/v1/${endpoint}`, { method: "POST", headers, body: JSON.stringify(body) });
    if (response.ok) { await response.body?.cancel(); return { model: model.id, status: "READY", code: response.status }; }
    const text = await response.text();
    const quota = response.status === 402 || /quota|exhausted|budget pool/i.test(text);
    const permission = response.status === 403 || /无权访问模型|no permission|not allowed to access|model.*not.*enabled/i.test(text);
    const channel = response.status === 503 || /无可用渠道|no available channel/i.test(text);
    const status = quota ? "QUOTA_EXHAUSTED" : permission ? "NOT_IN_TOKEN" : "ERROR";
    // Preserve upstream classifications but never expose arbitrary gateway text that may echo secrets.
    return { model: model.id, status, code: quota ? 402 : permission ? 403 : channel ? 503 : response.status,
      message: permission ? "Model not enabled for this token in AgentRouter dashboard" : channel ? "Channel temporarily paused on gateway (503)" : undefined };
  } catch (err) {
    return { model: model.id, status: "ERROR", code: 0, message: err instanceof Error ? err.name : "Network error" };
  }
}

export interface ManagementOptions {
  configFile?: string;
  stateDirectory?: string;
  getKey?: () => Promise<string>;
  fetch?: Fetch;
}
export async function manage(action: string, value?: string, options: ManagementOptions = {}): Promise<string> {
  action = normalizeCommandAction(action);
  const file = options.configFile ?? join(configDir(), "agentrouter.json");
  const dir = options.stateDirectory ?? stateDir();
  const config = await loadConfig(file);
  const settings = effectiveConfig(config);
  const key = environmentKey() || normalizeApiKey(config.apiKey) || normalizeApiKey(await options.getKey?.());
  const raw = options.fetch ?? fetch;
  if (action === "key") {
    const clean = normalizeApiKey(value);
    if (clean.length < 5) throw new Error("Provide a nonempty API key; prefer /connect → AgentRouter or AGENTROUTER_API_KEY");
    await saveConfig({ ...config, apiKey: clean }, file);
    return "API key saved with owner-only permissions. Restart OpenCode to reload provider credentials.";
  }
  if (action === "pacing") {
    if (value === undefined) return `Current pacing: ${settings.minIntervalMs} ms`;
    const interval = parseInt(value, 10);
    if (!value.trim() || !Number.isFinite(interval) || interval < 0) throw new Error("Pacing must be a non-negative number of milliseconds");
    await saveConfig({ ...config, minIntervalMs: interval }, file);
    return `Pacing set to ${interval} ms. Restart OpenCode to apply.`;
  }
  if (action === "pricing" || action === "sync" || action === "check") {
    const live = await fetchLivePricing(settings.baseURL, raw);
    if (live) await saveCachedPricing(live, dir);
    const pricing = live ?? await loadCachedPricing(dir);
    const models = toOpenCodeModelSpecs(pricing);
    if (action === "sync") return `Catalog ${live ? "refreshed from gateway" : "loaded from cache/upstream defaults"}. Restart OpenCode to register changes.\n${models.map(m => `${providerFor(m)}/${m.id}`).join("\n")}`;
    if (action === "pricing") return `AgentRouter ${live ? "live" : pricing ? "cached" : "upstream fallback"} pricing (USD / million tokens)\n\nModel | Input | Output | Cache read | Protocol\n${models.map(m => `${m.id} | $${m.cost.input.toFixed(2)} | $${m.cost.output.toFixed(2)} | ${m.cost.cache_read === undefined ? "unknown" : `$${m.cost.cache_read.toFixed(2)}`} | ${m.protocol}`).join("\n")}`;
    if (!key) throw new Error("No AgentRouter API key: use /connect → AgentRouter in OpenCode or AGENTROUTER_API_KEY for the CLI");
    // Probes are real generation requests; pacing is shared with normal traffic.
    // Disable retries for diagnostics so a health check reports the actual gateway status.
    const transport = installAgentRouterFetchHook({ baseURL: settings.baseURL, stateDir: dir, minIntervalMs: settings.minIntervalMs, maxRetries: 0, fetch: raw });
    const probeModels = live ? toOpenCodeModelSpecs(live).filter(model => live.some(item => item.model_name === model.id)) :
      FLAGSHIP_MODELS.map(id => models.find(model => `${providerFor(model)}/${model.id}` === id)!).filter(Boolean);
    const [usage, probes] = await Promise.all([
      fetchTokenUsage(key, settings.baseURL, raw),
      Promise.all(probeModels.map(model => probeModelQuota(model.id, key, model.protocol === "anthropic", model.protocol === "responses", settings.baseURL, transport))),
    ]);
    return `AgentRouter health (probes use tokens)\nMonthly spend: ${usage === null ? "unavailable" : `$${usage.toFixed(4)}`}\n${probes.map(p => `${p.model}: ${p.status} (HTTP ${p.code})${p.message ? ` — ${p.message}` : ""}`).join("\n")}`;
  }
  if (action !== "status" && action !== "") throw new Error(`Unknown action: ${action}. Use status, check, pricing, sync, pacing or key.`);
  const usage = key ? await fetchTokenUsage(key, settings.baseURL, raw) : null;
  return `AgentRouter for OpenCode\nEndpoint: ${settings.baseURL}\nAPI key: ${key ? "configured" : "not configured"}\nMonthly spend: ${usage === null ? "unavailable" : `$${usage.toFixed(4)}`}\nPacing: ${settings.minIntervalMs} ms\nMax transport retries: ${settings.maxRetries}\nCommands: /agentrouter_status, /agentrouter_pricing, /agentrouter_sync, /agentrouter_check (uses tokens), /agentrouter_pacing [ms], /agentrouter_key <key>.`;
}

export function normalizeCommandAction(action: string): string {
  switch (action.toLowerCase()) {
    case "set-key": return "key";
    case "enable-models": return "sync";
    case "order": return "fix-order";
    case "throttle": return "pacing";
    case "price": case "models": return "pricing";
    case "quota": case "status-live": return "check";
    default: return action.toLowerCase();
  }
}

// OpenCode provider/lifecycle conversion of upstream default extension registration.
const INTEGRATION_ID = "agentrouter";
function toModel(spec: OpenCodeModelSpec): Model.Info {
  const id = Model.ID.make(spec.id);
  const providerID = Provider.ID.make(providerFor(spec));
  return {
    ...Model.Info.default(providerID, id), name: spec.name,
    enabled: AVAILABLE_MODEL_IDS.has(spec.id),
    package: spec.protocol === "responses" ? "aisdk:@ai-sdk/openai" : spec.protocol === "anthropic" ? "aisdk:@ai-sdk/anthropic" : "aisdk:@ai-sdk/openai-compatible",
    capabilities: { tools: true, input: spec.image ? ["text", "image"] : ["text"], output: ["text"] },
    limit: { context: spec.context, output: spec.output },
    cost: [{ input: Money.USDPerMillionTokens.make(spec.cost.input), output: Money.USDPerMillionTokens.make(spec.cost.output), cache: { read: Money.USDPerMillionTokens.make(spec.cost.cache_read ?? 0), write: Money.USDPerMillionTokens.make(spec.cost.cache_write ?? 0) } }],
    ...(spec.id.includes("deepseek") ? { compatibility: { reasoningField: "reasoning_content", requireReasoning: true, maxTokensField: "max_tokens" } } : {}),
    variants: spec.protocol === "responses" ? ["low", "medium", "high"].map(level => ({ id: Model.VariantID.make(level), body: { reasoning: { effort: level } } })) :
      spec.protocol === "anthropic" ? ["low", "medium", "high"].map(level => ({ id: Model.VariantID.make(level), body: { thinking: { type: "adaptive" }, output_config: { effort: level } } })) : [],
  };
}
export default Plugin.define({
  id: "opencode-agentrouter",
  async setup(ctx) {
    const dir = stateDir();
    let settings = effectiveConfig(await loadConfig());
    const live = await fetchLivePricing(settings.baseURL);
    if (live) await saveCachedPricing(live, dir).catch(() => {});
    let models = toOpenCodeModelSpecs(live ?? await loadCachedPricing(dir));
    const pacer = new Pacer(dir, () => settings.minIntervalMs);
    const isOurs = (providerID: string) => Object.values(PROVIDERS).some(id => id === providerID);
    const getEffectiveApiKey = async () => {
      const env = environmentKey() || normalizeApiKey(settings.apiKey);
      if (env) return env;
      const connection = await ctx.integration.connection.active(INTEGRATION_ID);
      const credential = connection ? await ctx.integration.connection.resolve(connection) : undefined;
      return credential?.type === "key" ? normalizeApiKey(credential.key) : "";
    };
    await ctx.integration.transform(editor => {
      editor.method.update({ integrationID: INTEGRATION_ID, method: { type: "key", label: "AgentRouter API key" } });
      editor.method.update({ integrationID: INTEGRATION_ID, method: { type: "env", names: ["AGENTROUTER_API_KEY", "AGENT_ROUTER_API_KEY"] } });
      editor.update(INTEGRATION_ID, i => { i.name = "AgentRouter"; });
    });
    async function registerAgentRouterProviders() {
      await ctx.provider.transform(editor => {
      for (const value of Object.values(PROVIDERS)) {
        const id = Provider.ID.make(value);
        editor.add({
          info: { id, name: value === PROVIDERS.openai ? "AgentRouter OpenAI" : "AgentRouter Claude", activation: environmentKey() || settings.apiKey ? "enabled" : "auto", integrationID: Integration.ID.make(INTEGRATION_ID), package: value === PROVIDERS.openai ? "aisdk:@ai-sdk/openai-compatible" : "aisdk:@ai-sdk/anthropic", settings: { baseURL: `${settings.baseURL}/v1`, ...(environmentKey() || settings.apiKey ? { apiKey: environmentKey() || settings.apiKey } : {}), transport: "http", encoding: "none" } },
          models: models.filter(m => providerFor(m) === value).map(toModel),
        });
      }
    });
    }
    await registerAgentRouterProviders();
    await ctx.aisdk.hook("sdk", async event => {
      if (!isOurs(event.model.providerID)) return;
      const spec = models.find(m => m.id === event.model.modelID);
      const transport = installAgentRouterFetchHook({ ...settings, minIntervalMs: () => settings.minIntervalMs, stateDir: dir, pacer, fetch: event.options.fetch });
      const options = { ...event.options, baseURL: event.options.baseURL ?? `${settings.baseURL}/v1`, apiKey: normalizeApiKey(event.options.apiKey) || await getEffectiveApiKey(), fetch: ((input, init) => transport(input, init)) as typeof transport };
      if (spec?.protocol === "responses") event.sdk = createOpenAI(options);
      else if (spec?.protocol === "anthropic") event.sdk = createAnthropic(options);
      else event.sdk = createOpenAICompatible({ ...options, name: event.model.providerID });
    });
    await ctx.aisdk.hook("language", event => {
      if (!isOurs(event.model.providerID)) return;
      const protocol = models.find(m => m.id === event.model.modelID)?.protocol;
      event.language = protocol === "responses" ? event.sdk.responses(event.model.modelID) : event.sdk.languageModel(event.model.modelID);
    });
    for (const name of ["context", "compaction", "title", "generate"] as const) await ctx.session.hook(name, event => {
      if (!isOurs(event.model.providerID)) return;
      event.options.promptCacheKey = event.sessionID;
      if (name === "compaction") {
        // Native OpenCode request copy: keep call IDs/names, omit argument/result dumps, preserve persisted history.
        event.messages = event.messages.map(message => ({ ...message, content: message.content.filter(part => part.type !== "reasoning").map(part =>
          part.type === "tool-result" ? { ...part, result: { type: "text" as const, value: REDACTED_NOTE } } :
          part.type === "tool-call" ? { ...part, input: {} } :
          part.type === "text" ? { ...part, text: sanitizeCompactionText(cleanContent(part.text)) } : part) }));
      }
      if (!event.system[0]?.text.startsWith(CANONICAL_OPENCODE_HEADER)) event.system.unshift({ type: "text", text: CANONICAL_OPENCODE_HEADER });
    });
    const execute = async (action: string, value?: string) => {
      action = normalizeCommandAction(action);
      if (action === "key") {
        const report = await manage(action, value, { stateDirectory: dir, getKey: getEffectiveApiKey });
        settings = effectiveConfig(await loadConfig());
        await ctx.provider.reload();
        return report.replace("Restart OpenCode to reload provider credentials.", "Provider credentials reloaded.");
      }
      const report = await manage(action, value, { stateDirectory: dir, getKey: getEffectiveApiKey });
      if (action === "sync" || action === "pricing") {
        models = toOpenCodeModelSpecs(await loadCachedPricing(dir));
        await ctx.provider.reload();
      }
      if (action === "pacing" && value !== undefined) settings = effectiveConfig(await loadConfig());
      return report.replace("Restart OpenCode to register changes.", "Models reloaded.").replace("Restart OpenCode to apply.", "Applied to new requests.");
    };
    await ctx.command.transform(editor => {
      for (const [action, description] of [
        ["status", "Show AgentRouter status and monthly spend"],
        ["pricing", "Show AgentRouter model pricing and reload models"],
        ["sync", "Refresh the AgentRouter model catalog and reload models"],
        ["check", "Check AgentRouter model availability (uses tokens)"],
        ["pacing", "Show request pacing or set it with a millisecond value"],
        ["key", "Save an AgentRouter API key and reload credentials; prefer /connect"],
      ] as const) {
        editor.add({ name: `agentrouter_${action}`, description, async execute(input) {
          const value = action === "pacing" || action === "key" ? input.prompt.text.trim() || undefined : undefined;
          const text = await execute(action, value);
          await ctx.session.synthetic({ sessionID: input.sessionID, text, description: "AgentRouter", resume: false, delivery: input.delivery });
        } });
      }
    });
    await ctx.tool.transform(editor => {
      editor.add({ name: "agentrouter", description: "AgentRouter status/pricing/sync. check makes real API requests and consumes tokens.", options: { codemode: false }, input: Schema.Struct({ action: Schema.Literals(["status", "pricing", "sync", "check"]) }), async execute({ action }) { return { content: await execute(action) }; } });
    });
  },
});

async function arDebugLog(message: string, dir = stateDir()): Promise<void> {
  // Preserve upstream diagnostic filename; never log request bodies, secrets or gateway echoes.
  try {
    await mkdir(dir, { recursive: true });
    await appendFile(join(dir, ".agentrouter-debug.log"), `[${new Date().toISOString()}] ${message}\n`, { mode: 0o600 });
  } catch {}
}

export async function getLastRequestEndTime(target = "agentrouter", dir = stateDir()): Promise<number> {
  const file = target === "agentrouter" ? ".agentrouter-pacing" : `.pacing-${target.replace(/[^a-z0-9_-]/gi, "_")}`;
  try {
    const value = Number(await readFile(join(dir, file), "utf8"));
    return Number.isFinite(value) ? value : 0;
  } catch { return 0; }
}

export async function setLastRequestEndTime(ts: number, target = "agentrouter", dir = stateDir()): Promise<void> {
  const file = target === "agentrouter" ? ".agentrouter-pacing" : `.pacing-${target.replace(/[^a-z0-9_-]/gi, "_")}`;
  try { await mkdir(dir, { recursive: true }); await writeFile(join(dir, file), String(ts), { mode: 0o600 }); } catch {}
}
