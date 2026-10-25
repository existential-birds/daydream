/** Private readiness/accounting for synchronous native terminal admission. */
import { constants, closeSync, fstatSync, openSync, readFileSync } from "node:fs";
import { isDeepStrictEqual } from "node:util";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

export default function (pi: ExtensionAPI) {
  let packet: any;
  let failed = false;
  let ready = false;
  let terminalCallId: string | undefined;
  const admissions = new Map<string, { allowed: boolean; error?: string; consumed?: boolean }>();
  const request = async (path: string, value: any) => {
    const body = JSON.stringify(value);
    if (Buffer.byteLength(body) > packet.max_frame_bytes) throw new Error("Host frame limit");
    const response = await fetch(packet.url + path, {
      method: "POST",
      headers: { Authorization: packet.authorization, "Content-Type": "application/json" },
      body, signal: AbortSignal.timeout(packet.request_timeout_ms),
    });
    if (!response.ok) throw new Error("Host endpoint unavailable");
    const text = await response.text();
    if (Buffer.byteLength(text) > packet.max_frame_bytes) throw new Error("Host response frame limit");
    return JSON.parse(text) as any;
  };
  // Pi reports extension load errors without necessarily stopping. Install the
  // input barrier before parsing so missing/invalid authority stays fail closed.
  pi.on("input", async () => {
    try {
      if (failed || !packet) throw new Error("Host binding unavailable");
      const tool = pi.getAllTools().find(candidate => candidate.name === packet.tool_name);
      if (!tool || tool.exposure !== "direct" || !pi.getActiveTools().includes(packet.tool_name)
          || !isDeepStrictEqual(JSON.parse(JSON.stringify(tool.parameters)), packet.schema)) {
        throw new Error("Host terminal tool unavailable");
      }
      const status = await request("/ready", { tool_name: packet.tool_name, schema: tool.parameters });
      if (status.ready !== true) throw new Error("Host binding unavailable");
      ready = true;
      process.stdout.write(JSON.stringify({ type: "daydream_host_tools_ready" }) + "\n");
      return { action: "continue" as const };
    } catch { /* A thrown input handler is caught by Pi; explicitly handle below. */ }
    failed = true;
    process.stdout.write(JSON.stringify({ type: "daydream_host_tools_failed" }) + "\n");
    return { action: "handled" as const };
  });
  // This public event is awaited before tool execution. It carries the complete
  // assistant generation, so a terminal callback cannot admit a mixed batch or
  // a generation interrupted by length/error/abort. It never edits native history.
  pi.on("message_end", (event) => {
    if (event.message.role !== "assistant") return;
    terminalCallId = undefined;
    const calls = event.message.content.filter(block => block.type === "toolCall");
    if ((event.message.stopReason === "stop" || event.message.stopReason === "toolUse")
        && calls.length === 1 && calls[0].name === packet?.tool_name) {
      terminalCallId = calls[0].id;
    }
  });
  // Real starts, including native schema failures and nested starts, count once.
  // The host receives them before native execution and remains cumulative authority.
  pi.on("tool_execution_start", async event => {
    try {
      if (!ready || failed) throw new Error("Host authority unavailable");
      if (admissions.has(event.toolCallId)) {
        admissions.set(event.toolCallId, { allowed: false, error: "Native tool identity was already charged" });
        return;
      }
      let argumentsValue = event.toolName === packet.tool_name ? event.args : null;
      if (Buffer.byteLength(JSON.stringify(argumentsValue) ?? "null") > packet.max_argument_bytes) {
        argumentsValue = null;
      }
      admissions.set(event.toolCallId, await request("/admit", {
        id: event.toolCallId, name: event.toolName, arguments: argumentsValue,
        terminal: !event.parentToolCallId && terminalCallId === event.toolCallId,
      }));
    } catch {
      failed = true;
    }
  });
  pi.on("tool_call", async event => {
    const status = admissions.get(event.toolCallId);
    if (!ready || failed || !status || status.consumed) {
      return { block: true, reason: "Host-tool authority is unavailable or expired" };
    }
    status.consumed = true;
    if (status.allowed) return;
    return { block: true, reason: String(status.error).slice(0, 1024) };
  });
  try {
    const packetPath = process.env.DAYDREAM_PI_HOST_PACKET;
    delete process.env.DAYDREAM_PI_HOST_PACKET;
    if (!packetPath) throw new Error("Missing host packet");
    const fd = openSync(packetPath, constants.O_RDONLY | constants.O_NOFOLLOW);
    try {
      const stat = fstatSync(fd);
      if (!stat.isFile() || stat.size > 8 * 1024 * 1024) throw new Error("Invalid host packet");
      packet = JSON.parse(readFileSync(fd).toString("utf8"));
    } finally { closeSync(fd); }
    if (typeof pi.getAllTools !== "function" || typeof pi.getActiveTools !== "function"
        || packet?.tool_name !== "structured_output") throw new Error("Unsupported native terminal capability");
  } catch { failed = true; }
}
