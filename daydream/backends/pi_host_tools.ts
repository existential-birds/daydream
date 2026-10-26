/** Private readiness/accounting for synchronous native terminal admission. */
import { constants, closeSync, fstatSync, openSync, readFileSync } from "node:fs";
import { isDeepStrictEqual } from "node:util";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

export default function (pi: ExtensionAPI) {
  let packet: any;
  let failed = false;
  let ready = false;
  let terminalCallId: string | undefined;
  type Admission = { id?: string; allowed?: boolean; outcome?: string; code?: string; error?: string; consumed?: boolean };
  const admissions = new Map<string, Admission>();
  let terminalFailure: Admission | undefined;
  const stop = (id: string, status: Admission) => {
    failed = true;
    terminalFailure ??= status;
    process.stdout.write(JSON.stringify({ type: "daydream_host_tool_fatal", id,
      outcome: status.outcome, code: status.code, error: status.error }) + "\n");
  };
  const request = async (path: string, value: any) => {
    const body = JSON.stringify(value);
    if (Buffer.byteLength(body) > packet.max_frame_bytes) throw new Error("Host frame limit");
    const response = await fetch(packet.url + path, {
      method: "POST",
      headers: { Authorization: packet.authorization, "Content-Type": "application/json" },
      body, signal: AbortSignal.timeout(packet.request_timeout_ms),
    });
    const text = await response.text();
    if (Buffer.byteLength(text) > packet.max_frame_bytes) throw new Error("Host response frame limit");
    const result = JSON.parse(text);
    if (!result || typeof result !== "object" || Array.isArray(result)) throw new Error("Invalid host response");
    if (!response.ok && result.outcome !== "fatal" && result.outcome !== "indeterminate") {
      throw new Error("Host endpoint unavailable");
    }
    return result as any;
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
      if (packet.allowed_tools && (!isDeepStrictEqual([...pi.getActiveTools()].sort(), packet.allowed_tools)
          || process.versions.node !== "24.18.1")) {
        throw new Error("Unsupported hosted native tool or Node configuration");
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
      let argumentsValue = event.toolName === packet.tool_name ? event.args : null;
      const argumentsOversized = Buffer.byteLength(JSON.stringify(argumentsValue) ?? "null") > packet.max_argument_bytes;
      if (argumentsOversized) {
        argumentsValue = null;
      }
      const status = await request("/admit", {
        id: event.toolCallId, name: event.toolName, arguments: argumentsValue,
        terminal: !event.parentToolCallId && terminalCallId === event.toolCallId,
        ...(argumentsOversized ? { arguments_oversized: true } : {}),
      });
      if (status.id !== event.toolCallId) throw new Error("Uncorrelated host admission response");
      const rejection = ["correctable", "fatal", "indeterminate"].includes(status.outcome)
        && typeof status.error === "string" && status.error.length > 0 && typeof status.code === "string";
      if (!(status.allowed === true && status.outcome === undefined)
          && !(rejection && (status.outcome !== "correctable" || status.allowed === false))) {
        throw new Error("Unconfirmed host admission response");
      }
      admissions.set(event.toolCallId, status);
      if (status.outcome === "fatal" || status.outcome === "indeterminate") stop(event.toolCallId, status);
    } catch {
      const status = terminalFailure ?? { outcome: "indeterminate", code: "native_correlation",
        error: "Host tool admission could not be confirmed; changing the submission cannot resolve it. Invocation aborted." };
      admissions.set(event.toolCallId, status);
      stop(event.toolCallId, status);
    }
  });
  pi.on("tool_call", async event => {
    const status = admissions.get(event.toolCallId);
    if (terminalFailure) return { block: true, reason: String(terminalFailure.error).slice(0, 1024), terminate: true };
    if (!ready || failed || !status || status.consumed) {
      const failure = { outcome: "indeterminate", code: "native_correlation",
        error: "Host tool execution has no unused admission receipt; changing the submission cannot resolve it. Invocation aborted." };
      stop(event.toolCallId, failure);
      return { block: true, reason: failure.error, terminate: true };
    }
    status.consumed = true;
    if (status.allowed) return;
    return { block: true, reason: String(status.error).slice(0, 1024) };
  });
  pi.on("tool_execution_end", (_event, ctx) => {
    // Abort after Pi retains the correlated native error. This also covers native
    // schema rejection, which bypasses tool_call, and disables session recovery.
    if (terminalFailure) ctx.abort();
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
