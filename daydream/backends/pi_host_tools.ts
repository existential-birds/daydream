/** Private readiness/accounting for synchronous native host tools and terminal admission. */
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
  const request = async (path: string, value: any, timeoutMs: number = packet.request_timeout_ms) => {
    const body = JSON.stringify(value);
    if (Buffer.byteLength(body) > packet.max_frame_bytes) throw new Error("Host frame limit");
    const response = await fetch(packet.url + path, {
      method: "POST",
      headers: { Authorization: packet.authorization, "Content-Type": "application/json" },
      body, signal: AbortSignal.timeout(timeoutMs),
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
  const snippets: Record<string, string> = {
    review_source: "Look up the frozen captured review's targets and exact change bytes",
    review_admit: "Admit a bounded batch of review results without ending the assignment",
  };
  const descriptions: Record<string, string> = {
    review_source: "Read-only host lookup of the frozen captured review: omit target_id/path for the paged target "
      + "catalog; target_id returns exact captured change bytes with its source_options; path lists that changed "
      + "path's targets. Pass the returned cursor for the next page. Every call is charged and reports current host "
      + "capacity.",
    review_admit: "Admit a bounded batch of decisions/claims (or verdicts) into the host review without ending the "
      + "assignment. Use structured_output for the final batch. Every call is charged and returns a compact receipt "
      + "and current capacity.",
  };
  // A charged nonterminal host tool. Its start was admitted by tool_execution_start;
  // the host serves it once for that identity and remains the domain authority.
  const registerHostTool = (name: string, schema: any) => {
    const description = descriptions[name];
    if (!description || !snippets[name] || !schema || typeof schema !== "object" || Array.isArray(schema)) {
      throw new Error("Unsupported host tool");
    }
    pi.registerTool({
      name, label: name, description, promptSnippet: snippets[name],
      parameters: schema,
      async execute(toolCallId, params, _signal, _onUpdate, ctx) {
        try {
          if (!ready || failed) throw new Error("Host authority unavailable");
          // Oversized arguments are a correctable payload fault: the start is charged, nothing installs.
          if (Buffer.byteLength(JSON.stringify(params) ?? "null") > packet.max_argument_bytes) {
            return { content: [{ type: "text" as const, text: `The entire ${name} argument object exceeds `
              + `${packet.max_argument_bytes} UTF-8 bytes; split it into smaller batches before retrying` }],
              details: { host_tool: name, code: "invalid_submission" }, isError: true };
          }
          // Bound the synchronous host callback by the unchanged assignment deadline.
          const statusStarted = performance.now();
          const statusResponse = await fetch(packet.url + "/status", {
            headers: { Authorization: packet.authorization },
            signal: AbortSignal.timeout(packet.request_timeout_ms),
          });
          const current = await statusResponse.json() as any;
          if (!statusResponse.ok || current.ready !== true || !Number.isSafeInteger(current.remaining_ms)
              || current.remaining_ms <= 0 || current.remaining_ms > 2_147_483_647) {
            throw new Error("Host deadline unavailable");
          }
          const remaining = Math.floor(current.remaining_ms - (performance.now() - statusStarted));
          if (remaining <= 0) throw new Error("Host deadline expired");
          const status = await request("/call", { id: toolCallId, name, arguments: params }, remaining);
          if (status.outcome === undefined && status.id === toolCallId && status.result
              && typeof status.result === "object" && !Array.isArray(status.result)) {
            const { content, ...metadata } = status.result;
            // Frozen captured bytes are their own text block, never JSON-escaped.
            return { content: [{ type: "text" as const, text: JSON.stringify(metadata) },
              ...(typeof content === "string" ? [{ type: "text" as const, text: content }] : [])],
              details: { host_tool: name } };
          }
          const rejection = typeof status.error === "string" && status.error.length > 0
            && typeof status.code === "string";
          if (rejection && status.outcome === "correctable" && status.allowed === false) {
            return { content: [{ type: "text" as const, text: status.error.slice(0, 1024) }],
              details: { host_tool: name, code: status.code }, isError: true };
          }
          if (rejection && (status.outcome === "fatal" || status.outcome === "indeterminate")) {
            stop(toolCallId, status);
            ctx.abort();
            return { content: [{ type: "text" as const, text: status.error.slice(0, 1024) }],
              details: { host_tool: name, host_tool_status: status.outcome, code: status.code }, isError: true };
          }
          throw new Error("Unconfirmed host tool outcome");
        } catch {
          const failure = terminalFailure ?? { outcome: "indeterminate", code: "native_correlation",
            error: "Host tool outcome could not be confirmed; changing the request cannot resolve it. Invocation aborted." };
          stop(toolCallId, failure);
          ctx.abort();
          return { content: [{ type: "text" as const, text: String(failure.error).slice(0, 1024) }],
            details: { host_tool: name, host_tool_status: failure.outcome }, isError: true };
        }
      },
    });
  };
  // Pi reports extension load errors without necessarily stopping. Install the
  // input barrier before parsing so missing/invalid authority stays fail closed.
  pi.on("input", async (_event, ctx) => {
    let failureCode: string | undefined;
    let failureError: string | undefined;
    try {
      if (failed || !packet) throw new Error("Host binding unavailable");
      const tool = pi.getAllTools().find(candidate => candidate.name === packet.tool_name);
      if (!tool || tool.exposure !== "direct" || !pi.getActiveTools().includes(packet.tool_name)
          || !isDeepStrictEqual(JSON.parse(JSON.stringify(tool.parameters)), packet.schema)) {
        throw new Error("Host terminal tool unavailable");
      }
      const hostTools: Record<string, any> = {};
      for (const [name, schema] of Object.entries(packet.host_tools ?? {})) {
        const hosted = pi.getAllTools().find(candidate => candidate.name === name);
        if (!hosted || hosted.exposure !== "direct" || !pi.getActiveTools().includes(name)
            || !isDeepStrictEqual(JSON.parse(JSON.stringify(hosted.parameters)), schema)) {
          throw new Error("Host lookup tool unavailable");
        }
        hostTools[name] = hosted.parameters;
      }
      if (packet.allowed_tools && (!isDeepStrictEqual([...pi.getActiveTools()].sort(), packet.allowed_tools)
          || process.versions.node !== "24.18.1")) {
        throw new Error("Unsupported hosted native tool or Node configuration");
      }
      if (packet.hosted_model && (ctx.model?.provider !== packet.hosted_model.provider
          || ctx.model?.id !== packet.hosted_model.id)) {
        failureCode = "hosted_model_selection";
        throw new Error("Unsupported hosted provider/model selection");
      }
      const compaction: any = pi.getSettings().compaction ?? null;
      const status = await request("/ready", {
        tool_name: packet.tool_name, schema: tool.parameters,
        ...(Object.keys(hostTools).length ? { host_tools: hostTools } : {}),
        // Pi reports the selected model's registry limits; the host refuses an
        // invocation that cannot fit them before any provider request.
        native_limits: {
          model: { context_window: ctx.model?.contextWindow, max_tokens: ctx.model?.maxTokens },
          compaction: compaction && { enabled: compaction.enabled, reserve_tokens: compaction.reserveTokens,
            keep_recent_tokens: compaction.keepRecentTokens },
        },
      });
      if (status.ready !== true) {
        if ((status.outcome === "fatal" || status.outcome === "indeterminate") && typeof status.code === "string") {
          failureCode = status.code;
          failureError = typeof status.error === "string" ? status.error.slice(0, 1024) : undefined;
        }
        throw new Error("Host binding unavailable");
      }
      ready = true;
      process.stdout.write(JSON.stringify({ type: "daydream_host_tools_ready", native_limits: status.native_limits }) + "\n");
      return { action: "continue" as const };
    } catch { /* A thrown input handler is caught by Pi; explicitly handle below. */ }
    failed = true;
    process.stdout.write(JSON.stringify({ type: "daydream_host_tools_failed", code: failureCode,
      ...(failureError ? { error: failureError } : {}) }) + "\n");
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
    const hostTools = packet.host_tools ?? {};
    if (!hostTools || typeof hostTools !== "object" || Array.isArray(hostTools)) throw new Error("Invalid host tools");
    for (const [name, schema] of Object.entries(hostTools)) registerHostTool(name, schema);
  } catch { failed = true; }
}
