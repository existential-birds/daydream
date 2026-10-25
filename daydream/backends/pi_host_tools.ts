/** Native MCP registration and pre-provider readiness for one private invocation. */
import { constants, closeSync, fstatSync, openSync, readFileSync } from "node:fs";
import { isDeepStrictEqual } from "node:util";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

export default function (pi: ExtensionAPI) {
  let packet: any;
  let failed = false;
  let ready = false;
  const admissions = new Map<string, { allowed: boolean; error?: string; consumed?: boolean }>();
  const request = async (path: string, value?: any) => {
    const body = value === undefined ? undefined : JSON.stringify(value);
    if (body && Buffer.byteLength(body) > packet.max_frame_bytes) throw new Error("Host frame limit");
    const response = await fetch(packet.url + path, {
      method: body === undefined ? "GET" : "POST",
      headers: { Authorization: packet.authorization, "Content-Type": "application/json" },
      body, signal: AbortSignal.timeout(packet.request_timeout_ms),
    });
    if (!response.ok) throw new Error("Host endpoint unavailable");
    return await response.json() as any;
  };
  // Register the fail-closed input barrier before parsing or capability checks.
  pi.on("input", async () => {
    try {
      if (failed || !packet) throw new Error("Host binding unavailable");
      const deadline = Date.now() + packet.setup_timeout_ms;
      while (Date.now() < deadline) {
        const status = await request("/status");
        const tool = pi.getAllTools().find(candidate => candidate.name === packet.tool_name);
        if (status.ready && tool && tool.exposure === "direct"
            && pi.getActiveTools().includes(packet.tool_name)
            && isDeepStrictEqual(JSON.parse(JSON.stringify(tool.parameters)), packet.schema)) {
          ready = true;
          process.stdout.write(JSON.stringify({ type: "daydream_host_tools_ready" }) + "\n");
          return { action: "continue" as const };
        }
        await new Promise(resolve => setTimeout(resolve, 100));
      }
    } catch { /* A thrown input handler is caught by Pi; explicitly handle below. */ }
    failed = true;
    process.stdout.write(JSON.stringify({ type: "daydream_host_tools_failed" }) + "\n");
    return { action: "handled" as const };
  });
  // Pi emits and awaits this event before native argument validation. Charging
  // only tool_call would replenish capacity after malformed/unknown calls.
  pi.on("tool_execution_start", async event => {
    try {
      if (!ready || failed) throw new Error("Host authority unavailable");
      if (admissions.has(event.toolCallId)) {
        admissions.set(event.toolCallId, { allowed: false, error: "Native tool identity was already charged" });
        return;
      }
      if (admissions.size >= 256) { failed = true; return; }
      let argumentsValue = event.args;
      if (Buffer.byteLength(JSON.stringify(argumentsValue) ?? "null") > packet.max_frame_bytes - 1024) {
        // Retain the native failure while still charging its small identity.
        // No matching host callback can consume an admission for null args.
        argumentsValue = null;
      }
      admissions.set(event.toolCallId, await request("/admit", {
        id: event.toolCallId, name: event.toolName, arguments: argumentsValue,
      }));
    } catch {
      // Without a confirmed host admission this invocation cannot authorize
      // later calls, even when Pi rejects this call before tool_call fires.
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
    return { block: true, reason: String(status.error).slice(0, 256) };
  });
  try {
    const packetPath = process.env.DAYDREAM_PI_HOST_PACKET;
    delete process.env.DAYDREAM_PI_HOST_PACKET;
    if (!packetPath) throw new Error("Missing host packet");
    const fd = openSync(packetPath, constants.O_RDONLY | constants.O_NOFOLLOW);
    try {
      const stat = fstatSync(fd);
      if (!stat.isFile() || stat.size > 16384) throw new Error("Invalid host packet");
      packet = JSON.parse(readFileSync(fd).toString("utf8"));
    } finally { closeSync(fd); }
    if (typeof pi.registerMcpServer !== "function" || typeof pi.getAllTools !== "function"
        || typeof pi.getActiveTools !== "function") throw new Error("Unsupported native MCP capability");
    pi.registerMcpServer(packet.server_name, {
      url: packet.url + "/mcp", headers: { Authorization: packet.authorization },
      exposure: "direct", timeout: packet.request_timeout_ms / 1000,
    });
  } catch { failed = true; }
}
