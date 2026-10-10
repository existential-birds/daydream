/** Invocation-local frozen-source access and exact-schema final submission. */
import { constants, closeSync, fstatSync, openSync, readFileSync } from "node:fs";
import { createHash } from "node:crypto";
import { Type } from "@earendil-works/pi-ai";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

export default function (pi: ExtensionAPI) {
  const packetPath = process.env.DAYDREAM_PI_SOURCE_PACKET;
  delete process.env.DAYDREAM_PI_SOURCE_PACKET;
  if (!packetPath) throw new Error("Frozen source packet unavailable");
  const fd = openSync(packetPath, constants.O_RDONLY | constants.O_NOFOLLOW);
  let packet: any;
  let packetDigest: string;
  try {
    const stat = fstatSync(fd);
    if (!stat.isFile() || stat.size > 8 * 1024 * 1024) throw new Error("Frozen source packet unavailable");
    const raw = readFileSync(fd);
    packetDigest = createHash("sha256").update(raw).digest("hex");
    packet = JSON.parse(raw.toString("utf8"));
  } finally {
    closeSync(fd);
  }
  if (!packet || typeof packet !== "object" || Array.isArray(packet)) throw new Error("Invocation packet unavailable");
  if ("windows" in packet) {
    const windows = packet.windows;
    if (!Array.isArray(windows) || windows.some((entry: any) => {
      const source = entry?.source;
      return !source ||
        !["before", "after"].includes(source.side) ||
        !Array.isArray(source.target_ids) || source.target_ids.length === 0 ||
        source.target_ids.some((id: any) => typeof id !== "string" || !id) ||
        ["file", "source_path", "revision", "content_sha256", "blob_oid"].some(
          key => typeof source[key] !== "string" || !source[key]) ||
        ["start_line", "end_line", "start_byte", "end_byte"].some(
          key => !Number.isSafeInteger(source[key]) || source[key] < 0) ||
        source.start_line < 1 || source.end_line < source.start_line || source.end_byte < source.start_byte;
    })) throw new Error("Frozen source packet unavailable");
    const reader = packet.reader;
    if (!reader || typeof reader.url !== "string" ||
        !/^http:\/\/127\.0\.0\.1:[0-9]+\/mcp$/.test(reader.url) ||
        typeof reader.token !== "string" || !reader.token) throw new Error("Frozen source reader unavailable");
    pi.registerTool({
      name: "read_source",
      label: "Read frozen source",
      description: "Read one frozen window using a listed source selector and side. Output assignment IDs are not source selectors unless explicitly listed in the source catalog.",
      promptSnippet: "Read frozen before/after source windows using supplied target_id and side.",
      defaultActive: false,
      parameters: Type.Object({ target_id: Type.String(), side: Type.Union([Type.Literal("before"), Type.Literal("after")]) },
                              { additionalProperties: false }),
      async execute(_callId, params, signal) {
        if (signal?.aborted) throw new Error("Frozen source read cancelled");
        const response = await fetch(reader.url, {
          method: "POST", signal,
          headers: { Authorization: `Bearer ${reader.token}`, "Content-Type": "application/json",
                     Accept: "application/json, text/event-stream" },
          body: JSON.stringify({ jsonrpc: "2.0", id: _callId, method: "tools/call",
                                 params: { name: "read_source", arguments: params } }),
        });
        if (!response.ok) throw new Error("Frozen source read failed");
        const raw = await response.text();
        if (Buffer.byteLength(raw, "utf8") > 4 * 1024 * 1024 + 4096) throw new Error("Frozen source window exceeds bound");
        const result = JSON.parse(raw).result;
        if (!result || !Array.isArray(result.content) || result.content.length !== 1 ||
            result.content[0]?.type !== "text" || typeof result.content[0].text !== "string" ||
            typeof result.isError !== "boolean") throw new Error("Frozen source result unavailable");
        if (Buffer.byteLength(result.content[0].text, "utf8") > 2 * 1024 * 1024) {
          throw new Error("Frozen source window exceeds bound");
        }
        const details = result.isError && result._meta?.source_free_disposition === "zero_match"
          ? { source_free_disposition: "zero_match", packet_sha256: packetDigest } : undefined;
        return { content: result.content, isError: result.isError, details };
      },
    });
  }
  if ("output_schema" in packet) {
    if (!packet.output_schema || typeof packet.output_schema !== "object" || Array.isArray(packet.output_schema)) {
      throw new Error("Output schema unavailable");
    }
    let submitted = false;
    let reminded = false;
    pi.on("tool_execution_end", (event) => {
      if (event.toolName === "structured_output" && !event.isError) submitted = true;
    });
    pi.on("agent_before_settle", (event) => {
      if (event.outcome !== "completed" || submitted || reminded) return;
      reminded = true;
      return {
        entries: [...event.entries, {
          type: "custom_message",
          customType: "daydream_missing_submission",
          content: "Submit the final result now by calling structured_output alone. Assistant prose is not a submission.",
          display: false,
        }],
        continue: true,
      };
    });
    pi.registerTool({
      name: "structured_output",
      label: "Structured Output",
      description: "Submit the final result. Call this tool alone as your final action.",
      promptSnippet: "Submit the final structured result",
      promptGuidelines: ["Complete this invocation by calling structured_output alone with the final result; assistant prose does not submit it."],
      parameters: packet.output_schema,
      async execute(_toolCallId, params) {
        return { content: [{ type: "text", text: "Structured result submitted." }], details: params, terminate: true };
      },
    });
  }
}
