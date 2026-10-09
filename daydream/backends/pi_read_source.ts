/** Invocation-local frozen-source access and exact-schema final submission. */
import { constants, closeSync, fstatSync, openSync, readFileSync } from "node:fs";
import { Type } from "@earendil-works/pi-ai";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

export default function (pi: ExtensionAPI) {
  const packetPath = process.env.DAYDREAM_PI_SOURCE_PACKET;
  delete process.env.DAYDREAM_PI_SOURCE_PACKET;
  if (!packetPath) throw new Error("Frozen source packet unavailable");
  const fd = openSync(packetPath, constants.O_RDONLY | constants.O_NOFOLLOW);
  let packet: any;
  try {
    const stat = fstatSync(fd);
    if (!stat.isFile() || stat.size > 8 * 1024 * 1024) throw new Error("Frozen source packet unavailable");
    packet = JSON.parse(readFileSync(fd, "utf8"));
  } finally {
    closeSync(fd);
  }
  if (!packet || typeof packet !== "object" || Array.isArray(packet)) throw new Error("Invocation packet unavailable");
  if ("windows" in packet) {
    const windows = packet.windows;
    if (!Array.isArray(windows)) throw new Error("Frozen source packet unavailable");
    pi.registerTool({
      name: "read_source",
      label: "Read frozen source",
      description: "Read one supplied before/after source window by assigned target ID and captured side.",
      promptSnippet: "Read frozen before/after source windows using supplied target_id and side.",
      defaultActive: false,
      parameters: Type.Object({ target_id: Type.String(), side: Type.Union([Type.Literal("before"), Type.Literal("after")]) },
                              { additionalProperties: false }),
      async execute(_callId, params, signal) {
        if (signal?.aborted) throw new Error("Frozen source read cancelled");
        const matches = windows.filter((entry: any) => entry.source.side === params.side &&
                                      entry.source.target_ids.includes(params.target_id));
        if (matches.length !== 1) throw new Error("Frozen source selector unavailable");
        const text = JSON.stringify(matches[0]);
        if (Buffer.byteLength(text, "utf8") > 2 * 1024 * 1024) throw new Error("Frozen source window exceeds bound");
        return { content: [{ type: "text", text }] };
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
