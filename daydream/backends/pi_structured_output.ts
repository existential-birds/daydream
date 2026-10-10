/** Invocation-local exact-schema final submission for the packaged Pi provider. */
import { constants, closeSync, fstatSync, openSync, readFileSync } from "node:fs";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

export default function (pi: ExtensionAPI) {
  const packetPath = process.env.DAYDREAM_PI_OUTPUT_PACKET;
  delete process.env.DAYDREAM_PI_OUTPUT_PACKET;
  if (!packetPath) throw new Error("Output packet unavailable");
  const fd = openSync(packetPath, constants.O_RDONLY | constants.O_NOFOLLOW);
  let packet: any;
  try {
    const stat = fstatSync(fd);
    if (!stat.isFile() || stat.size > 8 * 1024 * 1024) throw new Error("Output packet unavailable");
    packet = JSON.parse(readFileSync(fd).toString("utf8"));
  } finally {
    closeSync(fd);
  }
  if (!packet || typeof packet !== "object" || Array.isArray(packet)) throw new Error("Output packet unavailable");
  if ("output_schema" in packet) {
    if (!packet.output_schema || typeof packet.output_schema !== "object" || Array.isArray(packet.output_schema)) {
      throw new Error("Output schema unavailable");
    }
    if ("tool_call_budget" in packet) {
      if (!Number.isSafeInteger(packet.tool_call_budget) || packet.tool_call_budget < 0) {
        throw new Error("Tool allowance unavailable");
      }
      const allowance = packet.tool_call_budget;
      const noteType = "daydream_native_tool_budget";
      let starts = 0;
      // Native starts include failed/invalid calls and every member of a parallel batch.
      // The Python host receives these asynchronously and remains the hard-limit authority.
      pi.on("tool_execution_start", () => { starts += 1; });
      pi.on("context", (event) => ({
        messages: [...event.messages.filter((message) => message.role !== "custom" || message.customType !== noteType), {
          role: "custom" as const,
          customType: noteType,
          content: `Live native Pi invocation budget: ${Math.max(0, allowance - starts)} tool starts remain from ${allowance}; `
            + `${starts} local native starts observed. This dispatch counter may lead the host's received-start ledger; `
            + "the host still enforces hard limits. structured_output costs 1 tool start. Every parallel batch member counts, "
            + "including failed calls. Preserve the submission start. Once concrete checks are settled, submit the assigned "
            + "decisions; honestly mark unfinished work instead of continuing exploration.",
          display: false,
          timestamp: Date.now(),
        }],
      }));
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
