/** Attempt-local exact-schema final submission for the packaged Pi provider. */
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
    let allowance: number | undefined;
    if ("tool_call_budget" in packet) {
      if (!Number.isSafeInteger(packet.tool_call_budget) || packet.tool_call_budget < 0) {
        throw new Error("Tool allowance unavailable");
      }
      allowance = packet.tool_call_budget;
    }
    let starts = 0;
    let submitted = false;
    let reminded = false;
    let warned = false;
    let pendingRecovery = false;
    const recoveryType = "daydream_submission_recovery_continuation";
    // Failed/invalid calls and every parallel batch member count. The host receives
    // these asynchronously and remains the cumulative hard-limit authority.
    pi.on("tool_execution_start", () => { starts += 1; });
    pi.on("context_with_system", (event) => {
      // Pi 1.0.4 requires runnable non-assistant context to continue after prose.
      // The empty boundary sentinel satisfies that guard only; remove it before
      // conversion to provider messages, even if the leading system is absent.
      const messages = event.messages.filter((message) =>
        message.role !== "custom" || message.customType !== recoveryType);
      const leading = messages[0];
      if (leading?.role !== "system") return { messages };
      const instructions: string[] = [];
      // A batch may skip directly past one remaining call. Never suggest submission
      // capacity at zero/negative remaining, or renew capacity after a retry.
      if (!submitted && !warned && allowance !== undefined && allowance - starts === 1) {
        warned = true;
        instructions.push("Final-submission budget warning: Only submission capacity remains. "
          + "Submit now using structured_output alone; report unfinished assigned work with truthful incomplete decisions."
          + (packet.submission_assignment
            ? " Assigned objective and identities: " + JSON.stringify(packet.submission_assignment) : ""));
      }
      if (pendingRecovery) {
        pendingRecovery = false;
        instructions.push("Submit the final result now by calling structured_output alone. Assistant prose is not a submission.");
      }
      if (!instructions.length) return { messages };
      const text = "\n\n" + instructions.join("\n\n");
      // Project once for this request. Do not persist synthetic turns or edit any
      // genuine history; preserve the leading system's sections/tool declarations.
      const content = typeof leading.content === "string"
        ? leading.content + text
        : [...leading.content, { type: "text" as const, text }];
      return { messages: [{ ...leading, content }, ...messages.slice(1)] };
    });
    pi.on("tool_execution_end", (event) => {
      if (event.toolName === "structured_output" && !event.isError) submitted = true;
    });
    pi.on("agent_before_settle", (event) => {
      if (event.outcome !== "completed" || submitted || reminded
          || (allowance !== undefined && allowance - starts <= 0)) return;
      reminded = true;
      pendingRecovery = true;
      return {
        entries: [...event.entries, {
          type: "custom_message",
          customType: recoveryType,
          content: [],
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
