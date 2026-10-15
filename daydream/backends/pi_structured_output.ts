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
    // Only durations cross the monotonic-clock boundary. Unix time estimates
    // attachment/spawn/load delay once, then Node's independent monotonic clock
    // drives the countdown. Clock shifts during that bridge are unknowable;
    // this is advisory context, never the host's enforced deadline.
    const wall = packet.wall_time;
    const receivedUnixMs = Date.now();
    const nativeOrigin = process.hrtime.bigint();
    const validWall = wall && typeof wall === "object" && !Array.isArray(wall)
      && [wall.remaining_s, wall.elapsed_s, wall.sent_unix_ms].every(
        (value) => typeof value === "number" && Number.isFinite(value) && value >= 0)
      && wall.sent_unix_ms <= receivedUnixMs;
    const transitS = validWall ? (receivedUnixMs - wall.sent_unix_ms) / 1000 : 0;
    let nativeElapsedS = 0;
    if (allowance !== undefined || validWall) {
      const noteType = "daydream_native_tool_budget";
      let starts = 0;
      // Native starts include failed/invalid calls and every member of a parallel batch.
      // The Python host receives these asynchronously and remains the hard-limit authority.
      pi.on("tool_execution_start", () => { starts += 1; });
      pi.on("context", (event) => {
        const remaining = allowance === undefined ? undefined : Math.max(0, allowance - starts);
        nativeElapsedS = Math.max(nativeElapsedS, Number(process.hrtime.bigint() - nativeOrigin) / 1e9);
        const wallRemaining = validWall ? Math.max(0, wall.remaining_s - transitS - nativeElapsedS) : undefined;
        const wallNote = validWall
          ? `Wall estimate: ${Math.floor(wall.elapsed_s + transitS + nativeElapsedS)}s elapsed since dispatch; `
            + `${Math.floor(wallRemaining!)}s remaining. Epoch bridge then native monotonic; clock shifts during `
            + "handoff may cause earlier host cutoff. Host deadline/cancellation remain authoritative. "
          : wall === null ? "Wall allowance is unbounded. " : "";
        const guidance = wallRemaining === 0
          ? "Estimated wall allowance is exhausted; no extra time or submission capacity is granted. "
            + "Unfinished assigned work remains incomplete."
          : remaining === undefined || remaining > 1
          ? "Keep one call for structured_output. Continue concrete unresolved checks for assigned work while tool/time "
            + "capacity permits; submit when those checks settle. Unused capacity is expected."
          : remaining === 1
            ? "Only submission capacity remains. Submit now using structured_output alone; report unfinished work "
              + "with truthful incomplete decisions."
            : "Tool-call capacity is exhausted; no call remains for structured_output. Assigned work that is unfinished "
              + "remains incomplete; do not claim completion or additional submission capacity.";
        return {
          messages: [...event.messages.filter((message) => message.role !== "custom" || message.customType !== noteType), {
            role: "custom" as const,
            customType: noteType,
            content: "Live native Pi attempt budget: " + (remaining === undefined
              ? "Tool-call allocation is uncapped. "
              : `This attempt has ${remaining} tool calls remaining`
                + (remaining > 0 ? ", including final submission. " : ". ")
                + "Each individual call counts, including failures and parallel batch members. "
                + "Use this refreshed count without recounting history. ") + wallNote + guidance,
            display: false,
            timestamp: Date.now(),
          }],
        };
      });
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
