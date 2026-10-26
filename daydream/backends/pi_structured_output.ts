/** Attempt-local exact-schema final submission for the packaged Pi provider. */
import { constants, closeSync, fstatSync, openSync, readFileSync } from "node:fs";
import { join } from "node:path";
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
  if (packet.native_selection) {
    pi.on("input", async (_event, ctx) => {
      try {
        const sdk = await import("@earendil-works/pi-coding-agent");
        const frozen = packet.native_selection;
        const agentDir = sdk.getAgentDir();
        const global = sdk.SettingsManager.inMemory(frozen.global);
        const storedTrust = new sdk.ProjectTrustStore(agentDir).get(frozen.source);
        const trusted = !sdk.hasTrustRequiringProjectResources(frozen.source)
          || (storedTrust ?? global.getDefaultProjectTrust() === "always");
        const settings = sdk.SettingsManager.fromStorage({
          withLock(scope, apply) {
            const updated = apply(JSON.stringify(scope === "global"
              ? { ...frozen.global, cacheWarming: "off" } : frozen.project));
            if (updated !== undefined) throw new Error("Selection settings are immutable");
          },
        }, { projectTrusted: trusted });
        // Never reload this loader: selection uses no resources, tools or model
        // generation, and its settings/session cannot persist to the operator.
        const resourceLoader = new sdk.DefaultResourceLoader({ cwd: ctx.cwd, agentDir, settingsManager: settings,
          noExtensions: true, noSkills: true, noPromptTemplates: true, noThemes: true, noContextFiles: true });
        const modelRuntime = await sdk.ModelRuntime.create({
          authPath: join(agentDir, "auth.json"), modelsPath: join(agentDir, "models.json") });
        const patterns = settings.getEnabledModels();
        let modelText = frozen.model;
        let provider = frozen.provider;
        if (!modelText && settings.getDefaultModel() === undefined && !patterns?.length) {
          modelText = frozen.fallback_model;
          provider ??= frozen.fallback_provider;
        } else if (!modelText && provider) {
          modelText = settings.getDefaultModel();
          if (!modelText) throw new Error("Native provider selection requires a model");
        }
        const resolved = modelText ? sdk.resolveCliModel({ cliModel: modelText,
          cliProvider: provider, cliThinking: frozen.thinking, modelRuntime }) : undefined;
        if (resolved?.error || (resolved && !resolved.model)) throw new Error("Native model selection unavailable");
        const scopedModels = patterns?.length
          ? (await sdk.resolveModelScopeWithDiagnostics(patterns, modelRuntime)).scopedModels : [];
        let model = resolved?.model;
        let thinking = frozen.thinking ?? resolved?.thinkingLevel;
        if (!model && scopedModels.length) {
          const saved = scopedModels.find(item => item.model.provider === settings.getDefaultProvider()
            && item.model.id === settings.getDefaultModel());
          const initial = saved ?? scopedModels[0];
          model = initial.model;
          thinking ??= initial.thinkingLevel;
        }
        const selected = await sdk.createAgentSession({ cwd: ctx.cwd, agentDir, settingsManager: settings,
          sessionManager: sdk.SessionManager.inMemory(ctx.cwd), resourceLoader, noTools: "all", tools: [], modelRuntime,
          scopedModels, ...(model ? { model } : {}), ...(thinking ? { thinkingLevel: thinking } : {}) });
        try {
          if (!selected.session.model || !await pi.setModel(selected.session.model)) {
            throw new Error("Native model selection unavailable");
          }
          pi.setThinkingLevel(selected.session.thinkingLevel);
        } finally { selected.session.dispose(); }
        return { action: "continue" as const };
      } catch {
        process.stdout.write(JSON.stringify({ type: "daydream_host_tools_failed" }) + "\n");
        return { action: "handled" as const };
      }
    });
  }
  if (packet.host_tools) {
    // The paired helper owns the active-schema handshake. A missing or broken
    // helper cannot silently turn a required host admission into serialization.
    pi.on("input", async () => {
      try {
        const host = packet.host_tools;
        const response = await fetch(host.url + "/status", {
          headers: { Authorization: host.authorization },
          signal: AbortSignal.timeout(host.request_timeout_ms),
        });
        const status = await response.json() as any;
        if (response.ok && status.ready === true) return { action: "continue" as const };
      } catch { /* Explicit handling is required when Pi catches handler errors. */ }
      process.stdout.write(JSON.stringify({ type: "daydream_host_tools_failed" }) + "\n");
      return { action: "handled" as const };
    });
  }
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
    let submissionIndeterminate = false;
    let submissionFatal = false;
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
      if (event.outcome !== "completed" || submitted || submissionIndeterminate || submissionFatal || reminded
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
      async execute(toolCallId, params, _signal, _onUpdate, ctx) {
        if (packet.host_tools) {
          const host = packet.host_tools;
          try {
            const body = JSON.stringify({ id: toolCallId, arguments: params });
            if (Buffer.byteLength(body) > host.max_frame_bytes) throw new Error("Host frame limit");
            const response = await fetch(host.url + "/submit", {
              method: "POST",
              headers: { Authorization: host.authorization, "Content-Type": "application/json" },
              body, signal: AbortSignal.timeout(host.request_timeout_ms),
            });
            const text = await response.text();
            if (Buffer.byteLength(text) > host.max_frame_bytes) throw new Error("Host response frame limit");
            const result = JSON.parse(text);
            if (!result || typeof result !== "object" || Array.isArray(result)
                || result.id !== toolCallId) throw new Error("Unconfirmed host response");
            if (response.ok && result.accepted === true && result.outcome === undefined && result.result
                && typeof result.result === "object" && !Array.isArray(result.result)) {
              return { content: [{ type: "text", text: "Review result accepted." }],
                details: result.result, terminate: true };
            }
            if ((result.outcome === "fatal" || result.outcome === "indeterminate")
                && typeof result.error === "string" && result.error.length > 0) {
              submissionFatal = true;
              ctx.abort();
              return { content: [{ type: "text", text: result.error.slice(0, 1024) }],
                details: { host_submission_status: result.outcome, code: result.code }, isError: true, terminate: true };
            }
            if (response.ok && result.accepted === false && result.outcome === "correctable"
                && typeof result.error === "string" && result.error.length > 0) {
              return { content: [{ type: "text", text: result.error.slice(0, 1024) }], isError: true };
            }
            throw new Error("Unconfirmed host submission outcome");
          } catch {
            submissionIndeterminate = true;
            ctx.abort();
            // This is an unsuccessful native stop, never an acceptance receipt.
            // The error stop hint prevents another core generation; abort also
            // disables session retry/recovery and before-settle continuation.
            return { content: [{ type: "text", text: "Host submission outcome could not be confirmed; invocation aborted." }],
              details: { host_submission_status: "indeterminate" }, isError: true, terminate: true };
          }
        }
        return { content: [{ type: "text", text: "Structured result submitted." }], details: params, terminate: true };
      },
    });
  }
}
