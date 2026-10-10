/** Synthetic external provider. The actual installed Pi CLI and tools still run. */
import { createAssistantMessageEventStream } from "@earendil-works/pi-ai";

export default function (pi: any) {
  pi.registerProvider("source-fixture", {
    api: "openai-completions", baseUrl: "https://fixture.invalid", apiKey: "synthetic-provider-key",
    models: [{ id: "source-model", name: "Source fixture", reasoning: false, input: ["text"],
               contextWindow: 32768, maxTokens: 8192,
               cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 } }],
    streamSimple(model: any, context: any) {
      const stream = createAssistantMessageEventStream();
      queueMicrotask(() => {
        const result = context.messages.findLast((entry: any) => entry.role === "toolResult");
        const message: any = {
          role: "assistant", api: model.api, provider: model.provider, model: model.id,
          content: [], timestamp: Date.now(), stopReason: result ? "stop" : "toolUse",
          usage: { input: 1, output: 1, cacheRead: 0, cacheWrite: 0, totalTokens: 2,
                   cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } },
        };
        stream.push({ type: "start", partial: message });
        if (!result) {
          const toolCall = { type: "toolCall", id: "native-frozen-source-001",
                             name: "read_source",
                             arguments: JSON.parse(process.env.DAYDREAM_TEST_SOURCE_SELECTOR!) };
          message.content = [toolCall];
          stream.push({ type: "toolcall_start", contentIndex: 0, partial: message });
          stream.push({ type: "toolcall_delta", contentIndex: 0,
                        delta: JSON.stringify(toolCall.arguments), partial: message });
          stream.push({ type: "toolcall_end", contentIndex: 0, toolCall, partial: message });
        } else {
          const output = {
            observed_body: result.isError ? "" : JSON.parse(result.content[0].text).body,
            source_error: result.isError,
          };
          const text = JSON.stringify(output);
          const toolCall = { type: "toolCall", id: "native-output-001", name: "structured_output", arguments: output };
          message.content = [toolCall];
          message.stopReason = "toolUse";
          stream.push({ type: "toolcall_start", contentIndex: 0, partial: message });
          stream.push({ type: "toolcall_delta", contentIndex: 0, delta: text, partial: message });
          stream.push({ type: "toolcall_end", contentIndex: 0, toolCall, partial: message });
        }
        stream.push({ type: "done", reason: message.stopReason, message });
        stream.end(message);
      });
      return stream;
    },
  });
}
