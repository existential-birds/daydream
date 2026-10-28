/** Observe response metadata without changing requests, tools or model history. */
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

export default function (pi: ExtensionAPI) {
  pi.on("after_provider_response", event => {
    const headers: Record<string, string> = {};
    // Never send cookies, credentials or arbitrary gateway headers to the recorder.
    for (const name of ["x-request-id", "request-id", "x-amzn-requestid", "x-amz-request-id",
                        "cf-ray", "retry-after", "retry-after-ms"]) {
      if (typeof event.headers[name] === "string") headers[name] = event.headers[name];
    }
    process.stdout.write(JSON.stringify({ type: "daydream_provider_response", status: event.status, headers }) + "\n");
  });
}
