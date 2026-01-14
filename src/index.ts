// Cloudflare Worker: POST /house-mask
// Expects multipart/form-data:
// - image: File
// - scene: "exterior" | "interior" (optional, default "exterior")
// Returns JSON: { maskPngBase64: "<base64png>" }

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    try {
      if (request.method === "OPTIONS") return handleOptions(request);

      const url = new URL(request.url);

      // Health check
      if (request.method === "GET" && url.pathname === "/") {
        return new Response("worker alive", { headers: corsHeaders(request) });
      }

      if (url.pathname !== "/house-mask") {
        return json(request, { error: "Not Found" }, 404);
      }

      if (request.method !== "POST") {
        return json(request, { error: "Method Not Allowed" }, 405);
      }

      if (!env.REPLICATE_API_TOKEN) {
        return json(request, { error: "Missing REPLICATE_API_TOKEN secret" }, 500);
      }

      const contentType = request.headers.get("content-type") || "";
      if (!contentType.toLowerCase().includes("multipart/form-data")) {
        return json(request, { error: "Expected multipart/form-data" }, 400);
      }

      const form = await request.formData();
      const file = form.get("image");

      if (!(file instanceof File)) {
        return json(request, { error: "Missing image file field 'image'" }, 400);
      }

      // Keep runtime and costs predictable
      if (file.size > 6_000_000) {
        return json(
          request,
          { error: "Image too large. Please upload an image under 6MB." },
          413
        );
      }

      const scene = String(form.get("scene") || "exterior").toLowerCase();
      const textPrompt =
        scene === "interior" ? "wall" : "house, building exterior";

      // Read file and convert to data URL
      const ab = await file.arrayBuffer();
      const base64 = arrayBufferToBase64(ab);

      // Use the model schema's expected field name: input_image
      // FastSAM version id (Replicate model version) from Replicate docs UI
      // casia-iva-lab/fastsam version: 53373d1908dedb50...  [oai_citation:1‡Replicate](https://replicate.com/casia-iva-lab/fastsam/versions/53373d1908dedb50c25180ce99f2488d4fc2bfab1b3dc1637352862887956281/api?utm_source=chatgpt.com)
      const replicateVersion =
        "53373d1908dedb50c25180ce99f2488d4fc2bfab1b3dc1637352862887956281";

      const createRes = await fetch("https://api.replicate.com/v1/predictions", {
        method: "POST",
        headers: {
          Authorization: `Bearer ${env.REPLICATE_API_TOKEN}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          version: replicateVersion,
          input: {
            input_image: `data:${file.type || "image/jpeg"};base64,${base64}`,
            text_prompt: textPrompt,
            conf: 0.4,
            iou: 0.9,
            retina: true,
          },
        }),
      });

      const created = await safeJson(createRes);
      if (!createRes.ok) {
        return json(
          request,
          {
            error: "Replicate create prediction failed",
            status: createRes.status,
            details: created,
          },
          502
        );
      }

      const getUrl = created?.urls?.get;
      if (!getUrl) {
        return json(
          request,
          { error: "Replicate response missing urls.get", details: created },
          502
        );
      }

      const result = await pollReplicate(getUrl, env.REPLICATE_API_TOKEN);

      const maskUrl = pickMaskUrl(result?.output);
      if (!maskUrl) {
        return json(
          request,
          { error: "Could not find mask URL in Replicate output", details: result },
          502
        );
      }

      // Fetch mask image and return as base64 PNG bytes
      const maskResp = await fetch(maskUrl);
      if (!maskResp.ok) {
        return json(
          request,
          { error: "Failed to fetch mask image", status: maskResp.status },
          502
        );
      }

      const maskBuf = await maskResp.arrayBuffer();

      // For now: return the mask as received.
      // Next iteration: largest-component cleanup, hole fill, and smoothing.
      const maskB64 = arrayBufferToBase64(maskBuf);

      return json(request, { maskPngBase64: maskB64 }, 200);
    } catch (err: any) {
      return json(
        request,
        { error: err?.message || "Unhandled error" },
        500
      );
    }
  },
};

type Env = {
  REPLICATE_API_TOKEN?: string;
};

/* ---------------- CORS ---------------- */

function corsHeaders(request: Request) {
  const origin = request.headers.get("origin") || "*";
  return {
    "Access-Control-Allow-Origin": origin === "null" ? "*" : origin,
    "Access-Control-Allow-Methods": "GET,POST,OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type",
    "Vary": "Origin",
  };
}

function handleOptions(request: Request) {
  return new Response(null, { headers: corsHeaders(request) });
}

function json(request: Request, data: any, status = 200) {
  return new Response(JSON.stringify(data), {
    status,
    headers: {
      ...corsHeaders(request),
      "Content-Type": "application/json",
    },
  });
}

/* ---------------- Replicate polling ---------------- */

async function pollReplicate(getUrl: string, token: string) {
  // Replicate predictions are async; poll the prediction get URL until succeeded/failed.  [oai_citation:2‡Replicate](https://replicate.com/docs/topics/predictions/create-a-prediction?utm_source=chatgpt.com)
  const maxAttempts = 30;
  const delayMs = 1000;

  for (let i = 0; i < maxAttempts; i++) {
    const res = await fetch(getUrl, {
      headers: { Authorization: `Bearer ${token}` },
    });
    const body = await safeJson(res);

    if (!res.ok) {
      throw new Error(`Replicate poll failed: ${res.status}`);
    }

    const status = body?.status;
    if (status === "succeeded") return body;
    if (status === "failed" || status === "canceled") {
      throw new Error(`Replicate status: ${status}`);
    }

    await sleep(delayMs);
  }

  throw new Error("Replicate prediction timed out");
}

function sleep(ms: number) {
  return new Promise((r) => setTimeout(r, ms));
}

/* ---------------- Output parsing ---------------- */

function pickMaskUrl(output: any): string | null {
  // Output can be:
  // - string URL
  // - array of URL strings
  // - object with fields including urls
  if (!output) return null;

  if (typeof output === "string" && output.startsWith("http")) return output;

  if (Array.isArray(output)) {
    for (const item of output) {
      if (typeof item === "string" && item.startsWith("http")) return item;
    }
  }

  if (typeof output === "object") {
    // try common patterns
    const candidates = [
      output.mask,
      output.masks,
      output.output,
      output.image,
      output.images,
      output.url,
      output.urls,
    ];

    for (const c of candidates) {
      const url = pickMaskUrl(c);
      if (url) return url;
    }
  }

  return null;
}

/* ---------------- Safe JSON ---------------- */

async function safeJson(res: Response) {
  const text = await res.text();
  try {
    return JSON.parse(text);
  } catch {
    return { raw: text };
  }
}

/* ---------------- Safe base64 ---------------- */

function arrayBufferToBase64(buffer: ArrayBuffer) {
  // Safe base64 encoder that avoids building a massive binary string.
  const bytes = new Uint8Array(buffer);
  const base64abc =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  let result = "";
  let i = 0;

  for (; i + 2 < bytes.length; i += 3) {
    result += base64abc[bytes[i] >> 2];
    result += base64abc[((bytes[i] & 3) << 4) | (bytes[i + 1] >> 4)];
    result += base64abc[((bytes[i + 1] & 15) << 2) | (bytes[i + 2] >> 6)];
    result += base64abc[bytes[i + 2] & 63];
  }

  if (i < bytes.length) {
    result += base64abc[bytes[i] >> 2];
    if (i === bytes.length - 1) {
      result += base64abc[(bytes[i] & 3) << 4];
      result += "==";
    } else {
      result += base64abc[((bytes[i] & 3) << 4) | (bytes[i + 1] >> 4)];
      result += base64abc[(bytes[i + 1] & 15) << 2];
      result += "=";
    }
  }

  return result;
}
