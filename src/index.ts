export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    if (request.method === "OPTIONS") return new Response(null, { headers: cors(request) });

    const url = new URL(request.url);

    if (request.method === "GET" && url.pathname === "/") {
      return new Response("worker alive", { headers: cors(request) });
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

    const ct = request.headers.get("content-type") || "";
    if (!ct.toLowerCase().includes("application/json")) {
      return json(request, { error: "Expected application/json" }, 400);
    }

    let body: any;
    try {
      body = await request.json();
    } catch {
      return json(request, { error: "Invalid JSON" }, 400);
    }

    const scene = String(body.scene || "exterior").toLowerCase();
    const imageDataUri = String(body.imageDataUri || "");

    if (!imageDataUri.startsWith("data:image/")) {
      return json(request, { error: "imageDataUri must be a data:image/* base64 data URI" }, 400);
    }

    // Replicate recommends data URIs only for smaller files. Keep it tight.  [oai_citation:2‡Replicate](https://replicate.com/docs/topics/predictions/input-files?utm_source=chatgpt.com)
    if (imageDataUri.length > 1_500_000) {
      return json(request, { error: "Image too large. Resize/compress to under ~1MB and retry." }, 413);
    }

    const textPrompt = scene === "interior" ? "wall" : "house, building exterior";

    // FastSAM model version id (pin it to avoid changes)
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
          input_image: imageDataUri,
          text_prompt: textPrompt,
          conf: 0.4,
          iou: 0.9,
          retina: true,
        },
      }),
    });

    const created = await safeJson(createRes);
    if (!createRes.ok) {
      return json(request, { error: "Replicate create failed", status: createRes.status, details: created }, 502);
    }

    const getUrl = created?.urls?.get;
    if (!getUrl) {
      return json(request, { error: "Replicate response missing urls.get", details: created }, 502);
    }

    const result = await pollReplicate(getUrl, env.REPLICATE_API_TOKEN);
    const maskUrl = pickFirstUrl(result?.output);

    if (!maskUrl) {
      return json(request, { error: "No mask URL found in output", details: result }, 502);
    }

    return json(request, { maskUrl }, 200);
  },
};

type Env = { REPLICATE_API_TOKEN?: string };

function cors(request: Request) {
  const origin = request.headers.get("origin") || "*";
  return {
    "Access-Control-Allow-Origin": origin === "null" ? "*" : origin,
    "Access-Control-Allow-Methods": "GET,POST,OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type",
    "Vary": "Origin",
  };
}

function json(request: Request, data: any, status = 200) {
  return new Response(JSON.stringify(data), {
    status,
    headers: { ...cors(request), "Content-Type": "application/json" },
  });
}

async function safeJson(res: Response) {
  const text = await res.text();
  try {
    return JSON.parse(text);
  } catch {
    return { raw: text };
  }
}

async function pollReplicate(getUrl: string, token: string) {
  for (let i = 0; i < 30; i++) {
    const res = await fetch(getUrl, { headers: { Authorization: `Bearer ${token}` } });
    const body = await safeJson(res);
    if (!res.ok) throw new Error(`Replicate poll failed: ${res.status}`);
    if (body.status === "succeeded") return body;
    if (body.status === "failed" || body.status === "canceled") throw new Error(`Replicate status: ${body.status}`);
    await new Promise((r) => setTimeout(r, 1000));
  }
  throw new Error("Replicate prediction timed out");
}

function pickFirstUrl(output: any): string | null {
  if (!output) return null;
  if (typeof output === "string" && output.startsWith("http")) return output;
  if (Array.isArray(output)) {
    const u = output.find((x) => typeof x === "string" && x.startsWith("http"));
    return u || null;
  }
  if (typeof output === "object") {
    return pickFirstUrl(output.url || output.urls || output.mask || output.masks || output.image || output.images);
  }
  return null;
}
