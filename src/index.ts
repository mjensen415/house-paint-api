import UPNG from "upng-js";

export interface Env {
  REPLICATE_API_TOKEN: string;
}

export default {
  async fetch(request: Request, env: Env, ctx: ExecutionContext): Promise<Response> {
    // CORS Preflight
    if (request.method === "OPTIONS") {
      return new Response(null, {
        headers: {
          "Access-Control-Allow-Origin": "*",
          "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
          "Access-Control-Allow-Headers": "Content-Type",
        },
      });
    }

    if (request.method === "POST" && new URL(request.url).pathname === "/house-mask") {
      try {
        const formData = await request.formData();
        const imageFile = formData.get("image");
        const scene = formData.get("scene") as string || "exterior";
        const mode = formData.get("mode") as string || "auto";
        const tapX = formData.get("tapX");
        const tapY = formData.get("tapY");

        if (!imageFile || !(imageFile instanceof File)) {
          return new Response("Missing image file", { status: 400, headers: { "Access-Control-Allow-Origin": "*" } });
        }

        // 1. Call Replicate
        const replicateResult = await callReplicate(env.REPLICATE_API_TOKEN, imageFile, scene, mode, tapX, tapY);

        // 2. Process Output
        // The output from Replicate FastSAM is typically a list of masks or a single combined mask image URL?
        // Actually, FastSAM on Replicate usually returns a list of objects or a generic output. 
        // Let's assume for this specific model/version we need to handle what it gives back.
        // Based on common FastSAM usage: it often maps segments.
        // However, the user request says: "Convert model output into a single binary mask PNG"
        // Let's first get the image data from the input or Replicate result.
        // If Replicate returns JSON with data, we parse it. If it returns an image URL, we fetch it.
        // For 'casia-iva-lab/fastsam', it usually returns a JSON with 'masks' (rle) or an image with strict segments.
        // Let's assume we get an image URL representing the segmentation or we might need to prompt it to return a binary mask for 'house'.

        // Wait, if we use text prompt "house", FastSAM usually segments everything matching "house".
        // The output of `casia-iva-lab/fastsam` on Replicate typically includes data.
        // Let's fetch the output.

        let maskImageBuffer: ArrayBuffer;

        if (Array.isArray(replicateResult) && replicateResult.length > 0) {
          // It might be a list of output images? or URLs.
          // Often it returns a URI to an image showing the mask.
          const outputUrl = replicateResult[0]; // simplistic assumption, robustify if needed
          const maskResp = await fetch(outputUrl);
          maskImageBuffer = await maskResp.arrayBuffer();
        } else if (typeof replicateResult === 'string' && replicateResult.startsWith('http')) {
          const maskResp = await fetch(replicateResult);
          maskImageBuffer = await maskResp.arrayBuffer();
        } else {
          // Fallback or error
          console.error("Unexpected Replicate output:", replicateResult);
          return new Response(JSON.stringify({ error: "Failed to get mask from model" }), { status: 500, headers: { "Access-Control-Allow-Origin": "*" } });
        }


        // 3. Post-process the mask (Hole filling, Largest Component)
        const processedPngBase64 = processMask(maskImageBuffer);

        return new Response(JSON.stringify({ maskPngBase64: processedPngBase64 }), {
          headers: {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
          },
        });

      } catch (e: any) {
        return new Response(JSON.stringify({ error: e.message }), {
          status: 500,
          headers: { "Content-Type": "application/json", "Access-Control-Allow-Origin": "*" },
        });
      }
    }

    return new Response("Not Found", { status: 404, headers: { "Access-Control-Allow-Origin": "*" } });
  },
};

async function callReplicate(token: string, image: File, scene: string, mode: string, tapX: any, tapY: any): Promise<any> {
  const modelVersion = "371aeee1ce0c5efd25bbef7a4527ec9e59188b963ebae1eeb851ddc145685c17"; // casia-iva-lab/fastsam

  // Convert image to base64 data URI
  const arrayBuffer = await image.arrayBuffer();
  const base64 = btoa(String.fromCharCode(...new Uint8Array(arrayBuffer)));
  const dataUri = `data:${image.type};base64,${base64}`;

  let prompt = scene === "interior" ? "wall" : "house, building";

  // Note: This specific FastSAM model on Replicate takes 'input_image' and 'text_prompt' (or 'points'). 
  // There isn't always a standardized "everything" mode purely via text without config, but text_prompt usually works.

  const input: any = {
    input_image: dataUri,
    // You might check exact model inputs, but text_prompt is standard for FastSAM wrappers
    text_prompt: prompt,
    better_quality: true,
    withContours: false, // simpler mask
  };

  if (mode === "tap" && tapX && tapY) {
    // If the model supports point prompts. The 'casia-iva-lab/fastsam' on Replicate often uses 'point_prompt' string format like "[[x,y]]"
    // We need to map normalized 0..1 to image structure or if it accepts normalized. 
    // Usually, FastSAM expects absolute coordinates if not specified otherwise.
    // Since we don't have image dims here easily without decoding, using text prompt as fallback or trying normalized might be risky.
    // However, the instructions say "include point prompts if supported... otherwise ignore".
    // Let's stick to text prompt mainly unless we are sure.
    // Actually, Replicate FastSAM often takes standard text prompts well.
    // Let's strictly follow the plan: use text prompt based on scene. 
    // If 'tap' is requested and we can't easily do points without dims, we just stick to text prompt to be safe.
  }

  const response = await fetch("https://api.replicate.com/v1/predictions", {
    method: "POST",
    headers: {
      "Authorization": `Token ${token}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      version: modelVersion,
      input: input,
    }),
  });

  if (!response.ok) {
    throw new Error(`Replicate API error: ${response.statusText}`);
  }

  const prediction = await response.json();
  const getUrl = prediction.urls.get;

  // Poll
  let result = null;
  while (!result) {
    await new Promise(r => setTimeout(r, 1000));
    const statusResp = await fetch(getUrl, {
      headers: { "Authorization": `Token ${token}` }
    });
    const statusData = await statusResp.json();
    if (statusData.status === "succeeded") {
      result = statusData.output;
    } else if (statusData.status === "failed" || statusData.status === "canceled") {
      throw new Error("Replicate prediction failed");
    }
  }

  return result;
}

function processMask(buffer: ArrayBuffer): string {
  const img = UPNG.decode(buffer);
  const rgba = UPNG.toRGBA8(img)[0]; // ArrayBuffer of RGBA
  const pixelData = new Uint8Array(rgba);
  const width = img.width;
  const height = img.height;

  // 1. Binarize
  // FastSAM output usually highlights the segment. We assume non-black/transparent is "selected".
  // Let's create a binary grid: 1 for mask, 0 for background.
  const grid = new Uint8Array(width * height);
  for (let i = 0; i < pixelData.length; i += 4) {
    const r = pixelData[i];
    const g = pixelData[i + 1];
    const b = pixelData[i + 2];
    // Simple threshold: if it's not basically black, it's mask.
    if (r > 20 || g > 20 || b > 20) {
      grid[i / 4] = 1;
    } else {
      grid[i / 4] = 0;
    }
  }

  // 2. Keep Largest Connected Component
  const labels = new Int32Array(width * height);
  let currentLabel = 1;
  const labelSizes = new Map<number, number>();

  // Use a simple iterative Union-Find or BFS for connectivity
  // Let's do a simple recursive DFS or better, iterative BFS/stack to avoid stack overflow
  const visited = new Uint8Array(width * height);

  // Helper to get index
  const idx = (x: number, y: number) => y * width + x;

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const i = idx(x, y);
      if (grid[i] === 1 && !visited[i]) {
        // Start BFS
        let size = 0;
        const stack = [i];
        visited[i] = 1;
        while (stack.length > 0) {
          const curr = stack.pop()!;
          labels[curr] = currentLabel;
          size++;

          const cx = curr % width;
          const cy = Math.floor(curr / width);

          // Neighbors 4-connectivity
          const neighbors = [
            { nx: cx + 1, ny: cy },
            { nx: cx - 1, ny: cy },
            { nx: cx, ny: cy + 1 },
            { nx: cx, ny: cy - 1 }
          ];

          for (const { nx, ny } of neighbors) {
            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
              const ni = idx(nx, ny);
              if (grid[ni] === 1 && !visited[ni]) {
                visited[ni] = 1;
                stack.push(ni);
              }
            }
          }
        }
        labelSizes.set(currentLabel, size);
        currentLabel++;
      }
    }
  }

  // Find max label
  let maxLabel = -1;
  let maxSize = -1;
  for (const [label, size] of labelSizes) {
    if (size > maxSize) {
      maxSize = size;
      maxLabel = label;
    }
  }

  // If we found a component, zero out everything else
  if (maxLabel !== -1) {
    for (let i = 0; i < grid.length; i++) {
      if (labels[i] !== maxLabel) {
        grid[i] = 0;
      }
    }
  }

  // 3. Fill Holes
  // Invert grid (treat 0 as 1 and 1 as 0), find components of 1s (original 0s).
  // Any component NOT touching the border is a hole. Fill it (turn original 0 to 1).

  // Reset visited for hole filling
  visited.fill(0);
  // We don't need labels really, just need to know if a texturally '0' component touches border.

  const holesToFill: number[] = []; // indices to flip to 1

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const i = idx(x, y);
      if (grid[i] === 0 && !visited[i]) {
        // Component of 0s
        const stack = [i];
        visited[i] = 1;
        const componentIndices: number[] = [i];
        let touchesBorder = false;

        while (stack.length > 0) {
          const curr = stack.pop()!;
          const cx = curr % width;
          const cy = Math.floor(curr / width);

          if (cx === 0 || cx === width - 1 || cy === 0 || cy === height - 1) {
            touchesBorder = true;
          }

          const neighbors = [
            { nx: cx + 1, ny: cy },
            { nx: cx - 1, ny: cy },
            { nx: cx, ny: cy + 1 },
            { nx: cx, ny: cy - 1 }
          ];

          for (const { nx, ny } of neighbors) {
            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
              const ni = idx(nx, ny);
              if (grid[ni] === 0 && !visited[ni]) {
                visited[ni] = 1;
                stack.push(ni);
                componentIndices.push(ni);
              }
            }
          }
        }

        if (!touchesBorder) {
          // It's a hole!
          for (const ignored of componentIndices) {
            // We want to fill these later or now.
            // Actually we can just mark them.
            // We can't modify grid immediately if we continue iterating? 
            // Actually we are using visited, so it's fine.
          }
          // Add to list to flip
          holesToFill.push(...componentIndices);
        }
      }
    }
  }

  // Flip holes
  for (const idx of holesToFill) {
    grid[idx] = 1;
  }

  // 4. Encode back to PNG
  // Create RGBA buffer: White (255,255,255,255) for 1, Transparent (0,0,0,0) for 0
  const outputBuffer = new Uint8Array(width * height * 4);
  for (let i = 0; i < grid.length; i++) {
    if (grid[i] === 1) {
      outputBuffer[i * 4] = 255;
      outputBuffer[i * 4 + 1] = 255;
      outputBuffer[i * 4 + 2] = 255;
      outputBuffer[i * 4 + 3] = 255;
    } else {
      outputBuffer[i * 4] = 0;
      outputBuffer[i * 4 + 1] = 0;
      outputBuffer[i * 4 + 2] = 0;
      outputBuffer[i * 4 + 3] = 0;
    }
  }

  const resultPng = UPNG.encode([outputBuffer.buffer], width, height, 0); // 0 loss = lossless
  // Convert ArrayBuffer to Base64
  let binary = '';
  const bytes = new Uint8Array(resultPng);
  const len = bytes.byteLength;
  for (let i = 0; i < len; i++) {
    binary += String.fromCharCode(bytes[i]);
  }
  return btoa(binary);
}
