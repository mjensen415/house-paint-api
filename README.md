# House Paint API Worker

A Cloudflare Worker that generates masking for house painting using Replicate's FastSAM model.

## Features
- **Endpoint**: `POST /house-mask`
- **Input**: `multipart/form-data`
    - `image`: The image file to mask (JPEG/PNG).
    - `scene`: "exterior" (default) or "interior".
    - `mode`: "auto" (default) or "tap".
    - `tapX`, `tapY`: Optional normalized coordinates for tap-based masking (not currently strictly enforced, falls back to text prompt if model limitations exist).
- **Output**: JSON `{ "maskPngBase64": "..." }` containing the base64-encoded binary mask PNG.
- **Image Processing**:
    - Automatically keeps the largest connected component (main structure).
    - Fills small holes in the mask.
    - Returns a cleaned binary mask (white = subject, transparent = background).

## Setup & Deployment

1.  **Install Dependencies**:
    ```bash
    npm install
    ```

2.  **Set Replicate API Token**:
    You need a Replicate API token to use the FastSAM model.
    ```bash
    npx wrangler secret put REPLICATE_API_TOKEN
    ```
    (Paste your token when prompted).

3.  **Deploy**:
    ```bash
    npx wrangler deploy
    ```

## Example Usage

### cURL

```bash
curl -X POST https://house-paint-api.<YOUR_SUBDOMAIN>.workers.dev/house-mask \
  -F "image=@/path/to/house.jpg" \
  -F "scene=exterior" \
  -F "mode=auto"
```

### Response

```json
{
  "maskPngBase64": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/w8AAwAB/AL+f4AAAAASUVORK5CYII="
}
```

## Development

- Run locally:
  ```bash
  npx wrangler dev
  ```
