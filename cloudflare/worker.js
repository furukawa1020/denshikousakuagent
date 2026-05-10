const CORS_HEADERS = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
  "Access-Control-Allow-Headers": "Content-Type, Authorization, X-Requested-With",
  "Access-Control-Max-Age": "86400",
};

export default {
  async fetch(request, env) {
    if (request.method === "OPTIONS") {
      return new Response(null, { status: 204, headers: CORS_HEADERS });
    }

    const url = new URL(request.url);
    if (url.pathname === "/api/health") {
      return json({
        ok: true,
        edge: true,
        service: "Lチカのつづき",
        gpuApiConfigured: Boolean(env.GPU_API_BASE),
      });
    }

    if (url.pathname.startsWith("/api/")) {
      return proxyToGpuBackend(request, env, url);
    }

    return json({ ok: false, error: "Not found" }, 404);
  },
};

async function proxyToGpuBackend(request, env, incomingUrl) {
  if (!env.GPU_API_BASE) {
    return json({
      ok: false,
      error: "GPU_API_BASE is not configured.",
      hint: "Set it with: wrangler secret put GPU_API_BASE",
    }, 503);
  }

  const upstream = new URL(env.GPU_API_BASE);
  upstream.pathname = joinPath(upstream.pathname, incomingUrl.pathname);
  upstream.search = incomingUrl.search;

  const headers = new Headers(request.headers);
  headers.delete("host");
  headers.set("x-forwarded-host", incomingUrl.host);
  headers.set("x-edge-provider", "cloudflare-workers");

  const init = {
    method: request.method,
    headers,
    redirect: "manual",
  };
  if (request.method !== "GET" && request.method !== "HEAD") {
    init.body = request.body;
  }

  try {
    const response = await fetch(upstream.toString(), init);
    const nextHeaders = new Headers(response.headers);
    for (const [key, value] of Object.entries(CORS_HEADERS)) {
      nextHeaders.set(key, value);
    }
    nextHeaders.set("Cache-Control", "no-store");
    return new Response(response.body, {
      status: response.status,
      statusText: response.statusText,
      headers: nextHeaders,
    });
  } catch (error) {
    return json({
      ok: false,
      error: "GPU backend is unreachable.",
      detail: String(error?.message || error),
    }, 502);
  }
}

function json(payload, status = 200) {
  return new Response(JSON.stringify(payload, null, 2), {
    status,
    headers: {
      "Content-Type": "application/json; charset=utf-8",
      "Cache-Control": "no-store",
      ...CORS_HEADERS,
    },
  });
}

function joinPath(basePath, nextPath) {
  const base = basePath.endsWith("/") ? basePath.slice(0, -1) : basePath;
  const next = nextPath.startsWith("/") ? nextPath : `/${nextPath}`;
  return `${base}${next}` || "/";
}
