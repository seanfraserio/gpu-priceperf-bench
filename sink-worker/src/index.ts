/**
 * Result sink. Rented GPUs PUT their BenchResult here; the Worker
 * authenticates the write and stores the object in R2.
 *
 * The instances are ephemeral and powered off the moment a run finishes, so
 * this is the only place a result survives. Writes are authenticated because
 * an open endpoint would let anyone forge a published benchmark number.
 */

export interface Env {
  RESULTS: R2Bucket;
  SINK_TOKEN: string;
}

// run_id is used verbatim as the object key, so it is constrained here rather
// than trusted: no slashes, no traversal, no surprises in the bucket layout.
const KEY_PATTERN = /^[A-Za-z0-9][A-Za-z0-9._-]{0,127}\.json$/;

// A full sweep result is a few hundred KB. Anything far past that is not ours.
const MAX_BYTES = 5 * 1024 * 1024;

function unauthorized(): Response {
  return new Response("unauthorized\n", { status: 401 });
}

/** Constant-time compare, so a wrong token leaks nothing through timing. */
function tokenMatches(presented: string, expected: string): boolean {
  const a = new TextEncoder().encode(presented);
  const b = new TextEncoder().encode(expected);
  if (a.byteLength !== b.byteLength) return false;
  let diff = 0;
  for (let i = 0; i < a.byteLength; i++) diff |= a[i] ^ b[i];
  return diff === 0;
}

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    const url = new URL(request.url);
    const key = url.pathname.replace(/^\/+/, "");

    if (
      request.method !== "PUT" &&
      request.method !== "GET" &&
      request.method !== "DELETE"
    ) {
      return new Response("method not allowed\n", {
        status: 405,
        headers: { Allow: "GET, PUT, DELETE" },
      });
    }

    if (!env.SINK_TOKEN) {
      // Fail closed. A Worker deployed without its secret must never accept
      // writes just because the comparison has nothing to compare against.
      return new Response("sink misconfigured\n", { status: 500 });
    }

    const header = request.headers.get("Authorization") ?? "";
    const presented = header.startsWith("Bearer ") ? header.slice(7) : "";
    if (!presented || !tokenMatches(presented, env.SINK_TOKEN)) {
      return unauthorized();
    }

    // Reading is authenticated too: results are the published record, and the
    // operator needs to enumerate what a run uploaded before committing it.
    if (request.method === "GET") {
      if (key === "" || key === "_list") {
        const listed = await env.RESULTS.list({ limit: 1000 });
        return Response.json({
          objects: listed.objects.map((o) => ({
            key: o.key,
            size: o.size,
            uploaded: o.uploaded,
          })),
          truncated: listed.truncated,
        });
      }
      const object = await env.RESULTS.get(key);
      if (!object) return new Response("not found\n", { status: 404 });
      return new Response(object.body, {
        headers: { "Content-Type": "application/json" },
      });
    }

    if (!KEY_PATTERN.test(key)) {
      return new Response("invalid key\n", { status: 400 });
    }

    // Deleting is authenticated and deliberate. A result that turns out to
    // have been measured wrongly is worse than no result — it is the published
    // record — but removing one should never be something a stray request can
    // do, so it takes an explicit method and a well-formed key.
    if (request.method === "DELETE") {
      const existing = await env.RESULTS.head(key);
      if (!existing) return new Response("not found\n", { status: 404 });
      await env.RESULTS.delete(key);
      return Response.json({ ok: true, deleted: key });
    }

    const declared = Number(request.headers.get("Content-Length") ?? "0");
    if (declared > MAX_BYTES) {
      return new Response("payload too large\n", { status: 413 });
    }

    const body = await request.arrayBuffer();
    if (body.byteLength > MAX_BYTES) {
      return new Response("payload too large\n", { status: 413 });
    }

    // Reject anything that is not the JSON we expect, so a corrupt upload
    // fails here rather than at report time.
    try {
      JSON.parse(new TextDecoder().decode(body));
    } catch {
      return new Response("body is not valid JSON\n", { status: 400 });
    }

    // Repeat PUTs of the same run_id overwrite by design: each one carries a
    // more complete result than the last.
    await env.RESULTS.put(key, body, {
      httpMetadata: { contentType: "application/json" },
    });

    return new Response(JSON.stringify({ ok: true, key }), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });
  },
};
