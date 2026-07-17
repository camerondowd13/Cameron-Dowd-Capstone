// Proxies the public "try it free" page (contact.html) to the real
// prospecting agent running on Render (server.py -> account_finder.find_accounts).
//
// Two things live server-side here, on purpose, never in browser JS:
//   - RENDER_SERVICE_API_KEY: gates a real Anthropic (Opus) + Exa spend per call.
//   - SUPABASE_SERVICE_ROLE_KEY: reads/writes demo_rate_limits, which has no
//     anon-key access at all (see supabase-migrations.sql).
//
// Rate limit: 8 live runs per IP per 24h. Checked BEFORE calling Render so a
// capped-out visitor never triggers a real agent run.

const SUPABASE_URL = 'https://jwwvmvanpsfcizivrjse.supabase.co';
const RATE_LIMIT = 8;
const WINDOW_MS = 24 * 60 * 60 * 1000;

function getClientIp(req) {
  const forwarded = req.headers['x-forwarded-for'];
  if (forwarded) return forwarded.split(',')[0].trim();
  return req.socket?.remoteAddress || 'unknown';
}

export default async function handler(req, res) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'POST only' });
  }

  const serviceRoleKey = process.env.SUPABASE_SERVICE_ROLE_KEY;
  const renderBaseUrl = process.env.RENDER_BASE_URL;
  const renderApiKey = process.env.RENDER_SERVICE_API_KEY;
  if (!serviceRoleKey || !renderBaseUrl || !renderApiKey) {
    return res.status(500).json({ error: 'server misconfigured' });
  }

  const ip = getClientIp(req);
  const since = new Date(Date.now() - WINDOW_MS).toISOString();

  const countResp = await fetch(
    `${SUPABASE_URL}/rest/v1/demo_rate_limits?select=id&ip=eq.${encodeURIComponent(ip)}&requested_at=gte.${encodeURIComponent(since)}`,
    {
      headers: {
        apikey: serviceRoleKey,
        Authorization: `Bearer ${serviceRoleKey}`,
      },
    }
  );

  if (!countResp.ok) {
    const text = await countResp.text();
    return res.status(502).json({ error: `rate-limit check failed: ${text}` });
  }

  const existingRows = await countResp.json();
  if (existingRows.length >= RATE_LIMIT) {
    return res.status(429).json({
      error: 'rate_limited',
      message: "You've hit today's free limit for live searches. Try again tomorrow, or get in touch for early access.",
    });
  }

  const body = req.body || {};
  const state = body.state;
  const industry = body.industry || null;
  const title = body.title || null;
  if (!state) {
    return res.status(400).json({ error: "'state' is required" });
  }

  // Record this run before calling Render -- if find_accounts throws partway
  // through, the visitor has still consumed a real Anthropic/Exa call, so the
  // count should reflect that regardless of whether it succeeds.
  const recordResp = await fetch(`${SUPABASE_URL}/rest/v1/demo_rate_limits`, {
    method: 'POST',
    headers: {
      apikey: serviceRoleKey,
      Authorization: `Bearer ${serviceRoleKey}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ ip }),
  });

  if (!recordResp.ok) {
    const text = await recordResp.text();
    return res.status(502).json({ error: `rate-limit record failed: ${text}` });
  }

  try {
    const renderResp = await fetch(`${renderBaseUrl}/find-accounts`, {
      method: 'POST',
      headers: {
        'X-API-Key': renderApiKey,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        state,
        industry,
        limit: 3,
        target_titles: title ? [title] : undefined,
      }),
    });

    const data = await renderResp.json();
    if (!renderResp.ok) {
      return res.status(renderResp.status).json(data);
    }
    return res.status(200).json(data);
  } catch (e) {
    return res.status(502).json({ error: `agent call failed: ${e.message}` });
  }
}
