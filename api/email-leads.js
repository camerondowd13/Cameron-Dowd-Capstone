// Proxies the "Email these leads to me" button on the public try-it-free
// page (contact.html) to Render (server.py -> /email-leads), which sends the
// AE's daily-report-style email to their own inbox for the demo.
//
// Same secrets-stay-server-side pattern as api/find-accounts.js:
//   - RENDER_SERVICE_API_KEY gates the Render call.
//   - SUPABASE_SERVICE_ROLE_KEY reads/writes demo_rate_limits.
//
// Rate limit: shares the same 8-per-IP-per-24h demo budget as searches
// (one table, no type column). The email only ever goes to the AE's own
// fixed inbox, so the worst abuse is inbox-flooding -- capping it at the
// shared demo budget is enough, and clicking the button counts as one demo
// action, same as a search.

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

  const body = req.body || {};
  const leads = Array.isArray(body.leads) ? body.leads : [];
  if (leads.length === 0) {
    return res.status(400).json({ error: "'leads' is required" });
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
      message: "You've hit today's free demo limit. Try again tomorrow.",
    });
  }

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
    const renderResp = await fetch(`${renderBaseUrl}/email-leads`, {
      method: 'POST',
      headers: {
        'X-API-Key': renderApiKey,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ leads, territory: body.territory || '' }),
    });

    const data = await renderResp.json();
    if (!renderResp.ok) {
      return res.status(renderResp.status).json(data);
    }
    return res.status(200).json(data);
  } catch (e) {
    return res.status(502).json({ error: `email send failed: ${e.message}` });
  }
}
