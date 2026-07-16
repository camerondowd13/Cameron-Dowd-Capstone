// Receives Apollo's async phone-reveal callback (see docs.apollo.io/docs/
// retrieve-mobile-phone-numbers-for-contacts): a people/match call with
// reveal_phone_number=true returns 200 immediately with NO phone number in
// it -- Apollo POSTs the actual number here, separately, a few minutes
// later. This just lands that payload in Supabase; apollo_client.py's
// poll_phone_reveal() picks it up from there.
//
// Token check: without it, this URL is public (Apollo's webhook_url must
// be reachable with no auth of its own), so anyone who finds it could POST
// fake phone numbers into the table. APOLLO_WEBHOOK_SECRET must match the
// token embedded in the APOLLO_WEBHOOK_URL query string that apollo_client.py sends to Apollo.

const SUPABASE_URL = 'https://jwwvmvanpsfcizivrjse.supabase.co';
const SUPABASE_ANON_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imp3d3ZtdmFucHNmY2l6aXZyanNlIiwicm9sZSI6ImFub24iLCJpYXQiOjE3ODM1OTk4MjAsImV4cCI6MjA5OTE3NTgyMH0.gZI7t6X8M-YyQ7YpLPfvLcuqLNXtB0e0hZM7bLxV91o';

export default async function handler(req, res) {
  if (req.method !== 'POST') {
    return res.status(405).json({ ok: false, error: 'POST only' });
  }
  if (!process.env.APOLLO_WEBHOOK_SECRET || req.query.token !== process.env.APOLLO_WEBHOOK_SECRET) {
    return res.status(401).json({ ok: false, error: 'invalid or missing token' });
  }

  const people = req.body?.people || [];
  const rows = people
    .filter((p) => p?.id && p?.phone_numbers?.[0]?.sanitized_number)
    .map((p) => ({
      apollo_person_id: p.id,
      phone: p.phone_numbers[0].sanitized_number,
      raw_payload: p,
    }));

  if (rows.length === 0) {
    // Apollo still calls this a "success" webhook even when a lookup came
    // back empty (e.g. no valid number found) -- nothing to store, not an error.
    return res.status(200).json({ ok: true, stored: 0 });
  }

  const resp = await fetch(
    `${SUPABASE_URL}/rest/v1/apollo_phone_reveals?on_conflict=apollo_person_id`,
    {
      method: 'POST',
      headers: {
        apikey: SUPABASE_ANON_KEY,
        Authorization: `Bearer ${SUPABASE_ANON_KEY}`,
        'Content-Type': 'application/json',
        Prefer: 'resolution=merge-duplicates',
      },
      body: JSON.stringify(rows),
    }
  );

  if (!resp.ok) {
    const text = await resp.text();
    return res.status(502).json({ ok: false, error: `Supabase write failed: ${text}` });
  }

  res.status(200).json({ ok: true, stored: rows.length });
}
