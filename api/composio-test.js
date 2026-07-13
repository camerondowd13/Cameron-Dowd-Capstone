import { Composio } from '@composio/core';

export default async function handler(req, res) {
  if (!process.env.COMPOSIO_API_KEY) {
    return res.status(500).json({ ok: false, error: 'COMPOSIO_API_KEY is not set' });
  }

  const composio = new Composio({ apiKey: process.env.COMPOSIO_API_KEY });

  try {
    const toolkits = await composio.toolkits.get();
    res.status(200).json({ ok: true, toolkitCount: toolkits.length });
  } catch (err) {
    res.status(500).json({ ok: false, error: err.message });
  }
}
