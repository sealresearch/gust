// Shared by the page, the worker and the Node test harness. JSON-schema tools (OpenAI / Needle style, plus optional
// `triggers`: regex strings) -> Relay tool records, trigger routing, and the run record.

function param(name, s, required) {
  const p = {name, type: s.type || 'string', description: s.description || '', required: !!required};
  if (s.enum) p.enum = s.enum;
  if ('default' in s) p.default = s.default;
  if (s.items) p.items = param('item', s.items, false);
  if (s.properties) p.properties = Object.entries(s.properties).map(([k, v]) => param(k, v, (s.required || []).includes(k)));
  if (s.minimum !== undefined || s.maximum !== undefined)
    p.description = `${p.description}${p.description ? ' ' : ''}(${s.minimum ?? ''}-${s.maximum ?? ''})`.trim();
  return p;
}

/** JSON-schema tools -> Relay's {name, description, params[]} */
export function toRelayTools(tools) {
  return tools.map(t => {
    const ps = (t.parameters && t.parameters.properties) || {};
    const req = (t.parameters && t.parameters.required) || [];
    return {name: t.name, description: t.description || '', params: Object.entries(ps).map(([k, v]) => param(k, v, req.includes(k)))};
  });
}

/** Tools whose triggers match the query (case-insensitive), or [] */
export function routed(tools, query) {
  return tools.filter(t => (t.triggers || []).some(rx => { try { return new RegExp(rx, 'i').test(query); } catch { return false; } }));
}

/** Local ISO time without zone, e.g. 2026-10-17T12:41:00 */
export function nowISO(d = new Date()) {
  const z = n => String(n).padStart(2, '0');
  return `${d.getFullYear()}-${z(d.getMonth() + 1)}-${z(d.getDate())}T${z(d.getHours())}:${z(d.getMinutes())}:${z(d.getSeconds())}`;
}

/**
 * The record Relay sees. Trigger routing (like Needle's): if any tool's trigger matches, only the matched tools are
 * offered and the best of them is always called (Relay's extraction mode forces a call).
 */
export function buildRecord(tools, query, {now = nowISO(), mode = 'call'} = {}) {
  const hit = mode === 'call' ? routed(tools, query) : [];
  const use = hit.length ? hit : tools;
  const day = new Date(now).toLocaleDateString('en-US', {weekday: 'long'});
  return {
    record: {id: 'demo', text: query, tools: toRelayTools(use), kind: hit.length || mode === 'extract' ? 'extract' : 'call',
      context: {now, system: `Current date and time: ${now}\nDay of week is ${day}`}},
    routedBy: hit.map(t => t.name),
  };
}
