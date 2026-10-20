// Host-side decoding fixes from the 2026-10-17 audit (demo/audit/REPORT.md). Model weights are untouched; these change
// how head outputs become calls. All off = the Python-parity decoder (golden tests); Relay.create({fixes}) sets them.
export const FIX = {
  enumOnly: false,     // closed enums: only the enum op (no copied words like position "Close")
  numReject: false,    // integer/number params never take a non-numeric value ("Max" volume -> 0%)
  numSpan: false,      // one span fills one numeric param too ("an hour" -> minutes=60 and seconds=60)
  perCallHold: false,  // a call missing a required value is held alone; complete sibling calls still run
  orderTie: false,     // calls on the same anchor are ordered by where their values are stated, not tool index
  dedupe: false,       // drop exact duplicate calls
  chunks: false,       // balanced tool chunks; calls from later chunks must compete with the best chunk
  caps: false,         // lower-case ALL-CAPS requests (the cased tokenizer fragments capitals)
  locale: false,       // number parser: currency symbols, decimal comma, "and a half", "point five"
};
export const ALL_ON = Object.fromEntries(Object.keys(FIX).filter(k => k !== 'orderTie').map(k => [k, true]));  // orderTie measured no gain
export function setFixes(f) {
  const v = f === true ? ALL_ON : f === false || !f ? {} : f;
  for (const k of Object.keys(FIX)) FIX[k] = !!v[k];
  return {...FIX};
}
