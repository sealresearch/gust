// Structured extraction: messy text in, typed fields out. Relay picks the best-fitting schema (like SNIPS 7-way)
// and fills its fields; every value is a span or a parsed number/date from the text, so nothing is invented.

const tool = (name, description, props, required = []) => ({name, description, parameters: {type: 'object', properties: props, required}});
const s = d => ({type: 'string', description: d});
const n = d => ({type: 'number', description: d});

export default {
  key: 'extract',
  mode: 'extract',
  label: 'Extract',
  subtitle: 'messy text into typed fields',
  tagline: 'Paste text; Relay picks the matching schema and fills it. Edit the schemas on the left to extract anything.',
  placeholder: 'paste an email, receipt, message or notification...',
  tools: [
    tool('invoice', 'An invoice or bill to pay.', {vendor: s('Company that sent the invoice.'), invoice_number: s('Invoice number or ID.'),
      amount: n('Total amount due.'), currency: s('Currency code or symbol.'), due_date: s('Payment due date.')}),
    tool('restaurant_booking', 'A table reservation at a restaurant.', {restaurant: s('Restaurant name.'), party_size: {type: 'integer', description: 'Number of people.'},
      date: s('Date of the booking.'), time: s('Time of the booking.'), name: s('Name the booking is under.')}),
    tool('card_payment', 'A card transaction notification from a bank.', {merchant: s('Where the money was spent.'), amount: n('Amount charged.'),
      currency: s('Currency.'), card_last4: s('Last four digits of the card.'), date: s('When it happened.')}),
    tool('shipment', 'A parcel or delivery update.', {carrier: s('Delivery company.'), tracking_number: s('Tracking number.'),
      status: {type: 'string', enum: ['shipped', 'out for delivery', 'delivered', 'delayed'], description: 'Delivery status.'}, eta: s('Expected delivery date or time.')}),
    tool('meeting', 'A meeting or calendar invitation.', {title: s('What the meeting is about.'), date: s('Meeting date.'), time: s('Start time.'),
      location: s('Room, address or video link.'), organizer: s('Who sent the invitation.')}),
    tool('contact', 'Contact details of a person.', {name: s('Full name.'), phone: s('Phone number.'), email: s('Email address.'), company: s('Company or organisation.')}),
    tool('flight', 'A flight booking or boarding pass.', {airline: s('Airline.'), flight_number: s('Flight number.'), from: s('Departure city or airport.'),
      to: s('Arrival city or airport.'), date: s('Departure date.'), seat: s('Seat number.')}),
  ],
  init: () => ({schema: '', fields: null}),
  async apply(c, st) { st.schema = c.tool; st.fields = c.args; return `extracted ${Object.keys(c.args).length} fields as ${c.tool}`; },
  render(st) {
    const esc = t => String(t ?? '').replace(/[&<>"]/g, ch => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;'}[ch]));
    const schema = this.tools.find(t => t.name === st.schema);
    const keys = schema ? Object.keys(schema.parameters.properties) : [];
    const rows = keys.map((k, i) => {
      const has = st.fields && k in st.fields;
      const y = 120 + i * 50;
      return `<rect x="60" y="${y}" width="440" height="40" rx="10" fill="#15151a" stroke="${has ? 'var(--acc)' : '#24242b'}"/>
<text x="76" y="${y + 25}" fill="#8b8b96" font-size="12" font-family="ui-monospace,monospace">${esc(k)}</text>
<text x="484" y="${y + 26}" fill="${has ? '#f2f2f4' : '#55555f'}" font-size="15" text-anchor="end" font-weight="${has ? 600 : 400}">${has ? esc(JSON.stringify(st.fields[k])) : '—'}</text>`;
    }).join('');
    return `<svg viewBox="0 0 560 460" xmlns="http://www.w3.org/2000/svg" font-family="Inter,system-ui,sans-serif">
<text x="60" y="64" fill="#8b8b96" font-size="12" font-family="ui-monospace,monospace" letter-spacing="1">SCHEMA</text>
<text x="60" y="96" fill="#f2f2f4" font-size="26" font-weight="700">${schema ? esc(schema.name) : 'pick an example →'}</text>
${rows}
${schema ? '' : '<text x="280" y="260" fill="#55555f" font-size="13" text-anchor="middle">Relay chooses among the schemas on the left and fills the one that fits.</text>'}
</svg>`;
  },
  examples: [
    {title: 'Bank alert', sub: 'card payment', q: 'Your card ending 4821 was charged 23.40 EUR at Albert Heijn on 14 October.'},
    {title: 'Invoice email', sub: 'numbers + dates', q: 'Hi, please find attached invoice INV-2207 from Brightline Studio. The total of 1,250.00 USD is due on 30 November 2026. Thanks!'},
    {title: 'Booking', sub: 'restaurant', q: 'Confirmed: table for 4 at Bella Napoli on Friday 23 October at 19:30, under the name Jansen.'},
    {title: 'Parcel', sub: 'enum status', q: 'DHL update: your parcel JD014600006281 is out for delivery and should arrive today between 14:00 and 16:00.'},
    {title: 'Invite', sub: 'meeting', q: 'Priya Shah invited you to Q4 roadmap review on 2 November at 10:00 in Room 4B.'},
    {title: 'Signature', sub: 'contact card', q: 'Best regards, Tom Becker | Northwind Logistics | +31 6 1234 5678 | tom.becker@northwind.example'},
    {title: 'Boarding pass', sub: 'flight', q: 'KLM KL1023 from Amsterdam to London Heathrow, 21 October 2026, seat 14C.'},
  ],
  help: 'Pick an example or paste your own text. Add a schema on the left (any name, description and fields) to extract something new.',
};
