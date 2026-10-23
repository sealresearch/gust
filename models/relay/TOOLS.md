# The request Relay takes

A request is one JSON object: the message, the tools the app offers, and optionally the date.

```json
{
  "text": "Set an alarm for 6:45 tomorrow and text Sam I'm running late",
  "tools": [
    { "name": "set_alarm", "description": "Sets an alarm.",
      "params": [
        { "name": "time", "type": "string", "description": "When it rings, as an ISO 8601 time.", "required": true },
        { "name": "label", "type": "string", "description": "A name for the alarm." }
      ] },
    { "name": "send_message", "description": "Sends a text message.",
      "params": [
        { "name": "recipient", "type": "string", "description": "Who to send it to.", "required": true },
        { "name": "body", "type": "string", "description": "The message.", "required": true }
      ] },
    { "name": "turn_on_flashlight", "description": "Turns the flashlight on.", "params": [] }
  ],
  "context": { "now": "2026-10-23T09:00:00" }
}
```

Relay answers with the calls to make, or `[]` when no tool fits:

```json
[{"tool": "set_alarm", "args": {"time": "2026-10-24T06:45:00"}},
 {"tool": "send_message", "args": {"recipient": "Sam", "body": "I'm running late"}}]
```

## The request

| Field | | |
|---|---|---|
| `text` | string, required | What the person said. |
| `tools` | array, required | The tools the app offers. It may be empty, and then there is no call. |
| `context.now` | string | The local date and time, like `2026-10-23T09:00:00`. Relay resolves "tomorrow" and "at 6:45" against it. Without it, a request that needs a date has nothing to resolve it against and gets no call, so always pass it. |
| `kind` | string | `"call"` (the default) makes no call when nothing fits. `"extract"` always calls the best fitting tool: use it to fill in a form, where the answer is never "none". It also reads every tool instead of ranking a long list first. |

## A tool

| Field | | |
|---|---|---|
| `name` | string, required | The function's name. It is what comes back as `tool`. |
| `description` | string | What it does, in a sentence. Relay reads it, so a clear one helps. |
| `params` | array | The parameters. Leave it empty or out for a tool that takes none. |

## A parameter

| Field | | |
|---|---|---|
| `name` | string, required | The argument's name. |
| `type` | string | `string` (the default), `integer`, `number`, `boolean`, `array` or `object`. |
| `description` | string | What it is. Relay reads this too. |
| `required` | boolean | A call that leaves a required argument empty is dropped. |
| `enum` | array | The allowed values. The argument is one of them. |
| `default` | any | The value the tool uses when the argument is left out. Relay can read it, and a call may leave the argument out. |
| `items` | parameter | For an `array`, the element: `{ "name": "item", "type": "string" }`. |
| `properties` | array of parameters | For an `object`, its fields. An object argument is left out or takes its default; Relay does not build one from the request. |

Every value Relay returns comes from the request or from the schema (an enum value, a default, a date computed from
`context.now`). It cannot write text that is not in the request, such as a composed email.

## Tools written as JSON Schema

Most apps already describe their tools the way OpenAI, Anthropic or MCP does. Relay accepts those as they are and
converts them to the format above before it reads them, so these all give the same calls:

```js
// Relay's own format
{ name: 'get_weather', description: 'Weather for a city.',
  params: [{ name: 'city', type: 'string', description: 'The city.', required: true }] }

// OpenAI, as the Chat Completions API takes it ...
{ type: 'function', function: { name: 'get_weather', description: 'Weather for a city.',
  parameters: { type: 'object', properties: { city: { type: 'string', description: 'The city.' } }, required: ['city'] } } }

// ... or without the wrapper (the Responses API, and most function lists)
{ name: 'get_weather', description: 'Weather for a city.', parameters: { type: 'object', properties: { ... }, required: [...] } }

// Anthropic
{ name: 'get_weather', description: 'Weather for a city.', input_schema: { type: 'object', properties: { ... }, required: [...] } }

// MCP
{ name: 'get_weather', description: 'Weather for a city.', inputSchema: { type: 'object', properties: { ... }, required: [...] } }
```

You can mix formats in one list. What the conversion reads from a schema property: `type`, `description`, `enum`,
`default`, `items`, nested `properties` with their `required`, and `minimum` and `maximum` (added to the description
as `(1-10)`, because Relay reads the description). A `type` that is a list, like `["string", "null"]`, becomes its
first type that is not `"null"`. It does not follow `$ref`, and it does not read `anyOf`, `oneOf` or `allOf`: a property
with only one of those is read as a string. For those, write the parameter in Relay's format.

A tool with neither `params` nor a schema is a tool without parameters.
