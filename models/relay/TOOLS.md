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

Relay reads the format above, with the parameters in `params`. Most apps already describe their tools as JSON Schema, the
way OpenAI (`parameters`, bare or inside `{ type: 'function', function: {...} }`), Anthropic (`input_schema`) and MCP
(`inputSchema`) do. **Relay does not read those directly: a tool with `parameters` and no `params` is a tool without
parameters, so the request gets no call.** Convert them first. `sealpup run --tools FILE` does it for you, and in code it
is a few lines.

```js
function relayTools(tools) {
  const param = (name, s, required) => ({
    name,
    type: [].concat(s.type ?? []).find((t) => t !== 'null') ?? 'string',
    description: s.description ?? '',
    required,
    ...(s.enum && { enum: s.enum }),
    ...(s.items && { items: param('item', s.items, false) }),
  });
  return tools.map((t) => {
    t = t.function ?? t;
    const schema = t.parameters ?? t.input_schema ?? t.inputSchema ?? {};
    return {
      name: t.name,
      description: t.description ?? '',
      params: Object.entries(schema.properties ?? {}).map(([k, s]) => param(k, s, (schema.required ?? []).includes(k))),
    };
  });
}

relay.run({ text, tools: relayTools(openaiTools) });
```

```python
def relay_tools(tools):
    def param(name, s, required):
        t = s.get("type", "string")
        if isinstance(t, list):                      # ["string", "null"]: the first type that is not null
            t = next((x for x in t if x != "null"), "string")
        p = {"name": name, "type": t, "description": s.get("description", ""), "required": required}
        if "enum" in s:
            p["enum"] = s["enum"]
        if "items" in s:
            p["items"] = param("item", s["items"], False)
        return p

    out = []
    for t in tools:
        t = t.get("function", t)
        schema = t.get("parameters") or t.get("input_schema") or t.get("inputSchema") or {}
        out.append({"name": t["name"], "description": t.get("description", ""),
                    "params": [param(k, s, k in schema.get("required", [])) for k, s in schema.get("properties", {}).items()]})
    return out

relay.run({"text": text, "tools": relay_tools(openai_tools)})
```

Both give the same calls as the same tools written in Relay's format, for OpenAI, Anthropic and MCP definitions. They do not
follow `$ref` or read `anyOf`, `oneOf` or `allOf`: a property with only one of those is read as a string, so write that
parameter in Relay's format. A tool with neither `params` nor a schema is a tool without parameters.
