//! The tools a request is offered. A tools file may be written the way Relay reads them (`params`), or as the JSON
//! Schema tools most APIs use (`parameters`, or `input_schema`, with or without a `function` wrapper).

use chrono::{Local, NaiveDateTime};
use serde_json::{json, Map, Value};

const SAMPLE: &str = include_str!("../sample-tools.json");

pub struct ToolSet {
    pub tools: Vec<Value>,
    /// What to call it on screen: a file name, or the built-in set.
    pub source: String,
    pub is_sample: bool,
}

/// The tools from `path`, or the built-in phone tools when there is no path.
pub fn load(path: Option<&str>) -> Result<ToolSet, String> {
    match path {
        None => Ok(ToolSet { tools: parse(SAMPLE)?, source: "built-in phone tools".into(), is_sample: true }),
        Some(p) => {
            let text = std::fs::read_to_string(p).map_err(|e| format!("cannot read {p}: {e}"))?;
            Ok(ToolSet {
                tools: parse(&text).map_err(|e| format!("{p}: {e}"))?,
                source: p.to_string(),
                is_sample: false,
            })
        }
    }
}

/// Tools from JSON text: an array of tools, or an object with a `tools` array.
pub fn parse(text: &str) -> Result<Vec<Value>, String> {
    let v: Value = serde_json::from_str(text).map_err(|e| format!("not valid JSON ({e})"))?;
    let list = match &v {
        Value::Array(a) => a,
        Value::Object(o) => match o.get("tools") {
            Some(Value::Array(a)) => a,
            _ => return Err("expected an array of tools, or an object with a \"tools\" array".into()),
        },
        _ => return Err("expected an array of tools".into()),
    };
    if list.is_empty() {
        return Err("the tools list is empty".into());
    }
    list.iter().enumerate().map(|(i, t)| convert(t).map_err(|e| format!("tool {}: {e}", i + 1))).collect()
}

fn convert(v: &Value) -> Result<Value, String> {
    let t = v.get("function").unwrap_or(v);
    let name = t.get("name").and_then(Value::as_str).filter(|s| !s.is_empty()).ok_or("a tool needs a \"name\"")?;
    let description = t.get("description").and_then(Value::as_str).unwrap_or("");
    if let Some(Value::Array(params)) = t.get("params") {
        return Ok(json!({"name": name, "description": description, "params": params}));
    }
    let schema = ["parameters", "input_schema", "inputSchema"].iter().find_map(|k| t.get(*k));
    let props = schema.and_then(|s| s.get("properties")).and_then(Value::as_object);
    let required: Vec<&str> = schema
        .and_then(|s| s.get("required"))
        .and_then(Value::as_array)
        .map(|a| a.iter().filter_map(Value::as_str).collect())
        .unwrap_or_default();
    let params: Vec<Value> =
        props.map(|p| p.iter().map(|(k, s)| param(k, s, required.contains(&k.as_str()))).collect()).unwrap_or_default();
    Ok(json!({"name": name, "description": description, "params": params}))
}

/// One parameter, in Relay's shape, from a JSON Schema property.
fn param(name: &str, s: &Value, required: bool) -> Value {
    let kind = match s.get("type") {
        Some(Value::String(t)) => t.clone(),
        Some(Value::Array(a)) => {
            a.iter().filter_map(Value::as_str).find(|t| *t != "null").unwrap_or("string").to_string()
        }
        _ => "string".to_string(),
    };
    let mut description = s.get("description").and_then(Value::as_str).unwrap_or("").to_string();
    let (min, max) = (s.get("minimum"), s.get("maximum"));
    if min.is_some() || max.is_some() {
        let n = |v: Option<&Value>| v.map(|x| x.to_string()).unwrap_or_default();
        let range = format!("({}-{})", n(min), n(max));
        description = if description.is_empty() { range } else { format!("{description} {range}") };
    }
    let mut p = Map::new();
    p.insert("name".into(), json!(name));
    p.insert("type".into(), json!(kind));
    p.insert("description".into(), json!(description));
    p.insert("required".into(), json!(required));
    if let Some(e) = s.get("enum") {
        p.insert("enum".into(), e.clone());
    }
    if let Some(d) = s.get("default") {
        p.insert("default".into(), d.clone());
    }
    if let Some(items) = s.get("items") {
        p.insert("items".into(), param("item", items, false));
    }
    if let Some(Value::Object(props)) = s.get("properties") {
        let req: Vec<&str> = s
            .get("required")
            .and_then(Value::as_array)
            .map(|a| a.iter().filter_map(Value::as_str).collect())
            .unwrap_or_default();
        let nested: Vec<Value> = props.iter().map(|(k, v)| param(k, v, req.contains(&k.as_str()))).collect();
        p.insert("properties".into(), Value::Array(nested));
    }
    Value::Object(p)
}

/// Local time as Relay wants it, e.g. 2026-10-18T09:00:00.
pub fn now() -> String {
    Local::now().format("%Y-%m-%dT%H:%M:%S").to_string()
}

/// The record Relay reads: the request, the tools and the date it should resolve "tomorrow" against.
pub fn record(text: &str, tools: &[Value], now: &str) -> Result<Value, String> {
    let when = NaiveDateTime::parse_from_str(now, "%Y-%m-%dT%H:%M:%S")
        .map_err(|_| format!("--now should look like 2026-10-18T09:00:00, not \"{now}\""))?;
    Ok(json!({
        "id": "seal",
        "text": text,
        "tools": tools,
        "kind": "call",
        "context": {"now": now, "system": format!("Current date and time: {now}\nDay of week is {}", when.format("%A"))},
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_built_in_tools_parse() {
        let tools = load(None).unwrap().tools;
        assert_eq!(tools.len(), 9);
        assert_eq!(tools[0]["name"], "set_alarm");
        assert_eq!(tools[0]["params"][0]["required"], true);
    }

    #[test]
    fn json_schema_tools_become_relay_tools() {
        let tools = parse(
            r#"[{"type":"function","function":{"name":"set_volume","description":"Volume.",
                "parameters":{"type":"object","properties":{
                  "level":{"type":"integer","description":"Percent.","minimum":0,"maximum":100},
                  "stream":{"type":"string","enum":["media","ring"]}},"required":["level"]}}}]"#,
        )
        .unwrap();
        let p = &tools[0]["params"];
        assert_eq!(tools[0]["name"], "set_volume");
        assert_eq!(p[0]["name"], "level");
        assert_eq!(p[0]["required"], true);
        assert_eq!(p[0]["description"], "Percent. (0-100)");
        assert_eq!(p[1]["required"], false);
        assert_eq!(p[1]["enum"], json!(["media", "ring"]));
    }

    #[test]
    fn properties_keep_the_order_they_were_written_in() {
        let tools = parse(r#"[{"name":"t","parameters":{"properties":{"zeta":{},"alpha":{}}}}]"#).unwrap();
        assert_eq!(tools[0]["params"][0]["name"], "zeta");
        assert_eq!(tools[0]["params"][1]["name"], "alpha");
    }

    #[test]
    fn anthropic_style_and_wrapped_lists_are_understood() {
        let t = parse(r#"{"tools":[{"name":"a","input_schema":{"properties":{"x":{"type":"number"}}}}]}"#).unwrap();
        assert_eq!(t[0]["params"][0]["type"], "number");
    }

    #[test]
    fn bad_tools_say_what_is_wrong() {
        assert!(parse("[]").unwrap_err().contains("empty"));
        assert!(parse("[{}]").unwrap_err().contains("name"));
        assert!(parse("{").unwrap_err().contains("JSON"));
    }

    #[test]
    fn records_carry_the_date_and_its_weekday() {
        let r = record("hi", &[], "2026-10-18T09:00:00").unwrap();
        assert_eq!(r["context"]["now"], "2026-10-18T09:00:00");
        assert!(r["context"]["system"].as_str().unwrap().ends_with("Sunday"));
        assert!(record("hi", &[], "tomorrow").is_err());
    }
}
