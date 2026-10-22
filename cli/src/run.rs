//! Asking a model: hand the request to the program that runs it, read back what it decided and why, and show that.

use crate::ui::{bar, millis, pad, rpad, Style, Term};
use serde_json::Value;
use std::path::Path;
use std::process::Command;
use std::sync::atomic::{AtomicU32, Ordering};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Timing {
    pub n: u32,
    pub first: f64,
    pub median: f64,
    pub best: f64,
}

/// What came back from one run of the program.
pub struct Outcome {
    /// Relay's `--explain` answer: `{calls, dropped, chunks}`.
    pub answer: Value,
    pub load_ms: f64,
    pub graph_mb: f64,
    /// The instruction set the runtime chose, e.g. "x86-64 AVX-512 VNNI".
    pub isa: String,
    pub timing: Option<Timing>,
}

static COUNTER: AtomicU32 = AtomicU32::new(0);

/// Run `program` on one request.
pub fn invoke(
    program: &Path,
    model_dir: &Path,
    record: &Value,
    threads: usize,
    prefilter: bool,
    repeat: u32,
) -> Result<Outcome, String> {
    let file = std::env::temp_dir().join(format!(
        "seal-{}-{}.json",
        std::process::id(),
        COUNTER.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::write(&file, record.to_string()).map_err(|e| format!("cannot write {}: {e}", file.display()))?;
    let mut cmd = Command::new(program);
    cmd.arg(model_dir).arg("--record-file").arg(&file).arg("--explain");
    cmd.arg("--threads").arg(threads.to_string());
    cmd.arg("--repeat").arg(repeat.to_string());
    if prefilter {
        cmd.arg("--prefilter");
    }
    let output = cmd.output();
    let _ = std::fs::remove_file(&file);
    let output = output.map_err(|e| format!("cannot start {}: {e}", program.display()))?;
    let stderr = String::from_utf8_lossy(&output.stderr);
    if !output.status.success() {
        let why = stderr.lines().rev().find(|l| !l.trim().is_empty()).unwrap_or("it stopped without saying why");
        return Err(format!("the model program failed: {}", why.trim_start_matches("relay: ")));
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    let answer: Value = serde_json::from_str(stdout.trim())
        .map_err(|_| "the model program gave an answer that is not JSON".to_string())?;
    let (load_ms, graph_mb, isa) = parse_loaded(&stderr).unwrap_or((0.0, 0.0, String::new()));
    Ok(Outcome { answer, load_ms, graph_mb, isa, timing: parse_timing(&stderr) })
}

/// "loaded in 48 ms, graphs 21.2 MB, x86-64 AVX-512 VNNI"
fn parse_loaded(stderr: &str) -> Option<(f64, f64, String)> {
    let line = stderr.lines().find(|l| l.starts_with("loaded in "))?;
    let rest = line.strip_prefix("loaded in ")?;
    let (ms, rest) = rest.split_once(" ms, graphs ")?;
    let (mb, isa) = rest.split_once(" MB, ")?;
    Some((ms.trim().parse().ok()?, mb.trim().parse().ok()?, isa.trim().to_string()))
}

/// "timing: n=30 first=7.83 median=5.83 best=5.14 ms"
fn parse_timing(stderr: &str) -> Option<Timing> {
    let line = stderr.lines().find(|l| l.starts_with("timing: "))?;
    let value = |key: &str| -> Option<f64> {
        line.split_whitespace().find_map(|w| w.strip_prefix(key)?.strip_prefix('=')?.parse().ok())
    };
    Some(Timing { n: value("n")? as u32, first: value("first")?, median: value("median")?, best: value("best")? })
}

/// The calls Relay made, as `{tool, args}`.
pub fn calls(answer: &Value) -> Vec<&Value> {
    answer["calls"].as_array().map(|a| a.iter().collect()).unwrap_or_default()
}

fn dropped(answer: &Value) -> Vec<&Value> {
    answer["dropped"].as_array().map(|a| a.iter().collect()).unwrap_or_default()
}

/// Every tool's score for the request, best first.
pub fn ranking(answer: &Value) -> Vec<(String, f64)> {
    let mut all: Vec<(String, f64)> = Vec::new();
    for chunk in answer["chunks"].as_array().into_iter().flatten() {
        for m in chunk["margins"].as_array().into_iter().flatten() {
            if let (Some(t), Some(v)) = (m["tool"].as_str(), m["margin"].as_f64()) {
                all.push((t.to_string(), v));
            }
        }
    }
    all.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    all
}

/// Spans of the request (UTF-16 offsets, as the host reports them) that some value was read from, merged.
fn spans(answer: &Value) -> Vec<(usize, usize)> {
    let mut all: Vec<(usize, usize)> = Vec::new();
    for call in calls(answer).into_iter().chain(dropped(answer)) {
        for why in call["why"].as_object().into_iter().flat_map(|o| o.values()) {
            if let Some([a, b]) = why["span"].as_array().map(|s| s.as_slice()) {
                if let (Some(a), Some(b)) = (a.as_u64(), b.as_u64()) {
                    if b > a {
                        all.push((a as usize, b as usize));
                    }
                }
            }
        }
    }
    all.sort();
    let mut merged: Vec<(usize, usize)> = Vec::new();
    for (a, b) in all {
        match merged.last_mut() {
            Some(last) if a <= last.1 => last.1 = last.1.max(b),
            _ => merged.push((a, b)),
        }
    }
    merged
}

/// The request with the words values were read from underlined.
pub fn highlight(term: Term, request: &str, answer: &Value) -> String {
    let units: Vec<u16> = request.encode_utf16().collect();
    let mut out = String::new();
    let mut at = 0;
    for (a, b) in spans(answer) {
        let (a, b) = (a.min(units.len()), b.min(units.len()));
        if a < at {
            continue;
        }
        out.push_str(&String::from_utf16_lossy(&units[at..a]));
        out.push_str(&term.underline(Style::Accent, &String::from_utf16_lossy(&units[a..b])));
        at = b;
    }
    out.push_str(&String::from_utf16_lossy(&units[at..]));
    out
}

fn show_value(v: &Value) -> String {
    let s = v.to_string();
    if s.chars().count() > 44 {
        format!("{}…", s.chars().take(43).collect::<String>())
    } else {
        s
    }
}

fn op_label(op: &str) -> &str {
    match op {
        "copy" => "copied",
        "enum" => "option",
        "ref" => "reference",
        other => other,
    }
}

fn arg_rows(term: Term, call: &Value, muted: bool) -> Vec<String> {
    let args = match call["args"].as_object() {
        Some(a) if !a.is_empty() => a,
        _ => return vec![],
    };
    let name_w = args.keys().map(|k| k.chars().count()).max().unwrap_or(0);
    let val_w = args.values().map(|v| show_value(v).chars().count()).max().unwrap_or(0);
    args.iter()
        .map(|(k, v)| {
            let why = &call["why"][k];
            let op = why["op"].as_str().map(op_label).unwrap_or("");
            let conf = why["conf"].as_f64();
            let value = show_value(v);
            let value = if muted { term.dim(&value) } else { value };
            let conf_text = conf.map(|c| format!("{:>3.0}%", c * 100.0)).unwrap_or_default();
            let conf_text = match conf {
                Some(c) if c < 0.8 => term.warn(&conf_text),
                _ => term.dim(&conf_text),
            };
            format!(
                "      {}  {}  {}  {}",
                term.dim(&pad(k, name_w)),
                pad(&value, val_w),
                term.dim(&pad(op, 9)),
                conf_text
            )
        })
        .collect()
}

/// What Relay decided, as lines of text.
pub fn render(term: Term, request: &str, out: &Outcome, tools: usize, verbose: bool) -> Vec<String> {
    let mut lines = vec![format!("  {} {}", term.dim("›"), highlight(term, request, &out.answer)), String::new()];
    let made = calls(&out.answer);
    let held = dropped(&out.answer);
    for call in &made {
        let tool = call["tool"].as_str().unwrap_or("?");
        let mut head = format!("  {} {}", term.ok("✓"), term.accent(tool));
        if verbose {
            if let Some(m) = call["margin"].as_f64() {
                head.push_str(&term.dim(&format!("   margin {m:+.1}")));
            }
        }
        lines.push(head);
        lines.extend(arg_rows(term, call, false));
    }
    for call in &held {
        let tool = call["tool"].as_str().unwrap_or("?");
        lines.push(format!(
            "  {} {}  {}",
            term.warn("△"),
            term.accent(tool),
            term.warn("held back: a required value is missing")
        ));
        lines.extend(arg_rows(term, call, true));
    }
    let rank = ranking(&out.answer);
    if made.is_empty() && held.is_empty() {
        lines.push(format!("  {} {}", term.dim("○"), "no call"));
        let closest = rank
            .first()
            .map(|(t, m)| format!("closest: {t} (score {m:+.1}); a call needs a score above zero"))
            .unwrap_or_else(|| "no tool was left to consider".into());
        lines.push(format!("      {}", term.dim(&format!("nothing fits this request. {closest}"))));
    }
    if verbose && !rank.is_empty() {
        lines.push(String::new());
        lines.push(format!("  {}", term.dim("tool scores")));
        let name_w = rank.iter().map(|(t, _)| t.chars().count()).max().unwrap_or(0);
        let top = rank.iter().map(|(_, m)| *m).fold(0.0_f64, f64::max);
        for (t, m) in &rank {
            let cells = if top > 0.0 { bar(m.max(0.0) / top, 16) } else { " ".repeat(16) };
            lines.push(format!("    {}  {}  {}", pad(t, name_w), term.accent(&cells), term.dim(&format!("{m:+.1}"))));
        }
    }
    lines.push(String::new());
    lines.push(format!("  {}", term.dim(&footer(out, tools))));
    lines
}

fn footer(out: &Outcome, tools: usize) -> String {
    let noun = if tools == 1 { "tool" } else { "tools" };
    match out.timing {
        Some(t) if t.n > 1 => {
            format!("{} median, {} best over {} runs · {tools} {noun}", millis(t.median), millis(t.best), t.n)
        }
        Some(t) => format!(
            "{} · {tools} {noun} read · loaded in {:.0} ms ({:.1} MB)",
            millis(t.first),
            out.load_ms,
            out.graph_mb
        ),
        None => format!("{tools} {noun} read · loaded in {:.0} ms", out.load_ms),
    }
}

/// The answer as JSON for scripts: the calls, what was held back, and the numbers.
pub fn to_json(request: &str, out: &Outcome, tools: usize, threads: usize) -> Value {
    let plain = |c: &Value| serde_json::json!({"tool": c["tool"], "args": c["args"]});
    serde_json::json!({
        "request": request,
        "calls": calls(&out.answer).into_iter().map(plain).collect::<Vec<_>>(),
        "held_back": dropped(&out.answer).into_iter().map(plain).collect::<Vec<_>>(),
        "stats": {
            "run_ms": out.timing.map(|t| t.first),
            "load_ms": out.load_ms,
            "tools": tools,
            "threads": threads,
            "runtime": out.isa,
        },
    })
}

/// Speed table: the same request at several thread counts.
pub struct BenchRow {
    pub threads: usize,
    pub timing: Timing,
}

pub fn render_bench(term: Term, rows: &[BenchRow], load_ms: f64, isa: &str, tools: usize) -> Vec<String> {
    let slowest = rows.iter().map(|r| r.timing.median).fold(0.0_f64, f64::max).max(f64::MIN_POSITIVE);
    let single = rows.iter().find(|r| r.threads == 1).map(|r| r.timing.median);
    let mut lines = vec![format!(
        "  {}   {}   {}   {}",
        pad(&term.dim("threads"), 7),
        rpad(&term.dim("median"), 9),
        rpad(&term.dim("best"), 9),
        rpad(&term.dim("speed-up"), 8)
    )];
    for r in rows {
        let speedup = match single {
            Some(one) if r.threads > 1 => format!("{:.1}×", one / r.timing.median),
            _ => String::new(),
        };
        lines.push(format!(
            "  {}   {}   {}   {}   {}",
            pad(&r.threads.to_string(), 7),
            rpad(&millis(r.timing.median), 9),
            rpad(&millis(r.timing.best), 9),
            rpad(&speedup, 8),
            term.accent(&bar(r.timing.median / slowest, 24)),
        ));
    }
    let runs = rows.first().map(|r| r.timing.n).unwrap_or(0);
    lines.push(String::new());
    lines.push(format!(
        "  {}",
        term.dim(&format!("{runs} runs each · {tools} tools · {isa} · the model loads in {load_ms:.0} ms"))
    ));
    lines
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn answer() -> Value {
        json!({
          "calls": [{"tool": "show_map", "args": {"query": "Eiffel Tower"},
            "why": {"query": {"op": "copy", "conf": 0.9999, "span": [12, 24]}}, "margin": 5.5}],
          "dropped": [],
          "chunks": [{"margins": [{"tool": "show_map", "margin": 5.5}, {"tool": "flashlight", "margin": -9.0}]}]
        })
    }

    #[test]
    fn the_program_s_notes_on_stderr_are_read() {
        let (ms, mb, isa) = parse_loaded("loaded in 48 ms, graphs 21.2 MB, x86-64 AVX-512 VNNI\n").unwrap();
        assert_eq!((ms, mb, isa.as_str()), (48.0, 21.2, "x86-64 AVX-512 VNNI"));
        let t = parse_timing("loaded in 1 ms, graphs 1.0 MB, x\ntiming: n=30 first=7.83 median=5.83 best=5.14 ms\n")
            .unwrap();
        assert_eq!(t, Timing { n: 30, first: 7.83, median: 5.83, best: 5.14 });
        assert!(parse_timing("nothing").is_none());
    }

    #[test]
    fn spans_are_utf16_offsets_into_the_request() {
        // The emoji is two UTF-16 units, so the span starts one character later than a count of characters says.
        let request = "😀 Show Eiffel Tower now";
        let a = json!({"calls": [{"tool": "t", "args": {}, "why": {"q": {"span": [8, 20]}}}], "dropped": []});
        let shown = highlight(Term::plain(), request, &a);
        assert_eq!(shown, "😀 Show [Eiffel Tower] now");
    }

    #[test]
    fn overlapping_spans_merge() {
        let a = json!({"calls": [{"why": {"a": {"span": [0, 5]}, "b": {"span": [3, 8]}, "c": {"span": [10, 12]}}}]});
        assert_eq!(spans(&a), vec![(0, 8), (10, 12)]);
    }

    #[test]
    fn a_call_shows_its_values_where_they_came_from() {
        let out = Outcome {
            answer: answer(),
            load_ms: 48.0,
            graph_mb: 21.2,
            isa: "x".into(),
            timing: Some(Timing { n: 1, first: 5.8, median: 5.8, best: 5.8 }),
        };
        let text = render(Term::plain(), "Show the Eiffel Tower now", &out, 9, false).join("\n");
        assert!(text.contains("✓ show_map"));
        assert!(text.contains("\"Eiffel Tower\""));
        assert!(text.contains("copied"));
        assert!(text.contains("100%"));
        assert!(text.contains("5.80 ms · 9 tools read · loaded in 48 ms (21.2 MB)"));
    }

    #[test]
    fn no_call_says_what_came_closest() {
        let none = json!({"calls": [], "dropped": [], "chunks": [{"margins": [{"tool": "a", "margin": -3.0}, {"tool": "b", "margin": -9.0}]}]});
        let out = Outcome { answer: none, load_ms: 1.0, graph_mb: 1.0, isa: String::new(), timing: None };
        let text = render(Term::plain(), "tell me a joke", &out, 2, false).join("\n");
        assert!(text.contains("no call"));
        assert!(text.contains("closest: a"));
    }

    #[test]
    fn verbose_adds_the_tool_scores() {
        let out = Outcome { answer: answer(), load_ms: 1.0, graph_mb: 1.0, isa: String::new(), timing: None };
        let text = render(Term::plain(), "x", &out, 2, true).join("\n");
        assert!(text.contains("tool scores"));
        assert!(text.contains("flashlight"));
        assert!(text.contains("-9.0"));
    }

    #[test]
    fn json_output_has_just_the_calls_and_the_numbers() {
        let out = Outcome { answer: answer(), load_ms: 48.0, graph_mb: 21.2, isa: "x86".into(), timing: None };
        let j = to_json("req", &out, 9, 4);
        assert_eq!(j["calls"][0]["tool"], "show_map");
        assert!(j["calls"][0].get("why").is_none());
        assert_eq!(j["stats"]["threads"], 4);
    }
}
