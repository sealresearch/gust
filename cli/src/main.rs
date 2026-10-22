//! seal: install Seal's on-device models and run them.

mod art;
mod net;
mod registry;
mod run;
mod store;
mod tools;
mod ui;

use registry::Registry;
use std::collections::{HashMap, HashSet};
use std::io::{BufRead, IsTerminal, Read, Write};
use std::path::PathBuf;
use ui::{bytes, millis, pad, rpad, Stream, Term};

const VERSION: &str = env!("CARGO_PKG_VERSION");
const TAGLINE: &str = "tiny models, on your device";

struct Fail {
    message: String,
    hint: Option<String>,
    code: i32,
}

type Res = Result<(), Fail>;

fn fail(message: impl Into<String>) -> Fail {
    Fail { message: message.into(), hint: None, code: 1 }
}

fn hinted(message: impl Into<String>, hint: impl Into<String>) -> Fail {
    Fail { message: message.into(), hint: Some(hint.into()), code: 1 }
}

fn usage(message: impl Into<String>, hint: impl Into<String>) -> Fail {
    Fail { message: message.into(), hint: Some(hint.into()), code: 2 }
}

impl From<String> for Fail {
    fn from(message: String) -> Fail {
        fail(message)
    }
}

struct Args {
    positional: Vec<String>,
    values: HashMap<String, String>,
    flags: HashSet<String>,
}

impl Args {
    fn flag(&self, name: &str) -> bool {
        self.flags.contains(name)
    }
    fn value(&self, name: &str) -> Option<&str> {
        self.values.get(name).map(|s| s.as_str())
    }
    fn number(&self, name: &str) -> Result<Option<usize>, Fail> {
        match self.value(name) {
            None => Ok(None),
            Some(v) => match v.parse::<usize>() {
                Ok(n) if n > 0 => Ok(Some(n)),
                _ => Err(usage(format!("--{name} needs a whole number above zero, not \"{v}\""), "seal help")),
            },
        }
    }
}

const SHORT: [(&str, &str); 6] =
    [("v", "verbose"), ("y", "yes"), ("n", "runs"), ("t", "tools"), ("h", "help"), ("V", "version")];
const WITH_VALUE: [&str; 4] = ["tools", "threads", "now", "runs"];

fn parse_args(raw: &[String]) -> Result<Args, Fail> {
    let mut a = Args { positional: vec![], values: HashMap::new(), flags: HashSet::new() };
    let mut i = 0;
    while i < raw.len() {
        let arg = &raw[i];
        i += 1;
        if arg == "--" {
            a.positional.extend(raw[i..].iter().cloned());
            break;
        }
        let name = if let Some(long) = arg.strip_prefix("--") {
            long.to_string()
        } else if arg.len() == 2 && arg.starts_with('-') && arg != "-" {
            let short = &arg[1..];
            match SHORT.iter().find(|(s, _)| *s == short) {
                Some((_, long)) => long.to_string(),
                None => return Err(usage(format!("unknown option {arg}"), "seal help")),
            }
        } else {
            a.positional.push(arg.clone());
            continue;
        };
        if let Some((k, v)) = name.split_once('=') {
            a.values.insert(k.to_string(), v.to_string());
        } else if WITH_VALUE.contains(&name.as_str()) {
            match raw.get(i) {
                Some(v) => {
                    a.values.insert(name, v.clone());
                    i += 1;
                }
                None => return Err(usage(format!("--{name} needs a value"), "seal help")),
            }
        } else {
            a.flags.insert(name);
        }
    }
    Ok(a)
}

fn main() {
    ui::init();
    let out = ui::detect(Stream::Out);
    let err = ui::detect(Stream::Err);
    let raw: Vec<String> = std::env::args().skip(1).collect();
    let code = match dispatch(out, err, &raw) {
        Ok(()) => 0,
        Err(f) => {
            eprintln!("\n  {} {}", err.err("✗"), f.message);
            if let Some(h) = f.hint {
                eprintln!("    {}", err.dim(&h));
            }
            eprintln!();
            f.code
        }
    };
    std::process::exit(code);
}

fn dispatch(out: Term, err: Term, raw: &[String]) -> Res {
    let Some(first) = raw.first() else {
        banner(out);
        return Ok(());
    };
    let rest = &raw[1..];
    match first.as_str() {
        "help" | "--help" | "-h" => {
            banner(out);
            Ok(())
        }
        "version" | "--version" | "-V" => {
            println!("seal {VERSION}");
            Ok(())
        }
        "install" => install(out, err, &parse_args(rest)?),
        "run" => run_cmd(out, err, &parse_args(rest)?),
        "bench" => bench(out, err, &parse_args(rest)?),
        "list" | "ls" => list(out),
        "info" => info(out, &parse_args(rest)?),
        "remove" | "rm" | "uninstall" => remove(out, &parse_args(rest)?),
        "doctor" => doctor(out),
        other => Err(usage(format!("unknown command \"{other}\""), "seal help")),
    }
}

/// The seal, the commands, and one line to try.
fn banner(t: Term) {
    let art = art::seal();
    let side = [
        String::new(),
        format!("{}  {}", t.bold("seal"), t.dim(VERSION)),
        t.dim(TAGLINE),
        String::new(),
        String::new(),
        String::new(),
    ];
    println!();
    for (line, text) in art.iter().zip(side.iter()) {
        println!("  {}   {}", line, text);
    }
    println!();
    let rows = [
        ("install <model>", "download a model and the program that runs it"),
        ("run <model> [text]", "ask a model; without text, start a conversation"),
        ("bench <model>", "measure its speed on this machine"),
        ("list", "the models, and which are installed"),
        ("info <model>", "what a model is, and what it needs"),
        ("remove <model>", "delete a model"),
        ("doctor", "check this machine and what is installed"),
    ];
    for (cmd, what) in rows {
        println!("  {}  {}", t.accent(&pad(cmd, 20)), t.dim(what));
    }
    println!();
    println!("  {}", t.dim("options for run:  --tools FILE   --threads N   --now TIME   --json   --lines   -v"));
    println!();
    println!("  {}  seal install relay-1 && seal run relay-1 \"Set an alarm for 6:45 tomorrow\"", t.dim("try"));
    println!();
}

fn model_arg<'a>(reg: &'a Registry, args: &Args, command: &str) -> Result<(String, &'a registry::Model), Fail> {
    let name = args.positional.first().ok_or_else(|| {
        let known: Vec<&str> = reg.models.keys().map(|s| s.as_str()).collect();
        usage(format!("seal {command} needs a model"), format!("models: {}", known.join(", ")))
    })?;
    let id = reg.resolve(name).map_err(|m| usage(m, "seal list"))?;
    let model = &reg.models[&id];
    Ok((id, model))
}

/// All files of a model are present with the right size.
fn is_installed(id: &str, model: &registry::Model) -> bool {
    let dir = store::model_dir(id);
    store::read_manifest(id).is_some()
        && model.files.iter().all(|f| std::fs::metadata(dir.join(&f.name)).map(|m| m.len() == f.size).unwrap_or(false))
}

fn status_cell(t: Term, installed: bool) -> String {
    if installed {
        format!("{} {}", t.ok("✓"), "installed")
    } else {
        t.dim("· not installed")
    }
}

// ---------------------------------------------------------------------------------------------------------------- install

fn install(out: Term, err: Term, args: &Args) -> Res {
    let reg = registry::load();
    let (id, model) = model_arg(&reg, args, "install")?;
    let runner = reg.runner(model);
    let asset = runner.asset()?;
    let force = args.flag("force");
    let started = std::time::Instant::now();

    println!();
    println!(
        "  Installing {}  {}",
        out.bold(&model.name),
        out.dim(&format!("{} · {}", bytes(model.total_size()), model.licence))
    );
    println!();

    // The program that runs the model, from the release of the Gust repository, checked against the release's checksums.
    let program = store::runner_path(&model.runner, &runner.tag);
    if program.exists() && !force {
        println!(
            "  {} {}  {}  {}",
            out.ok("✓"),
            pad(&format!("{} {}", model.runner, runner.tag), 22),
            rpad("", 8),
            out.dim("already here")
        );
    } else {
        let release = format!("{}/{}/releases/download/{}", net::github_base(), runner.repo, runner.tag);
        let sums = net::parse_sums(&net::fetch_text(&format!("{release}/SHA256SUMS"))?);
        let sha =
            sums.get(asset).ok_or_else(|| fail(format!("the release {} has no checksum for {asset}", runner.tag)))?;
        let label = format!("{} {}", model.runner, runner.tag);
        let done = net::download(&format!("{release}/{asset}"), &program, Some(sha), None, &label, err)?;
        make_executable(&program)?;
        println!(
            "  {} {}  {}  {}",
            out.ok("✓"),
            pad(&label, 22),
            rpad(&bytes(done.bytes), 8),
            out.dim(&format!("{:.1}s", done.secs))
        );
    }

    // The weights, from Hugging Face, checked against the checksums this version of seal was released with.
    let dir = store::model_dir(&id);
    for f in &model.files {
        let dest = dir.join(&f.name);
        if !force && dest.exists() && net::sha256_file(&dest).map(|h| h == f.sha256).unwrap_or(false) {
            println!(
                "  {} {}  {}  {}",
                out.ok("✓"),
                pad(&f.name, 22),
                rpad(&bytes(f.size), 8),
                out.dim("already here")
            );
            continue;
        }
        let url = format!("{}/{}/resolve/{}/{}", net::hf_base(), model.repo, model.revision, f.name);
        let done = net::download(&url, &dest, Some(&f.sha256), Some(f.size), &f.name, err)?;
        println!(
            "  {} {}  {}  {}",
            out.ok("✓"),
            pad(&f.name, 22),
            rpad(&bytes(done.bytes), 8),
            out.dim(&format!("{:.1}s", done.secs))
        );
    }

    let manifest = store::Installed {
        id: id.clone(),
        installed_at: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0),
        runtime_tag: runner.tag.clone(),
        files: model.files.iter().map(|f| f.name.clone()).collect(),
    };
    store::write_manifest(&dir, &manifest)?;

    println!();
    println!(
        "  Installed {} in {:.1}s  {}",
        out.bold(&id),
        started.elapsed().as_secs_f64(),
        out.dim(&dir.display().to_string())
    );
    println!();
    println!("  {}  seal run {id} \"Set an alarm for 6:45 tomorrow\"", out.dim("try"));
    println!();
    Ok(())
}

#[cfg(unix)]
fn make_executable(path: &std::path::Path) -> Res {
    use std::os::unix::fs::PermissionsExt;
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o755))
        .map_err(|e| fail(format!("cannot make {} executable: {e}", path.display())))
}

#[cfg(not(unix))]
fn make_executable(_: &std::path::Path) -> Res {
    Ok(())
}

// --------------------------------------------------------------------------------------------------------------------- run

fn default_threads() -> usize {
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1).min(8)
}

/// What a command needs to talk to an installed model.
struct Ready {
    id: String,
    program: PathBuf,
    dir: PathBuf,
}

fn ready(args: &Args, command: &str) -> Result<Ready, Fail> {
    let reg = registry::load();
    let (id, model) = model_arg(&reg, args, command)?;
    if !is_installed(&id, model) {
        return Err(hinted(format!("{id} is not installed"), format!("seal install {id}")));
    }
    let runner = reg.runner(model);
    let program = store::runner_path(&model.runner, &runner.tag);
    if !program.exists() {
        return Err(hinted(
            format!("the program that runs {id} is missing ({} {})", model.runner, runner.tag),
            format!("seal install {id} --force"),
        ));
    }
    Ok(Ready { id: id.clone(), program, dir: store::model_dir(&id) })
}

fn run_cmd(out: Term, _err: Term, args: &Args) -> Res {
    let r = ready(args, "run")?;
    let toolset = tools::load(args.value("tools"))?;
    let threads = args.number("threads")?.unwrap_or_else(default_threads);
    let now = args.value("now").map(|s| s.to_string()).unwrap_or_else(tools::now);
    let prefilter = !args.flag("no-prefilter");
    let verbose = args.flag("verbose");
    let json = args.flag("json");
    let n_tools = toolset.tools.len();

    let ask = |request: &str| -> Result<run::Outcome, Fail> {
        let record = tools::record(request, &toolset.tools, &now)?;
        Ok(run::invoke(&r.program, &r.dir, &record, threads, prefilter, 1)?)
    };

    // Where the requests come from: the command line, a conversation at the keyboard, or standard input.
    let text = args.positional[1..].join(" ");
    let stdin = std::io::stdin();
    let conversation = text.is_empty() && stdin.is_terminal();
    let mut requests: Vec<String> = Vec::new();
    if !text.is_empty() {
        requests.push(text);
    } else if !conversation {
        let mut all = String::new();
        stdin
            .lock()
            .read_to_string(&mut all)
            .map_err(|e| fail(format!("cannot read the requests from standard input: {e}")))?;
        if args.flag("lines") {
            requests.extend(all.lines().map(str::trim).filter(|l| !l.is_empty()).map(str::to_string));
        } else if !all.trim().is_empty() {
            requests.push(all.trim().to_string());
        }
        if requests.is_empty() {
            return Err(usage("there is no request to run", format!("seal run {} \"your request\"", r.id)));
        }
    }
    if conversation && json {
        return Err(usage("--json needs a request", format!("seal run {} --json \"your request\"", r.id)));
    }

    if json {
        for request in &requests {
            println!("{}", run::to_json(request, &ask(request)?, n_tools, threads));
        }
        return Ok(());
    }

    let plural = if threads == 1 { "thread" } else { "threads" };
    println!();
    println!("  {}", out.dim(&format!("{} · {} ({n_tools} tools) · {threads} {plural}", r.id, toolset.source)));
    println!();
    for request in &requests {
        for line in run::render(out, request, &ask(request)?, n_tools, verbose) {
            println!("{line}");
        }
        println!();
    }
    if !requests.is_empty() {
        if toolset.is_sample {
            println!("  {}", out.dim("these are built-in phone tools; give your own with --tools FILE"));
            println!();
        }
        return Ok(());
    }

    println!("  {}", out.dim("Type a request. An empty line or Ctrl-D ends the conversation."));
    println!();
    loop {
        print!("  {} ", out.accent("›"));
        let _ = std::io::stdout().flush();
        let mut line = String::new();
        if stdin.lock().read_line(&mut line).map(|n| n == 0).unwrap_or(true) {
            println!();
            break;
        }
        let request = line.trim();
        if request.is_empty() || matches!(request, "exit" | "quit" | ":q") {
            break;
        }
        if out.level != ui::Level::Plain {
            // Replace the line that was typed with the same words, underlined where the values came from.
            print!("[1A[2K");
        }
        match ask(request) {
            Ok(outcome) => {
                for l in run::render(out, request, &outcome, n_tools, verbose) {
                    println!("{l}");
                }
                println!();
            }
            Err(f) => eprintln!(
                "  {} {}
",
                out.err("✗"),
                f.message
            ),
        }
    }
    Ok(())
}

// ------------------------------------------------------------------------------------------------------------------- bench

fn bench(out: Term, _err: Term, args: &Args) -> Res {
    let r = ready(args, "bench")?;
    let toolset = tools::load(args.value("tools"))?;
    let text = if args.positional.len() > 1 {
        args.positional[1..].join(" ")
    } else {
        "What is the weather in Rotterdam?".to_string()
    };
    let runs = args.number("runs")?.unwrap_or(100).min(10_000) as u32;
    let now = args.value("now").map(|s| s.to_string()).unwrap_or_else(tools::now);
    let record = tools::record(&text, &toolset.tools, &now)?;

    let cpus = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1).min(8);
    let mut counts: Vec<usize> = [1, 2, 4, 8].into_iter().filter(|n| *n <= cpus).collect();
    if !counts.contains(&cpus) {
        counts.push(cpus);
    }
    if let Some(t) = args.number("threads")? {
        counts = vec![t];
    }

    println!();
    println!("  Speed of {} on this machine", out.bold(&r.id));
    println!("  {}", out.dim(&format!("\"{text}\" with {} ({} tools)", toolset.source, toolset.tools.len())));
    println!();
    let mut rows = Vec::new();
    let (mut load_ms, mut isa) = (0.0, String::new());
    for t in counts {
        let outcome = run::invoke(&r.program, &r.dir, &record, t, true, runs)?;
        let timing = outcome.timing.ok_or_else(|| fail("the model program did not report its timing"))?;
        load_ms = outcome.load_ms;
        isa = outcome.isa.clone();
        rows.push(run::BenchRow { threads: t, timing });
    }
    for line in run::render_bench(out, &rows, load_ms, &isa, toolset.tools.len()) {
        println!("{line}");
    }
    println!();
    Ok(())
}

// -------------------------------------------------------------------------------------------------------- list, info, remove

fn list(out: Term) -> Res {
    let reg = registry::load();
    println!();
    println!(
        "  {}  {}  {}  {}",
        out.dim(&pad("model", 10)),
        out.dim(&pad("status", 15)),
        out.dim(&rpad("size", 8)),
        out.dim("parameters")
    );
    for (id, m) in &reg.models {
        let ok = is_installed(id, m);
        println!(
            "  {}  {}  {}  {}",
            out.accent(&pad(id, 10)),
            pad(&status_cell(out, ok), 15),
            rpad(&bytes(m.total_size()), 8),
            m.parameters
        );
        println!("  {}", out.dim(&format!("{}{}", " ".repeat(10 + 2), m.task)));
    }
    println!();
    println!("  {}", out.dim("seal install <model>   ·   seal info <model>"));
    println!();
    Ok(())
}

fn when(secs: u64) -> String {
    chrono::DateTime::from_timestamp(secs as i64, 0)
        .map(|d| d.with_timezone(&chrono::Local).format("%Y-%m-%d %H:%M").to_string())
        .unwrap_or_default()
}

fn info(out: Term, args: &Args) -> Res {
    let reg = registry::load();
    let (id, m) = model_arg(&reg, args, "info")?;
    let runner = reg.runner(m);
    let key = |k: &str, v: String| println!("  {}  {}", out.dim(&pad(k, 11)), v);
    println!();
    println!("  {}", out.bold(&m.name));
    println!("  {}", out.dim(&m.task));
    println!();
    key("id", id.clone());
    key("parameters", m.parameters.clone());
    key("size", format!("{} in {} files", bytes(m.total_size()), m.files.len()));
    key("licence", m.licence.clone());
    key("weights", format!("huggingface.co/{}", m.repo));
    key("write-up", m.research.clone());
    key("program", format!("{} {}", m.runner, runner.tag));
    let status = match store::read_manifest(&id).filter(|_| is_installed(&id, m)) {
        Some(i) => format!(
            "{} installed {}  {}",
            out.ok("✓"),
            when(i.installed_at),
            out.dim(&store::model_dir(&id).display().to_string())
        ),
        None => out.dim("not installed"),
    };
    key("status", status);
    println!();
    for f in &m.files {
        println!("  {}  {}  {}", pad(&f.name, 22), rpad(&bytes(f.size), 8), out.dim(&f.sha256[..12]));
    }
    println!();
    Ok(())
}

fn remove(out: Term, args: &Args) -> Res {
    let reg = registry::load();
    let (id, _) = model_arg(&reg, args, "remove")?;
    let dir = store::model_dir(&id);
    if !dir.exists() {
        return Err(fail(format!("{id} is not installed")));
    }
    let size = store::disk_use(&dir);
    if !args.flag("yes") && std::io::stdin().is_terminal() {
        print!("\n  Remove {id} ({})? [y/N] ", bytes(size));
        let _ = std::io::stdout().flush();
        let mut answer = String::new();
        let _ = std::io::stdin().read_line(&mut answer);
        if !matches!(answer.trim().to_lowercase().as_str(), "y" | "yes") {
            println!("\n  Nothing removed.\n");
            return Ok(());
        }
    }
    std::fs::remove_dir_all(&dir).map_err(|e| fail(format!("cannot remove {}: {e}", dir.display())))?;
    println!("\n  {} Removed {} {}\n", out.ok("✓"), out.bold(&id), out.dim(&format!("({} freed)", bytes(size))));
    Ok(())
}

// ------------------------------------------------------------------------------------------------------------------ doctor

fn doctor(out: Term) -> Res {
    let reg = registry::load();
    let mut problems = 0;
    let mut line = |good: Option<bool>, name: &str, detail: String| {
        let mark = match good {
            Some(true) => out.ok("✓"),
            Some(false) => out.err("✗"),
            None => out.dim("·"),
        };
        println!("  {} {}  {}", mark, out.dim(&pad(name, 10)), detail);
        if good == Some(false) {
            problems += 1;
        }
    };
    println!();
    let cpus = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    line(None, "system", format!("{} {} · {cpus} threads", std::env::consts::OS, std::env::consts::ARCH));
    let home = store::home();
    line(None, "home", format!("{}  {}", home.display(), out.dim(&bytes(store::disk_use(&home)))));

    for (rname, runner) in &reg.runners {
        let path = store::runner_path(rname, &runner.tag);
        match runner.asset() {
            Err(e) => line(Some(false), rname, e),
            Ok(_) if path.exists() => {
                line(Some(true), rname, format!("{} {}", runner.tag, out.dim(&path.display().to_string())))
            }
            Ok(_) => line(None, rname, out.dim("not downloaded yet")),
        }
    }
    for (id, m) in &reg.models {
        if store::read_manifest(id).is_none() {
            line(None, id, out.dim("not installed"));
            continue;
        }
        let dir = store::model_dir(id);
        let bad: Vec<&str> = m
            .files
            .iter()
            .filter(|f| net::sha256_file(&dir.join(&f.name)).map(|h| h != f.sha256).unwrap_or(true))
            .map(|f| f.name.as_str())
            .collect();
        if !bad.is_empty() {
            line(Some(false), id, format!("damaged: {}. Run: seal install {id} --force", bad.join(", ")));
            continue;
        }
        line(Some(true), id, format!("{} files verified", m.files.len()));
        let args = Args { positional: vec![id.clone()], values: HashMap::new(), flags: HashSet::new() };
        match ready(&args, "doctor") {
            Ok(r) => {
                let set = tools::load(None)?;
                let record = tools::record("Set an alarm for 6:45 tomorrow", &set.tools, "2026-10-18T09:00:00")?;
                match run::invoke(&r.program, &r.dir, &record, 1, true, 3) {
                    Ok(o) => {
                        let first =
                            run::calls(&o.answer).first().and_then(|c| c["tool"].as_str().map(|s| s.to_string()));
                        let t = o.timing.map(|t| millis(t.median)).unwrap_or_default();
                        line(
                            Some(first.as_deref() == Some("set_alarm")),
                            "self-test",
                            format!("{} answered in {t} · {}", r.id, out.dim(&o.isa)),
                        );
                    }
                    Err(e) => line(Some(false), "self-test", e),
                }
            }
            Err(f) => line(Some(false), "self-test", f.message),
        }
    }
    println!();
    if problems > 0 {
        return Err(fail(format!("{problems} problem{} found", if problems == 1 { "" } else { "s" })));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(list: &[&str]) -> Args {
        parse_args(&list.iter().map(|s| s.to_string()).collect::<Vec<_>>()).ok().unwrap()
    }

    #[test]
    fn words_after_the_model_are_the_request() {
        let a = args(&["relay-1", "set", "an", "alarm"]);
        assert_eq!(a.positional, vec!["relay-1", "set", "an", "alarm"]);
    }

    #[test]
    fn options_can_come_anywhere_and_take_values() {
        let a = args(&["relay-1", "--tools", "t.json", "hello", "-v", "--threads=4", "--json"]);
        assert_eq!(a.positional, vec!["relay-1", "hello"]);
        assert_eq!(a.value("tools"), Some("t.json"));
        assert_eq!(a.number("threads").ok().flatten(), Some(4));
        assert!(a.flag("verbose") && a.flag("json"));
    }

    #[test]
    fn a_double_dash_ends_the_options() {
        let a = args(&["relay-1", "--", "--tools", "is", "a", "word"]);
        assert_eq!(a.positional, vec!["relay-1", "--tools", "is", "a", "word"]);
        assert!(a.value("tools").is_none());
    }

    #[test]
    fn bad_options_are_usage_errors() {
        assert_eq!(parse_args(&["-z".to_string()]).err().map(|f| f.code), Some(2));
        assert_eq!(parse_args(&["--tools".to_string()]).err().map(|f| f.code), Some(2));
        assert!(args(&["--threads", "0"]).number("threads").is_err());
        assert!(args(&["--threads", "x"]).number("threads").is_err());
    }
}
