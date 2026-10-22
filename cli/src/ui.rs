//! Terminal output: colour that follows the brand when the terminal can show it, plain text when it cannot, and a
//! download progress bar. Everything here degrades to unadorned text for pipes, `NO_COLOR` and dumb terminals.

use std::io::{IsTerminal, Write};
use std::time::{Duration, Instant};

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum Level {
    /// No escape codes at all.
    Plain,
    /// The 16 basic colours.
    Basic,
    /// 24-bit colour, used to match the brand palette.
    True,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Style {
    /// Secondary text: labels, units, hints.
    Dim,
    /// The one accent: tool names, highlights.
    Accent,
    Ok,
    Warn,
    Err,
    Bold,
}

#[derive(Clone, Copy)]
pub struct Term {
    pub level: Level,
    pub tty: bool,
}

#[derive(Clone, Copy)]
pub enum Stream {
    Out,
    Err,
}

/// Terminal state for one stream. Call once at start-up.
pub fn detect(stream: Stream) -> Term {
    let tty = match stream {
        Stream::Out => std::io::stdout().is_terminal(),
        Stream::Err => std::io::stderr().is_terminal(),
    };
    let env = |k: &str| std::env::var(k).unwrap_or_default();
    // FORCE_COLOR turns colour on when the output is not a terminal (logs, screenshots); NO_COLOR still wins.
    let forced = !matches!(env("FORCE_COLOR").as_str(), "" | "0");
    let level = if std::env::var_os("NO_COLOR").is_some() || env("TERM") == "dumb" || !(tty || forced) {
        Level::Plain
    } else if env("COLORTERM").contains("truecolor")
        || env("COLORTERM").contains("24bit")
        || std::env::var_os("WT_SESSION").is_some()
        || cfg!(windows)
    {
        Level::True
    } else {
        Level::Basic
    };
    Term { level, tty }
}

/// Set up the console: UTF-8 output and escape-code processing on Windows. A no-op elsewhere.
pub fn init() {
    #[cfg(windows)]
    windows::init();
}

#[cfg(windows)]
mod windows {
    use std::ffi::c_void;
    extern "system" {
        fn GetStdHandle(n: u32) -> *mut c_void;
        fn GetConsoleMode(h: *mut c_void, mode: *mut u32) -> i32;
        fn SetConsoleMode(h: *mut c_void, mode: u32) -> i32;
        fn SetConsoleOutputCP(cp: u32) -> i32;
    }
    pub fn init() {
        // SAFETY: plain kernel32 calls with valid arguments; failure only means the console stays as it was.
        unsafe {
            SetConsoleOutputCP(65001);
            for id in [0xFFFF_FFF5u32, 0xFFFF_FFF4u32] {
                // STD_OUTPUT_HANDLE, STD_ERROR_HANDLE
                let h = GetStdHandle(id);
                let mut mode = 0u32;
                if !h.is_null() && GetConsoleMode(h, &mut mode) != 0 {
                    SetConsoleMode(h, mode | 0x0004); // ENABLE_VIRTUAL_TERMINAL_PROCESSING
                }
            }
        }
    }
}

impl Term {
    #[cfg(test)]
    pub fn plain() -> Term {
        Term { level: Level::Plain, tty: false }
    }

    /// `text` in `style`, with the escape codes this terminal can take.
    pub fn paint(&self, style: Style, text: &str) -> String {
        let code = match (self.level, style) {
            (Level::Plain, _) => return text.to_string(),
            (Level::True, Style::Dim) => "38;2;139;147;149",
            (Level::True, Style::Accent) => "38;2;126;179;255",
            (Level::True, Style::Ok) => "38;2;143;209;158",
            (Level::True, Style::Warn) => "38;2;217;165;79",
            (Level::True, Style::Err) => "38;2;224;122;122",
            (Level::Basic, Style::Dim) => "90",
            (Level::Basic, Style::Accent) => "94",
            (Level::Basic, Style::Ok) => "32",
            (Level::Basic, Style::Warn) => "33",
            (Level::Basic, Style::Err) => "31",
            (_, Style::Bold) => "1",
        };
        format!("\x1b[{code}m{text}\x1b[0m")
    }

    pub fn dim(&self, t: &str) -> String {
        self.paint(Style::Dim, t)
    }
    pub fn accent(&self, t: &str) -> String {
        self.paint(Style::Accent, t)
    }
    pub fn ok(&self, t: &str) -> String {
        self.paint(Style::Ok, t)
    }
    pub fn warn(&self, t: &str) -> String {
        self.paint(Style::Warn, t)
    }
    pub fn err(&self, t: &str) -> String {
        self.paint(Style::Err, t)
    }
    pub fn bold(&self, t: &str) -> String {
        self.paint(Style::Bold, t)
    }

    /// Underline, for the words of the request a value was read from.
    pub fn underline(&self, style: Style, text: &str) -> String {
        match self.level {
            Level::Plain => format!("[{text}]"),
            _ => format!("\x1b[4m{}\x1b[24m", self.paint(style, text)),
        }
    }
}

/// Number of characters on screen (the output only uses single-width characters).
pub fn width(s: &str) -> usize {
    let mut n = 0;
    let mut in_escape = false;
    for c in s.chars() {
        if in_escape {
            if c == 'm' {
                in_escape = false;
            }
        } else if c == '\x1b' {
            in_escape = true;
        } else {
            n += 1;
        }
    }
    n
}

/// `s` padded on the right to `w` columns, counting only what is visible.
pub fn pad(s: &str, w: usize) -> String {
    let n = width(s);
    format!("{s}{}", " ".repeat(w.saturating_sub(n)))
}

/// `s` padded on the left to `w` columns.
pub fn rpad(s: &str, w: usize) -> String {
    let n = width(s);
    format!("{}{s}", " ".repeat(w.saturating_sub(n)))
}

/// A bar of `cells` characters, `fraction` (0 to 1) of it filled, in eighths of a cell.
pub fn bar(fraction: f64, cells: usize) -> String {
    const PARTS: [char; 8] = [' ', '▏', '▎', '▍', '▌', '▋', '▊', '▉'];
    let eighths = (fraction.clamp(0.0, 1.0) * cells as f64 * 8.0).round() as usize;
    let (full, part) = (eighths / 8, eighths % 8);
    let mut s = "█".repeat(full);
    if full < cells {
        s.push(PARTS[part]);
        s.push_str(&" ".repeat(cells - full - 1));
    }
    s
}

pub fn bytes(n: u64) -> String {
    const UNITS: [&str; 4] = ["B", "KB", "MB", "GB"];
    let mut v = n as f64;
    let mut i = 0;
    while v >= 1000.0 && i < UNITS.len() - 1 {
        v /= 1000.0;
        i += 1;
    }
    if i == 0 {
        format!("{n} B")
    } else if v >= 100.0 {
        format!("{v:.0} {}", UNITS[i])
    } else {
        format!("{v:.1} {}", UNITS[i])
    }
}

pub fn millis(ms: f64) -> String {
    if ms >= 100.0 {
        format!("{ms:.0} ms")
    } else if ms >= 10.0 {
        format!("{ms:.1} ms")
    } else {
        format!("{ms:.2} ms")
    }
}

/// A download progress line, redrawn in place on a terminal and silent otherwise.
pub struct Progress {
    term: Term,
    label: String,
    total: u64,
    done: u64,
    started: Instant,
    last_draw: Instant,
}

impl Progress {
    pub fn new(term: Term, label: &str, total: u64) -> Progress {
        let now = Instant::now();
        Progress {
            term,
            label: label.to_string(),
            total,
            done: 0,
            started: now,
            last_draw: now - Duration::from_secs(1),
        }
    }

    pub fn advance(&mut self, n: u64) {
        self.done += n;
        if self.term.tty && self.last_draw.elapsed() >= Duration::from_millis(60) {
            self.draw();
        }
    }

    fn draw(&mut self) {
        self.last_draw = Instant::now();
        let secs = self.started.elapsed().as_secs_f64().max(0.001);
        let rate = self.done as f64 / secs;
        let frac = if self.total > 0 { self.done as f64 / self.total as f64 } else { 0.0 };
        let line = format!(
            "  {} {}  {}  {}  {}",
            self.term.dim("↓"),
            pad(&self.label, 22),
            self.term.accent(&bar(frac, 24)),
            rpad(&format!("{:.0}%", frac * 100.0), 4),
            self.term.dim(&format!("{} / {}  {}/s", bytes(self.done), bytes(self.total), bytes(rate as u64))),
        );
        let mut e = std::io::stderr();
        let _ = write!(e, "\r\x1b[2K{line}");
        let _ = e.flush();
    }

    /// Clear the progress line. The caller prints the final status.
    pub fn finish(&mut self) {
        if self.term.tty {
            let mut e = std::io::stderr();
            let _ = write!(e, "\r\x1b[2K");
            let _ = e.flush();
        }
    }

    pub fn elapsed(&self) -> Duration {
        self.started.elapsed()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bars_fill_in_eighths() {
        assert_eq!(bar(0.0, 4), "    ");
        assert_eq!(bar(1.0, 4), "████");
        assert_eq!(bar(0.5, 4), "██  ");
        assert_eq!(bar(0.125, 4), "▌   ");
        assert_eq!(width(&bar(0.37, 10)), 10);
    }

    #[test]
    fn widths_ignore_escape_codes() {
        let t = Term { level: Level::True, tty: true };
        assert_eq!(width(&t.accent("tool")), 4);
        assert_eq!(pad(&t.accent("tool"), 6).chars().count(), t.accent("tool").chars().count() + 2);
    }

    #[test]
    fn plain_terminals_get_no_escape_codes() {
        let t = Term::plain();
        assert_eq!(t.accent("x"), "x");
        assert_eq!(t.underline(Style::Accent, "x"), "[x]");
    }

    #[test]
    fn sizes_and_times_read_naturally() {
        assert_eq!(bytes(11_474_672), "11.5 MB");
        assert_eq!(bytes(512), "512 B");
        assert_eq!(bytes(2_581_792), "2.6 MB");
        assert_eq!(millis(5.83), "5.83 ms");
        assert_eq!(millis(39.4), "39.4 ms");
        assert_eq!(millis(210.0), "210 ms");
    }
}
