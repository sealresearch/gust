//! Downloads: from Hugging Face for the weights and from a GitHub release for the program that runs them. Every file
//! is written beside its destination, checked, and only then moved into place.

use crate::ui::{Progress, Term};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;

/// Where Hugging Face is. `SEAL_HF_URL` points it at a mirror.
pub fn hf_base() -> String {
    base("SEAL_HF_URL", "https://huggingface.co")
}

/// Where GitHub releases are. `SEAL_GITHUB_URL` points it at a mirror.
pub fn github_base() -> String {
    base("SEAL_GITHUB_URL", "https://github.com")
}

fn base(var: &str, default: &str) -> String {
    std::env::var(var).unwrap_or_else(|_| default.to_string()).trim_end_matches('/').to_string()
}

fn agent() -> ureq::Agent {
    ureq::AgentBuilder::new()
        .user_agent(concat!("seal/", env!("CARGO_PKG_VERSION")))
        .timeout_connect(Duration::from_secs(15))
        .timeout_read(Duration::from_secs(60))
        .build()
}

fn explain(url: &str, e: ureq::Error) -> String {
    match e {
        ureq::Error::Status(404, _) => format!("not found: {url}"),
        ureq::Error::Status(401 | 403, _) => format!("access was refused for {url} (is the repository public yet?)"),
        ureq::Error::Status(code, _) => format!("the server answered {code} for {url}"),
        ureq::Error::Transport(t) => format!("cannot reach {url}: {t}"),
    }
}

pub fn fetch_text(url: &str) -> Result<String, String> {
    agent().get(url).call().map_err(|e| explain(url, e))?.into_string().map_err(|e| format!("cannot read {url}: {e}"))
}

pub struct Fetched {
    pub bytes: u64,
    pub secs: f64,
}

/// Download `url` to `dest`. With a SHA-256 and a size the file is only kept if it matches them.
pub fn download(
    url: &str,
    dest: &Path,
    sha256: Option<&str>,
    size: Option<u64>,
    label: &str,
    term: Term,
) -> Result<Fetched, String> {
    let resp = agent().get(url).call().map_err(|e| explain(url, e))?;
    let total = resp.header("Content-Length").and_then(|v| v.parse::<u64>().ok()).or(size).unwrap_or(0);
    if let Some(dir) = dest.parent() {
        fs::create_dir_all(dir).map_err(|e| format!("cannot create {}: {e}", dir.display()))?;
    }
    let mut part = dest.as_os_str().to_owned();
    part.push(".part");
    let part = PathBuf::from(part);
    let result = (|| -> Result<Fetched, String> {
        let mut file = fs::File::create(&part).map_err(|e| format!("cannot write {}: {e}", part.display()))?;
        let mut reader = resp.into_reader();
        let mut hasher = Sha256::new();
        let mut buf = vec![0u8; 64 * 1024];
        let mut progress = Progress::new(term, label, total);
        let mut bytes = 0u64;
        loop {
            let n = reader.read(&mut buf).map_err(|e| format!("download of {label} broke off: {e}"))?;
            if n == 0 {
                break;
            }
            file.write_all(&buf[..n]).map_err(|e| format!("cannot write {}: {e}", part.display()))?;
            hasher.update(&buf[..n]);
            bytes += n as u64;
            progress.advance(n as u64);
        }
        progress.finish();
        file.flush().map_err(|e| e.to_string())?;
        let got = hex(&hasher.finalize());
        if size.is_some_and(|n| n != bytes) || sha256.is_some_and(|sha| sha != got) {
            let short = |h: &str| h.chars().take(12).collect::<String>();
            return Err(format!(
                "{label} does not match its checksum (expected {}…, got {}…); it was not installed",
                short(sha256.unwrap_or("-")),
                short(&got)
            ));
        }
        Ok(Fetched { bytes, secs: progress.elapsed().as_secs_f64() })
    })();
    match result {
        Ok(done) => {
            if dest.exists() {
                fs::remove_file(dest).map_err(|e| format!("cannot replace {}: {e}", dest.display()))?;
            }
            fs::rename(&part, dest).map_err(|e| format!("cannot move {} into place: {e}", dest.display()))?;
            Ok(done)
        }
        Err(e) => {
            let _ = fs::remove_file(&part);
            Err(e)
        }
    }
}

pub fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// SHA-256 of a file on disk.
pub fn sha256_file(path: &Path) -> Result<String, String> {
    let mut file = fs::File::open(path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 64 * 1024];
    loop {
        let n = file.read(&mut buf).map_err(|e| e.to_string())?;
        if n == 0 {
            return Ok(hex(&hasher.finalize()));
        }
        hasher.update(&buf[..n]);
    }
}

/// The checksums of a `SHA256SUMS` file: name to SHA-256.
pub fn parse_sums(text: &str) -> HashMap<String, String> {
    text.lines()
        .filter_map(|l| {
            let (hash, name) = l.split_once(char::is_whitespace)?;
            Some((name.trim().trim_start_matches('*').to_string(), hash.trim().to_lowercase()))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checksum_files_parse_in_both_styles() {
        let sums = parse_sums("aa11 *relay-linux-x64\nbb22  relay-windows-x64.exe\n\n");
        assert_eq!(sums["relay-linux-x64"], "aa11");
        assert_eq!(sums["relay-windows-x64.exe"], "bb22");
        assert_eq!(sums.len(), 2);
    }

    #[test]
    fn hashing_a_file_matches_a_known_digest() {
        let path = std::env::temp_dir().join(format!("seal-sha-test-{}", std::process::id()));
        fs::write(&path, b"abc").unwrap();
        assert_eq!(sha256_file(&path).unwrap(), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
        fs::remove_file(&path).unwrap();
    }
}
