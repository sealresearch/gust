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

/// The file a `file://` URL names, for mirrors on disk (`file:///C:/mirror/hf` or `file:///home/me/mirror/hf`).
fn local_path(url: &str) -> Option<PathBuf> {
    let rest = url.strip_prefix("file://")?;
    let mut decoded = Vec::with_capacity(rest.len());
    let bytes = rest.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        let hex = |b: u8| (b as char).to_digit(16);
        if bytes[i] == b'%' && i + 2 < bytes.len() && hex(bytes[i + 1]).is_some() && hex(bytes[i + 2]).is_some() {
            decoded.push((hex(bytes[i + 1]).unwrap() * 16 + hex(bytes[i + 2]).unwrap()) as u8);
            i += 3;
        } else {
            decoded.push(bytes[i]);
            i += 1;
        }
    }
    let path = String::from_utf8_lossy(&decoded).into_owned();
    // "/C:/mirror" is how a Windows drive looks inside a URL.
    let b = path.as_bytes();
    let path = if b.len() > 2 && b[0] == b'/' && b[2] == b':' && b[1].is_ascii_alphabetic() {
        path[1..].to_string()
    } else {
        path
    };
    Some(PathBuf::from(path))
}

/// Open `url` for reading, and say how long it is when that is known.
fn open(url: &str) -> Result<(Box<dyn Read + Send + Sync>, Option<u64>), String> {
    if let Some(path) = local_path(url) {
        let file = fs::File::open(&path).map_err(|e| format!("not found: {} ({e})", path.display()))?;
        let len = file.metadata().ok().map(|m| m.len());
        return Ok((Box::new(file), len));
    }
    let resp = agent().get(url).call().map_err(|e| explain(url, e))?;
    let len = resp.header("Content-Length").and_then(|v| v.parse::<u64>().ok());
    Ok((resp.into_reader(), len))
}

pub fn fetch_text(url: &str) -> Result<String, String> {
    let (mut reader, _) = open(url)?;
    let mut text = String::new();
    reader.read_to_string(&mut text).map_err(|e| format!("cannot read {url}: {e}"))?;
    Ok(text)
}

#[derive(Debug)]
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
    let (mut reader, length) = open(url)?;
    let total = length.or(size).unwrap_or(0);
    if let Some(dir) = dest.parent() {
        fs::create_dir_all(dir).map_err(|e| format!("cannot create {}: {e}", dir.display()))?;
    }
    let mut part = dest.as_os_str().to_owned();
    part.push(".part");
    let part = PathBuf::from(part);
    let result = (|| -> Result<Fetched, String> {
        let mut file = fs::File::create(&part).map_err(|e| format!("cannot write {}: {e}", part.display()))?;
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
    fn file_urls_name_files_on_disk() {
        assert_eq!(local_path("file:///C:/mirror/hf/a.bin"), Some(PathBuf::from("C:/mirror/hf/a.bin")));
        assert_eq!(local_path("file:///home/me/mirror"), Some(PathBuf::from("/home/me/mirror")));
        assert_eq!(local_path("file:///C:/my%20mirror/a"), Some(PathBuf::from("C:/my mirror/a")));
        assert_eq!(local_path("https://huggingface.co"), None);
    }

    #[test]
    fn a_mirror_on_disk_downloads_and_verifies_like_a_server() {
        let dir = std::env::temp_dir().join(format!("seal-mirror-test-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("a.bin"), b"abc").unwrap();
        let url = format!("file://{}", dir.join("a.bin").display()).replace(std::path::MAIN_SEPARATOR, "/");
        let abc = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
        let dest = dir.join("out").join("a.bin");
        let done = download(&url, &dest, Some(abc), Some(3), "a.bin", Term::plain()).unwrap();
        assert_eq!(done.bytes, 3);
        assert_eq!(fs::read(&dest).unwrap(), b"abc");
        let bad = download(&url, &dir.join("out").join("b.bin"), Some("00"), None, "b.bin", Term::plain());
        assert!(bad.unwrap_err().contains("checksum"));
        assert!(!dir.join("out").join("b.bin").exists() && !dir.join("out").join("b.bin.part").exists());
        assert_eq!(fetch_text(&url).unwrap(), "abc");
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn hashing_a_file_matches_a_known_digest() {
        let path = std::env::temp_dir().join(format!("seal-sha-test-{}", std::process::id()));
        fs::write(&path, b"abc").unwrap();
        assert_eq!(sha256_file(&path).unwrap(), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
        fs::remove_file(&path).unwrap();
    }
}
