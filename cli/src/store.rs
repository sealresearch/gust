//! Where things live: `~/.sealpup` (or `$SEALPUP_HOME`), with one folder per model and one per downloaded runtime.

use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};

pub fn home() -> PathBuf {
    if let Some(h) = std::env::var_os("SEALPUP_HOME") {
        return PathBuf::from(h);
    }
    let user = std::env::var_os("USERPROFILE").or_else(|| std::env::var_os("HOME")).map(PathBuf::from);
    user.unwrap_or_else(|| PathBuf::from(".")).join(".sealpup")
}

pub fn model_dir(id: &str) -> PathBuf {
    home().join("models").join(id)
}

/// Where the downloaded program of a runtime is kept, by release tag.
pub fn runner_path(runner: &str, tag: &str) -> PathBuf {
    let exe = if cfg!(windows) { format!("{runner}.exe") } else { runner.to_string() };
    home().join("runtimes").join(runner).join(tag).join(exe)
}

/// Where the downloaded Gust library is kept, by release tag.
pub fn engine_path(engine: &str, tag: &str) -> PathBuf {
    let file = if cfg!(windows) {
        format!("{engine}.dll")
    } else if cfg!(target_os = "macos") {
        format!("lib{engine}.dylib")
    } else {
        format!("lib{engine}.so")
    };
    home().join("runtimes").join(engine).join(tag).join(file)
}

/// What an install leaves next to the files, so `list` and `info` do not have to guess.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Installed {
    pub id: String,
    /// Seconds since the Unix epoch.
    pub installed_at: u64,
    pub runtime_tag: String,
    /// The tag of the library the install fetched, when there was one.
    #[serde(default)]
    pub engine_tag: String,
    pub files: Vec<String>,
}

const MANIFEST: &str = "sealpup-install.json";

pub fn write_manifest(dir: &Path, m: &Installed) -> Result<(), String> {
    let text = serde_json::to_string_pretty(m).map_err(|e| e.to_string())?;
    fs::write(dir.join(MANIFEST), text + "\n").map_err(|e| format!("cannot write {}: {e}", dir.display()))
}

pub fn read_manifest(id: &str) -> Option<Installed> {
    let text = fs::read_to_string(model_dir(id).join(MANIFEST)).ok()?;
    serde_json::from_str(&text).ok()
}

/// Bytes used by everything under `path`.
pub fn disk_use(path: &Path) -> u64 {
    let mut total = 0;
    if let Ok(entries) = fs::read_dir(path) {
        for e in entries.flatten() {
            match e.metadata() {
                Ok(m) if m.is_dir() => total += disk_use(&e.path()),
                Ok(m) => total += m.len(),
                Err(_) => {}
            }
        }
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manifests_round_trip() {
        let dir = std::env::temp_dir().join(format!("sealpup-store-test-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let m = Installed {
            id: "relay-1".into(),
            installed_at: 7,
            runtime_tag: "v0.1.0".into(),
            engine_tag: "v0.2.0".into(),
            files: vec!["a".into()],
        };
        write_manifest(&dir, &m).unwrap();
        let back: Installed = serde_json::from_str(&fs::read_to_string(dir.join(MANIFEST)).unwrap()).unwrap();
        assert_eq!(back.id, "relay-1");
        assert_eq!(back.files, vec!["a"]);
        assert!(disk_use(&dir) > 0);
        fs::remove_dir_all(&dir).unwrap();
    }
}
