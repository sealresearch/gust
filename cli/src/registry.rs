//! What can be installed: models, the files each is made of with their checksums, and the runtime that runs them.
//! The list is compiled into the program, so the checksums of the weights are the ones this version was released with.

use serde::Deserialize;
use std::collections::BTreeMap;

const REGISTRY: &str = include_str!("../registry.json");

#[derive(Deserialize, Clone, Debug)]
pub struct File {
    pub name: String,
    pub sha256: String,
    pub size: u64,
}

#[derive(Deserialize, Clone, Debug)]
pub struct Model {
    pub name: String,
    pub task: String,
    pub parameters: String,
    pub licence: String,
    /// Hugging Face repository holding the files.
    pub repo: String,
    pub revision: String,
    /// The program that runs it, one per model (kept as a fallback).
    pub runner: String,
    /// The Gust library that runs it in this process, and the model's id inside that library.
    #[serde(default)]
    pub engine: Option<String>,
    #[serde(default)]
    pub engine_id: Option<String>,
    pub research: String,
    pub files: Vec<File>,
}

#[derive(Deserialize, Clone, Debug)]
pub struct Runner {
    /// GitHub repository whose release holds the program.
    pub repo: String,
    pub tag: String,
    /// The release file for each "os-arch".
    pub assets: BTreeMap<String, String>,
}

#[derive(Deserialize, Debug)]
pub struct Registry {
    pub models: BTreeMap<String, Model>,
    pub aliases: BTreeMap<String, String>,
    /// Shared libraries that run models in this process: one file serves every model built into it.
    #[serde(default)]
    pub engines: BTreeMap<String, Runner>,
    pub runners: BTreeMap<String, Runner>,
}

impl Model {
    /// All the bytes an install downloads, apart from the runtime.
    pub fn total_size(&self) -> u64 {
        self.files.iter().map(|f| f.size).sum()
    }
}

pub fn load() -> Registry {
    serde_json::from_str(REGISTRY).expect("registry.json is valid")
}

impl Registry {
    /// The id of a model from what the user typed: an id, an alias, in any case.
    pub fn resolve(&self, name: &str) -> Result<String, String> {
        let n = name.trim().to_lowercase();
        if self.models.contains_key(&n) {
            return Ok(n);
        }
        if let Some(id) = self.aliases.get(&n) {
            return Ok(id.clone());
        }
        let known: Vec<&str> = self.models.keys().map(|s| s.as_str()).collect();
        Err(format!("unknown model \"{name}\". Models: {}", known.join(", ")))
    }

    pub fn runner(&self, model: &Model) -> &Runner {
        &self.runners[&model.runner]
    }

    /// The library that runs a model in this process, with the model's id inside it, if it has one.
    pub fn engine(&self, model: &Model, id: &str) -> Option<(&str, &Runner, String)> {
        let (name, engine) = self.engines.get_key_value(model.engine.as_deref()?)?;
        Some((name.as_str(), engine, model.engine_id.clone().unwrap_or_else(|| id.to_string())))
    }
}

impl Runner {
    /// The release file for this machine, if there is one.
    pub fn asset(&self) -> Result<&str, String> {
        let key = platform();
        self.assets.get(&key).map(|s| s.as_str()).ok_or_else(|| {
            format!("no build for {key} yet. Builds: {}", self.assets.keys().cloned().collect::<Vec<_>>().join(", "))
        })
    }
}

/// "os-arch" of this machine, as the registry spells it.
pub fn platform() -> String {
    format!("{}-{}", std::env::consts::OS, std::env::consts::ARCH)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_registry_parses_and_is_consistent() {
        let r = load();
        for (id, m) in &r.models {
            assert!(r.runners.contains_key(&m.runner), "{id} names a runner that exists");
            if let Some(e) = &m.engine {
                assert!(r.engines.contains_key(e), "{id} names an engine that exists");
            }
            assert!(!m.files.is_empty());
            for f in &m.files {
                assert_eq!(f.sha256.len(), 64, "{} has a SHA-256", f.name);
                assert!(f.sha256.bytes().all(|b| b.is_ascii_hexdigit()));
            }
        }
        for (name, e) in &r.engines {
            assert!(e.tag.starts_with('v'), "{name} is pinned to a release tag");
            assert!(
                e.assets.contains_key("windows-x86_64") && e.assets.contains_key("linux-x86_64"),
                "{name} covers the main platforms"
            );
        }
        for target in r.aliases.values() {
            assert!(r.models.contains_key(target));
        }
    }

    #[test]
    fn names_resolve_in_any_case_and_through_aliases() {
        let r = load();
        assert_eq!(r.resolve("relay-1").unwrap(), "relay-1");
        assert_eq!(r.resolve("Relay").unwrap(), "relay-1");
        assert!(r.resolve("nope").unwrap_err().contains("relay-1"));
    }

    #[test]
    fn relay_is_about_eleven_and_a_half_megabytes_of_model() {
        let r = load();
        let m = &r.models["relay-1"];
        let model_only: u64 = m.files.iter().filter(|f| f.name.ends_with(".gust")).map(|f| f.size).sum();
        assert_eq!(model_only, 11_474_672);
    }
}
