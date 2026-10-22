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
    /// Which runtime runs it.
    pub runner: String,
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
            assert!(!m.files.is_empty());
            for f in &m.files {
                assert_eq!(f.sha256.len(), 64, "{} has a SHA-256", f.name);
                assert!(f.sha256.bytes().all(|b| b.is_ascii_hexdigit()));
            }
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
