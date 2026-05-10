//! Thin wrapper over the OS-native secret store (macOS Keychain / Windows
//! Credential Manager / Linux Secret Service).
//!
//! All secrets are scoped under the service id `com.screenieai.app` and keyed
//! by a stable `name` (e.g. `"anthropic_api_key"`). Secrets are never written
//! to disk in plaintext — the OS handles encryption and access prompts.

use serde::{Serialize, Serializer};

const SERVICE: &str = "com.screenieai.app";
const ALLOWED_NAMES: &[&str] = &["anthropic_api_key", "openai_api_key", "gemini_api_key"];

#[derive(Debug, thiserror::Error)]
pub enum SecretError {
    #[error("invalid secret name")]
    InvalidName,
    #[error("keyring: {0}")]
    Keyring(String),
}

impl Serialize for SecretError {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(&self.to_string())
    }
}

impl From<keyring::Error> for SecretError {
    fn from(e: keyring::Error) -> Self {
        SecretError::Keyring(e.to_string())
    }
}

fn entry(name: &str) -> Result<keyring::Entry, SecretError> {
    if !ALLOWED_NAMES.contains(&name) {
        return Err(SecretError::InvalidName);
    }
    Ok(keyring::Entry::new(SERVICE, name)?)
}

pub fn set(name: &str, value: &str) -> Result<(), SecretError> {
    entry(name)?.set_password(value)?;
    Ok(())
}

pub fn get(name: &str) -> Result<Option<String>, SecretError> {
    match entry(name)?.get_password() {
        Ok(v) => Ok(Some(v)),
        Err(keyring::Error::NoEntry) => Ok(None),
        Err(e) => Err(e.into()),
    }
}

pub fn delete(name: &str) -> Result<(), SecretError> {
    match entry(name)?.delete_credential() {
        Ok(()) => Ok(()),
        Err(keyring::Error::NoEntry) => Ok(()),
        Err(e) => Err(e.into()),
    }
}
