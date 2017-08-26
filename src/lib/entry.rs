use ash;
use ash::version;
use std::ops;
use std::sync::Arc;

pub type AshEntry = ash::Entry<version::V1_0>;

pub struct Entry {
    handle: AshEntry,
}

impl Entry {
    pub fn new() -> Result<Arc<Entry>, ash::LoadingError> {
        let entry = AshEntry::new()?;

        Ok(Arc::new(Entry { handle: entry }))
    }

    pub fn vk(&self) -> &AshEntry {
        &self.handle
    }
}

impl ops::Deref for Entry {
    type Target = AshEntry;

    fn deref(&self) -> &AshEntry {
        &self.handle
    }
}