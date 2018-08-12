use ash::{self, version::V1_0};
use std::ops::Deref;

pub type AshEntry = ash::Entry<V1_0>;

pub struct Entry {
    handle: AshEntry,
}

impl Entry {
    pub fn new() -> Result<Entry, ash::LoadingError> {
        let entry = AshEntry::new()?;

        Ok(Entry { handle: entry })
    }

    pub fn vk(&self) -> &AshEntry {
        &self.handle
    }
}

impl Deref for Entry {
    type Target = AshEntry;

    fn deref(&self) -> &AshEntry {
        &self.handle
    }
}
