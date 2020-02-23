use std::ops::Deref;

type AshEntry = ash::Entry;

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
