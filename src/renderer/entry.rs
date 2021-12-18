use std::ops::Deref;

type AshEntry = ash::Entry;

pub(crate) struct Entry {
    handle: AshEntry,
}

impl Entry {
    pub(crate) fn new() -> Entry {
        let entry = unsafe { AshEntry::new() };

        Entry { handle: entry }
    }

    pub(crate) fn vk(&self) -> &AshEntry {
        &self.handle
    }
}

impl Deref for Entry {
    type Target = AshEntry;

    fn deref(&self) -> &AshEntry {
        &self.handle
    }
}
