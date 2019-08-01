#![allow(warnings)]

use rayon;
use std::{
    borrow::BorrowMut,
    collections::btree_map,
    mem::swap,
    ops::{Deref, DerefMut},
};

#[derive(PartialEq, Eq, Debug)]
struct Device(pub u32);
#[derive(PartialEq, Eq, Debug)]
struct DepthImage(pub u32);
#[derive(PartialEq, Eq, Debug)]
struct Set(pub u32);

struct Renderer {
    device: Device,
    depth_image: DepthImage,
    set: Set,
}

#[macro_export]
macro_rules! par_custom {
    ($first:block $second:block $($tail:block)+) => {
        rayon::join(
            || $first,
            || { par_custom!($second $($tail)+) },
        );
    };
    ($first:block $second:block) => {
        rayon::join(
            || $first,
            || $second,
        );
    };
}

pub struct EntitiesStorage {
    mask: croaring::Bitmap,
    deleted: croaring::Bitmap,
}

const MAX_ENTITIES: u32 = 512;

impl EntitiesStorage {
    pub fn new() -> EntitiesStorage {
        let mut mask = croaring::Bitmap::create();
        EntitiesStorage {
            mask: mask,
            deleted: croaring::Bitmap::create(),
        }
    }

    pub fn mask(&self) -> &croaring::Bitmap {
        &self.mask
    }

    pub fn allocate(&mut self) -> u32 {
        let free_slots = self.mask.flip(0..MAX_ENTITIES as u64) - &self.deleted;
        let free = free_slots
            .iter()
            .next()
            .expect("no space to allocate entities");
        self.mask.add(free);
        free
    }

    pub fn allocate_many(&mut self, n: u32) -> Vec<u32> {
        let free_slots = self.mask.flip(0..MAX_ENTITIES as u64) - &self.deleted;
        let free_ids = free_slots.iter().take(n as usize).collect::<Vec<u32>>();
        self.mask.add_many(&free_ids);
        free_ids
    }

    pub fn allocate_mask(&mut self, n: u32) -> croaring::Bitmap {
        let free_slots = self.mask.flip(0..MAX_ENTITIES as u64) - &self.deleted;
        let free_ids = free_slots.iter().take(n as usize).collect::<Vec<u32>>();
        let mut mask = croaring::Bitmap::create();
        mask.add_many(&free_ids);
        self.mask.or_inplace(&mask);
        mask
    }

    pub fn remove(&mut self, ix: u32) {
        self.mask.remove(ix);
        self.deleted.add(ix);
    }

    pub fn maintain(&mut self) -> croaring::Bitmap {
        let mut x = croaring::Bitmap::create();
        swap(&mut self.deleted, &mut x);
        x
    }
}

pub struct ComponentStorage<T> {
    mask: croaring::Bitmap,
    data: btree_map::BTreeMap<u32, T>,
}

impl<T> ComponentStorage<T> {
    pub fn new() -> ComponentStorage<T> {
        ComponentStorage {
            mask: croaring::Bitmap::create(),
            data: btree_map::BTreeMap::new(),
        }
    }

    pub fn mask(&self) -> &croaring::Bitmap {
        &self.mask
    }

    pub fn allocate_mask(&mut self, mask: &croaring::Bitmap) {
        debug_assert_eq!(
            self.mask.and_cardinality(mask),
            0,
            "ComponentStorage::allocate_mask() interescts with stored components"
        );
        self.mask.or_inplace(mask);
    }

    pub fn replace_mask(&mut self, mask: &croaring::Bitmap) {
        let diff = self.mask.andnot(mask);
        self.mask.clone_from(mask);
        for ix in diff.iter() {
            self.data.remove(&ix);
        }
    }

    pub fn allocate_many(&mut self, ixes: &[u32]) {
        debug_assert_eq!(
            {
                let mut temp = croaring::Bitmap::create();
                temp.add_many(ixes);
                self.mask.and_cardinality(&temp)
            },
            0,
            "ComponentStorage::allocate_many() interescts with stored components"
        );
        self.mask.add_many(ixes);
    }

    pub fn get<'a>(&'a self, ix: u32) -> Option<&'a T> {
        debug_assert!(self.mask.contains(ix), "fetching dead component");
        self.data.get(&ix)
    }

    pub fn entry<'a>(&'a mut self, ix: u32) -> ComponentEntry<'a, T> {
        ComponentEntry {
            key: ix,
            btree_entry: self.data.entry(ix),
            mask: &mut self.mask,
        }
    }

    pub fn insert<'a>(&'a mut self, ix: u32, val: T) -> Option<T> {
        self.mask.add(ix);
        self.data.insert(ix, val)
    }

    pub fn maintain(&mut self, freed: &croaring::Bitmap) {
        for x in freed.iter() {
            self.data.remove(&x);
        }
        self.mask.andnot_inplace(freed);
    }
}

pub struct ComponentEntry<'a, T> {
    key: u32,
    btree_entry: btree_map::Entry<'a, u32, T>,
    mask: &'a mut croaring::Bitmap,
}

impl<'a, T> ComponentEntry<'a, T> {
    pub fn remove(self) {
        self.mask.remove(self.key);
        match self.btree_entry {
            btree_map::Entry::Occupied(slot) => {
                slot.remove();
                ()
            }
            btree_map::Entry::Vacant(_) => (),
        }
    }

    pub fn or_insert(self, fallback: T) -> &'a mut T {
        self.mask.add(self.key);
        self.btree_entry.or_insert(fallback)
    }

    pub fn or_insert_with<F: FnOnce() -> T>(self, inserter: F) -> &'a mut T {
        self.mask.add(self.key);
        self.btree_entry.or_insert_with(inserter)
    }
}

pub struct FlagStorage {
    pub present: croaring::Bitmap,
}

impl FlagStorage {
    pub fn new() -> FlagStorage {
        FlagStorage {
            present: croaring::Bitmap::create(),
        }
    }

    pub fn invert(&self) -> croaring::Bitmap {
        self.present.flip(0..self.present.maximum() as u64 + 1)
    }
}

#[derive(PartialEq, Clone, Debug)]
struct Position(glm::Vec3);

#[test]
fn test_flags() {
    let mut x = FlagStorage::new();
    x.present.add_range(0..40);
    x.present.add_range(45..60);
    x.present.add_range(63..80);
    assert_eq!(
        (40..45).chain(60..63).collect::<Vec<_>>(),
        x.invert().iter().collect::<Vec<_>>()
    );
}

#[test]
fn test_no_reuse() {
    let mut entities = EntitiesStorage::new();
    let first = entities.allocate();
    let second = entities.allocate();
    entities.remove(first);
    assert_eq!(entities.allocate(), second + 1);
}

#[test]
fn test_reuse_after_maintain() {
    let mut entities = EntitiesStorage::new();
    let first = entities.allocate();
    let second = entities.allocate();
    entities.remove(first);
    assert_eq!(entities.maintain().to_vec(), vec![first]);
    assert_eq!(entities.allocate(), first);
}

#[test]
fn test_components() {
    let entities = &mut EntitiesStorage::new();
    let positions = &mut ComponentStorage::<glm::Vec3>::new();
    let velocity = &mut ComponentStorage::<glm::Vec3>::new();
    let timedelta = &mut 0.0f32;

    let ixes = entities.allocate_many(5);
    positions.allocate_many(&[ixes[0], ixes[3], ixes[4], 8]);

    *timedelta = 3.0;
    par_custom! {
        {
            *timedelta = 3.0;
                    *timedelta = 5.0;
        }
        {
            velocity.allocate_mask(entities.mask());
            for x in entities.mask().iter() {
                *velocity.entry(x).or_insert(na::zero()) = glm::vec3(10.0, 20.0, 0.0);
            }
        }
    }
    for ix in (positions.mask() & velocity.mask() & entities.mask()).iter() {
        *positions.entry(ix).or_insert(na::zero()) += velocity.data.get(&ix).unwrap() * *timedelta;
    }

    assert_eq!(*timedelta, 5.0);
    assert_eq!(positions.data.get(&3), Some(&glm::vec3(50.0, 100.0, 0.0)));
}
