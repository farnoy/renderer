use std::ops::{Index, IndexMut};

use smallvec::SmallVec;

#[derive(Clone)]
pub(crate) struct DoubleBuffered<T> {
    data: SmallVec<[T; 3]>,
}

impl<T> DoubleBuffered<T> {
    pub(crate) fn new<F: FnMut(u32) -> T>(count: usize, mut creator: F) -> DoubleBuffered<T> {
        DoubleBuffered {
            data: (0..count).map(|ix| creator(ix as u32)).collect(),
        }
    }

    pub(crate) fn current(&self, ix: u32) -> &T {
        &self.data[ix as usize % self.data.len()]
    }

    pub(crate) fn current_mut(&mut self, ix: u32) -> &mut T {
        let len = self.data.len();
        &mut self.data[ix as usize % len]
    }

    #[allow(unused)]
    pub(crate) fn iter<'a>(&'a self) -> impl Iterator<Item = &T> + 'a {
        self.data.iter()
    }

    #[allow(unused)]
    pub(crate) fn iter_mut<'a>(&'a mut self) -> impl Iterator<Item = &mut T> + 'a {
        self.data.iter_mut()
    }

    pub(crate) fn into_iter(self) -> impl Iterator<Item = T> {
        self.data.into_iter()
    }
}

impl<T> Index<usize> for DoubleBuffered<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

impl<T> IndexMut<usize> for DoubleBuffered<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.data[index]
    }
}
