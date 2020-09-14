use smallvec::SmallVec;

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

    pub(crate) fn iter<'a>(&'a self) -> impl std::iter::Iterator<Item = &T> + 'a {
        self.data.iter()
    }
}
