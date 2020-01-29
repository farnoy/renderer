use smallvec::SmallVec;

pub struct DoubleBuffered<T> {
    data: SmallVec<[T; 3]>,
}

impl<T> DoubleBuffered<T> {
    pub fn new<F: FnMut(u32) -> T>(count: usize, mut creator: F) -> DoubleBuffered<T> {
        DoubleBuffered {
            data: (0..count).map(|ix| creator(ix as u32)).collect(),
        }
    }

    pub fn current(&self, ix: u32) -> &T {
        &self.data[ix as usize % self.data.len()]
    }

    pub fn current_mut(&mut self, ix: u32) -> &mut T {
        let len = self.data.len();
        &mut self.data[ix as usize % len]
    }

    pub fn iter<'a>(&'a self) -> impl std::iter::Iterator<Item = &T> + 'a {
        self.data.iter()
    }
}
