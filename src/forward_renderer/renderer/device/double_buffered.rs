use smallvec::SmallVec;

pub struct DoubleBuffered<T> {
    // Supports up to triple-buffered inline, should be enough
    amount: u8,
    data: SmallVec<[T; 3]>,
}

impl<T> DoubleBuffered<T> {
    pub fn new<F: FnMut(u32) -> T>(mut creator: F, amount: u8) -> DoubleBuffered<T> {
        DoubleBuffered {
            amount,
            data: (0..amount).map(|ix| creator(u32::from(ix))).collect(),
        }
    }

    pub fn current(&self, ix: u32) -> &T {
        &self.data[ix as usize % self.amount as usize]
    }

    pub fn current_mut(&mut self, ix: u32) -> &mut T {
        &mut self.data[ix as usize % self.amount as usize]
    }
}
