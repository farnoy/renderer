pub struct DoubleBuffered<T> {
    data: [T; 2],
}

impl<T> DoubleBuffered<T> {
    pub fn new<F: FnMut(u32) -> T>(mut creator: F) -> DoubleBuffered<T> {
        DoubleBuffered {
            data: [creator(0), creator(1)],
        }
    }

    pub fn current(&self, ix: u32) -> &T {
        &self.data[ix as usize % 2]
    }

    pub fn current_mut(&mut self, ix: u32) -> &mut T {
        &mut self.data[ix as usize % 2]
    }
}
