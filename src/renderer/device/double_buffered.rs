#[cfg(windows)]
const SIZE: usize = 2;
#[cfg(not(windows))]
const SIZE: usize = 3;

pub struct DoubleBuffered<T> {
    data: [T; SIZE],
}

impl<T> DoubleBuffered<T> {
    pub fn new<F: FnMut(u32) -> T>(mut creator: F) -> DoubleBuffered<T> {
        #[cfg(windows)]
        return DoubleBuffered {
            data: [creator(0), creator(1)],
        };

        #[cfg(not(windows))]
        return DoubleBuffered {
            data: [creator(0), creator(1), creator(2)],
        };
    }

    pub fn current(&self, ix: u32) -> &T {
        &self.data[ix as usize % SIZE]
    }

    pub fn current_mut(&mut self, ix: u32) -> &mut T {
        &mut self.data[ix as usize % SIZE]
    }

    pub fn iter<'a>(&'a self) -> impl std::iter::Iterator<Item = &T> + 'a {
        self.data.iter()
    }
}
