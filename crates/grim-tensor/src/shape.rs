//! `Shape` — fully-static n-d tensor shape.

use crate::error::{Error, Result};

/// Fully-static shape. Multi-dim layout with row-major walks (this is the
/// only layout Grim supports in v1; permuted strides for a few specific ops
/// like attention come via temporary reshape/transpose).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Shape {
    dims: Vec<usize>,
}

impl Shape {
    pub fn new(dims: impl Into<Vec<usize>>) -> Self {
        Self { dims: dims.into() }
    }

    pub fn from_slice(dims: &[usize]) -> Self {
        Self { dims: dims.to_vec() }
    }

    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    pub fn rank(&self) -> usize {
        self.dims.len()
    }

    pub fn elem_count(&self) -> usize {
        self.dims.iter().product()
    }

    pub fn dim(&self, axis: usize) -> Result<usize> {
        self.dims
            .get(axis)
            .copied()
            .ok_or_else(|| Error::IndexOutOfBounds(format!("axis {axis} out of rank {}", self.rank())))
    }

    /// Return a new shape with `axis` removed.
    pub fn remove_dim(&self, axis: usize) -> Shape {
        let mut d = self.dims.clone();
        if axis < d.len() {
            d.remove(axis);
        }
        Shape { dims: d }
    }

    /// Generic "broadcast" check used by elementwise ops — both must agree on
    /// every dim or one must be 1.
    pub fn broadcast_compatible(&self, other: &Shape) -> bool {
        let a = self.dims();
        let b = other.dims();
        let n = a.len().max(b.len());
        for i in 0..n {
            let ad = *a.get(a.len().saturating_sub(i + 1)).unwrap_or(&1);
            let bd = *b.get(b.len().saturating_sub(i + 1)).unwrap_or(&1);
            if ad != bd && ad != 1 && bd != 1 {
                return false;
            }
        }
        true
    }
}

impl From<Vec<usize>> for Shape {
    fn from(v: Vec<usize>) -> Self {
        Self::new(v)
    }
}

impl From<&[usize]> for Shape {
    fn from(v: &[usize]) -> Self {
        Self::from_slice(v)
    }
}

impl<const N: usize> From<[usize; N]> for Shape {
    fn from(v: [usize; N]) -> Self {
        Self::new(v.to_vec())
    }
}
