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
            let ad = if i < a.len() { a[a.len() - 1 - i] } else { 1 };
            let bd = if i < b.len() { b[b.len() - 1 - i] } else { 1 };
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shape_elem_count_and_rank() {
        let s1 = Shape::new(vec![4, 8, 16]);
        assert_eq!(s1.rank(), 3);
        assert_eq!(s1.elem_count(), 512);

        let scalar = Shape::new(vec![]);
        assert_eq!(scalar.rank(), 0);
        assert_eq!(scalar.elem_count(), 1);

        let empty = Shape::new(vec![4, 0, 16]);
        assert_eq!(empty.rank(), 3);
        assert_eq!(empty.elem_count(), 0);
    }

    #[test]
    fn test_shape_dim_bounds_check() {
        let s = Shape::new(vec![2, 4, 8]);
        assert_eq!(s.dim(0).unwrap(), 2);
        assert_eq!(s.dim(1).unwrap(), 4);
        assert_eq!(s.dim(2).unwrap(), 8);
        assert!(s.dim(3).is_err());
    }

    #[test]
    fn test_shape_remove_dim() {
        let s = Shape::new(vec![2, 4, 8]);
        assert_eq!(s.remove_dim(1).dims(), &[2, 8]);
        assert_eq!(s.remove_dim(0).dims(), &[4, 8]);
        assert_eq!(s.remove_dim(2).dims(), &[2, 4]);
        assert_eq!(s.remove_dim(99).dims(), &[2, 4, 8]);
    }

    #[test]
    fn test_shape_broadcast_compatible() {
        let a = Shape::new(vec![32, 1, 64]);
        let b = Shape::new(vec![1, 16, 64]);
        assert!(a.broadcast_compatible(&b));

        let c = Shape::new(vec![64]);
        assert!(a.broadcast_compatible(&c));

        let incompatible = Shape::new(vec![32, 8, 65]);
        assert!(!a.broadcast_compatible(&incompatible));
    }
}
