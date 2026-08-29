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
        Self {
            dims: dims.to_vec(),
        }
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
        self.dims.get(axis).copied().ok_or_else(|| {
            Error::IndexOutOfBounds(format!("axis {axis} out of rank {}", self.rank()))
        })
    }

    /// Return a new shape with `axis` removed.
    pub fn remove_dim(&self, axis: usize) -> Shape {
        let mut d = self.dims.clone();
        if axis < d.len() {
            d.remove(axis);
        }
        Shape { dims: d }
    }

    /// Return the standard row-major contiguous strides for this shape.
    pub fn strides(&self) -> Vec<usize> {
        let mut s = vec![1; self.dims.len()];
        for i in (0..self.dims.len().saturating_sub(1)).rev() {
            s[i] = s[i + 1] * self.dims[i + 1];
        }
        s
    }

    /// Reshape to new dimensions with the same total element count.
    pub fn reshape(&self, new_dims: impl Into<Vec<usize>>) -> Result<Shape> {
        let new_shape = Shape::new(new_dims);
        if new_shape.elem_count() != self.elem_count() {
            return Err(Error::Shape(format!(
                "cannot reshape from total elements {} to {}",
                self.elem_count(),
                new_shape.elem_count()
            )));
        }
        Ok(new_shape)
    }

    /// Permute dimensions according to a permutation order.
    pub fn transpose(&self, permutation: &[usize]) -> Result<Shape> {
        if permutation.len() != self.rank() {
            return Err(Error::Shape(format!(
                "transpose permutation length {} does not match shape rank {}",
                permutation.len(),
                self.rank()
            )));
        }
        let mut seen = vec![false; self.rank()];
        let mut new_dims = Vec::with_capacity(self.rank());
        for &axis in permutation {
            if axis >= self.rank() {
                return Err(Error::IndexOutOfBounds(format!(
                    "transpose axis {axis} out of bounds for rank {}",
                    self.rank()
                )));
            }
            if seen[axis] {
                return Err(Error::Shape(format!(
                    "duplicate axis {axis} in transpose permutation"
                )));
            }
            seen[axis] = true;
            new_dims.push(self.dims[axis]);
        }
        Ok(Shape::new(new_dims))
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

    /// Compute the common broadcasted target shape of two shapes, if compatible.
    pub fn broadcast_shape(&self, other: &Shape) -> Result<Shape> {
        let a = self.dims();
        let b = other.dims();
        let n = a.len().max(b.len());
        let mut out_dims = vec![0usize; n];
        for i in 0..n {
            let ad = if i < a.len() { a[a.len() - 1 - i] } else { 1 };
            let bd = if i < b.len() { b[b.len() - 1 - i] } else { 1 };
            if ad == bd {
                out_dims[n - 1 - i] = ad;
            } else if ad == 1 {
                out_dims[n - 1 - i] = bd;
            } else if bd == 1 {
                out_dims[n - 1 - i] = ad;
            } else {
                return Err(Error::Shape(format!(
                    "shapes {:?} and {:?} are not broadcast compatible",
                    self.dims(),
                    other.dims()
                )));
            }
        }
        Ok(Shape::new(out_dims))
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
