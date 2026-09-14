//! Bounded, allocation-light maximum-consensus search. No device I/O.

fn consensus(xs: &[f64], ys: &[f64], limit: f64) -> Option<Vec<usize>> {
    if xs.len() < 3 { return None; }
    let mut usable = false;
    let mut best = (0usize, f64::INFINITY);
    let mut selected: Vec<usize> = (0..xs.len()).collect();
    let mut indices = Vec::with_capacity(xs.len());
    for i in 0..xs.len() {
        for j in i + 1..xs.len() {
            if ys[i] == ys[j] { continue; }
            usable = true;
            let slope = (xs[j] - xs[i]) / (ys[j] - ys[i]);
            indices.clear();
            let mut sum = 0.0;
            for k in 0..xs.len() {
                let residual = (xs[k] - (xs[i] + slope * (ys[k] - ys[i]))).abs();
                if residual <= limit { indices.push(k); sum += residual; }
            }
            let n = indices.len();
            let mean = sum / n.max(1) as f64;
            if n >= 3 && (n > best.0 || (n == best.0 && mean < best.1)) {
                best = (n, mean);
                selected.clone_from(&indices);
            }
        }
    }
    if usable { Some(selected) } else { None }
}

/// # Safety
/// Input buffers must contain n readable f64 values; output n writable usize
/// values. Buffers must not overlap. Returns zero for no usable hypothesis.
#[no_mangle]
pub unsafe extern "C" fn line_consensus(
    xs: *const f64, ys: *const f64, n: usize, limit: f64, output: *mut usize,
) -> usize {
    if n < 3 || xs.is_null() || ys.is_null() || output.is_null() { return 0; }
    let xs = std::slice::from_raw_parts(xs, n);
    let ys = std::slice::from_raw_parts(ys, n);
    match consensus(xs, ys, limit) {
        Some(indices) => {
            std::ptr::copy_nonoverlapping(indices.as_ptr(), output, indices.len());
            indices.len()
        }
        None => 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rejects_degenerate() {
        assert_eq!(consensus(&[1., 2., 3.], &[0., 0., 0.], 1.), None);
    }
    #[test]
    fn excludes_outlier() {
        assert_eq!(consensus(&[0., 1., 2., 99.], &[0., 1., 2., 3.], 0.1), Some(vec![0, 1, 2]));
    }
}
