mod pot;
use pot::PotK;
mod uij;
use uij::Uij;
mod bms;
use bms::Bms;
mod cms;
use cms::Cms;
mod euler;

/// A Python function implemented in Rust.
use pyo3::prelude::*;
#[pyfunction]
fn hello() -> PyResult<String> {
    Ok(String::from("Hello, Rust And Python."))
}

/// A Python module implemented in Rust.
#[pymodule]
#[pyo3(name = "rsms")] // Rename pymodule
fn pylib(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(hello, m)?)?;
    m.add_class::<PotK>()?;
    m.add_class::<Uij>()?;
    m.add_class::<Bms>()?;
    m.add_class::<Cms>()?;
    Ok(())
}

#[inline]
fn abs_xyz(xyz: &[f64; 3]) -> f64 {
    (xyz[0] * xyz[0] + xyz[1] * xyz[1] + xyz[2] * xyz[2]).sqrt()
}

fn centralize(xyz: Vec<[f64; 3]>) -> Vec<[f64; 3]> {
    let dx = xyz.iter().map(|[x, _, _]| x).sum::<f64>() / xyz.len() as f64;
    let dy = xyz.iter().map(|[_, y, _]| y).sum::<f64>() / xyz.len() as f64;
    let dz = xyz.iter().map(|[_, _, z]| z).sum::<f64>() / xyz.len() as f64;
    xyz.iter()
        .map(|[x, y, z]| [x - dx, y - dy, z - dz])
        .collect()
}
