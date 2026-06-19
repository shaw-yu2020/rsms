mod pot;
use pot::PotK;
mod uij;
use uij::Uij;
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
    Ok(())
}
