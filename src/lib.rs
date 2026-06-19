mod pot;
use pot::PotK;

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
    Ok(())
}
