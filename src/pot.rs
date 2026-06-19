use pyo3::{pyclass, pymethods};

#[pyclass]
pub enum PotK {
    R0(),
    R1(f64),
    R6R12(f64, f64),
    R1R6R12(f64, f64, f64),
}
impl PotK {
    pub fn calc(&self, r: f64) -> f64 {
        match self {
            PotK::R0() => 0.0,
            PotK::R1(r1) => r1 / r,
            PotK::R6R12(r6, r12) => r6 * r.powi(-6) + r12 * r.powi(-12),
            PotK::R1R6R12(r1, r6, r12) => r1 / r + r6 * r.powi(-6) + r12 * r.powi(-12),
        }
    }
}
#[pymethods]
impl PotK {
    #[new]
    fn from_py(flag: String, r: Vec<f64>) -> Self {
        if flag.eq("R1") {
            if r.len() == 1 {
                PotK::R1(r[0])
            } else {
                println!("Err<PotK::R1({r:?})> => PotK::R0");
                PotK::R0()
            }
        } else if flag.eq("R6R12") {
            if r.len() == 2 {
                PotK::R6R12(r[0], r[1])
            } else {
                println!("Err<PotK::R6R12({r:?})> => PotK::R0");
                PotK::R0()
            }
        } else if flag.eq("R1R6R12") {
            if r.len() == 3 {
                PotK::R1R6R12(r[0], r[1], r[2])
            } else {
                println!("Err<PotK::R1R6R12({r:?})> => PotK::R0");
                PotK::R0()
            }
        } else {
            println!("Err<PotK::R?({r:?})> => PotK::R0");
            PotK::R0()
        }
    }
}
