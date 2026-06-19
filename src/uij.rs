use super::PotK;
use super::euler::Euler;
use pyo3::{pyclass, pymethods};
use rand::prelude::*;
use std::f64::consts::TAU;
use std::time::Instant;

#[pyclass]
pub struct Uij {
    potk: Vec<Vec<PotK>>,
    moli: Vec<[f64; 3]>,
    molj: Vec<[f64; 3]>,
}

#[pymethods]
impl Uij {
    #[new]
    pub fn new(moli: Vec<[f64; 3]>, molj: Vec<[f64; 3]>, potk: Vec<Vec<PotK>>) -> Self {
        Uij { moli, molj, potk }
    }
    fn uij(&self, r: f64, phi1: f64, theta1: f64, phi2: f64, theta2: f64, psi2: f64) -> f64 {
        let moli = Euler::from_zy(&[phi1 * TAU, theta1 * TAU]).rotate(&self.moli);
        let molj = Euler::from_zyz(&[phi2 * TAU, theta2 * TAU, psi2 * TAU]).rotate(&self.molj);
        self.calc(&moli, &molj, &[0.0, 0.0, r])
    }
    fn u_r(
        &self,
        r: f64,
        dev_tol: f64,
        num_inner: usize,
        num_outer: usize,
        num_print: usize,
    ) -> f64 {
        let mut rng = rand::rng();
        let mut sum_uij = 0.0;
        let mut vec_u_r = [0.0; 10];
        let time_ini = Instant::now();
        for i in 1..=10 {
            for _ in 1..=num_inner {
                let (phi1, theta1, phi2, theta2, psi2) = rng.random::<(f64, f64, f64, f64, f64)>();
                sum_uij += self.uij(r, phi1, theta1, phi2, theta2, psi2);
            }
            vec_u_r[i % 10] = sum_uij / (i * num_inner) as f64;
        }
        let mut result = vec_u_r.iter().sum::<f64>() / 10.0;
        for i in 1..=num_outer {
            for _ in 1..=num_inner {
                let (phi1, theta1, phi2, theta2, psi2) = rng.random::<(f64, f64, f64, f64, f64)>();
                sum_uij += self.uij(r, phi1, theta1, phi2, theta2, psi2);
            }
            vec_u_r[i % 10] = sum_uij / ((i + 10) * num_inner) as f64;
            result = vec_u_r.iter().sum::<f64>() / 10.0;
            let deviation = vec_u_r
                .iter()
                .map(|u_r| (u_r - result).powi(2))
                .sum::<f64>()
                .sqrt();
            if i % num_print == 0 {
                let time_now = time_ini.elapsed().as_secs();
                println!(
                    "[Uij::u_r] i=({}/{})*{}, result={:.4}, deviation={:.4}, {}h{}m{}s",
                    i,
                    num_outer,
                    num_inner,
                    result,
                    deviation,
                    time_now / 3600,
                    time_now % 3600 / 60,
                    time_now % 60
                );
            }
            if deviation < f64::EPSILON * 1E8 || deviation / result.abs() < dev_tol {
                break;
            }
        }
        result
    }
}

impl Uij {
    pub fn calc(&self, moli: &[[f64; 3]], molj: &[[f64; 3]], drij: &[f64; 3]) -> f64 {
        moli.iter()
            .enumerate()
            .map(|(i, xyz1)| {
                molj.iter()
                    .enumerate()
                    .map(|(j, xyz2)| {
                        let rij = ((xyz1[0] - (xyz2[0] + drij[0])).powi(2)
                            + (xyz1[1] - (xyz2[1] + drij[1])).powi(2)
                            + (xyz1[2] - (xyz2[2] + drij[2])).powi(2))
                        .sqrt()
                        .max(1E-30);
                        self.potk[i][j].pot_r(rij)
                    })
                    .sum::<f64>()
            })
            .sum::<f64>()
    }
}
