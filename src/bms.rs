use super::PotK;
use super::Uij;
use super::euler::Euler;
use pyo3::{pyclass, pymethods};
use rand::prelude::*;
use rand_distr::StandardNormal;
use std::f64::consts::PI;
use std::time::Instant;

#[pyclass]
pub struct Bms {
    mol1: Vec<[f64; 3]>,
    mol2: Vec<[f64; 3]>,
    u12: Uij,
}

#[pymethods]
impl Bms {
    #[new]
    fn new(mol1: Vec<[f64; 3]>, mol2: Vec<[f64; 3]>, potk: Vec<Vec<PotK>>) -> Self {
        Self {
            mol1: mol1.clone(),
            mol2: mol2.clone(),
            u12: Uij::new(mol1, mol2, potk),
        }
    }
    fn run(&self, t_kelvin: f64, u_min: f64, dev_tol: f64, order_max: usize) -> Vec<f64> {
        let t_recip = t_kelvin.recip();
        let mut rng = rand::rng();
        // hard-sphere
        let sigma_hs =
            1.0 + self.mol1.iter().fold(0_f64, |a, &xyz| {
                a.max((xyz[0] * xyz[0] + xyz[1] * xyz[1] + xyz[2] * xyz[2]).sqrt())
            }) + self.mol2.iter().fold(0_f64, |a, &xyz| {
                a.max((xyz[0] * xyz[0] + xyz[1] * xyz[1] + xyz[2] * xyz[2]).sqrt())
            });
        let virial_hs = 2.0 / 3.0 * PI * sigma_hs.powi(3);
        let time_ini = Instant::now();
        // ini
        let mut xyz_new = [0.0, 0.0, 0.0];
        let mut zyz_new = [0.0, 0.0, 0.0];
        let mut u12_new = 0_f64;
        let mut f12_new = 0_f64;
        while f12_new.abs() < 0.1 || f12_new.abs() > 10.0 {
            xyz_new = [
                rng.sample::<f64, StandardNormal>(StandardNormal) * sigma_hs,
                rng.sample::<f64, StandardNormal>(StandardNormal) * sigma_hs,
                rng.sample::<f64, StandardNormal>(StandardNormal) * sigma_hs,
            ];
            zyz_new = [
                rng.sample::<f64, StandardNormal>(StandardNormal) * PI,
                rng.sample::<f64, StandardNormal>(StandardNormal) * PI,
                rng.sample::<f64, StandardNormal>(StandardNormal) * PI,
            ];
            u12_new = self.u12.calc(
                &self.mol1,
                &Euler::from_zyz(&zyz_new).rotate(&self.mol2),
                &xyz_new,
            );
            if u12_new > u_min {
                f12_new = (-u12_new * t_recip).exp_m1();
            }
        }
        let time_now = time_ini.elapsed().as_secs();
        println!(
            "[Bms::run::ini] sigma_hs={:.6},virial_hs={:.6},{}h{}m{}s",
            sigma_hs,
            virial_hs,
            time_now / 3600,
            time_now % 3600 / 60,
            time_now % 60
        );
        // opt
        let mut scale = 1_f64;
        let mut xyz_old = xyz_new;
        let mut zyz_old = zyz_new;
        let mut f12_old = f12_new;
        let mut counter = 0_i32;
        for i in 1..=1_000_000 {
            xyz_new = [
                xyz_old[0] + rng.sample::<f64, StandardNormal>(StandardNormal) * sigma_hs * scale,
                xyz_old[1] + rng.sample::<f64, StandardNormal>(StandardNormal) * sigma_hs * scale,
                xyz_old[2] + rng.sample::<f64, StandardNormal>(StandardNormal) * sigma_hs * scale,
            ];
            zyz_new = [
                zyz_old[0] + rng.sample::<f64, StandardNormal>(StandardNormal) * PI,
                zyz_old[1] + rng.sample::<f64, StandardNormal>(StandardNormal) * PI,
                zyz_old[2] + rng.sample::<f64, StandardNormal>(StandardNormal) * PI,
            ];
            u12_new = self.u12.calc(
                &self.mol1,
                &Euler::from_zyz(&zyz_new).rotate(&self.mol2),
                &xyz_new,
            );
            if u12_new > u_min {
                f12_new = (-u12_new * t_recip).exp_m1();
            } else {
                u12_new = f64::INFINITY;
                f12_new = -1.0;
            }
            if rng.random::<f64>() < f12_new.abs() / f12_old.abs() {
                xyz_old = xyz_new;
                zyz_old = zyz_new;
                f12_old = f12_new;
                counter += 1;
            }
            if i % 100_000 == 0 {
                let time_now = time_ini.elapsed().as_secs();
                println!(
                    "[Bms::run::opt] scale={:.6},acceptance=1E-5*{},{}h{}m{}s",
                    scale,
                    counter,
                    time_now / 3600,
                    time_now % 3600 / 60,
                    time_now % 60
                );
                scale *= 1.0
                    + ((counter - 50_000).signum() * (counter - 50_000).abs().min(20_000)) as f64
                        / 100_000.0;
                counter = 0;
            }
        }
        // run
        let mut u12_old = u12_new;
        let mut sum_hs = 0.0;
        let mut sum_taylor = vec![0_f64; order_max + 1];
        let mut mean_hs = [0.0; 10];
        let mut mean_taylor = [const { Vec::<f64>::new() }; 10];
        let mut flag = 0_usize;
        let num_inner = 1_000_000_usize;
        for i in 1..=1000 {
            for _ in 1..=num_inner {
                xyz_new = [
                    xyz_old[0]
                        + rng.sample::<f64, StandardNormal>(StandardNormal) * sigma_hs * scale,
                    xyz_old[1]
                        + rng.sample::<f64, StandardNormal>(StandardNormal) * sigma_hs * scale,
                    xyz_old[2]
                        + rng.sample::<f64, StandardNormal>(StandardNormal) * sigma_hs * scale,
                ];
                zyz_new = [
                    zyz_old[0] + rng.sample::<f64, StandardNormal>(StandardNormal) * PI,
                    zyz_old[1] + rng.sample::<f64, StandardNormal>(StandardNormal) * PI,
                    zyz_old[2] + rng.sample::<f64, StandardNormal>(StandardNormal) * PI,
                ];
                u12_new = self.u12.calc(
                    &self.mol1,
                    &Euler::from_zyz(&zyz_new).rotate(&self.mol2),
                    &xyz_new,
                );
                if u12_new > u_min {
                    f12_new = (-u12_new * t_recip).exp_m1();
                } else {
                    u12_new = f64::INFINITY;
                    f12_new = -1.0;
                }
                if rng.random::<f64>() < f12_new.abs() / f12_old.abs() {
                    xyz_old = xyz_new;
                    zyz_old = zyz_new;
                    u12_old = u12_new;
                    f12_old = f12_new;
                }
                sum_hs += if (xyz_old[0] * xyz_old[0]
                    + xyz_old[1] * xyz_old[1]
                    + xyz_old[2] * xyz_old[2])
                    .sqrt()
                    < sigma_hs
                {
                    -1.0
                } else {
                    0.0
                } / f12_old.abs();
                sum_taylor[0] += f12_old.signum();
                let mut taylor = (-u12_old * t_recip).exp() / f12_old.abs();
                for sum_taylor_order in sum_taylor.iter_mut().skip(1) {
                    taylor *= if u12_old.is_infinite() {
                        0.0
                    } else {
                        -u12_old * t_recip
                    };
                    *sum_taylor_order += taylor;
                }
            }
            mean_hs[i % 10] = sum_hs / (i * num_inner) as f64;
            mean_taylor[i % 10] = (0..=order_max)
                .map(|order| sum_taylor[order] / (i * num_inner) as f64)
                .collect();
            let time_now = time_ini.elapsed().as_secs();
            if i > 10 {
                let result = (0..10)
                    .map(|ii| mean_taylor[ii][0] / mean_hs[ii])
                    .collect::<Vec<f64>>();
                let mean_result = result.iter().sum::<f64>() / 10.0;
                let deviation = result
                    .iter()
                    .map(|value| (value - mean_result).powi(2))
                    .sum::<f64>()
                    .sqrt();
                println!(
                    "[Bms::run::run] i={:0>4}*1E6, fraction={:.4}, deviation={:.4}, {}h{}m{}s",
                    i,
                    mean_result,
                    deviation,
                    time_now / 3600,
                    time_now % 3600 / 60,
                    time_now % 60
                );
                if deviation < f64::EPSILON * 1E4 || deviation / mean_result.abs() < dev_tol {
                    flag = i;
                    break;
                }
            } else {
                println!(
                    "[Bms::run::run] i={:0>4}*1E6, {}h{}m{}s",
                    i,
                    time_now / 3600,
                    time_now % 3600 / 60,
                    time_now % 60
                );
            }
        }
        (0..=order_max)
            .map(|order| {
                mean_taylor[flag % 10][order] / mean_hs[flag % 10] * virial_hs
                    / (1..=order).product::<usize>() as f64
            })
            .collect()
    }
}
