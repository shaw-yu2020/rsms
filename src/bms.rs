use super::PotK;
use super::Uij;
use super::abs_xyz;
use super::centralize;
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
        let mol1 = centralize(mol1);
        let mol2 = centralize(mol2);
        Self {
            u12: Uij::new(mol1.clone(), mol2.clone(), potk),
            mol1,
            mol2,
        }
    }
    fn run(&self, t_kelvin: f64, u_min: f64, dev_tol: f64, order_max: usize) -> Vec<f64> {
        let t_recip = t_kelvin.recip();
        let u_plus_max = u_min * -t_recip;
        let mut rng = rand::rng();
        // hard-sphere
        let sigma_hs = 1.0
            + self.mol1.iter().fold(0_f64, |a, &xyz| a.max(abs_xyz(&xyz)))
            + self.mol2.iter().fold(0_f64, |a, &xyz| a.max(abs_xyz(&xyz)));
        let virial_hs = 2.0 / 3.0 * PI * sigma_hs.powi(3);
        let time_ini = Instant::now();
        // ini
        let mut xyz_new = [0.0, 0.0, 0.0];
        let mut zyz_new = [0.0, 0.0, 0.0];
        let mut gamma_new = 0_f64;
        while gamma_new.abs() < 0.1 || gamma_new.abs() > 10.0 {
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
            let u12_plus = self.u12.calc(
                &self.mol1,
                &Euler::from_zyz(&zyz_new).rotate(&self.mol2),
                &xyz_new,
            ) * -t_recip;
            if u12_plus < u_plus_max {
                gamma_new = u12_plus.exp_m1();
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
        let mut xyz_old = xyz_new;
        let mut zyz_old = zyz_new;
        let mut gamma_old = gamma_new;
        let mut scale = 1_f64;
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
            let u12_plus = self.u12.calc(
                &self.mol1,
                &Euler::from_zyz(&zyz_new).rotate(&self.mol2),
                &xyz_new,
            ) * -t_recip;
            gamma_new = if u12_plus < u_plus_max {
                u12_plus.exp_m1()
            } else {
                -1.0
            };
            if rng.random::<f64>() < gamma_new.abs() / gamma_old.abs() {
                xyz_old = xyz_new;
                zyz_old = zyz_new;
                gamma_old = gamma_new;
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
        let mut old_hs = if abs_xyz(&xyz_old) < sigma_hs {
            -1.0
        } else {
            0.0
        } / gamma_old.abs();
        let mut f12_taylor = vec![0.0; order_max + 1];
        f12_taylor[0] = gamma_old.signum();
        let mut u12_plus = self.u12.calc(
            &self.mol1,
            &Euler::from_zyz(&zyz_old).rotate(&self.mol2),
            &xyz_old,
        ) * -t_recip;
        if u12_plus < u_plus_max {
            f12_taylor[1] = u12_plus.exp() / gamma_old.abs() * u12_plus;
            for i in 2..=order_max {
                f12_taylor[i] = f12_taylor[i - 1] * u12_plus;
            }
        }
        let mut sum_hs = 0.0;
        let mut sum_taylor = vec![0_f64; order_max + 1];
        let mut mean_hs = [0.0; 10];
        let mut mean_target = [0.0; 10];
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
                u12_plus = self.u12.calc(
                    &self.mol1,
                    &Euler::from_zyz(&zyz_new).rotate(&self.mol2),
                    &xyz_new,
                ) * -t_recip;
                gamma_new = if u12_plus < u_plus_max {
                    u12_plus.exp_m1()
                } else {
                    -1.0
                };
                if rng.random::<f64>() < gamma_new.abs() / gamma_old.abs() {
                    xyz_old = xyz_new;
                    zyz_old = zyz_new;
                    gamma_old = gamma_new;
                    old_hs = if abs_xyz(&xyz_old) < sigma_hs {
                        -1.0
                    } else {
                        0.0
                    } / gamma_old.abs();
                    if u12_plus < u_plus_max {
                        f12_taylor[1] = u12_plus.exp() / gamma_old.abs() * u12_plus;
                        for i in 2..=order_max {
                            f12_taylor[i] = f12_taylor[i - 1] * u12_plus;
                        }
                    } else {
                        f12_taylor = vec![0.0; order_max + 1];
                    }
                    f12_taylor[0] = gamma_old.signum();
                }
                sum_hs += old_hs;
                sum_taylor
                    .iter_mut()
                    .zip(&f12_taylor)
                    .map(|(sum, old)| *sum += old)
                    .count();
            }
            mean_hs[i % 10] = sum_hs / (i * num_inner) as f64;
            mean_target[i % 10] = sum_taylor[0] / (i * num_inner) as f64;
            let time_now = time_ini.elapsed().as_secs();
            if i > 9 {
                let result: Vec<f64> = mean_target
                    .iter()
                    .zip(mean_hs)
                    .map(|(target, hs)| target / hs)
                    .collect();
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
        sum_taylor
            .iter()
            .enumerate()
            .map(|(i, taylor)| taylor / sum_hs * virial_hs / (1..=i).product::<usize>() as f64)
            .collect()
    }
}
