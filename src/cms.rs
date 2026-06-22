use super::PotK;
use super::Uij;
use super::abs_xyz;
use super::euler::Euler;
use pyo3::{pyclass, pymethods};
use rand::prelude::*;
use rand_distr::StandardNormal;
use std::f64::consts::PI;
use std::time::Instant;

#[pyclass]
pub struct Cms {
    mol1: Vec<[f64; 3]>,
    mol2: Vec<[f64; 3]>,
    mol3: Vec<[f64; 3]>,
    u12: Uij,
    u13: Uij,
    u23: Uij,
}

#[pymethods]
impl Cms {
    #[new]
    fn new(
        mol1: Vec<[f64; 3]>,
        mol2: Vec<[f64; 3]>,
        mol3: Vec<[f64; 3]>,
        potk12: Vec<Vec<PotK>>,
        potk13: Vec<Vec<PotK>>,
        potk23: Vec<Vec<PotK>>,
    ) -> Self {
        Self {
            u12: Uij::new(mol1.clone(), mol2.clone(), potk12),
            u13: Uij::new(mol1.clone(), mol3.clone(), potk13),
            u23: Uij::new(mol2.clone(), mol3.clone(), potk23),
            mol1,
            mol2,
            mol3,
        }
    }
    fn run(&self, t_kelvin: f64, dev_tol: f64, order_max: usize) -> Vec<f64> {
        let t_recip = t_kelvin.recip();
        let mut rng = rand::rng();
        let factorials: Vec<usize> = (0..=order_max)
            .map(|order| (1..=order).product::<usize>())
            .collect();
        // hard-sphere
        let sigma_hs = 1.0
            + self.mol1.iter().fold(0_f64, |a, &xyz| a.max(abs_xyz(&xyz))) * 2.0 / 3.0
            + self.mol2.iter().fold(0_f64, |a, &xyz| a.max(abs_xyz(&xyz))) * 2.0 / 3.0
            + self.mol3.iter().fold(0_f64, |a, &xyz| a.max(abs_xyz(&xyz))) * 2.0 / 3.0;
        let virial_hs = 5.0 / 18.0 * PI.powi(2) * sigma_hs.powi(6);
        let time_ini = Instant::now();
        // ini
        let (mut d12_new, mut zyz2_new) = ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]);
        let (mut d13_new, mut zyz3_new) = ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]);
        let (mut d23_new, mut gamma_new) = ([0.0, 0.0, 0.0], 0_f64);
        while gamma_new.abs() < 0.1 || gamma_new.abs() > 10.0 {
            d12_new = [
                rng.sample::<f64, StandardNormal>(StandardNormal) * sigma_hs,
                rng.sample::<f64, StandardNormal>(StandardNormal) * sigma_hs,
                rng.sample::<f64, StandardNormal>(StandardNormal) * sigma_hs,
            ];
            zyz2_new = [
                rng.sample::<f64, StandardNormal>(StandardNormal) * PI,
                rng.sample::<f64, StandardNormal>(StandardNormal) * PI,
                rng.sample::<f64, StandardNormal>(StandardNormal) * PI,
            ];
            d13_new = [
                rng.sample::<f64, StandardNormal>(StandardNormal) * sigma_hs,
                rng.sample::<f64, StandardNormal>(StandardNormal) * sigma_hs,
                rng.sample::<f64, StandardNormal>(StandardNormal) * sigma_hs,
            ];
            zyz3_new = [
                rng.sample::<f64, StandardNormal>(StandardNormal) * PI,
                rng.sample::<f64, StandardNormal>(StandardNormal) * PI,
                rng.sample::<f64, StandardNormal>(StandardNormal) * PI,
            ];
            d23_new = [
                d13_new[0] - d12_new[0],
                d13_new[1] - d12_new[1],
                d13_new[2] - d12_new[2],
            ];
            let mol2_new = Euler::from_zyz(&zyz2_new).rotate(&self.mol2);
            let mol3_new = Euler::from_zyz(&zyz3_new).rotate(&self.mol3);
            let u12_plus = self.u12.calc(&self.mol1, &mol2_new, &d12_new) * -t_recip;
            let u13_plus = self.u13.calc(&self.mol1, &mol3_new, &d13_new) * -t_recip;
            let u23_plus = self.u23.calc(&mol2_new, &mol3_new, &d23_new) * -t_recip;
            gamma_new = u12_plus.exp_m1() * u13_plus.exp_m1() * u23_plus.exp_m1();
        }
        let time_now = time_ini.elapsed().as_secs();
        println!(
            "[Cms::run::ini] sigma_hs={:.6},virial_hs={:.6},{}h{}m{}s",
            sigma_hs,
            virial_hs,
            time_now / 3600,
            time_now % 3600 / 60,
            time_now % 60
        );
        // opt
        let (mut d12_old, mut zyz2_old) = (d12_new, zyz2_new);
        let (mut d13_old, mut zyz3_old) = (d13_new, zyz3_new);
        let (mut d23_old, mut gamma_old) = (d23_new, gamma_new);
        let mut scale = 1_f64;
        let mut counter = 0_i32;
        for i in 1..=1_000_000 {
            if rng.random::<bool>() {
                d12_new = [
                    d12_old[0]
                        + rng.sample::<f64, StandardNormal>(StandardNormal) * sigma_hs * scale,
                    d12_old[1]
                        + rng.sample::<f64, StandardNormal>(StandardNormal) * sigma_hs * scale,
                    d12_old[2]
                        + rng.sample::<f64, StandardNormal>(StandardNormal) * sigma_hs * scale,
                ];
                zyz2_new = [
                    zyz2_old[0] + rng.sample::<f64, StandardNormal>(StandardNormal) * PI,
                    zyz2_old[1] + rng.sample::<f64, StandardNormal>(StandardNormal) * PI,
                    zyz2_old[2] + rng.sample::<f64, StandardNormal>(StandardNormal) * PI,
                ];
            } else {
                d13_new = [
                    d13_old[0]
                        + rng.sample::<f64, StandardNormal>(StandardNormal) * sigma_hs * scale,
                    d13_old[1]
                        + rng.sample::<f64, StandardNormal>(StandardNormal) * sigma_hs * scale,
                    d13_old[2]
                        + rng.sample::<f64, StandardNormal>(StandardNormal) * sigma_hs * scale,
                ];
                zyz3_new = [
                    zyz3_old[0] + rng.sample::<f64, StandardNormal>(StandardNormal) * PI,
                    zyz3_old[1] + rng.sample::<f64, StandardNormal>(StandardNormal) * PI,
                    zyz3_old[2] + rng.sample::<f64, StandardNormal>(StandardNormal) * PI,
                ];
            }
            d23_new = [
                d13_new[0] - d12_new[0],
                d13_new[1] - d12_new[1],
                d13_new[2] - d12_new[2],
            ];
            let mol2_new = Euler::from_zyz(&zyz2_new).rotate(&self.mol2);
            let mol3_new = Euler::from_zyz(&zyz3_new).rotate(&self.mol3);
            let u12_plus = self.u12.calc(&self.mol1, &mol2_new, &d12_new) * -t_recip;
            let u13_plus = self.u13.calc(&self.mol1, &mol3_new, &d13_new) * -t_recip;
            let u23_plus = self.u23.calc(&mol2_new, &mol3_new, &d23_new) * -t_recip;
            gamma_new = u12_plus.exp_m1() * u13_plus.exp_m1() * u23_plus.exp_m1();
            if rng.random::<f64>() < gamma_new.abs() / gamma_old.abs() {
                (d12_old, zyz2_old) = (d12_new, zyz2_new);
                (d13_old, zyz3_old) = (d13_new, zyz3_new);
                (d23_old, gamma_old) = (d23_new, gamma_new);
                counter += 1;
            }
            if i % 100_000 == 0 {
                let time_now = time_ini.elapsed().as_secs();
                println!(
                    "[Cms::run::opt] scale={:.6},acceptance=1E-5*{},{}h{}m{}s",
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
        let mol2_plus = Euler::from_zyz(&zyz2_old).rotate(&self.mol2);
        let mol3_plus = Euler::from_zyz(&zyz3_old).rotate(&self.mol3);
        let u12_plus = self.u12.calc(&self.mol1, &mol2_plus, &d12_old) * -t_recip;
        let u13_plus = self.u13.calc(&self.mol1, &mol3_plus, &d13_old) * -t_recip;
        let u23_plus = self.u23.calc(&mol2_plus, &mol3_plus, &d23_old) * -t_recip;
        let mut f12_taylor = vec![0.0; order_max + 1];
        let mut f13_taylor = vec![0.0; order_max + 1];
        let mut f23_taylor = vec![0.0; order_max + 1];
        f12_taylor[0] = u12_plus.exp_m1();
        f13_taylor[0] = u13_plus.exp_m1();
        f23_taylor[0] = u23_plus.exp_m1();
        f12_taylor[1] = u12_plus.exp() * u12_plus;
        f13_taylor[1] = u13_plus.exp() * u13_plus;
        f23_taylor[1] = u23_plus.exp() * u23_plus;
        for i in 2..=order_max {
            f12_taylor[i] = f12_taylor[i - 1] * u12_plus;
            f13_taylor[i] = f13_taylor[i - 1] * u13_plus;
            f23_taylor[i] = f23_taylor[i - 1] * u23_plus;
        }
        let mut old_taylor = vec![0_f64; order_max + 1];
        for i12 in 0..=order_max {
            for i13 in 0..=order_max - i12 {
                for i23 in 0..=order_max - i12 - i13 {
                    old_taylor[i12 + i13 + i23] += (factorials[i12 + i13 + i23]
                        / factorials[i12]
                        / factorials[i13]
                        / factorials[i23])
                        as f64
                        * f12_taylor[i12]
                        * f13_taylor[i13]
                        * f23_taylor[i23];
                }
            }
        }
        let mut sum_hs = 0.0;
        let mut sum_taylor = vec![0.0; order_max + 1];
        let mut mean_hs = [0.0; 10];
        let mut mean_target = [0.0; 10];
        let num_inner = 1_000_000_usize;
        for i in 1..=1000 {
            for _ in 1..num_inner {
                if rng.random::<bool>() {
                    d12_new = [
                        d12_old[0]
                            + rng.sample::<f64, StandardNormal>(StandardNormal) * sigma_hs * scale,
                        d12_old[1]
                            + rng.sample::<f64, StandardNormal>(StandardNormal) * sigma_hs * scale,
                        d12_old[2]
                            + rng.sample::<f64, StandardNormal>(StandardNormal) * sigma_hs * scale,
                    ];
                    zyz2_new = [
                        zyz2_old[0] + rng.sample::<f64, StandardNormal>(StandardNormal) * PI,
                        zyz2_old[1] + rng.sample::<f64, StandardNormal>(StandardNormal) * PI,
                        zyz2_old[2] + rng.sample::<f64, StandardNormal>(StandardNormal) * PI,
                    ];
                } else {
                    d13_new = [
                        d13_old[0]
                            + rng.sample::<f64, StandardNormal>(StandardNormal) * sigma_hs * scale,
                        d13_old[1]
                            + rng.sample::<f64, StandardNormal>(StandardNormal) * sigma_hs * scale,
                        d13_old[2]
                            + rng.sample::<f64, StandardNormal>(StandardNormal) * sigma_hs * scale,
                    ];
                    zyz3_new = [
                        zyz3_old[0] + rng.sample::<f64, StandardNormal>(StandardNormal) * PI,
                        zyz3_old[1] + rng.sample::<f64, StandardNormal>(StandardNormal) * PI,
                        zyz3_old[2] + rng.sample::<f64, StandardNormal>(StandardNormal) * PI,
                    ];
                }
                d23_new = [
                    d13_new[0] - d12_new[0],
                    d13_new[1] - d12_new[1],
                    d13_new[2] - d12_new[2],
                ];
                let mol2_new = Euler::from_zyz(&zyz2_new).rotate(&self.mol2);
                let mol3_new = Euler::from_zyz(&zyz3_new).rotate(&self.mol3);
                let u12_plus = self.u12.calc(&self.mol1, &mol2_new, &d12_new) * -t_recip;
                let u13_plus = self.u13.calc(&self.mol1, &mol3_new, &d13_new) * -t_recip;
                let u23_plus = self.u23.calc(&mol2_new, &mol3_new, &d23_new) * -t_recip;
                gamma_new = u12_plus.exp_m1() * u13_plus.exp_m1() * u23_plus.exp_m1();
                if rng.random::<f64>() < gamma_new.abs() / gamma_old.abs() {
                    (d12_old, zyz2_old) = (d12_new, zyz2_new);
                    (d13_old, zyz3_old) = (d13_new, zyz3_new);
                    (d23_old, gamma_old) = (d23_new, gamma_new);
                    f12_taylor[0] = u12_plus.exp_m1();
                    f13_taylor[0] = u13_plus.exp_m1();
                    f23_taylor[0] = u23_plus.exp_m1();
                    f12_taylor[1] = u12_plus.exp() * u12_plus;
                    f13_taylor[1] = u13_plus.exp() * u13_plus;
                    f23_taylor[1] = u23_plus.exp() * u23_plus;
                    for i in 2..=order_max {
                        f12_taylor[i] = f12_taylor[i - 1] * u12_plus;
                        f13_taylor[i] = f13_taylor[i - 1] * u13_plus;
                        f23_taylor[i] = f23_taylor[i - 1] * u23_plus;
                    }
                    old_taylor = vec![0_f64; order_max + 1];
                    for i12 in 0..=order_max {
                        for i13 in 0..=order_max - i12 {
                            for i23 in 0..=order_max - i12 - i13 {
                                old_taylor[i12 + i13 + i23] += (factorials[i12 + i13 + i23]
                                    / factorials[i12]
                                    / factorials[i13]
                                    / factorials[i23])
                                    as f64
                                    * f12_taylor[i12]
                                    * f13_taylor[i13]
                                    * f23_taylor[i23];
                            }
                        }
                    }
                }
                sum_hs += if abs_xyz(&d12_old) < sigma_hs
                    && abs_xyz(&d13_old) < sigma_hs
                    && abs_xyz(&d23_old) < sigma_hs
                {
                    -1.0
                } else {
                    0.0
                } / gamma_old.abs();
                sum_taylor[0] += gamma_old.signum();
                sum_taylor
                    .iter_mut()
                    .zip(&old_taylor)
                    .skip(1)
                    .map(|(sum, old)| *sum += old / gamma_old.abs())
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
                    "[Cms::run::run] i={:0>4}*1E6, fraction={:.4}, deviation={:.4}, {}h{}m{}s",
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
                    "[Cms::run::run] i={:0>4}*1E6, {}h{}m{}s",
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
            .map(|(i, taylor)| taylor / sum_hs * virial_hs / factorials[i] as f64)
            .collect()
    }
}
