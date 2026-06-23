use rayon::prelude::*;

pub struct Euler {
    x: [f64; 3],
    y: [f64; 3],
    z: [f64; 3],
}
impl Euler {
    #[allow(dead_code)]
    fn from_x(x: f64) -> Self {
        let sin_x = x.sin();
        let cos_x = x.cos();
        Euler {
            x: [1.0, 0.0, 0.0],
            y: [0.0, cos_x, -sin_x],
            z: [0.0, sin_x, cos_x],
        }
    }
    #[allow(dead_code)]
    fn from_y(y: f64) -> Self {
        let sin_y = y.sin();
        let cos_y = y.cos();
        Euler {
            x: [cos_y, 0.0, sin_y],
            y: [0.0, 1.0, 0.0],
            z: [-sin_y, 0.0, cos_y],
        }
    }
    #[allow(dead_code)]
    fn from_z(z: f64) -> Self {
        let sin_z = z.sin();
        let cos_z = z.cos();
        Euler {
            x: [cos_z, -sin_z, 0.0],
            y: [sin_z, cos_z, 0.0],
            z: [0.0, 0.0, 1.0],
        }
    }
    pub fn from_zy([z, y]: &[f64; 2]) -> Self {
        let sin_z = z.sin();
        let cos_z = z.cos();
        let sin_y = y.sin();
        let cos_y = y.cos();
        Euler {
            x: [cos_z * cos_y, -sin_z * cos_y, sin_y],
            y: [sin_z, cos_z, 0.0],
            z: [-cos_z * sin_y, sin_z * sin_y, cos_y],
        }
    }
    pub fn from_zyz([z, y, x]: &[f64; 3]) -> Self {
        let sin_z = z.sin();
        let cos_z = z.cos();
        let sin_y = y.sin();
        let cos_y = y.cos();
        let sin_x = x.sin();
        let cos_x = x.cos();
        Euler {
            x: [
                -sin_x * sin_z + cos_x * cos_y * cos_z,
                -sin_x * cos_z - cos_x * cos_y * sin_z,
                cos_x * sin_y,
            ],
            y: [
                cos_x * sin_z + sin_x * cos_y * cos_z,
                cos_x * cos_z - sin_x * cos_y * sin_z,
                sin_x * sin_y,
            ],
            z: [-sin_y * cos_z, sin_y * sin_z, cos_y],
        }
    }
    pub fn rotate(&self, mol: &[[f64; 3]]) -> Vec<[f64; 3]> {
        mol.par_iter()
            .map(|xyz| {
                [
                    self.x[0] * xyz[0] + self.x[1] * xyz[1] + self.x[2] * xyz[2],
                    self.y[0] * xyz[0] + self.y[1] * xyz[1] + self.y[2] * xyz[2],
                    self.z[0] * xyz[0] + self.z[1] * xyz[1] + self.z[2] * xyz[2],
                ]
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;
    #[test]
    fn test_euler() {
        // test euler::from_x
        let from_x = Euler::from_x(PI / 6.0).rotate(&vec![[0.0, 1.0, 0.0]]);
        assert_eq!(from_x[0], [0.0, (PI / 6.0).cos(), (PI / 6.0).sin()]);
        // test euler::from_y
        let from_y = Euler::from_y(PI / 6.0).rotate(&vec![[0.0, 0.0, 1.0]]);
        assert_eq!(from_y[0], [(PI / 6.0).sin(), 0.0, (PI / 6.0).cos()]);
        // test euler::from_z
        let from_z = Euler::from_z(PI / 6.0).rotate(&vec![[1.0, 0.0, 0.0]]);
        assert_eq!(from_z[0], [(PI / 6.0).cos(), (PI / 6.0).sin(), 0.0]);
        // test euler::from_zy
        let from_zy = Euler::from_zy(&[PI / 6.0, PI / 4.0]).rotate(&vec![[1.0, 2.0, 3.0]]);
        assert_eq!(2.026586, (from_zy[0][0] * 1e6).round() / 1e6);
        assert_eq!(2.232051, (from_zy[0][1] * 1e6).round() / 1e6);
        assert_eq!(2.216055, (from_zy[0][2] * 1e6).round() / 1e6);
        // test euler::from_zyz
        let from_zyz =
            Euler::from_zyz(&[PI / 6.0, PI / 4.0, PI / 6.0 * 11.0]).rotate(&vec![[1.0, 2.0, 3.0]]);
        assert_eq!(2.871100, (from_zyz[0][0] * 1e6).round() / 1e6);
        assert_eq!(0.919720, (from_zyz[0][1] * 1e6).round() / 1e6);
        assert_eq!(2.216055, (from_zyz[0][2] * 1e6).round() / 1e6);
    }
}
