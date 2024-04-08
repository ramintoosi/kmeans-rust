mod utils;

use ndarray::{Array1, Array2};
use crate::kmeans::utils::{
    kmeans_pp,
    random_centers,
    assign_cluster_to_sample,
    update_centers
};



#[derive(Default)]
pub struct KMeans {
    pub num_cluster: usize,
    pub kpp: bool,
    pub tolerance: f32,
    pub max_iter: i32,
    pub iter: Option<i32>,
    pub max_change: Option<f32>

}

impl KMeans {
    pub fn fit(&mut self, data: &Array2<f32>) -> Array1<usize> {
        // initiate centers random or using kmeans++
        let mut centers = if self.kpp {
            kmeans_pp(&data, self.num_cluster)
        } else {
            random_centers(&data, self.num_cluster)
        };

        // main loop, assign indices and update centers
        let mut indices: Array1<usize> = Array1::zeros(data.nrows());
        let mut iter = 0;
        let mut max_change = f32::INFINITY;
        while  (max_change > self.tolerance) & (iter < self.max_iter) {
            iter += 1;
            assign_cluster_to_sample(&data, &centers, &mut indices);
            max_change = update_centers(&data, &mut centers, &indices, self.num_cluster);
        }
        let _ = self.iter.insert(iter);
        let _ = self.max_change.insert(max_change);
        indices
    }

    pub fn get_iter(&self) -> i32 {
        if self.iter.is_some() {
            self.iter.unwrap()
        }
        else {
            0
        }
    }

    pub fn get_max_change(&self) -> f32 {
        if self.max_change.is_some() {
            self.max_change.unwrap()
        } else {
            f32::INFINITY
        }
    }

}