use std::ops::IndexMut;
use ndarray::{Array1, Array2, ArrayView1, AssignElem};
use rand::seq::index::sample;
use rand::{Rng,thread_rng};

pub fn random_centers(data : &Array2<f32>, n_cluster: usize) -> Array2<f32> {
    // centers are randomly selected from the current set of samples
    let n_rows = data.nrows();
    let mut rng = thread_rng();
    let mut selected_rows =
        Array2::<f32>::zeros((n_cluster, data.ncols()));
    let indices: Array1<usize> = sample(&mut rng, n_rows, n_cluster).into_iter().collect();
    for (i, &index) in indices.iter().enumerate() {
        selected_rows.row_mut(i).assign(&data.row(index));
    }

    selected_rows
}

pub fn kmeans_pp(data : &Array2<f32>, n_cluster: usize) -> Array2<f32> {
    // centers are selected based on KMeans++ algorithm
    let n_rows = data.nrows();
    let mut chosen_points: Vec<usize> = vec!();
    let mut centers: Array2<f32> = Array2::zeros((n_cluster, data.ncols()));

    // first center
    chosen_points.push(thread_rng().gen_range(0..n_rows));
    centers.row_mut(1).assign(&data.row(chosen_points[0]));

    // other centers
    for i_center in 1..n_cluster {

        let mut max_dist: f32 = -1.0;
        let mut max_index: usize = 0;
        for (i_sample, sample) in data.rows().into_iter().enumerate() {
            if chosen_points.contains(&i_sample) {
                continue
            }
            let mut c_dist: f32 = 0.0;
            for i_prev_centers in 0..i_center {
                c_dist += euclidean_distance(&sample, &centers.row(i_prev_centers));
            }
            if c_dist > max_dist {
                max_dist = c_dist;
                max_index = i_sample
            }
        }

        chosen_points.push(max_index);
        centers.row_mut(i_center).assign(&data.row(max_index))
    }
    centers
}

fn euclidean_distance(a: &ArrayView1<f32>, b: &ArrayView1<f32>) -> f32 {
    // calculate Euclidean distance between two arrays
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum::<f32>().sqrt()
}

pub fn assign_cluster_to_sample(data: &Array2<f32>, centers: &Array2<f32>, indices: &mut Array1<usize>) {
    // finds the closest cluster to each sample using centers
    for (index_sample, row) in data.rows().into_iter().enumerate(){
        let mut min_distance = f32::INFINITY;
        let mut min_index: usize = 0;

        for (index, center_row) in centers.rows().into_iter().enumerate(){
            let distance = euclidean_distance(&row, &center_row);
            if distance < min_distance{
                min_distance = distance;
                min_index = index;
            }
        }
        indices.index_mut(index_sample).assign_elem(min_index)
    }
}

pub fn update_centers(data: &Array2<f32>,
                  centers: &mut Array2<f32>,
                  indices: &Array1<usize>, n_clusters: usize) -> f32 {
    // update centers as the average of the samples within the cluster
    let mut max_change = f32::INFINITY;
    for index in 0..n_clusters {
        let matched_indices: Vec<usize> = indices.iter()
            .enumerate()
            .filter(|&(_, &value)| value == index as usize)
            .map(|(i, _)| i)
            .collect();
        let mut c: Array1<f32> = Array1::zeros(data.ncols());
        for m_index in &matched_indices {
            c += &data.row(*m_index);
        }
        c /= matched_indices.len() as f32;
        let distance = euclidean_distance(&centers.row(index), &c.view());
        centers.row_mut(index as usize).assign(&c);
        if distance < max_change{
            max_change = distance;
        }
    }
    max_change
}