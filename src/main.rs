use std::{
    env,
    error::Error,
};
use std::ops::IndexMut;
use csv::ReaderBuilder;
use ndarray::{Array1, Array2, ArrayView1, AssignElem};
use ndarray_csv::{Array2Reader};
use rand::seq::index::sample;
use rand::thread_rng;

fn main() {
    // TODO: change it according to: https://docs.rs/csv/latest/csv/tutorial/index.html#reading-csv
    let args: Vec<String> = env::args().collect();
    if args.len() != 2 {
        println!("Usage:\n./main [file_path]");
        return
    }
    let n_cluster = 4;
    let file_path: &String = &args[1];
    let data = load_data(&file_path).expect("Error reading csv");
    // println!("shape of data: {:?}", data.shape());
    let mut centers = random_centroids(&data, n_cluster);
    // println!("shape of centers {:?}", centers.shape());
    let mut indices: Array1<usize> = Array1::zeros(data.nrows());
    for _ in 0..100{
        assign_cluster_to_sample(&data, &centers, &mut indices);
        update_centers(&data, &mut centers, &indices, n_cluster);
    }
    println!("Centers:\n{:?}",centers)


}

fn load_data(file_path: &str) -> Result<Array2<f32>, Box<dyn Error>> {

    let reader = ReaderBuilder::new().has_headers(false).from_path(file_path);
    let array_read: Array2<f32> = reader?.deserialize_array2_dynamic()?;
    Ok(array_read)
}


fn random_centroids(data : &Array2<f32>, n_cluster: usize) -> Array2<f32> {
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

fn euclidean_distance(a: ArrayView1<f32>, b: ArrayView1<f32>) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum::<f32>().sqrt()
}

fn assign_cluster_to_sample(data: &Array2<f32>, centers: &Array2<f32>, indices: &mut Array1<usize>) {
    for (index_sample, row) in data.rows().into_iter().enumerate(){
        let mut min_distance = f32::INFINITY;
        let mut min_index: usize = 0;

        for (index, center_row) in centers.rows().into_iter().enumerate(){
            let distance = euclidean_distance(row, center_row);
            if distance < min_distance{
                min_distance = distance;
                min_index = index;
            }
        }
        indices.index_mut(index_sample).assign_elem(min_index)
    }
}

fn update_centers(data: &Array2<f32>, centers: &mut Array2<f32>, indices: &Array1<usize>, n_clusters: usize) {
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
        centers.row_mut(index as usize).assign(&c);
    }
}