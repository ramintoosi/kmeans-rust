use std::{
    error::Error,
};
use std::ops::IndexMut;
use csv::{ReaderBuilder, Writer};
use ndarray::{Array1, Array2, ArrayView1, AssignElem};
use ndarray_csv::{Array2Reader};
use rand::seq::index::sample;
use rand::{Rng, thread_rng};
use clap::{Parser};

// define arguments using clap
#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(short, long, help = "Path to the csv file")]
    data_path: String,

    #[arg(short, long, help = "Number of clusters")]
    num_cluster: usize,

    #[arg(short, long, help = "Use Kmeans++ to initialize centers")]
    kpp: bool,

    #[arg(short, long, help = "Maximum number of iterations", default_value = "1000")]
    max_iter: i32,

    #[arg(short, long, help = "Maximum center change tolerance", default_value = "1e-4")]
    tolerance: f32,

    #[arg(short, long, help = "Path to save indices as csv", default_value = "indices.csv")]
    output_path: String

}

fn main() {

    // parse and get arguments
    let cli = Args::parse();

    let kpp = cli.kpp;
    let n_cluster = cli.num_cluster;
    let file_path = cli.data_path;

    // load data
    let data = load_data(&file_path).expect("Error reading csv");

    // initiate centers random or using kmeans++
    let mut centers = if kpp {
        kmeans_pp(&data, n_cluster)
    } else {
        random_centers(&data, n_cluster)
    };

    // main loop, assign indices and update centers
    let mut indices: Array1<usize> = Array1::zeros(data.nrows());
    let mut iter = 0;
    let mut max_change = f32::INFINITY;
    while  (max_change > cli.tolerance) & (iter < cli.max_iter) {
        iter += 1;
        assign_cluster_to_sample(&data, &centers, &mut indices);
        max_change = update_centers(&data, &mut centers, &indices, n_cluster);
    }
    println!("Number of Iters: {iter} with max change: {max_change}");

    // write to csv file
    let _ = write_csv(&indices, &cli.output_path);

}

fn load_data(file_path: &str) -> Result<Array2<f32>, Box<dyn Error>> {
    // this function loads data from the csv file
    let reader = ReaderBuilder::new().has_headers(false).from_path(file_path);
    let array_read: Array2<f32> = reader?.deserialize_array2_dynamic()?;
    Ok(array_read)
}


fn random_centers(data : &Array2<f32>, n_cluster: usize) -> Array2<f32> {
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

fn kmeans_pp(data : &Array2<f32>, n_cluster: usize) -> Array2<f32> {
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

fn assign_cluster_to_sample(data: &Array2<f32>, centers: &Array2<f32>, indices: &mut Array1<usize>) {
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

fn update_centers(data: &Array2<f32>,
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

fn write_csv(indices: &Array1<usize>, output_path: &str) -> Result<(), Box<dyn Error>>{
    // write the result into a csv file
    let mut writer = Writer::from_path(output_path)?;

    for &value in indices.iter() {
        writer.write_record(&[value.to_string()])?;
    }

    writer.flush()?;
    println!("Results wrote to {output_path}");
    Ok(())
}
