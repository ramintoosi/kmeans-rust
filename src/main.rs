use clap::Parser;
use crate::dataloader::*;
use crate::kmeans::KMeans;
use crate::parser::DataParser;

mod kmeans;
mod dataloader;
mod parser;




fn main() {

    // parse and get arguments
    let cli = DataParser::parse();
    let file_path = cli.data_path;

    // load data
    let data = load_data(&file_path).expect("Error reading csv");

    let mut kmeans = KMeans {
        num_cluster: cli.num_cluster,
        kpp: cli.kpp,
        tolerance: cli.tolerance,
        max_iter: cli.max_iter,
        ..Default::default()
    };

    let indices = kmeans.fit(&data);

    println!("Number of Iters: {:?} with max change: {:?}",
             kmeans.get_iter(), kmeans.get_max_change());

    // write to csv file
    let _ = write_csv(&indices, &cli.output_path);

}
