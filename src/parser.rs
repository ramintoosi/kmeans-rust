use clap::Parser;

// define arguments using clap
#[derive(Parser)]
#[command(author, version, about, long_about = None)]
pub struct DataParser {
    #[arg(short, long, help = "Path to the csv file")]
    pub data_path: String,

    #[arg(short, long, help = "Number of clusters")]
    pub num_cluster: usize,

    #[arg(short, long, help = "Use Kmeans++ to initialize centers")]
    pub kpp: bool,

    #[arg(short, long, help = "Maximum number of iterations", default_value = "1000")]
    pub max_iter: i32,

    #[arg(short, long, help = "Maximum center change tolerance", default_value = "1e-4")]
    pub tolerance: f32,

    #[arg(short, long, help = "Path to save indices as csv", default_value = "indices.csv")]
    pub output_path: String

}