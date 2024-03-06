use std::{
    env,
    fs::File,
    error::Error
};
use csv::ReaderBuilder;
use ndarray::{Array2};
use ndarray_csv::{Array2Reader};

fn main() {
    // TODO: change it according to: https://docs.rs/csv/latest/csv/tutorial/index.html#reading-csv
    let args: Vec<String> = env::args().collect();
    if args.len() != 2 {
        println!("Usage:\n./main [file_path]");
        return
    }

    let file_path: &String = &args[1];
    let data = load_data(&file_path).expect("Error reading csv");
    println!("{:?}", data.shape())

}

fn load_data(file_path: &str) -> Result<Array2<f32>, Box<dyn Error>> {

    let file = File::open(file_path)?;
    let mut reader = ReaderBuilder::new().has_headers(false).from_reader(file);
    let array_read: Array2<f32> = reader.deserialize_array2_dynamic()?;
    Ok(array_read)
}
