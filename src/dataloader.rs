use std::error::Error;
use csv::{ReaderBuilder, Writer};
use ndarray::{Array1, Array2};
use ndarray_csv::Array2Reader;

pub fn load_data(file_path: &str) -> Result<Array2<f32>, Box<dyn Error>> {
    // this function loads data from the csv file
    let reader = ReaderBuilder::new().has_headers(false).from_path(file_path);
    let array_read: Array2<f32> = reader?.deserialize_array2_dynamic()?;
    Ok(array_read)
}


pub fn write_csv(indices: &Array1<usize>, output_path: &str) -> Result<(), Box<dyn Error>>{
    // write the result into a csv file
    let mut writer = Writer::from_path(output_path)?;

    for &value in indices.iter() {
        writer.write_record(&[value.to_string()])?;
    }

    writer.flush()?;
    println!("Results wrote to {output_path}");
    Ok(())
}