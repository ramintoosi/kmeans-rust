use std::io;
use csv::{ReaderBuilder, Writer};
use ndarray::{Array1, Array2};
use ndarray_csv::Array2Reader;

pub fn load_data(file_path: &str) -> Array2<f32> {
    // this function loads data from the csv file
    let mut reader = ReaderBuilder::new()
        .has_headers(false)
        .from_path(file_path)
        .expect("CSV file read error");
    let array_read: Array2<f32> = match reader.deserialize_array2_dynamic() {
        Ok(array) => array,
        Err(e) => panic!("CSV file content is corrupted: {:?}", e)   
    };
    array_read
}


pub fn write_csv(indices: &Array1<usize>, output_path: &str) -> Result<(), io::Error>{
    // write the result into a csv file
    let mut writer = Writer::from_path(output_path)?;

    for &value in indices.iter() {
        writer.write_record(&[value.to_string()])?;
    }

    writer.flush()?;
    Ok(())
}