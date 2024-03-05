use std::env;
fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 2 {
        println!("Usage:\n./main [file_path]");
        return
    }
    let file_path: &String = &args[1];
    println!("{file_path}")
}
