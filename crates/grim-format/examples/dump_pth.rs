use grim_format::torch::PthProvider;
use grim_tensor::provider::TensorProvider;
fn main() {
    let p = PthProvider::load_from_file(std::env::args().nth(1).unwrap()).unwrap();
    let mut names = p.tensor_names();
    names.sort();
    for n in &names {
        let m = p.meta(n).unwrap();
        println!("{n} {:?}", m.shape);
    }
}
