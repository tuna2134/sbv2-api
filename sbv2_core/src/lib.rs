pub mod bert;
pub mod error;
pub mod jtalk;
pub mod model;
pub mod mora;
pub mod nlp;
pub mod norm;
pub mod style;
pub mod tts;
pub mod utils;

pub fn add(left: usize, right: usize) -> usize {
    left + right
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }
}
