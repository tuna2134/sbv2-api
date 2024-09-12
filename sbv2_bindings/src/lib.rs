use pyo3::prelude::*;
mod sbv2;
pub mod style;

/// sbv2 bindings module
#[pymodule]
fn sbv2_bindings(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<sbv2::TTSModel>()?;
    m.add_class::<style::StyleVector>()?;
    Ok(())
}
