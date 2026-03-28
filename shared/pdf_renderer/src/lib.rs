use pyo3::prelude::*;
use pyo3::types::PyBytes;
use pdf_oxide::PdfDocument;
use pdf_oxide::rendering::{render_page, RenderOptions};

#[pyclass(unsendable)]
struct PdfRenderer {
    doc: PdfDocument,
}

#[pymethods]
impl PdfRenderer {
    #[new]
    fn new(path: &str) -> PyResult<Self> {
        let doc = PdfDocument::open(path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("{e}")))?;
        Ok(Self { doc })
    }

    #[staticmethod]
    fn from_bytes(data: &[u8]) -> PyResult<Self> {
        let doc = PdfDocument::from_bytes(data.to_vec())
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("{e}")))?;
        Ok(Self { doc })
    }

    fn page_count(&mut self) -> PyResult<usize> {
        self.doc.page_count()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{e}")))
    }

    /// Render a page to PNG bytes. Returns (png_bytes, width, height).
    #[pyo3(signature = (page, dpi=144))]
    fn render_page<'py>(&mut self, py: Python<'py>, page: usize, dpi: u32) -> PyResult<(Bound<'py, PyBytes>, u32, u32)> {
        let opts = RenderOptions::with_dpi(dpi);
        let image = render_page(&mut self.doc, page, &opts)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{e}")))?;
        let bytes = PyBytes::new(py, &image.data);
        Ok((bytes, image.width, image.height))
    }
}

#[pymodule]
fn pdf_renderer(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PdfRenderer>()?;
    Ok(())
}
