from typing import List, Dict, Any
import base64
from io import BytesIO

from docling.datamodel.base_models import Page
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling_core.types.doc import ImageRef, PictureItem, SectionHeaderItem, TableItem, TextItem

from processor.final_service.readingorder_model import ReadingOrderModel, ReadingOrderOptions
from processor.final_service.page_assemble_model import PageAssembleModel, PageAssembleOptions, AssembledUnit
from processor.shared.logging_config import get_logger
from processor.shared.telemetry import now, ms

logger = get_logger(__name__)


class DocumentAssembler:
    """Final assembly - can run on Lambda or GPU."""

    images_scale: int = 2

    def __init__(self, pipeline_options=None):
        self.pipeline_options = pipeline_options or PdfPipelineOptions()
        self.page_assembler = PageAssembleModel(options=PageAssembleOptions())
        self.reading_order_model = ReadingOrderModel(options=ReadingOrderOptions())

    def assemble_document(self, input_doc, processed_pages: List[Page]) -> ConversionResult:
        """Assemble final document from processed pages."""
        n = len(processed_pages)
        conv_res = ConversionResult(input=input_doc)
        conv_res.pages = processed_pages

        # Phase 1: page assembly
        t0 = now()
        assembled_pages = list(self.page_assembler(conv_res, processed_pages))
        conv_res.pages = assembled_pages
        t1 = now()

        all_elements = []
        all_headers = []
        all_body = []
        for page in conv_res.pages:
            if page.assembled:
                all_elements.extend(page.assembled.elements)
                all_headers.extend(page.assembled.headers)
                all_body.extend(page.assembled.body)

        conv_res.assembled = AssembledUnit(
            elements=all_elements,
            headers=all_headers,
            body=all_body
        )

        # Phase 2: reading order
        conv_res.document = self.reading_order_model(conv_res)
        t2 = now()

        # Phase 3: figure image extraction
        n_pics = self._extract_figure_images(conv_res)
        t3 = now()

        logger.info(
            f"assemble_document: {n} pages | "
            f"page_asm={ms(t1 - t0):.0f}ms ro={ms(t2 - t1):.0f}ms "
            f"figures={ms(t3 - t2):.0f}ms({n_pics}) total={ms(t3 - t0):.0f}ms"
        )
        return conv_res

    def _extract_figure_images(self, conv_res: ConversionResult) -> int:
        """Extract actual images for figure elements. Returns count extracted."""
        count = 0
        # Build page lookup to avoid O(n) scan per picture
        page_by_ix = {p.page_no: p for p in conv_res.pages}

        for element, _level in conv_res.document.iterate_items():
            if isinstance(element, PictureItem) and len(element.prov) > 0:
                page = page_by_ix.get(element.prov[0].page_no - 1)

                if page and page.image:
                    crop_bbox = (
                        element.prov[0]
                        .bbox.scaled(scale=self.images_scale)
                        .to_top_left_origin(
                            page_height=page.size.height * self.images_scale
                        )
                    )

                    cropped_im = page.image.crop(crop_bbox.as_tuple())
                    element.image = ImageRef.from_pil(
                        cropped_im,
                        dpi=int(72 * self.images_scale)
                    )
                    count += 1

        return count

    def extract_structured_elements(self, conv_res: ConversionResult) -> Dict[str, Any]:
        t0 = now()
        doc = conv_res.document
        pages_map: Dict[int, List[Dict[str, Any]]] = {}
        page_sizes: Dict[int, Dict[str, float]] = {}

        # Get page sizes
        for page in conv_res.pages:
            page_no = page.page_no + 1  # 1-indexed
            page_sizes[page_no] = {
                "width": page.size.width,
                "height": page.size.height
            }
            pages_map[page_no] = []

        # Iterate through document in reading order
        for element, _level in doc.iterate_items():
            if not hasattr(element, 'prov') or not element.prov:
                continue

            # Get primary provenance (first bbox)
            prov = element.prov[0]
            page_no = prov.page_no
            bbox = prov.bbox

            elem_data: Dict[str, Any] = {
                "bbox": {
                    "l": bbox.l,
                    "t": bbox.t,
                    "r": bbox.r,
                    "b": bbox.b
                }
            }

            if isinstance(element, TextItem):
                # Map label to cleaner type names
                label_str = str(element.label).lower().replace("docitemlabel.", "")
                elem_data["type"] = label_str
                elem_data["content"] = element.text or ""
                if isinstance(element, SectionHeaderItem):
                    elem_data["level"] = element.level

            elif isinstance(element, TableItem):
                elem_data["type"] = "table"
                # Export table to markdown
                if element.data:
                    elem_data["content"] = element.export_to_markdown(doc=doc)
                else:
                    elem_data["content"] = ""

            elif isinstance(element, PictureItem):
                elem_data["type"] = "picture"
                elem_data["content"] = ""

                # Include base64 image if available
                if element.image and element.image.pil_image:
                    buf = BytesIO()
                    element.image.pil_image.save(buf, format="PNG")
                    elem_data["image_base64"] = base64.b64encode(buf.getvalue()).decode("utf-8")

            else:
                # Other element types
                elem_data["type"] = type(element).__name__.lower().replace("item", "")
                elem_data["content"] = getattr(element, 'text', '') or ""

            if page_no in pages_map:
                pages_map[page_no].append(elem_data)

        # Build final structure
        result = {
            "pages": []
        }

        for page_no in sorted(pages_map.keys()):
            page_data = {
                "page_no": page_no,
                "width": page_sizes.get(page_no, {}).get("width", 0),
                "height": page_sizes.get(page_no, {}).get("height", 0),
                "elements": pages_map[page_no]
            }
            result["pages"].append(page_data)

        n_elems = sum(len(p) for p in pages_map.values())
        logger.info(f"extract_structured: {len(pages_map)} pages {n_elems} elements {ms(now()-t0):.0f}ms")
        return result
