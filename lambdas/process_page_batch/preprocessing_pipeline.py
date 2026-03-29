import gzip
import io
import os
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import boto3
import msgpack
from PIL import Image
import pypdfium2 as pdfium
from pypdfium2 import PdfPage
from docling_core.types.doc.base import BoundingBox, CoordOrigin, Size
from docling_parse.pdf_parser import DoclingPdfParser

from shared.page_serialization import pack_segmented_page

if TYPE_CHECKING:
    from docling_parse.pdf_parser import PdfDocument


_S3_CLIENT = boto3.client("s3")
_DOCLING_PARSER = DoclingPdfParser(loglevel="fatal")


class PreprocessingPipeline:

    MAX_2X_PIXELS = 3200

    def __init__(self, user_id: str):
        self.user_id = user_id
        self.bucket = os.environ["DOCUMENTS_BUCKET"]
        self.s3_client = _S3_CLIENT

    def process_batch(self, job_id: str, start_page: int, end_page: int, total_pages: int):
        """Processes a batch of pages"""
        t0 = time.time()
        file_bytes = self._load_from_s3(job_id=job_id)
        t_s3 = time.time()

        pdoc = None
        dp_doc = None
        job_short = job_id[:8]

        try:
            pdoc = pdfium.PdfDocument(io.BytesIO(file_bytes))
            t_pdfium = time.time()

            dp_doc = _DOCLING_PARSER.load(path_or_stream=io.BytesIO(file_bytes))
            t_load = time.time()
            print(
                f"[{job_short}] LOAD pdfium={t_pdfium-t_s3:.3f}s "
                f"docling_load={t_load-t_pdfium:.3f}s pdf_size={len(file_bytes)/1024:.0f}KB"
            )

            pages = self._preprocess_page_batch(
                pdoc=pdoc,
                dp_doc=dp_doc,
                start_page=start_page,
                end_page=end_page,
                job_short=job_short,
            )
            t_preprocess = time.time()

            batch_s3_key = self._upload_pages_to_s3(job_id, pages, start_page, end_page)
            t_upload = time.time()

            print(
                f"[{job_short}] TIMING s3_get={t_s3-t0:.3f}s pdf_load={t_load-t_s3:.3f}s "
                f"preprocess={t_preprocess-t_load:.3f}s s3_put={t_upload-t_preprocess:.3f}s "
                f"total={t_upload-t0:.3f}s pages={start_page}-{end_page}"
            )

            return {
                "batch_key": batch_s3_key,
                "start_page": start_page,
                "end_page": end_page,
            }
        finally:
            if dp_doc is not None:
                dp_doc.unload()
            if pdoc is not None:
                close = getattr(pdoc, "close", None)
                if callable(close):
                    close()

    def _preprocess_page_batch(
        self,
        pdoc: pdfium.PdfDocument,
        dp_doc: "PdfDocument",
        start_page: int,
        end_page: int,
        job_short: str = "",
    ) -> List[Dict[str, Any]]:
        """Segment a page range, calculate the size, and generate images"""
        out: List[Dict[str, Any]] = []

        for page_no in range(start_page, end_page + 1):
            t0 = time.time()

            seg = self._parse_page(dp_doc, page_no)
            t_parse = time.time()

            pdf_page = pdoc[page_no]
            size = self._get_size(pdf_page)

            long_side = max(size.width, size.height)
            scale_2x = min(2.0, self.MAX_2X_PIXELS / long_side)

            img2 = self._get_page_image(pdf_page, size, scale=scale_2x, cropbox=None)
            t_render = time.time()

            img1 = img2.resize((img2.width // 2, img2.height // 2), Image.Resampling.LANCZOS)

            img2_webp = self._convert_to_webp(img=img2)
            img1_webp = self._convert_to_webp(img=img1)
            t_encode = time.time()

            seg_packed = pack_segmented_page(seg)
            t_pack = time.time()

            print(f"[{job_short}] PAGE {page_no} parse={t_parse-t0:.3f}s render={t_render-t_parse:.3f}s "
                  f"encode={t_encode-t_render:.3f}s pack={t_pack-t_encode:.3f}s "
                  f"img={img2.width}x{img2.height} scale={scale_2x:.2f} "
                  f"cells=c{len(seg.char_cells)}/w{len(seg.word_cells)}/l{len(seg.textline_cells)}")

            page = {
                "page_index": page_no,
                "size": {"w": size.width, "h": size.height},
                "segmented": seg_packed,
                "images": {1: img1_webp.getvalue(), 2: img2_webp.getvalue()},
                "images_scale": scale_2x,
            }
            out.append(page)

        return out

    @staticmethod
    def _parse_page(doc: "PdfDocument", page_no: int) -> Any:
        """Parses the page and returns a segmented PDF page"""
        t0 = time.time()
        seg = doc.get_page(page_no + 1, create_words=True, create_textlines=True)
        t1 = time.time()

        H = seg.dimension.height
        for cells in (seg.textline_cells, seg.char_cells, seg.word_cells):
            for tc in cells:
                tc.to_top_left_origin(H)
        t2 = time.time()

        n = len(seg.char_cells) + len(seg.word_cells) + len(seg.textline_cells)
        if n > 500 or (t2 - t0) > 0.5:
            print(f"[parse] page={page_no} get_page={1000*(t1-t0):.0f}ms coord_flip={1000*(t2-t1):.0f}ms total={1000*(t2-t0):.0f}ms cells={n}")

        return seg

    @staticmethod
    def _get_size(page: PdfPage) -> Size:
        return Size(width=page.get_width(), height=page.get_height())

    @staticmethod
    def _get_page_image(page: PdfPage, page_size: Size, scale: float = 1,
                        cropbox: Optional[BoundingBox] = None) -> Image.Image:
        if not cropbox:
            cropbox = BoundingBox(l=0, r=page_size.width, t=0, b=page_size.height, coord_origin=CoordOrigin.TOPLEFT)
            padbox = BoundingBox(l=0, r=0, t=0, b=0, coord_origin=CoordOrigin.BOTTOMLEFT)
        else:
            padbox = cropbox.to_bottom_left_origin(page_size.height).model_copy()
            padbox.r = page_size.width - padbox.r
            padbox.t = page_size.height - padbox.t

        img = page.render(scale=scale, rotation=0, crop=padbox.as_tuple()).to_pil()

        return img.resize(size=(round(cropbox.width * scale), round(cropbox.height * scale)))

    @staticmethod
    def _convert_to_webp(img: Image.Image) -> io.BytesIO:
        """Converts the image to lossless webp with fast compression."""
        mode = "RGB" if img.mode not in ("L", "RGB", "RGBA") else img.mode
        if img.mode != mode:
            img = img.convert(mode)
        buf = io.BytesIO()
        img.save(buf, format="WEBP", lossless=True, method=1)
        buf.seek(0)

        return buf

    def _load_from_s3(self, job_id: str) -> bytes:
        """Loads the file from S3"""
        s3_key = f"uploads/{self.user_id}/{job_id}/source.pdf"

        response = self.s3_client.get_object(Bucket=self.bucket, Key=s3_key)
        return response['Body'].read()

    def _upload_pages_to_s3(self, job_id: str, pages: List[Dict[str, Any]], start_page: int, end_page: int) -> str:
        """Upload the page range to S3"""
        packed = msgpack.packb(pages, use_bin_type=True)
        blob = gzip.compress(packed, compresslevel=6)

        key = f"batches/{self.user_id}/{job_id}/pages_{start_page}_{end_page}.bin"
        self.s3_client.put_object(Bucket=self.bucket, Key=key, Body=blob,
                                  ContentType="application/octet-stream")
        return key
