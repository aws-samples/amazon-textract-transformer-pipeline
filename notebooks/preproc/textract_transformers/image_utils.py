# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Utilities for loading and handling images and raw documents

Use the `Document` class to handle raw images/documents: Providing a consistent interface whether
the input file is a PDF, single image, or multi-page image.
"""
# Python Built-Ins:
from __future__ import annotations
import dataclasses
import io
from logging import getLogger
import os
from typing import BinaryIO, Generator, List, Optional, Tuple, Union

# External Dependencies:
import pdf2image
from PIL import Image, ExifTags

# Local Dependencies:
from .file_utils import split_filename


logger = getLogger("image_utils")

# MIME/Media-Type mappings:
SINGLE_IMAGE_CONTENT_TYPES = {
    "image/jpeg": "JPG",
    "image/jpg": "JPG",
    "image/png": "PNG",
}
MULTI_IMAGE_CONTENT_TYPES = {
    "image/tiff": "TIFF",
}
PDF_CONTENT_TYPES = set(("application/pdf",))
CONTENT_TYPES_BY_EXT = {
    "jpg": "image/jpg",
    "jpeg": "image/jpeg",
    "pdf": "application/pdf",
    "png": "image/png",
    "tiff": "image/tiff",
}


def _get_exif_tag_id_by_name(name: str) -> Optional[str]:
    """Find a numeric EXIF tag ID by common name

    As per https://pillow.readthedocs.io/en/stable/reference/ExifTags.html
    """
    try:
        return next(k for k in ExifTags.TAGS.keys() if ExifTags.TAGS[k] == name)
    except StopIteration:
        return None


ORIENTATION_EXIF_ID = _get_exif_tag_id_by_name("Orientation")


def apply_exif_rotation(image: Image.Image) -> Tuple[Image.Image, int]:
    """If image has an EXIF metadata rotation, create a copy with the rotation actually applied

    Returns
    -------
    image :
        The original image if rotation is 0, else a copy of the image with pixels actually rotated
        per the original's EXIF metadata tag specification.
    applied_angle_deg :
        Applied rotation in degrees anticlockwise (0, 90, 180 or 270)
    """
    # Correct orientation from EXIF data:
    exif = dict((image.getexif() or {}).items())
    img_orientation = exif.get(ORIENTATION_EXIF_ID)
    if img_orientation == 3:
        return image.rotate(180, expand=True), 180
    elif img_orientation == 6:
        return image.rotate(270, expand=True), 270
    elif img_orientation == 8:
        return image.rotate(90, expand=True), 90
    else:
        return image, 0


def resize_image(
    image: Image.Image,
    size: Union[int, Tuple[int, int]] = (224, 224),
    default_square: bool = True,
    letterbox_color: Optional[Tuple[int, int, int]] = None,
    max_size: Optional[int] = None,
    resample: int = Image.BICUBIC,
) -> Image.Image:
    """Resize (stretch or letterbox) a PIL Image

    In the case no resizing was necessary, the original image object may be returned. Otherwise,
    the result will be a copy. This function is similar to the logic in Hugging Face
    image_utils.ImageFeatureExtractionMixin.resize - but defaults to bicubic resampling instead of
    bilinear and also supports letterboxing as well as aspect ratio stretch.

    Arguments
    ---------
    image :
        The (loaded PIL) image to resize
    size :
        The target size to output. May be a sequence of (width, height), or a single number. If
        `default_square` is `True`, a single number will be resized to (size, size). Otherwise, the
        **smaller** edge of the image will be matched to `size` and the aspect ratio preserved.
    default_square :
        Control how to interpret single-number `size`. Set `True` to target a square, or `False` to
        preserve aspect ratio.
    letterbox_color :
        Provide a 0-255 (R, G, B) tuple to letterbox the image and use this color as the background
        for any unused area. Leave unset (`None`) to stretch the image to match the target size.
    max_size :
        Maximum allowed size for longer edge when using single-`size` mode with `default_square` =
        `False`. If the longer edge of the image is greater than `max_size` after initial resize,
        the image is again proportionally resized so that the longer edge is equal to `max_size`.
        As a result, `size` might be overruled, i.e the smaller edge may be shorter than `size`.
        Only used if `default_to_square` is `False` and `size` is a single number.
    resample :
        Image.Resampling method, defaults to BICUBIC
    """
    if not isinstance(image, Image.Image):
        raise ValueError(f"resize_image accepts PIL.Image only. Got: {type(image)}")

    if not hasattr(size, "__len__"):
        size = (size,)

    if len(size) == 1:
        if default_square:
            # Treat as square:
            size = (size[0], size[0])
        else:
            # Specified target shortest edge size:
            short = size[0]
            iw, ih = image.size
            ishort, ilong = (iw, ih) if iw <= ih else (ih, iw)

            if short == ishort:
                return image

            long = int(short * ilong / ishort)

            # Check longer edge max_size limit if provided:
            if max_size is not None:
                if max_size <= short:
                    raise ValueError(
                        f"max_size = {max_size} must be strictly greater than the requested "
                        f"size for the smaller edge = {short}"
                    )
                if long > max_size:
                    short, long = int(max_size * short / long), max_size

            size = (short, long) if iw <= ih else (long, short)

    if letterbox_color:
        # Letterbox the image to the normalized `size` with given background color:
        iw, ih = image.size
        w, h = size
        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)
        result = Image.new("RGB", size, letterbox_color)
        return result.paste(
            image.resize((nw, nh), resample=resample),
            ((w - nw) // 2, (h - nh) // 2),
        )
    else:
        # Just stretch the image to fit:
        return image.resize(size, resample=resample)


@dataclasses.dataclass
class Page:
    """An individual page of a raw document, ready for OCR

    May be backed by either an in-memory PIL Image, or a local file. Includes useful information for
    OCR engines like pre-applied rotation, and link to the parent Document object for requesting
    alternative DPI views if available.
    """

    _image: Union[str, Image.Image]

    parent_doc: Document
    page_num: int
    rotation: int
    file_path: Optional[str] = None
    dpi: Optional[int] = None

    def __post_init__(self):
        if isinstance(self._image, str):
            if self.file_path is None:
                self.file_path = self._image
            self._image = None

    @property
    def image(self) -> Image.Image:
        """Fetch PIL Image backing this Page, either in-memory value or from filesystem"""
        if self._image:
            return self._image
        elif self.file_path:
            return Image.open(self.file_path)
        else:
            raise ValueError(
                "Couldn't fetch page %s of doc %s: file_path is not defined"
                % (self.page_num, self.parent_doc)
            )

    def copy_without_imdata(self) -> Page:
        """Create a copy of this Page in which the actual image data is not cached in memory

        Use this to save memory if you need to keep a reference to a page, but the actual image can
        be recovered on-demand from the filesystem. Will fail if `page.file_path` is not set.
        """
        if not self.file_path:
            raise RuntimeError(
                "Can't create a filesystem-linked record for in-memory-only Page image"
            )
        return dataclasses.replace(self, _image=None)


class Document:
    """A sequence of (page) images with APIs for OCR engines to request different views

    During processing, a particular OCR engine might find evidence to suggest the input image should
    be transformed: For example, to try again at higher/lower DPI or with a rotation. Although
    PIL.Image might seem a natural API for many of these actions, a page Image sourced from a PDF
    wouldn't have any way to request an up-scaled version of itself. Therefore this class provides a
    standardised wrapper over single-image, multi-image (TIFF) and PDF documents to produce PIL
    Images.

    Page images generated by this class have any EXIF rotations explicitly applied to the pixel
    grid, so they're ready to use with Amazon SageMaker Ground Truth or other rotation-unaware tools.
    """

    def __init__(
        self,
        file_or_path: Union[BinaryIO, bytes, str],
        ext_or_media_type: Optional[str] = None,
        default_doc_dpi: int = 300,
        default_image_ext: str = "png",
        convert_image_formats: bool = False,
        base_file_path: Optional[str] = None,
    ):
        """Create a Document

        Parameters
        ----------
        file_or_path :
            Either a string file-path, a readable BinaryIO (open file), or bytes in memory: Data of
            the original/raw document.
        ext_or_media_type :
            File extension (e.g. "pdf") or MIME/Media-Type (e.g. "application/pdf") indicating the
            raw document file type. This is optional if `file_or_path` is a file path and the type
            can be inferred from filename extension, but mandatory otherwise.
        default_doc_dpi :
            Default DPI to use for rendering variable-resolution (PDF) documents. This setting has
            no effect when the source document is a fixed resolution file e.g. a PNG/JPEG image.
        default_image_ext :
            Default image file extension for outputs (resized/normalized page images)
        convert_image_formats :
            Set True to always convert generated page images to the `default_image_ext`, rather than
            allowing propagation of single-image file types from the input.
        base_file_path :
            Optional base/reference folder for raw file path. If set, `file_or_path` must be a path
            beginning with `base_file_path` - and any subfolders under this base will be preserved
            when creating page views in the workspace folder (see `set_workspace()`).
        """
        self._file_or_path = file_or_path
        self._default_doc_dpi = default_doc_dpi
        self._default_image_ext = default_image_ext
        self._convert_image_formats = convert_image_formats
        self._media_type, self._ext, self._scalable = Document._infer_media_type_and_ext(
            ext_or_media_type, file_or_path
        )
        self._n_pages = None  # Will be filled in the first time file is read

        if isinstance(file_or_path, str):
            self._filename = os.path.basename(file_or_path)
            if base_file_path is None:
                self._subfolder = ""
            else:
                if not file_or_path.startswith(base_file_path):
                    raise ValueError(
                        "Provided file path does not start with base_file_path: Got '%s' and '%s'"
                        % (file_or_path, base_file_path)
                    )
                relpath = file_or_path[len(base_file_path) :]
                if relpath.startswith(os.path.sep):
                    relpath = relpath[1:]
                self._subfolder = os.path.dirname(relpath)
        else:
            self._filename = f"document.{self._ext}"
            self._subfolder = ""

        # Cache of filenames by dpi ONLY when workspace is set
        self._views_cache = {}
        self._workspace_folder = None
        self._workspace_multires = False

    @staticmethod
    def _infer_media_type_and_ext(
        ext_or_media_type: Optional[str] = None,
        file_or_path: Optional[Union[BinaryIO, bytes, str]] = None,
    ) -> Tuple[str, str, bool]:
        """Utility function to normalize file extension or MIME/Media-Type, given file path"""
        media_type = None
        ext = None
        scalable = False
        if ext_or_media_type is not None:
            # TODO: Lowercasing here could be a problem for MIME type arguments
            # ...But it's OK for simple types e.g. image/jpeg
            ext_or_media_type = ext_or_media_type.lower()

            if "/" in ext_or_media_type:
                # Looks like a MIME/Media-Type
                media_type = ext_or_media_type
                if media_type in PDF_CONTENT_TYPES:
                    ext = "pdf"
                    scalable = True
                elif media_type in SINGLE_IMAGE_CONTENT_TYPES:
                    ext = SINGLE_IMAGE_CONTENT_TYPES[media_type].lower()
                elif media_type in MULTI_IMAGE_CONTENT_TYPES:
                    ext = MULTI_IMAGE_CONTENT_TYPES[media_type].lower()
                else:
                    raise ValueError(
                        "MIME/Media-Type '%s' not recognised. Supported list: %s"
                        % (media_type, list(CONTENT_TYPES_BY_EXT.values()))
                    )
            else:
                if ext_or_media_type.startswith("."):
                    ext_or_media_type = ext_or_media_type[1:]
                ext = ext_or_media_type
                if ext in CONTENT_TYPES_BY_EXT:
                    media_type = CONTENT_TYPES_BY_EXT[ext]
                    if media_type in PDF_CONTENT_TYPES:
                        scalable = True
                else:
                    raise ValueError(
                        "File extension '%s' not recognised. Supported list: %s"
                        % (ext, list(CONTENT_TYPES_BY_EXT.keys()))
                    )
            return media_type, ext, scalable
        # Else try to infer from filename:

        if isinstance(file_or_path, str):
            filename = file_or_path.rpartition("/")[2]
            ext = filename.rpartition(".")[2].lower()
            if ext in (filename, ""):
                raise ValueError(
                    "Couldn't infer media type of filename without extension: %s" % file_or_path
                )
            if ext in CONTENT_TYPES_BY_EXT:
                media_type = CONTENT_TYPES_BY_EXT[ext]
                if media_type in PDF_CONTENT_TYPES:
                    scalable = True
            else:
                raise ValueError(
                    "File extension '%s' not recognised. Supported list: %s"
                    % (ext, list(CONTENT_TYPES_BY_EXT.keys()))
                )
            return media_type, ext, scalable
        # Else neither explicitly provided nor available through file path:

        raise ValueError(
            "Couldn't infer media type: ext_or_media_type must be explicitly provided when "
            "file_or_path is not a path with a valid extension (Got %s)" % type(file_or_path)
        )

    def _normalize_target_dpi(self, dpi: Optional[int] = None) -> Optional[int]:
        """Validate that the current document is scalable; return requested dpi or default

        Raises
        ------
        NotImplementedError
            If requested DPI is not None but the current document does not support scaling
        """
        if self.scalable:
            return dpi or self._default_doc_dpi
        else:
            if dpi is not None:
                raise NotImplementedError(
                    "Can't request specific DPI for documents of type %s" % self.media_type
                )
            return None

    def get_page(self, page_num: int, dpi: Optional[int] = None) -> Page:
        """Fetch a single Page from the document at specific (or default) resolution

        Note that for some document types (e.g. PDFs), new DPI views can only be generated for the
        whole document at once so this may still take time.

        Parameters
        ----------
        page_num :
            1-based page number that should be returned
        dpi :
            Optional specific DPI to request (only supported for scalable document types)

        Raises
        ------
        NotImplementedError
            If requested DPI is not None but the current document does not support scaling
        RuntimeError
            If the current workspace folder was initialized as single-resolution
        ValueError
            If `page_num` is out of range for the document length
        """
        target_dpi = self._normalize_target_dpi(dpi)
        if target_dpi in self._views_cache:
            return self._views_cache[target_dpi][page_num - 1]
        elif len(self._views_cache) > 0 and not self._workspace_multires:
            raise RuntimeError(
                "Can't request secondary views when initialized with a single-resolution workspace"
            )
        page = None
        for ix, page_view in self._create_new_view(target_dpi):
            if ix + 1 == page_num:
                # If the view is getting cached to a folder, it makes sense to allow the view to get
                # completed before returning so that any later random page accesses will see it.
                # Otherwise, may as well return as soon as we have the target page ready.
                if self._workspace_folder is None:
                    return page_view
                else:
                    page = page_view
        if page is None:
            raise ValueError(
                "Couldn't get page_num %s from document of length %s" % (page_num, len(self))
            )
        return page

    def get_pages(self, dpi: Optional[int] = None) -> Generator[Page, None, None]:
        """Generate the sequence of document Pages at specific (or default) resolution

        Use this to iterate through all Pages at a single resolution

        Parameters
        ----------
        dpi :
            Optional specific DPI to request (only supported for scalable document types)

        Raises
        ------
        NotImplementedError
            If requested DPI is not None but the current document does not support scaling
        RuntimeError
            If the current workspace folder was initialized as single-resolution
        """
        target_dpi = self._normalize_target_dpi(dpi)
        if target_dpi in self._views_cache:
            for page in self._views_cache[target_dpi]:
                yield page
        else:
            if len(self._views_cache) > 0 and not self._workspace_multires:
                raise RuntimeError(
                    "Can't request secondary views from a single-resolution workspace"
                )
            yield from self._create_new_view(target_dpi)

    def _create_new_view(self, dpi: Optional[int] = None) -> Generator[Page, None, None]:
        """Generate a view (sequence of Pages) to satisfy a request not present in cache

        A workspace folder is only required (`set_workspace()`)if the document is a PDF
        """
        target_dpi = self._normalize_target_dpi(dpi)
        if self.media_type in PDF_CONTENT_TYPES:
            # For PDFs, we must have files for both input and output so cannot create a view without
            # a workspace.
            if not self._workspace_folder:
                raise RuntimeError(
                    "Can't generate page images view of a PDF unless a workspace folder is set. "
                    "Call set_workspace first."
                )
            view_folder = os.path.join(
                self._workspace_folder, str(target_dpi) if self._workspace_multires else ""
            )
            pages = self._generate_pdf_view(view_folder, dpi=target_dpi)
            self._views_cache[target_dpi] = pages
            for page in pages:
                yield page
        else:
            # For images, a workspace might not be present and is not necessary.
            view_folder = self._workspace_folder  # No multi-DPI support for image inputs
            view_pages = []
            for page in self._generate_image_view(
                view_folder,
                convert_to_format=self._default_image_ext if self._convert_image_formats else None,
            ):
                if view_folder:
                    view_pages.append(page.copy_without_imdata())
                yield page
            if view_folder:
                self._views_cache[target_dpi] = view_pages

    def _get_pdf_source_path(self) -> str:
        """Get a filesystem path for source PDF file, storing it to workspace folder if necessary"""
        if isinstance(self._file_or_path, str):
            return self._file_or_path
        if not self._workspace_folder:
            raise RuntimeError(
                "Tried to buffer document to file but set_workspace has not been called"
            )
        buffer_path = os.path.join(self._workspace_folder, "_raw.pdf")
        if not os.path.isfile(buffer_path):
            with open(buffer_path, "wb") as f:
                if hasattr(self._file_or_path, "read"):
                    f.write(self._file_or_path.read())
                else:
                    f.write(self._file_or_path)
        return buffer_path

    def _generate_image_view(
        self,
        output_folder: Optional[str] = None,
        convert_to_format: Optional[str] = None,
    ) -> Generator[Page, None, None]:
        """Generate a view of a non-PDF image only (may be multi-page e.g. TIFF)

        EXIF rotations, if present, are made concrete in the generated view.
        """
        img_raw = self._read_image()
        n_pages = len(self)
        if output_folder:
            if self.media_type in PDF_CONTENT_TYPES:
                ext = convert_to_format or self._default_image_ext
            else:
                ext = convert_to_format or self._ext
        else:
            ext = None
        for ixpage in range(n_pages):
            if n_pages > 1:
                img_raw.seek(ixpage)
            page_image, rotated_angle = apply_exif_rotation(img_raw)
            if output_folder is not None:
                basename = split_filename(self._filename)[0]
                filename = "".join(
                    (
                        basename,
                        "-%04i" % (ixpage + 1) if n_pages > 1 else "",
                        ".",
                        ext,
                    )
                )
                page_path = os.path.join(
                    output_folder,
                    self._subfolder,
                    filename,
                )
                page_image.save(page_path)
                page_image.filename = page_path
            else:
                page_path = None
            yield Page(
                page_image,
                parent_doc=self,
                page_num=ixpage + 1,
                rotation=rotated_angle,
                file_path=page_path,
                dpi=None,
            )

    def _generate_pdf_view(self, output_folder: str, dpi: Optional[int] = None) -> List[Page]:
        """Generate a new view (of a PDF only) at requested zoom (DPI) level

        This method requires `set_workspace()` to have been called.
        """
        dpi = dpi or self._default_doc_dpi
        basename = split_filename(self._filename)[0]
        output_folder = os.path.join(output_folder, self._subfolder)
        source_path = self._get_pdf_source_path()
        os.makedirs(output_folder, exist_ok=True)
        image_filepaths = pdf2image.convert_from_path(
            source_path,
            output_folder=output_folder,
            output_file=basename + "-",
            paths_only=True,
            fmt=self._default_image_ext,
            dpi=dpi,
        )
        self._n_pages = len(image_filepaths)
        return [
            Page(path, parent_doc=self, page_num=ixpage + 1, rotation=0, dpi=dpi)
            for ixpage, path in enumerate(image_filepaths)
        ]

    def _read_image(self) -> Image.Image:
        """Load this document's (assumed non-PDF/document) source as a PIL Image

        This function works regardless of whether self.file_or_path is an open file object or a path
        """
        if isinstance(self._file_or_path, str) or hasattr(self._file_or_path, "read"):
            img = Image.open(self._file_or_path)
        else:
            # Cannot `with` the buffer, because PIL requires the buffer to still be available later:
            buffer = io.BytesIO(self._file_or_path)
            img = Image.open(buffer)
        self._n_pages = getattr(img, "n_frames", 1)
        return img

    @property
    def media_type(self) -> str:
        """Underlying MIME/Media-Type (e.g. 'image/jpeg') of this Document"""
        return self._media_type

    @property
    def file_extension(self) -> str:
        """Underlying file extension (e.g. 'jpg', 'pdf') of this Document"""
        return self._ext

    @property
    def scalable(self) -> bool:
        """True if this Document supports requesting views at alternative resolutions"""
        return self._scalable

    def set_workspace(self, folder: str, multi_res: bool = True):
        """Set up a working folder to use for operations that require filesystem space

        For ephemeral operations, you might like to use a `tempfile.TemporaryDirectory`. For
        persistent document view generation, you can use a normal local folder.

        Arguments
        ---------
        folder :
            Path to the existing local folder to use as a workspace
        multi_res :
            Set True to treat this workspace as multi-resolution, in which case outputs will be
            nested in a {DPI}/ subfolder. Set False to use the workspace folder directly, but this
            will cause an error if a second DPI/zoom view is requested for the document.
        """
        self._views_cache = {}
        self._workspace_folder = folder
        self._workspace_multires = multi_res

    def unset_workspace(self):
        """Remove old workspace configuration (Does not delete files)"""
        self._views_cache = {}
        self._workspace_folder = None
        self._workspace_multires = False

    def __len__(self) -> int:
        """Number of pages in the document

        Note: calculating document length on a PDF will result in rendering the doc at default page
        resolution - which could fail if no workspace has been set up, or the workspace is
        single-resolution and a particular (non-default) resolution has already been requested.

        TODO: Would be nice to have a non-view-generating method for PDFs, e.g. with PyPDF.
        """
        if self._n_pages is not None:
            return self._n_pages
        if self.media_type in PDF_CONTENT_TYPES:
            self._n_pages = sum(1 for _ in self.get_pages())
            return self._n_pages
        else:
            self._read_image()  # Sets length
            return self._n_pages
