# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Visualization utilities for OCR enrichment"""

# Python Built-Ins:
import io
import json
from operator import itemgetter
from typing import Dict, List, Optional, Tuple, Union

# External Dependencies:
import ipywidgets as widgets
from matplotlib.colors import Colormap
from matplotlib import pyplot as plt
import numpy as np
import PIL
import trp  # amazon-textract-response-parser


# Default RGBA colors for rendering Textract results:
UNLABELLED_COLOR = (0.8, 0.8, 0.8, 0.5)
ERROR_COLOR = (1.0, 0.0, 0.0, 0.7)
FIELD_KEY_COLOR = (0.0, 0.6, 0.8667, 0.4)
FIELD_VALUE_COLOR = (1.0, 0.48, 0.0, 0.4)
TABLE_OUTLINE_COLOR = (0.0, 0.0, 0.8667, 0.3)
TABLE_CELL_COLOR = (0.0, 0.7333, 0.0, 0.4)


def get_default_cmap(classes):
    return plt.get_cmap("nipy_spectral", len(classes) + 1)


def trp_polygon_to_plottable(poly: trp.Polygon, img_width: float, img_height: float):
    """Convert a Textract Response Parser 'Polygon' to an array plottable with pyplot/matplotlib"""
    return np.array([[p.x, p.y] for p in poly]) * np.array([[img_width, img_height]])


def draw_smgt_annotated_page(
    img_path: str,
    ann_classes: List[str],
    annotations=[],
    textract_result: Optional[Union[str, Dict, List[Dict], trp.Document]] = None,
    page_num: int = 1,
    plain_color: Tuple[float, float, float, float] = UNLABELLED_COLOR,
    err_color: Tuple[float, float, float, float] = ERROR_COLOR,
    figsize: Tuple[float, float] = (14, 16),
    class_cmap: Optional[Colormap] = None,
    render_forms: bool = False,
    render_tables: bool = False,
    render_words: bool = True,
    show_axes: bool = True,
) -> plt.Figure:
    """Draw a page image with Ground Truth and/or model prediction annotations.

    If `annotations` are provided, the ground truth bounding boxes annotations will be rendered.

    If `textract_result` is provided, detected WORD box geometries will be rendered with face color
    set by the `PredictedClass` attribute if present (i.e. if the Textract result has been fed
    through the predictor model); and edge color set by the `annotations` if present.

    Arguments
    ---------
    img_path : str
        Local path to the image (e.g. PNG, JPEG) of the page
    ann_classes : List[str]
        List of token class names (excluding the extra 'unlabelled' class)
    annotations : List[Dict]
        List of BBox 'annotations' as per SageMaker Ground Truth
    textract_result : Union[str, Dict, List[Dict], trp.Document], optional
        The Textract result for the doc - either as loaded JSON, a local path string, or a loaded
        trp.Document object. (Default None)
    page_num : int, optional
        1-based page number in the document to render (set to match img_path's source page)
    plain_color : Tuple[float, float, float, float], optional
        0-1 RGBA color tuple to render for 'unlabelled' tokens. (Default UNLABELLED_COLOR)
    err_color : Tuple[float, float, float, float], optional
        0-1 RGBA color tuple to render for tokens which annotation classifies more than once.
        (Default ERROR_COLOR)
    figsize : Tuple[float, float], optional
        matplotlib figure size to render. (Default (14, 16))
    class_cmap : matplotlib.colors.Colormap, optional
        A colormap to apply for labelled tokens. If not provided, will use this module's
        get_default_cmap() function.
    render_forms : bool, optional
        Set =True to draw field key-value pair geometries. This may be pretty noisy to try and
        display on the same plot as words/annotations. (Default False)
    render_tables : bool, optional
        Set =True to draw table outline and cell geometries. This may be pretty noisy to try and
        display on the same plot as words/annotations. (Default False)
    render_words : bool, optional
        Set =False to suppress word boxes (with labelled classes where present). (Default True)
    show_axes : bool, optional
        Set =False to hide axis ticks (pixel coordinates). (Default True)
    """
    if class_cmap is None:
        class_cmap = get_default_cmap(ann_classes)
    class_colors = [class_cmap(ix, alpha=0.5) for ix in range(len(ann_classes))]

    fig = plt.figure(figsize=figsize)
    img = PIL.Image.open(img_path)
    plt.imshow(img)
    ax = fig.axes[0]
    ax.set_title(img_path.rpartition("/")[2])

    if textract_result is not None:
        if isinstance(textract_result, str):
            # It's a file path: Read the contents
            with open(textract_result, "r") as ftextract:
                textract_result = trp.Document(json.loads(ftextract.read()))
        elif hasattr(textract_result, "pages"):
            pass  # Looks like a trp.Document - use as-is
        else:
            # Should be a trp-able loaded JSON object:
            textract_result = trp.Document(textract_result)

        page = textract_result.pages[page_num - 1]
        print(f"Page: {str(page.geometry)}")

        if render_forms:
            for field in page.form.fields:
                if field.key:
                    ax.add_patch(
                        plt.Polygon(
                            trp_polygon_to_plottable(
                                field.key.geometry.polygon,
                                img.width,
                                img.height,
                            ),
                            fill=True,
                            edgecolor=FIELD_KEY_COLOR,
                            facecolor=FIELD_KEY_COLOR[:3] + (FIELD_KEY_COLOR[3] / 2,),
                            linewidth=3,
                        )
                    )
                if field.value:
                    ax.add_patch(
                        plt.Polygon(
                            trp_polygon_to_plottable(
                                field.value.geometry.polygon,
                                img.width,
                                img.height,
                            ),
                            fill=True,
                            edgecolor=FIELD_VALUE_COLOR,
                            facecolor=FIELD_VALUE_COLOR[:3] + (FIELD_VALUE_COLOR[3] / 2,),
                            linewidth=3,
                        )
                    )

        if render_tables:
            for table in page.tables:
                ax.add_patch(
                    plt.Polygon(
                        trp_polygon_to_plottable(table.geometry.polygon, img.width, img.height),
                        fill=True,
                        edgecolor=TABLE_OUTLINE_COLOR,
                        facecolor=TABLE_OUTLINE_COLOR[:3] + (TABLE_OUTLINE_COLOR[3] / 4,),
                        linewidth=5,
                    )
                )
                for row in table.rows:
                    for cell in row.cells:
                        ax.add_patch(
                            plt.Polygon(
                                trp_polygon_to_plottable(
                                    cell.geometry.polygon,
                                    img.width,
                                    img.height,
                                ),
                                fill=True,
                                edgecolor=TABLE_CELL_COLOR,
                                facecolor=TABLE_CELL_COLOR[:3] + (TABLE_CELL_COLOR[3] / 4,),
                                linewidth=3,
                            )
                        )

        if render_words:
            for line in page.lines:
                for word in line.words:
                    word_box = word.geometry.boundingBox
                    abs_word_left = word_box.left * img.width
                    abs_word_top = word_box.top * img.height
                    abs_word_width = word_box.width * img.width
                    abs_word_height = word_box.height * img.height
                    abs_word_area = abs_word_width * abs_word_height
                    matched_annotations = []
                    for ann in annotations:
                        isect_left = max(abs_word_left, ann["left"])
                        isect_top = max(abs_word_top, ann["top"])
                        isect_right = min(
                            abs_word_left + abs_word_width,
                            ann["left"] + ann["width"],
                        )
                        isect_bottom = min(
                            abs_word_top + abs_word_height,
                            ann["top"] + ann["height"],
                        )
                        isect_area = max(0, isect_right - isect_left) * max(
                            0, isect_bottom - isect_top
                        )
                        if isect_area >= (abs_word_area / 2):
                            matched_annotations.append(ann)

                    if len(matched_annotations) > 1:
                        word_color = err_color
                        print("ERROR: Word matched multiple annotations")
                        print(str(word_box))
                        print(matched_annotations)
                    elif len(matched_annotations) > 0:
                        word_color = class_colors[matched_annotations[0]["class_id"]]
                    else:
                        word_color = plain_color

                    if "PredictedClass" in word._block:
                        predicted_class = word._block["PredictedClass"]
                        if predicted_class > len(ann_classes):
                            facecolor = plain_color[:3] + (plain_color[3] / 2,)
                        else:
                            facecolor = class_cmap(predicted_class, alpha=0.3)
                    else:
                        facecolor = None

                    word_rect = plt.Rectangle(
                        (word_box.left * img.width, word_box.top * img.height),
                        word_box.width * img.width,
                        word_box.height * img.height,
                        fill=bool(facecolor),
                        edgecolor=word_color,
                        facecolor=facecolor,
                        linewidth=1.5,
                    )
                    ax.add_patch(word_rect)

    for ann in annotations:
        ann_rect = plt.Rectangle(
            (ann["left"], ann["top"]),
            ann["width"],
            ann["height"],
            fill=False,
            edgecolor=class_colors[ann["class_id"]],
            linewidth=2.5,
        )
        ax.add_patch(ann_rect)

    if len(ann_classes):
        # Create a legend:
        # (We need something to tag the legend to, so will create a dummy patch for each class)
        dummy_patches = []
        dummy_patches.append(
            plt.Rectangle((0, 0), 0, 0, fill=False, edgecolor=plain_color, linewidth=1.5)
        )
        for clscolor in class_colors:
            dummy_patches.append(
                plt.Rectangle((0, 0), 0, 0, fill=False, edgecolor=clscolor, linewidth=1.5),
            )
        dummy_patches.append(
            plt.Rectangle((0, 0), 0, 0, fill=False, edgecolor=err_color, linewidth=1.5)
        )
        ax.legend(
            dummy_patches,
            ["UNLABELLED"] + ann_classes + ["ERROR"],
            loc="center left",
            bbox_to_anchor=(1.0, 0.5),
            fancybox=True,
            shadow=True,
            title="Token Label",
        )

    ax.axes.xaxis.set_visible(show_axes)
    ax.axes.yaxis.set_visible(show_axes)
    return fig


def local_paths_from_manifest_item(
    item: Dict,
    imgs_s3key_prefix: str,
    textract_s3key_prefix: Optional[str] = None,
    imgs_local_prefix=None,
    textract_local_prefix=None,
) -> Dict:
    """Translate Textract+image manifest S3 URIs to local paths

    Arguments
    ---------
    item : Dict
        Loaded JSON line from a manifest file.
    imgs_s3key_prefix : str
        The S3 key prefix (key only, not incl. e.g. s3://bucket-name/) where page images are stored
    textract_s3key_prefix : str, optional
        The S3 key prefix (key only, not incl. e.g. s3://bucket-name/) where Textract results for
        documents are stored. (Default "")
    imgs_local_prefix : str, optional
        The local folder to map to `imgs_s3key_prefix`. (Default "")
    textract_local_prefix : str, optional
        The local folder to map to `textract_s3key_prefix`. (Default "")

    Returns
    -------
    result : Dict
        Containing key `image` (str) and `textract` (None or str). Local paths to the image and the
        Textract result for this given `item` from the manifest.
    """
    img_s3uri = item["source-ref"]
    img_s3bucket, _, img_s3key = img_s3uri[len("s3://") :].partition("/")
    img_path = (imgs_local_prefix or "") + img_s3key[len(imgs_s3key_prefix) :]

    textract_s3uri = item.get("textract-ref")
    if textract_s3uri:
        textract_s3bucket, _, textract_s3key = textract_s3uri[len("s3://") :].partition("/")
        textract_path = (textract_local_prefix or "") + textract_s3key[
            len(textract_s3key_prefix or "") :
        ]
    else:
        textract_path = None

    return {"image": img_path, "textract": textract_path}


def draw_from_manifest_item(
    item,
    ann_field_name,
    ann_classes,
    imgs_s3key_prefix,
    textract_s3key_prefix=None,
    imgs_local_prefix=None,
    textract_local_prefix=None,
    **kwargs,
):
    ann_result = item.get(ann_field_name)
    if not ann_result:
        print(f"WARNING: Couldn't find annotations at '{ann_field_name}' on manifest item")
        annotations = []
    else:
        annotations = ann_result.get("annotations")
        if not annotations:
            print(f"WARNING: item.{ann_field_name}.annotations not found")
            annotations = []

    img_path, textract_path = itemgetter("image", "textract")(
        local_paths_from_manifest_item(
            item,
            imgs_s3key_prefix,
            textract_s3key_prefix=textract_s3key_prefix,
            imgs_local_prefix=imgs_local_prefix,
            textract_local_prefix=textract_local_prefix,
        ),
    )
    return draw_smgt_annotated_page(
        img_path,
        ann_classes,
        annotations=annotations,
        textract_result=textract_path,
        page_num=item.get("page-num", 1),
        **kwargs,
    )


def draw_from_manifest_items(
    items,
    *args,
    **kwargs,
):
    def draw(ix):
        draw_from_manifest_item(items[ix], *args, **kwargs)
        plt.show()

    return widgets.interact(
        draw,
        ix=widgets.IntSlider(min=0, max=len(items) - 1, step=1, value=0, description="Example:"),
    )


def draw_thumbnails_response(
    res: Union[np.ndarray, np.lib.npyio.NpzFile],
    figsize: Tuple[int, int] = (4, 4),
    n_cols: int = 5,
) -> None:
    """Plot results from a thumbnails endpoint request

    Arguments
    ---------
    res :
        Result from the endpoint (after deserializing with numpy.load)
    figsize :
        Reference size for an individual plot
    n_cols :
        Number of columns added for multi-page plots
    """
    data = res
    data_type = None

    if isinstance(res, np.lib.npyio.NpzFile):
        # A numpy archive
        if "images" in res:
            data = res["images"]
        elif "image" in res:
            data = res["image"]
        else:
            raise ValueError("Got .npz archive containing neither 'images' nor 'image'")

    if isinstance(data, np.ndarray):
        # A plain numpy array
        if data.dtype.kind == "S":
            data_type = "png-bytes"
        else:
            data_type = "pixel-array"
    else:
        raise ValueError(f"Expected a numpy array or .npz archive but got: {type(data)}")

    if data_type == "png-bytes":
        n_images = len(data)
        n_max_subplots = min(n_cols, n_images)
        fig = None
        axes = None
        for ix, bstr in enumerate(data):
            ix_subplot = ix % n_max_subplots
            if ix_subplot == 0:
                n_subplots = min(n_max_subplots, n_images - ix)
                fig, axes = plt.subplots(
                    ncols=n_subplots,
                    figsize=(figsize[0] * n_subplots, figsize[1]),
                )
            ax = axes[ix_subplot] if hasattr(axes, "__getitem__") else axes
            with io.BytesIO(bstr) as f:
                ax.imshow(np.array(PIL.Image.open(f)))
                ax.set_title(f"Page {ix + 1}")
        plt.show()
    else:
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(data)
        ax.set_title("Single Image")
        plt.show()
