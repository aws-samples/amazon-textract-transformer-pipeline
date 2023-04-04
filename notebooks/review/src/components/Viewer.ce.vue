<!-- Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
  SPDX-License-Identifier: MIT-0 -->
<!-- Vue component to render a PDF document using PDF.js
  In some cases you may be able to substitute this with a simple <iframe type="application/pdf">,
  but:

    1. This doesn't seem to work in some browsers with strict default security restrictions (since
       the UI itself is already running in an iframe)
    2. A custom component gives potential for implementing custom annotation tools over documents
  -->
<script setup lang="ts">
// legacy/build/pdf.js exports the core `pdfjsLib`, and legacy/web/pdf_viewer.js exports the
// `pdfjsViewer` namespace - to globals when run from CDN as in our HTML entrypoints.
import * as pdfjsLib from "pdfjs-dist/legacy/build/pdf";
import * as pdfjsViewer from "pdfjs-dist/legacy/web/pdf_viewer";
import type { Ref } from "vue";
import { onBeforeUnmount, onMounted, ref } from "vue";

// Local Dependencies:
import type { AnnotationLayerElement } from "./PdfPageAnnotationLayer.ce.vue";
import type {
  Detection,
  ModelResult,
  ModelResultMultiField,
  ModelResultSingleField,
} from "../util/model";
import { addValidateHandler } from "../util/store";

// Need to explicitly set this for PDFJS to find it:
pdfjsLib.GlobalWorkerOptions.workerSrc = `//cdn.jsdelivr.net/npm/pdfjs-dist@${pdfjsLib.version}/legacy/build/pdf.worker.js`;
// And likewise we need CMaps to translate fonts for non-native locales (foreign docs):
const CMAP_URL = `//cdn.jsdelivr.net/npm/pdfjs-dist@${pdfjsLib.version}/cmaps/`;

const props = defineProps<{
  /**
   * Location / URL of the PDF to load
   *
   * To generate a pre-signed S3 URL from input S3 URI with SMGT/A2I liquid, you'll probably want
   * to set this as something like {{ task.input.TaskObject | grant_read_access }}
   */
  src: string;
}>();

const containerRef: Ref<HTMLDivElement | undefined> = ref();
const viewerRef: Ref<HTMLDivElement | undefined> = ref();

const error = ref(false);
const pdfEventBus = new pdfjsViewer.EventBus();
let viewer: pdfjsViewer.PDFViewer;
let unsubValidate: undefined | (() => void);

let detectionsByPage: Detection[][] = [];

pdfEventBus.on("pagerendered", (data: { pageNumber: number; source: pdfjsViewer.PDFPageView }) => {
  // When a page is (re)-rendered, check if bounding box annotation layer has been initialized and
  // create it if required:
  const pageDetections = detectionsByPage[data.pageNumber - 1];
  if (
    pageDetections &&
    pageDetections.length &&
    !data.source.div.querySelector("custom-page-annotation-layer")
  ) {
    console.log(`Adding ${pageDetections.length} bboxes to PDF page ${data.pageNumber}`);
    const annEl = document.createElement(
      "custom-page-annotation-layer"
    ) as unknown as AnnotationLayerElement;
    annEl.detections = pageDetections;
    data.source.div.appendChild(annEl);
  }
});

onMounted(() => {
  // Load in detections from global (window) object provided in Liquid:
  detectionsByPage = [];
  const rawData = window.taskData.taskInput as { ModelResult: ModelResult };
  const rawFields = rawData.ModelResult.Fields;
  Object.keys(rawFields).forEach((fieldName) => {
    // Check single-value field case first:
    let fieldDets = (rawFields[fieldName] as ModelResultSingleField).Detections;
    if (!fieldDets) {
      // Try to retrieve multi-value field detections:
      const fieldMultiValues = (rawFields[fieldName] as ModelResultMultiField).Values;
      // (This will fail to [] if .Values doesn't exist either)
      fieldDets = Array.prototype.concat.apply(
        [],
        fieldMultiValues?.map((v) => v.Detections) || []
      );
    }
    fieldDets.forEach((det: Detection) => {
      if (typeof det.PageNum !== "number") {
        console.error("Detection has non-numeric PageNum: Skipping to avoid infinite loop", det);
        return;
      }
      while (detectionsByPage.length < det.PageNum) {
        detectionsByPage.push([]);
      }
      detectionsByPage[det.PageNum - 1].push(det);
    });
  });

  // Subscribe to form validation events:
  unsubValidate = addValidateHandler(onValidate);
  try {
    const containerEl = containerRef.value;
    const viewerEl = viewerRef.value;
    if (!(containerEl && viewerEl)) {
      // This should never happen - really just to keep TypeScript happy
      throw new Error(
        `Container and/or viewer element missing from template: ${containerEl}, ${viewerEl}`
      );
    }
    viewer = new pdfjsViewer.PDFViewer({
      container: containerEl,
      eventBus: pdfEventBus,
      l10n: pdfjsViewer.NullL10n,
      // Type checks won't let us leave linkService out, but also don't like null:
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      linkService: null as any,
      viewer: viewerEl,
      textLayerMode: 0, // No text layer
    });
    pdfEventBus.on("pagesinit", function () {
      // On initialisation, default zoom to page width:
      console.log("pagesinit Event: Zooming to page-width");
      viewer.currentScaleValue = "page-width";
    });

    // In previous template the URL from Liquid template seemed to get escaped and require e.g.
    // `.replaceAll("&amp;", "&")` - but seems not to be the case now.
    const loadingTask = pdfjsLib.getDocument({
      url: props.src,
      cMapPacked: true,
      cMapUrl: CMAP_URL, // Enable character mapping (font translation)
    });
    loadingTask.promise.then(
      (doc) => {
        console.log(`Loaded document with ${doc.numPages} page(s)`);
        // Pad out detectionsByPage to make sure it matches document length:
        while (detectionsByPage.length < doc.numPages) {
          detectionsByPage.push([]);
        }

        viewer.setDocument(doc);
      },
      (reason) => {
        console.error("Document load failed", reason);
        error.value = true;
      }
    );
  } catch (err) {
    console.error("Failed to initialize PDF viewer", err);
    error.value = true;
  }
});

// Remove form validation listener on component destroy
onBeforeUnmount(() => {
  if (unsubValidate) unsubValidate();
});

/**
 * Prevent submission of the <crowd-form> if the PDF failed to load
 *
 * We don't want the worker to submit if they haven't actually seen the document
 */
function onValidate() {
  if (error.value) {
    console.error("Preventing form submission: Document did not load");
    return false;
  } else {
    return true;
  }
}

function viewerZoomIn() {
  viewer.currentScale *= 1.1;
}

function viewerZoomOut() {
  viewer.currentScale /= 1.1;
}
</script>

<template>
  <div ref="containerRef" class="pdf-container">
    <div class="pdf-floating-toolbar">
      <crowd-button class="pdf-zoom-button" @click="viewerZoomOut()"
        ><iron-icon icon="zoom-out" alt="Zoom Out"
      /></crowd-button>
      <crowd-button class="pdf-zoom-button" @click="viewerZoomIn()"
        ><iron-icon icon="zoom-in" alt="Zoom In"
      /></crowd-button>
    </div>
    <div ref="viewerRef" class="pdfViewer"></div>
    <div class="load-error" :style="{ display: error ? '' : 'none' }">
      <iron-icon icon="error-outline"></iron-icon>
      <h3>Sorry, failed to load the PDF document for this task!</h3>
      <p>Try refreshing the page and contact the task administrator if the error persists.</p>
      <p>${src}</p>
    </div>
  </div>
</template>

<style lang="scss">
// Elements dynamically created by pdf.js won't see our scoped styles:
.page {
  margin: 8px auto;
  position: relative;

  .textLayer {
    display: none; // Above disabled text layers anyway, but just in case they accidentally show
  }
}

custom-page-annotation-layer {
  position: absolute;
  top: 0px;
  left: 0px;
  height: 100%;
  width: 100%;
}
</style>

<style scoped lang="scss">
// But scope down the component styling themselves:
@use "../assets/base.scss" as stylebase;

*,
*::before,
*::after {
  box-sizing: border-box;
  margin: 0;
  position: relative;
  font-weight: normal;
}

.pdf-container {
  background-color: #ddd;
  height: 100%;
  width: 100%;
  position: absolute;
  overflow: auto;
  // PDF.js v3 viewer seems to want to handle horizontal padding within the page divs themselves:
  padding: 8px 0px;
}

.pdf-floating-toolbar {
  position: sticky;
  left: 0px;
  top: 0px;
  text-align: center;
  width: 100%;
  z-index: 10;
  margin-bottom: 5px;

  > crowd-button {
    margin: 0px 2px;
    opacity: 0.7;
    transition: 0.3s;

    &:hover {
      opacity: 0.9;
    }
  }
}

.pdfViewer {
  // Since updating to PDF.js v3, without this border the viewer seems not to recognise that it's
  // visible and therefore to never render.
  // TODO: Can we find some alternative to get rid of this?
  border: 1px solid transparent;
}

.pdfViewer .page {
  margin-bottom: 10px;
}

.load-error {
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  padding: 40px;
  text-align: center;

  iron-icon {
    color: var(--color-error);
    height: 40px;
    width: 40px;
  }

  h3 {
    color: var(--color-error);
  }

  h3,
  p {
    margin-top: 5px;
  }
}
</style>
