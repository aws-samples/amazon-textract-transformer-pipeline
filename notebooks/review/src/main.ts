// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0
/**
 * Main script entrypoint for Vue.js example A2I UI template.
 */
// External Dependencies:
import { createApp, defineCustomElement } from "vue";
import type { App as VueApp } from "vue";
// If needed, you could also use TRP.js here or in any of the components - as:
// import { TextractDocument } from "amazon-textract-response-parser";

// Local Dependencies:
import App from "./App.vue";
import FieldMultiValue from "./components/FieldMultiValue.ce.vue";
import FieldSingleValue from "./components/FieldSingleValue.ce";
import MultiFieldValue from "./components/MultiFieldValue.ce.vue";
import ObjectValueInputElement from "./components/ObjectValueInput";
import PdfPageAnnotationLayer from "./components/PdfPageAnnotationLayer.ce.vue";
import Viewer from "./components/Viewer.ce.vue";
import type { ModelResult } from "./util/model";

declare global {
  interface Window {
    app: VueApp;
    // As per the setup script in index.html / index-noliquid.html:
    taskData: {
      taskObject: string;
      taskInput: { ModelResult: ModelResult };
    };
  }
}

// Register our Custom Element components (not needed for normal Vue components):
customElements.define("object-value-input", ObjectValueInputElement);
customElements.define("custom-field", FieldSingleValue);
const FieldMultiValueElement = defineCustomElement(FieldMultiValue);
customElements.define("custom-field-multivalue", FieldMultiValueElement);
const MultiFieldValueElement = defineCustomElement(MultiFieldValue);
customElements.define("custom-multifield-value", MultiFieldValueElement);
const PdfPageAnnotationLayerElement = defineCustomElement(PdfPageAnnotationLayer);
customElements.define("custom-page-annotation-layer", PdfPageAnnotationLayerElement);
const ViewerElement = defineCustomElement(Viewer);
customElements.define("custom-viewer", ViewerElement);

// Mount the Vue app (not that the app itself does very much):
createApp(App).mount("#app");
