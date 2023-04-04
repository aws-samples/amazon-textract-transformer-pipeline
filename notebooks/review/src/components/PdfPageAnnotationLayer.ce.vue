<!-- Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
  SPDX-License-Identifier: MIT-0 -->
<!-- Component to draw SVG-based entity detection bounding boxes over a rendered PDF page -->
<script lang="ts">
import type { Detection } from "../util/model";
import type { VueElement } from "vue";

/**
 * Typing for the Vue Custom Element defined by this component
 *
 * It would be nice if we could declare a single props interface and share it between this and the
 * defineProps() call below... But at the time of writing seems like that's not possible due to
 * https://github.com/vuejs/core/issues/4294 - so watch out for the duplication here:
 */
export type AnnotationLayerElement = VueElement & {
  /**
   * Field/entity detections for this page
   */
  detections: Detection[];
};
</script>

<script setup lang="ts">
// External Dependencies:
import { onBeforeUnmount, onMounted, ref } from "vue";

// Local Dependencies:
import { LABEL_CLASS_COLORS } from "../util/colors";

const CLASS_COLORS = LABEL_CLASS_COLORS;

// Note: Keep in sync with AnnotationLayerElement above
// Stop ESLint complaining when props are only used in template:
// eslint-disable-next-line @typescript-eslint/no-unused-vars
const props = defineProps<{
  /**
   * Field/entity detections for this page
   */
  detections: Detection[];
}>();

const root = ref<SVGElement>();
const rootHeight = ref(1);
const rootWidth = ref(1);

// Although it would be possible to just set `viewBox="0 0 1 1" preserveAspectRatio="none"` on our
// SVG element and avoid any dealings with absolute element size, this would mean that everything
// from rectangle stroke widths to any text we might want to add to the annotations would be
// stretched by some unknown aspect ratio. So instead, we'll use actual pixel sizes which means
// needing to observe the overall size of this element:
let resizeObserver: ResizeObserver;
let warnedMissingCanvas = false; // We'll raise this warning at most once

onMounted(() => {
  const rootEl = root.value;
  if (!rootEl) throw new Error("root SVG el missing from mounted template");
  resizeObserver = new ResizeObserver(onResize);
  resizeObserver.observe(rootEl);
});

onBeforeUnmount(() => {
  const rootEl = root.value;
  if (resizeObserver && rootEl) {
    resizeObserver.unobserve(rootEl);
  }
});

/**
 * Set annotation overlay SVG sizing (in response to parent/page resize events)
 *
 * Since v3 upgrade, it seems PDF.js PDFViewer sets actual page size via the core
 * `div.page > div.canvaswrapper > canvas` element instead of on the parent `div.page` to which
 * this custom component gets inserted. As a consequence the `div.page` may include arbitrary extra
 * space and we can't rely on just height=100%, width=100% to get this element the same size as the
 * PDF page.
 *
 * The result is the horrible encapsulation-breaking code below, which traverses up from `rootEl`
 * (the SVG), to `hostEl` (the `<custom...>` element you bound this component to), to `hostEl`'s
 * parent (assumed to be a PDF.js `div.page`) and from there tries to find the `<canvas>`.
 *
 * TODO: A better solution for drawing an SVG overlay that matches the rendered page size+position!
 */
function onResize() {
  const rootEl = root.value;
  if (rootEl) {
    const hostEl = (rootEl.getRootNode() as ShadowRoot).host;
    const pageCanvas = (hostEl.parentElement as HTMLElement).querySelector("canvas");
    if (pageCanvas) {
      rootHeight.value = pageCanvas.clientHeight;
      rootWidth.value = pageCanvas.clientWidth;
      rootEl.style.height = `${pageCanvas.clientHeight}px`;
      rootEl.style.width = `${pageCanvas.clientWidth}px`;
    } else {
      if (!warnedMissingCanvas) {
        console.warn(
          "Couldn't find PDF.js page <canvas> element to annotate - box sizes may be skewed"
        );
        // This function is called often, so warn at most once:
        warnedMissingCanvas = true;
      }
      rootHeight.value = rootEl.clientHeight;
      rootWidth.value = rootEl.clientWidth;
      rootEl.style.height = "100%";
      rootEl.style.width = "100%";
    }
  }
}
</script>

<template>
  <svg
    ref="root"
    class="page-annotation-layer"
    :viewBox="`0 0 ${rootWidth} ${rootHeight}`"
    preserveAspectRatio="none"
  >
    <g
      v-for="(det, ixDet) in detections"
      :key="ixDet"
      :transform="`translate(${det.BoundingBox.Left * rootWidth}  ${
        det.BoundingBox.Top * rootHeight
      })`"
    >
      <rect
        class="detection-box"
        :width="det.BoundingBox.Width * rootWidth"
        :height="det.BoundingBox.Height * rootHeight"
        :fill="CLASS_COLORS[det.ClassId]"
        :stroke="CLASS_COLORS[det.ClassId]"
      />
    </g>
  </svg>
</template>

<style scoped lang="scss">
rect.detection-box {
  fill-opacity: 0.1;
  stroke-opacity: 0.8;
  stroke-width: 2;

  // Increase transparency on hover so users can see what's behind more clearly if they want:
  &:hover {
    fill-opacity: 0;
    stroke-opacity: 0.25;
  }
}

svg.page-annotation-layer {
  height: 100%;
  width: 100%;
}
</style>
