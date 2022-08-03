<!-- Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
  SPDX-License-Identifier: MIT-0 -->
<!-- Vue CE component for an ordered multi-value text input
  In this custom element, initial values are transcluded into the template via <slot> (via Liquid
  templating) rather than bound to a reactive data property. This choice makes managing child
  events and state quite different (less direct data binding, more functional/event interfaces)
  from a typical Vue component. -->
<script setup lang="ts">
// External Dependencies:
import { nextTick, onBeforeUnmount, onMounted, ref } from "vue";

// Local Dependencies:
import type { MountedMultiFieldValueElement } from "./MultiFieldValue.ce.vue";
import { REMOVE_EVENT_TYPE } from "./MultiFieldValue.ce.vue";
import { LABEL_CLASS_COLORS } from "../util/colors";
import { addValidateHandler, store } from "../util/store";

// Templates can access constants we declare here, but not directly see imports, so:
const CLASS_COLORS = LABEL_CLASS_COLORS;
const VALUE_REMOVE_EVENT_TYPE = REMOVE_EVENT_TYPE;
const VALUE_REMOVABLE_ATTR_NAME = "removable";

const props = defineProps<{
  /**
   * Name of the field/entity type
   */
  name: string;
  /**
   * 0-1 confidence score of the model-detected value
   */
  confidence: number;
  /**
   * This is actually a boolean input coded as a number so it can be injected via Liquid while
   * keeping the un-resolved template valid HTML (which is required for our build toolchain):
   *
   *     <!-- TEMPLATE VALID LIQUID BUT INVALID HTML: -->
   *     <my-custom-component
   *       {% if fieldkv[1].Optional == true %}
   *         optional
   *       {% endif %}
   *     ></my-custom-component>
   *     <!-- TEMPLATE VALID LIQUID AND VALID HTML: -->
   *     <my-custom-component
   *       optional="{% if fieldkv[1].Optional == true %} 1 {% else %} 0 {% endif %}"
   *     ></my-custom-component>
   */
  optional?: number;
  /**
   * Index number of this field's model class (for color-keying with document annotations)
   */
  classId?: number;
}>();

const afterMultiValuesEl = ref<HTMLElement>();
const valuesContainerEl = ref<HTMLElement>();

// In Vue.js, data flow is one-directional on 'props'. Updates are handled by the child component
// emitting an event and the parent updating the prop value. We can't rely on that in this case
// because this component may be used without a parent Vue component - so instead treat the props
// just as inputs and create (reactive) internal states for anything we might want to update:
const isPresent = ref(true);

let unsubValidate: undefined | (() => void);

/**
 * List MultiFieldValue elements within this component
 *
 * By taking the decision to handle listing out values in Liquid templating, instead of in Vue, we
 * end up with inner components transcluded in via <slot> instead of bound into this component's
 * layout by reactive data (like an array of objects). Consequences of this are that every time we
 * want to do something with the child elements:
 *
 *  1. We have to query the element tree to collect the current list of them, and
 *  2. We have to interact with them as elements, rather than Vue components
 */
function listValueEls(): Array<MountedMultiFieldValueElement> {
  const valuesContainer = valuesContainerEl.value;
  if (!valuesContainer) {
    throw new Error("Couldn't find values container in FieldMultiValue component template");
  }

  const directEls = Array.from(valuesContainer.querySelectorAll("custom-multifield-value"));
  let transcludedEls: Element[] = [];
  const valuesSlot = valuesContainer.querySelector("slot");
  if (!valuesSlot) {
    throw new Error("Couldn't find <slot> in FieldMultiValue component template");
  }
  valuesSlot.assignedElements().forEach((el) => {
    if (el.tagName.toLowerCase() === "custom-multifield-value") {
      transcludedEls.push(el);
    } else {
      transcludedEls = transcludedEls.concat(
        Array.from(el.querySelectorAll("custom-multifield-value"))
      );
    }
  });
  return directEls.concat(transcludedEls) as MountedMultiFieldValueElement[];
}

/**
 * Handle toggling of the "field is present" checkbox
 *
 * Note need to listen to the 'change' event not 'click', because otherwise we'll be racing with
 * the component's own internal state handling so could end up double-toggling.
 */
function onCheckBoxChange() {
  if (isPresent.value) {
    isPresent.value = false;
  } else {
    isPresent.value = true;
  }
}

/**
 * Remove a value at given index (deleting the value component from the DOM)
 */
function onRemoveValueIndex(index: number) {
  console.log(`${props.name} removing value index ${index}`);
  const elsAscendingIndex = listValueEls().sort(
    // Only typed as HTMLElements here, but the concrete components will have 'index' attribute:
    (a, b) => a.index - b.index
  );
  let nRemoved = 0;
  elsAscendingIndex.forEach((el) => {
    const elIndex = parseInt(el.getAttribute("index") || "");
    if (elIndex === index) {
      console.log(`Removing ${el}`);
      el.remove();
      ++nRemoved;
    } else if (elIndex > index) {
      console.log(`Decrementing index ${elIndex}`);
      el.setAttribute("index", (elIndex - 1).toString());
    }
  });
  if (elsAscendingIndex.length - nRemoved < 2) {
    setCanDeleteValues({ enabled: false, els: elsAscendingIndex });
  }
}

/**
 * Native DOM CustomEvent handler for 'remove' events raised by child value input components
 */
function onRemoveEvent(this: HTMLElement, ev: CustomEvent<number[]>) {
  ev.detail.forEach((index) => onRemoveValueIndex(index));
}

/**
 * Append a new value control and update whether values should enable deletion
 */
function onClickAdd() {
  const existingEls = listValueEls();
  const nExistingEls = existingEls.length;
  const afterEl = afterMultiValuesEl.value;
  if (!afterEl) {
    throw new Error("Couldn't find expected afterMultiValuesEl in template");
  }
  const newValueEl = document.createElement("custom-multifield-value");
  newValueEl.setAttribute("name", props.name);
  newValueEl.setAttribute("index", nExistingEls.toString());
  newValueEl.setAttribute("value", "");
  afterEl.parentNode?.insertBefore(newValueEl, afterEl);

  // Defer the update to nextTick to try and make sure the element is mounted by the time we use it
  nextTick(() => {
    setCanDeleteValues({
      enabled: nExistingEls > 0,
      els: (existingEls as Element[]).concat([newValueEl]),
    });
  });
}

/**
 * Update child value controls on whether they should offer the option to remove themselves
 *
 * A value's remove button should only be active when there are 2 or more values remaining.
 *
 * @param config.enabled If not defined, will be calculated based on number of elements present
 * @param config.els If not defined, will be queried from DOM.
 */
function setCanDeleteValues(config?: { enabled?: boolean; els?: Element[] }) {
  if (!config) config = {};
  if (!config.els) config.els = listValueEls();
  if (typeof config.enabled === "undefined") {
    config.enabled = config.els.length > 1;
  }

  console.log(
    `${config.enabled ? "Enabling" : "Disabling"} delete for ${config.els.length} values`
  );
  // Boolean HTMLElement attributes use ""/remove, not true/false:
  if (config.enabled) {
    config.els.forEach((el) => el.setAttribute(VALUE_REMOVABLE_ATTR_NAME, ""));
  } else {
    config.els.forEach((el) => el.removeAttribute(VALUE_REMOVABLE_ATTR_NAME));
  }
}

onMounted(() => {
  // At this point, the 'value' components transcluded in our <slot> exist in DOM but are not
  // guaranteed to have completed their component setup yet... As a result:
  //
  //   1. To check value contents we can access their 'value' DOM attribute but not their .value
  //      state:
  const valueEls = listValueEls();
  const hasValues =
    valueEls.map((el) => el.getAttribute("value") as string | undefined).filter((value) => value)
      .length > 0;

  // Note that ("  0   " > 0) == false and ("  1   " > 0) == true, so this comparison should be
  // pretty safe even if the template includes whitespace or type conversion hasn't happened yet:
  const initIsPresent = props.optional && props.optional > 0 ? hasValues : true;
  isPresent.value = initIsPresent;
  store[`field-present-${props.name}`] = isPresent;

  if (!valueEls.length) onClickAdd();

  // ...and
  //
  //   2. Trying to update their 'removable' attribute right now seems to break reactivity for some
  //      reason on the transcluded components in <slot>. As a hacky workaround just set after a
  //      reasonable timeout... Doesn't matter if this is a bit slow because the default is that
  //      values can't be deleted anyway.
  setTimeout(() => setCanDeleteValues(), 2000);

  // Finally, register to receive form validate() events pre-submit:
  unsubValidate = addValidateHandler(onValidate);
});

/**
 * If this component is deleted, remove it from the <crowd-form> state store and listeners.
 */
onBeforeUnmount(() => {
  delete store[`field-present-${props.name}`];
  if (unsubValidate) unsubValidate();
});

/**
 * Send validation requests to all values, only if the 'present' checkbox is ticked.
 */
function onValidate(): boolean {
  if (isPresent.value) {
    console.log(`Validating field-multival-${props.name}`);
    return listValueEls()
      .map((el) => {
        if (typeof el.validate === "function") {
          return el.validate();
        } else {
          return true;
        }
      })
      .every((x) => x);
  } else {
    return true;
  }
}
</script>

<template>
  <!-- Multi-Value Field -->
  <div class="field-container">
    <div style="display: flex; align-items: center; padding-bottom: 8px">
      <crowd-checkbox
        name="field-present-${name}"
        @change="onCheckBoxChange()"
        style="flex: 0 0 auto; padding-top: 16px"
        value="checked"
        :checked="isPresent ? '' : null"
      ></crowd-checkbox>
      <label class="multivalue-field-label" style="flex: 1 0 auto">${name}</label>
    </div>
    <div
      class="confidence-bar"
      :style="{
        background: `linear-gradient(90deg, var(--color-confidence-bar-on) 0%, var(--color-confidence-bar-on) ${
          confidence * 100
        }%, var(--color-confidence-bar-off) ${
          confidence * 100
        }%, var(--color-confidence-bar-off) 100%)`,
      }"
    >
      ${ Math.round(confidence * 1000) / 10 }%
    </div>
    <div
      ref="valuesContainerEl"
      class="multivalue-field-values"
      @[VALUE_REMOVE_EVENT_TYPE]="onRemoveEvent"
    >
      <slot></slot>
      <div ref="afterMultiValuesEl" class="multivalue-field-adder">
        <div class="spacer"></div>
        <crowd-button class="field-multival-add" @click="onClickAdd()"
          ><iron-icon icon="add-box"
        /></crowd-button>
      </div>
    </div>
  </div>
</template>

<style scoped lang="scss">
@use "../assets/base.scss" as stylebase;

.field-container {
  // Colour checked present box as per the field's class ID colour, if known
  --paper-checkbox-checked-color: v-bind(
    'typeof classId !== "undefined" ? CLASS_COLORS[classId] : "inherit"'
  );
  // Could draw the confidence bar also in the field's color as well, but this gives pretty low
  // contrast on some classes which are low-saturation, and especially the pale ones. E.g. classes
  // 17, 7, 55... So here instead kept the primary UI accent color:
  --color-confidence-bar-on: var(--color-accent);
  /* --color-confidence-bar-on: v-bind(
    'typeof classId !== "undefined" ? CLASS_COLORS[classId] : "var(--color-accent)"'
  ); */
  --color-confidence-bar-off: rgba(92, 92, 92, 0.5);

  border-left: v-bind(
    "typeof classId !== 'undefined' ? `3px solid ${CLASS_COLORS[classId]}` : 'none'"
  );
  margin: 8px 0px;
  padding-left: v-bind("typeof classId !== 'undefined' ? '6px' : '0px'");
}

.confidence-bar {
  @include stylebase.confidence-bar;
}

.multivalue-field-label {
  border-bottom: 1px solid transparent;
  padding-bottom: 2px;
  padding-top: 16px;
}

.multivalue-field-values {
  padding-left: 26px;
}

.multivalue-field-adder {
  align-items: center;
  display: flex;
  padding-top: 10px;

  > crowd-button {
    flex: 0 0 auto;
  }
}

.spacer {
  flex: 1 0 auto;
}
</style>
