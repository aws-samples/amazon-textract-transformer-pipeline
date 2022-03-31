<!-- Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
  SPDX-License-Identifier: MIT-0 -->
<!-- Vue CE component for a single-valued text input with 'present' checkbox
  The text entry is required if and only if the 'present' box is ticked. Both the boolean presence
  value and the text should be passed through to the task result data.

  This component demonstrates *both* registering form outputs via the global `store` +
  <object-value-input> pattern, *and* directly registering as a form-associated Cusom Element
  (which as discussed only worked end-to-end with <crowd-form> in a subset of tested browsers).
  
  This component should be used through the wrapper in index.ts, not directly from this file. -->
<script setup lang="ts">
// External Dependencies:
import type { IElementInternals } from "element-internals-polyfill";
import { onBeforeMount, onBeforeUnmount, onMounted, ref, toRef, watch } from "vue";

// Local Dependencies:
import { LABEL_CLASS_COLORS } from "../../util/colors";
import { addValidateHandler, store } from "../../util/store";

// Template can access constants defined here, but not see imports directly:
const CLASS_COLORS = LABEL_CLASS_COLORS;

const props = defineProps<{
  /**
   * Name of the field/entity type
   */
  name: string;
  /**
   * Input, model-detected value of the field (not a 2-way binding)
   */
  value: string;
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

/**
 * SageMaker <crowd-input> and <crowd-checkbox> custom elements support the form validate() method
 */
interface ValidatableHTMLElement extends HTMLElement {
  validate: () => boolean;
}

const presentCheckbox = ref<ValidatableHTMLElement>();
const valueInput = ref<ValidatableHTMLElement>();
const root = ref<HTMLElement>();

// In Vue.js, data flow is one-directional on 'props'. Updates are handled by the child component
// emitting an event and the parent updating the prop value. We can't rely on that in this case
// because this component may be used without a parent Vue component - so instead treat the props
// just as inputs and create (reactive) internal states for anything we might want to update:
const isPresent = ref(false);
const valueState = ref("");

let hiddenValue = "";
let internals: IElementInternals | undefined;
let unsubValidate: undefined | (() => void);

onBeforeMount(() => {
  // Note that ("  0   " > 0) == false and ("  1   " > 0) == true, so this comparison should be
  // pretty safe even if the template includes whitespace or type conversion hasn't happened yet:
  isPresent.value = props.optional && props.optional > 0 ? !!props.value : true;
  valueState.value = props.value;
  unsubValidate = addValidateHandler(onValidate);
  console.log(`${props.name} isPresent: ${isPresent.value}`);
});

onMounted(() => {
  // Register this component's states on the global form data store (workaround for shadow-DOM
  // components not being picked up by <crowd-form>):
  store[`field-present-${props.name}`] = isPresent;
  store[`field-value-${props.name}`] = valueState;

  // Look up the actual host <custom-field> element this component is mounted to, check the element
  // constructor (see index.ts) attached the ElementInternals, and if so set the element's form
  // value to our reactive value state:
  try {
    const hostNode = (root.value?.getRootNode() as ShadowRoot | undefined)?.host as HTMLElement & {
      _internals: IElementInternals;
    };
    internals = hostNode?._internals;
    // Unlike <object-value-input> (which has post-processing), we can't just plug the reactive Vue
    // state in as the form value for a custom component (the result renders "[object Object]" in
    // the form). So instead we'll set the non-reactive value here and watch for changes below:
    if (internals) internals.setFormValue(valueState.value);
  } catch (err) {
    console.warn(
      [
        "Failed to attach control shadow root to parent form. You'll need to rely on store",
        "pattern fallback for transmitting results to SageMaker.",
      ].join(" "),
      (err as Error).stack
    );
  }
});

onBeforeUnmount(() => {
  // De-register from form data state store and validation listeners on component delete:
  delete store[`field-present-${props.name}`];
  delete store[`field-value-${props.name}`];
  if (unsubValidate) unsubValidate();
});

watch(toRef(props, "name"), (newName, oldName) => {
  // Keep global state store in sync with any incoming name prop changes:
  console.log(`Control name changed from ${oldName} to ${newName}`);
  store[`field-present-${newName}`] = isPresent;
  store[`field-value-${newName}`] = valueState;
  delete store[`field-present-${oldName}`];
  delete store[`field-value-${oldName}`];
});

// Keep value state in sync with any incoming prop changes:
watch(toRef(props, "value"), (newValue) => {
  valueState.value = newValue;
});

// Keep ElementInternals FormData in sync with component state changes:
watch(valueState, (newValue) => {
  if (internals && isPresent.value) internals.setFormValue(newValue);
});
watch(isPresent, (newValue) => {
  if (internals) internals.setFormValue(newValue ? valueState.value : "");
});

/**
 * Handle toggling of the "field is present" checkbox
 *
 * When True->False, clear the text field value but cache it in memory. When False->True, restore
 * the cached value.
 */
function onCheckBoxChange() {
  console.log(`Updating checkbox from ${isPresent.value}`);
  if (isPresent.value) {
    hiddenValue = valueState.value;
    valueState.value = "";
    isPresent.value = false;
  } else {
    valueState.value = hiddenValue;
    isPresent.value = true;
  }
}

/**
 * When the page form is submitted, validate both the text input and checkbox
 *
 * The crowd-* input components will automatically highlight themselves if validation fails.
 *
 * @returns true if submission should proceed, false if there's an issue preventing submission.
 */
function onValidate(): boolean {
  console.log(`Validating ${props.name}`);
  // (`or false` here to convert any undefined to false for TypeScript, but not expected)
  return (presentCheckbox.value?.validate() && valueInput.value?.validate()) || false;
}
</script>

<template>
  <!-- Single-Valued Field (all values in Liquid are truthy except nil & false) -->
  <div ref="root" class="field-container">
    <div style="display: flex; align-items: center">
      <!-- Why do we bind `checked` to the prop `value` and not the state `isPresent` here? Because
        the cbox component doesn't seem to like the double-update caused by the proper binding. -->
      <crowd-checkbox
        ref="presentCheckbox"
        :name="`field-present-${name}`"
        @change="onCheckBoxChange()"
        style="flex: 0 0 auto; padding-top: 16px"
        value="checked"
        :checked="isPresent ? '' : null"
      ></crowd-checkbox>
      <crowd-input
        ref="valueInput"
        :name="`field-value-${name}`"
        :label="name"
        style="flex: 1 0 auto"
        v-model="valueState"
        :required="isPresent"
        :disabled="!isPresent || null"
      ></crowd-input>
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
    <slot></slot>
  </div>
</template>

<style scoped lang="scss">
@use "../../assets/base.scss" as stylebase;

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
</style>
