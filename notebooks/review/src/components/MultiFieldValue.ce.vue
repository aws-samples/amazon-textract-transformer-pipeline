<!-- Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
  SPDX-License-Identifier: MIT-0 -->
<!-- Vue CE component for an individual item in a multi-value text input
  Since this is a web component transcluded into the parent by Liquid template (rather than a Vue
  component), the integration with FieldMultiValue is a little more complex than might otherwise
  be the case. We cannot simply bind to incoming prop data and publish change requests via Vue
  events (https://vuejs.org/guide/essentials/forms.html).
  -->
<script lang="ts">
type RemoveEventType = "remove";
export const REMOVE_EVENT_TYPE: RemoveEventType = "remove";
type AddedEventType = "value-added";
export const ADDED_EVENT_TYPE: AddedEventType = "value-added";

/**
 * Type interface for an element mounted with this component
 *
 * This component augments the Element onMounted (with a validate() method for checking form
 * submission if and only if the parent state makes it relevant) - so we'll define a type
 * interface here to make referencing it in TypeScript easier.
 */
export interface MountedMultiFieldValueElement extends Element {
  index: number;
  name: string;
  removable?: boolean;
  value: string;
  validate: () => boolean;
}
</script>

<script setup lang="ts">
// External Dependencies:
import { onMounted, onBeforeUnmount, nextTick, ref, toRef, watch } from "vue";

// Local Dependencies:
import store from "../util/store";

const props = defineProps<{
  /**
   * Name of the field/entity type
   */
  name: string;
  /**
   * Index of this particular value in the multi-value list
   */
  index: number;
  /**
   * Input, model-detected value of the field (not a 2-way binding)
   */
  value: string;
  /**
   * Whether this value should currently be able to remove itself from the field
   *
   * (i.e. is it *not* the last remaining value in the list)
   */
  removable?: boolean;
}>();

const thisElRef = ref<HTMLElement>();
const valueInput = ref<HTMLElement & { validate: () => boolean }>();

// In Vue.js, data flow is one-directional on 'props'. Updates are handled by the child component
// emitting an event and the parent updating the prop value. We can't rely on that in this case
// because this component may be used without a parent Vue component - so instead treat the props
// just as inputs and create (reactive) internal states for anything we might want to update:
const valueState = ref("");

onMounted(() => {
  valueState.value = props.value;
  // Register the state on the store for publishing to <crowd-form>:
  store[`field-multival-${props.index}-${props.name}`] = valueState;

  // This component doesn't directly register an addValidateHandler on `store`, because it only
  // needs to be validated if the overall field present checkbox is ticked. Instead, we'll bind
  // our onValidate handler to the <custom-multifield-value> element so that the parent component
  // can call it if needed:
  const hostNode = (thisElRef.value?.getRootNode() as ShadowRoot).host as HTMLElement & {
    validate: () => boolean;
  };
  hostNode.validate = onValidate;

  // Notify parent we exist:
  console.log("Sending onMounted event");
  return emitBubblingEvent(ADDED_EVENT_TYPE, thisElRef.value);
});

onBeforeUnmount(() => {
  // De-register the data from the <crowd-form>
  delete store[`field-multival-${props.index}-${props.name}`];
});

watch(toRef(props, "name"), (newName, oldName) => {
  console.log(`${oldName} value index ${props.index} renaming to ${newName}`);
  // When a middle value is removed, the parent FieldMultiValue should remaining values in
  // deterministic ascending index order - but still renaming an item on the reactive store is a
  // two-step process so we'll delete first and defer the write to next Vue cycle to avoid any
  // listeners getting confused about the ordering.
  delete store[`field-multival-${props.index}-${oldName}`];
  nextTick(() => {
    store[`field-multival-${props.index}-${newName}`] = valueState;
  });
});

watch(toRef(props, "index"), (newIx, oldIx) => {
  console.log(`${props.name} value index ${newIx} moving to ${oldIx}`);
  // Two-step rename as discussed in the `name` listener
  delete store[`field-multival-${oldIx}-${props.name}`];
  nextTick(() => {
    store[`field-multival-${newIx}-${props.name}`] = valueState;
  });
});

// Since control is not directly bound to input prop, update the control if the prop changes:
watch(toRef(props, "value"), (newValue) => {
  valueState.value = newValue;
});

/**
 * Transmit event that will bubble up the DOM to parent components (e.g. FieldMultiValue).
 *
 * @returns As per native dispatchEvent()
 *
 * You might think we could use native Vue events `emit(eventType, arg1, arg2, ...)`, by first
 * defining the emitter during setup:
 *
 *    const emit = defineEmits<{
 *      (e: "myEventType", arg1: number, arg2: any, ...): void;
 *    }>();
 *
 * It's true that Custom Element components in Vue would convert this to a native CustomEvent with
 * payload array of args. However, this pattern doesn't provide any way to specify whether the
 * event should bubble (rise up the DOM node tree) and the default is that it doesn't. So our
 * choices are either to add logic to FieldMultiValue so it's able to attach listeners to every
 * MultiFieldValue inside it (complex as these are transcluded in via Liquid) - or to raise a
 * native CustomEvent ourselves. Here we do the latter.
 */
function emitBubblingEvent<T = unknown>(eventType: string, ...args: T[]): boolean {
  const thisEl = thisElRef.value;
  if (!thisEl) {
    // Check really just for TypeScript
    throw new Error("Failed to find root element in MultiFieldValue template");
  }

  return thisEl.dispatchEvent(
    new CustomEvent(eventType, {
      detail: args, // Same (array) format as native Vue event would have been
      bubbles: true, // ...but let it bubble up
      // Because we're raising this on an element in the component's template, we need to mark it
      // as `composed` to allow it to bubble through the shadow root boundary. If this is a
      // problem for you, could raise it directly on the outer <custom-multifield-value> by
      // instead dispatching it on `(thisEl.getRootNode() as ShadowRoot | undefined)?.host`
      composed: true,
    })
  );
}

/**
 * Transmit value remove request to parent component via a bubbling DOM CustomEvent
 */
function onClickRemove() {
  const myIndex = props.index;
  return emitBubblingEvent(
    REMOVE_EVENT_TYPE,
    // Fix bug observed in FF SMStudio render preview where this was getting passed up as string,
    // preventing the parent from correctly identifying which element to delete:
    typeof myIndex === "string" ? parseInt(myIndex) : myIndex
  );
}

/**
 * Send validation requests to the inner text field component
 */
function onValidate() {
  console.log(`Validating field-multival-${props.index}-${props.name}`);
  const inputEl = valueInput.value;
  if (inputEl) {
    return inputEl.validate();
  } else {
    // Shouldn't really happen (just for TypeScript). If it does, allow form submission:
    return true;
  }
}
</script>

<template>
  <div ref="thisElRef" class="multivalue-field-value">
    <crowd-input
      name="field-multival-${index}-${name}"
      ref="valueInput"
      v-model="valueState"
      required
    ></crowd-input>
    <crowd-button class="field-multival-rm" @click="onClickRemove()" :disabled="!removable"
      ><iron-icon icon="remove-circle"
    /></crowd-button>
  </div>
</template>

<style scoped lang="scss">
.multivalue-field-value {
  align-items: center;
  display: flex;

  > crowd-input {
    flex: 1 0 auto;
  }

  > crowd-button {
    flex: 0 0 auto;
  }
}
</style>
