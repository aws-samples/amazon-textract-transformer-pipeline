// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0
/**
 * Vue Custom Element to proxy all data from global state store through to SageMaker <crowd-form>
 *
 * This CE must be registered with the special name <object-value-input> to work correctly with the
 * SageMaker <crowd-form> element.
 *
 * Since this element doesn't have any UI template or styling, there's no need to use a single-file
 * component '.vue': Plain TypeScript is fine.
 */

// External Dependencies:
import { defineCustomElement } from "vue";

// Local Dependencies:
import { emitValidate, store } from "../util/store";

// Base Vue component configuration (just binds the 'name' attribute)
const ObjectValueInputVueComponentBase = {
  props: {
    name: String,
  },
  /**
   * No UI/DOM on this component
   */
  render() {
    return;
  },
};

// The base component is not sufficient because it doesn't register itself as being form-associated
// or implement the validate() API for <crowd-form>.
const ObjectValueInputElementBase = defineCustomElement(ObjectValueInputVueComponentBase);

/**
 * Form-associated Custom Element class for the <object-value-input> data proxy
 */
export class ObjectValueInputElement extends ObjectValueInputElementBase {
  _internals: ElementInternals;
  value: Record<string, unknown>; // Form data value property

  /**
   * Required property to mark the element as associated to forms
   *
   * https://html.spec.whatwg.org/multipage/custom-elements.html#custom-elements-face-example
   */
  static get formAssociated() {
    return true;
  }

  constructor(initialProps?: Record<string, unknown> | undefined) {
    super(initialProps);
    this._internals = this.attachInternals();
    // No need to continuously watch() for changes here: Setting once is sufficient
    this.value = store;
  }

  /**
   * Validate data before <crowd-form> submission (return true for go, false for stop)
   *
   * Only allows submission to proceed if *all* registered listeners in the state store return true
   */
  validate(): boolean {
    return emitValidate().every((r) => r);
  }
}

export default ObjectValueInputElement;
