// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0
/**
 * Wrapper around FieldSingleValue Vue CE component to add ElementInternals DOM functionality
 */
// External Dependencies:
import { defineCustomElement } from "vue";

// Local Dependencies:
import FieldSingleValueBase from "./FieldSingleValue.ce.vue";

// Start with the vanilla Vue CE element:
const FieldSingleValueBaseElement = defineCustomElement(FieldSingleValueBase);

/**
 * Extend the vanilla Vue CE to implement a form-associated Custom Element
 *
 * The ElementInternals standard allows custom elements to register their
 * participation in <form>s as discussed at:
 *
 * https://html.spec.whatwg.org/multipage/custom-elements.html#the-elementinternals-interface
 * https://developer.mozilla.org/en-US/docs/Web/API/ElementInternals
 *
 * By defining the required static formAssociated property (here) and attaching ElementInternals
 * and setting Form Value (in the Vue component onMounted), we can demonstrate how this approach
 * might be used to send data to the SageMaker <crowd-form> without the central state store and
 * <object-value-input> pattern. However, at the time of writing this appeared to work end-to-end
 * in Firefox (v91 ESR, using ElementInternals polyfill) but not Chrome (v98, native EInternals).
 */
export class FieldSingleValue extends FieldSingleValueBaseElement {
  _internals?: ElementInternals;
  static get formAssociated() {
    return true;
  }
  constructor(initialProps?: Record<string, unknown> | undefined) {
    super(initialProps);
    this._internals = this.attachInternals();
  }
}

export default FieldSingleValue;
