// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0
/**
 * A reactive state store singleton and validation event bus for <object-value-input>.
 *
 * This state store, used with the custom <object-value-input> element, allows input components
 * under shadow DOM to still register output data to be picked up by the SageMaker <crowd-form>.
 * Pub/sub validation event handling is also provided, so components can handle and respond to
 * validation requests when the user clicks 'Submit'.
 */
import { reactive } from "vue";

/**
 * A simple event bus on which publish()ing collects the results from all registered listeners.
 */
class ResultCollectingEventBus<TEvent = unknown, TResponse = void> {
  private nextId = 0;
  private subscriptions: Record<string, Record<string, (event: TEvent) => TResponse>> = {};

  generateId() {
    return (this.nextId++).toString();
  }

  /**
   * Subscribe a new event listener/handler
   * @param eventType Name of event type to listen for
   * @param listener Function to be called when an event is published
   * @returns A function to call to de-register/remove the listener
   */
  subscribe(eventType: string, listener: (event: TEvent) => TResponse): () => void {
    if (!this.subscriptions[eventType]) this.subscriptions[eventType] = {};
    const id = this.generateId();
    this.subscriptions[eventType][id] = listener;
    return () => {
      delete this.subscriptions[eventType][id];
      if (!Object.keys(this.subscriptions[eventType]).length) {
        delete this.subscriptions[eventType];
      }
    };
  }

  /**
   * Publish an event and synchronously collect and return the results from all listeners.
   * @param eventType Name of event type to publish
   * @param event Event data to publish
   * @returns Array of listener responses (in the order they were called)
   */
  publish(eventType: string, event: TEvent): TResponse[] {
    const subs = this.subscriptions[eventType];
    if (!subs) return [];
    return Object.keys(subs).map((k) => subs[k](event));
  }
}

const storeBus = new ResultCollectingEventBus<void, boolean>();

/**
 * Register a listener/handler for form validation events when user clicks submit.
 * @param validateHandler A callback returning true to allow form submission, false to prevent it
 * @returns A callback to remove the event listener (e.g. when your component is deleted)
 */
export const addValidateHandler = (validateHandler: () => boolean) => {
  return storeBus.subscribe("validate", validateHandler);
};

/**
 * Publish a form validation event and collect the response from all registered components
 * @returns Boolean array in which any 'false' entry should prevent accepting the form.
 */
export const emitValidate = () => storeBus.publish("validate");

/**
 * Reactive state store to which components can save data inputs.
 *
 * The contents of this store will be continuously watched and synchronised to the
 * <object-value-input> element, to be included in output data when the SageMaker task is submitted
 */
export const store = reactive<Record<string, unknown>>({});
export default store;
