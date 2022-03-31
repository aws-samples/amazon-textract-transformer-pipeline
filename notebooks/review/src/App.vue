<!-- Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
  SPDX-License-Identifier: MIT-0 -->
<!-- Dummy Vue app
  In this task template, UI functionality is delivered mainly by independent native Web Components
  authored in Vue but not running in any one shared Vue app context. The Vue 'app' here is provided
  mainly as a placeholder to illustrate that placement is flexible (as shown here mounted in the
  <short-instructions>) and the app need not encapsulate the entire task UI implementation.

  Alternatively, you could for example pass one big, complex data object from Liquid into a Vue app
  (perhaps via a global variable such as window.taskData) and then keep as much of the UI logic in
  the shared Vue app render tree as possible.
  -->
<script setup lang="ts">
// Local Dependencies:
import HelloWorld from "./components/HelloWorld.vue";

// Demo data ingestion via Liquid template global variable:
// onMounted(() => {
//   const data = window.taskData;
//   // ...
// });
</script>

<template>
  <HelloWorld msg="Debug"></HelloWorld>
</template>

<style lang="scss">
@import "./assets/base.scss";

#app {
  max-width: 600px;
  margin: 0 auto;
  padding: 2rem;

  font-weight: normal;
}

a,
.accent {
  color: var(--color-accent);
}
</style>
