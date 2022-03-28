<!-- Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
  SPDX-License-Identifier: MIT-0 -->
<!-- Dummy Vue component
  This simple component is only here to demonstrate using a vanilla (non Custom Element) Vue
  component in the template. In these situations, you can use standard Vue data binding and event
  hooks to transmit information up and down the (virtual) DOM hierarchy.
  -->
<script setup lang="ts">
defineProps<{
  msg: string;
}>();
</script>

<template>
  <div class="greetings">
    <h3 class="accent">${ msg }</h3>
    <p>Vue app is running successfully.</p>
  </div>
</template>

<style scoped lang="scss">
h3 {
  font-size: 1.2rem;
}

.greetings {
  background-color: rgba(0, 0, 0, 0.05);
  padding: 10px;
  h3,
  p {
    text-align: center;
  }

  > :last-child {
    margin-bottom: 0px;
  }
}
</style>
