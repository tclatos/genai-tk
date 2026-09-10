<script setup lang="ts">
</script>

<template>
  <div class="slidev-layout atos-default">
    <div class="content-wrapper">
      <slot />
    </div>
  </div>
</template>

<style scoped>
.atos-default {
  position: relative;
  width: 100%;
  height: 100%;
  box-sizing: border-box;
  background-color: #ffffff;
  padding: var(--slide-padding-top, 3.5%) var(--slide-padding-x, 4.5%) var(--slide-padding-bottom, 6.5%);
}

.content-wrapper {
  width: 100%;
  height: 100%;
  display: flex;
  flex-direction: column;
}
</style>
