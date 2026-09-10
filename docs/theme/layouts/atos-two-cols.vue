<script setup lang="ts">
const props = defineProps<{
  class?: string
  layoutClass?: string
  columns?: string
}>()
</script>

<template>
  <div class="slidev-layout atos-two-cols">
    <!-- Optional header slot if explicitly separated -->
    <div v-if="$slots.header" class="col-header">
      <slot name="header" />
    </div>

    <!-- 2-Columns Grid -->
    <div class="two-columns-grid" :class="props.columns || ''">
      <div class="col-left">
        <slot name="left">
          <slot />
        </slot>
      </div>
      <div class="col-right">
        <slot name="right" />
      </div>
    </div>
  </div>
</template>

<style scoped>
.atos-two-cols {
  position: relative;
  width: 100%;
  height: 100%;
  box-sizing: border-box;
  background-color: #ffffff;
  padding: var(--slide-padding-top, 3.5%) var(--slide-padding-x, 4.5%) var(--slide-padding-bottom, 6.5%);
  display: flex;
  flex-direction: column;
}

.col-header {
  margin-bottom: 0.5em;
}

.two-columns-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  column-gap: 2.5rem;
  row-gap: 1rem;
  width: 100%;
  flex: 1;
}

.col-left,
.col-right {
  display: flex;
  flex-direction: column;
}

.col-left :deep(h1),
.col-right :deep(h1) {
  font-size: 32px;
}

.col-left :deep(h2),
.col-right :deep(h2) {
  font-size: 20px;
}
</style>
