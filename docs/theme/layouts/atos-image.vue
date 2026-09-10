<script setup lang="ts">
const props = defineProps<{
  image?: string
  alt?: string
  imageClass?: string
}>()
</script>

<template>
  <div class="slidev-layout atos-image">
    <div class="image-layout-grid">
      <!-- Left Content Column -->
      <div class="content-col">
        <slot />
      </div>

      <!-- Right Image Column -->
      <div class="image-col">
        <slot name="image">
          <slot name="right">
            <img
              v-if="props.image"
              :src="props.image"
              :alt="props.alt || 'Slide Illustration'"
              :class="props.imageClass"
              class="rounded-image"
            />
          </slot>
        </slot>
      </div>
    </div>
  </div>
</template>

<style scoped>
.atos-image {
  position: relative;
  width: 100%;
  height: 100%;
  box-sizing: border-box;
  background-color: #ffffff;
  padding: var(--slide-padding-top, 3.5%) var(--slide-padding-x, 4.5%) var(--slide-padding-bottom, 6.5%);
}

.image-layout-grid {
  display: grid;
  grid-template-columns: 1.1fr 0.9fr;
  column-gap: 2rem;
  width: 100%;
  height: 100%;
  align-items: center;
}

.content-col {
  display: flex;
  flex-direction: column;
  justify-content: flex-start;
  height: 100%;
}

.image-col {
  display: flex;
  justify-content: center;
  align-items: center;
  height: 100%;
  overflow: hidden;
}

.image-col :deep(img),
.rounded-image {
  max-width: 100%;
  max-height: 85%;
  object-fit: contain;
  border-radius: 4px;
  border: 1px solid var(--atos-border, #e6e6e6);
}
</style>
