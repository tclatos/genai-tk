<script setup lang="ts">
import { computed } from 'vue'
import coverCurve from '../assets/cover-curve.svg'
import atosWhite from '../assets/atos-white.png'

const props = defineProps<{
  author?: string
  date?: string
}>()
</script>

<template>
  <div class="slidev-layout atos-cover">
    <!-- Decorative curve visual on the right -->
    <img
      class="cover-curve"
      :src="coverCurve"
      alt="Atos Cover Geometry"
    />

    <!-- Main Title & Subtitle Area -->
    <div class="cover-content">
      <slot />
    </div>

    <!-- Author & Date Metadata -->
    <div class="cover-author">
      <slot name="author">
        <div v-if="props.author" class="author-name">{{ props.author }}</div>
        <div v-if="props.date || $slidev.configs.presentationDate" class="author-date">
          {{ props.date || $slidev.configs.presentationDate }}
        </div>
      </slot>
    </div>

    <!-- White Atos Corporate Logo -->
    <img
      class="cover-logo"
      :src="atosWhite"
      alt="Atos Logo"
    />
  </div>
</template>

<style scoped>
.atos-cover {
  position: relative;
  width: 100%;
  height: 100%;
  overflow: hidden;
  background-color: var(--atos-navy, #00005b);
  color: #ffffff;
  padding: 6% var(--slide-padding-x, 4.5%) 6%;
  box-sizing: border-box;
}

.cover-curve {
  position: absolute;
  top: 0;
  right: 0;
  width: 48%;
  height: 100%;
  object-fit: cover;
  object-position: right top;
  pointer-events: none;
  z-index: 1;
}

.cover-content {
  position: relative;
  z-index: 2;
  width: 62%;
  max-width: 720px;
}

.cover-content :deep(h1) {
  color: #ffffff !important;
  font-size: 42px;
  font-weight: 700;
  line-height: 1.15;
  margin: 0 0 0.4em 0;
  letter-spacing: -0.01em;
}

.cover-content :deep(h2) {
  color: var(--atos-cyan, #43c7f4) !important;
  font-size: 24px;
  font-weight: 500;
  line-height: 1.3;
  margin: 0 0 1em 0;
}

.cover-content :deep(p) {
  color: #e0e6f5;
  font-size: 18px;
  line-height: 1.4;
}

.cover-author {
  position: absolute;
  left: var(--slide-padding-x, 4.5%);
  bottom: 8%;
  z-index: 2;
  font-size: 16px;
  color: #d1dcfa;
  line-height: 1.4;
}

.author-name {
  font-weight: 700;
  color: #ffffff;
}

.author-date {
  color: var(--atos-cyan, #43c7f4);
  font-size: 14px;
}

.cover-logo {
  position: absolute;
  right: var(--slide-padding-x, 4.5%);
  bottom: 6%;
  width: 140px;
  height: auto;
  object-fit: contain;
  z-index: 2;
}
</style>
