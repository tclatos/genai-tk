<script setup lang="ts">
import sectionCurve from '../assets/section-curve.svg'
import atosWhite from '../assets/atos-white.png'
</script>

<template>
  <div class="slidev-layout atos-section">
    <!-- Curved visual on right -->
    <img
      class="section-curve"
      :src="sectionCurve"
      alt="Atos Section Graphic"
    />

    <!-- Main Section Content -->
    <div class="section-content">
      <slot />
    </div>

    <!-- White Atos Corporate Logo -->
    <img
      class="section-logo"
      :src="atosWhite"
      alt="Atos Logo"
    />
  </div>
</template>

<style scoped>
.atos-section {
  position: relative;
  width: 100%;
  height: 100%;
  overflow: hidden;
  background-color: var(--atos-blue, #0073e6);
  color: #ffffff;
  padding: 10% var(--slide-padding-x, 4.5%) 6%;
  box-sizing: border-box;
  display: flex;
  align-items: center;
}

.section-curve {
  position: absolute;
  top: 0;
  right: 0;
  width: 42%;
  height: 100%;
  object-fit: cover;
  object-position: right center;
  pointer-events: none;
  z-index: 1;
}

.section-content {
  position: relative;
  z-index: 2;
  width: 65%;
  max-width: 750px;
}

.section-content :deep(h1) {
  color: #ffffff !important;
  font-size: 40px;
  font-weight: 700;
  line-height: 1.15;
  margin: 0 0 0.3em 0;
}

.section-content :deep(h2) {
  color: var(--atos-navy, #00005b) !important;
  font-size: 24px;
  font-weight: 600;
  line-height: 1.3;
  margin: 0 0 0.8em 0;
}

.section-content :deep(p) {
  color: #ffffff;
  font-size: 18px;
  opacity: 0.95;
}

.section-logo {
  position: absolute;
  right: var(--slide-padding-x, 4.5%);
  bottom: 6%;
  width: 130px;
  height: auto;
  object-fit: contain;
  z-index: 2;
}
</style>
