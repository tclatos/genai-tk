<script setup lang="ts">
import { computed } from 'vue'
import atosBlue from './assets/atos-blue.png'

const hiddenLayouts = [
  'atos-cover',
  'cover',
  'atos-section',
  'section',
  'intro',
  'end',
  'full',
  'none'
]

// Computed check for showing footer
const showFooter = computed(() => {
  // @ts-ignore
  const layout = window?.__slidev__?.nav?.currentLayout || ''
  return !hiddenLayouts.includes(layout)
})
</script>

<template>
  <footer
    v-if="!hiddenLayouts.includes($slidev.nav.currentLayout)"
    class="atos-global-footer"
  >
    <div class="footer-left">
      <span class="footer-date">{{ $slidev.configs.presentationDate || $slidev.configs.date || 'Atos Presentation' }}</span>
      <span class="footer-divider">|</span>
      <span class="footer-title">{{ $slidev.configs.title || 'Corporate Presentation' }}</span>
      <span class="footer-divider">|</span>
      <span class="footer-notice">{{ $slidev.configs.confidentiality || '© Atos Group - for internal use' }}</span>
    </div>

    <div class="footer-right">
      <span v-if="$slidev.configs.pageNumbers !== false" class="footer-page-num">
        {{ $slidev.nav.currentPage }}
      </span>
      <img
        class="footer-logo"
        :src="atosBlue"
        alt="Atos"
      />
    </div>
  </footer>
</template>

<style scoped>
.atos-global-footer {
  position: absolute;
  left: var(--slide-padding-x, 4.5%);
  right: var(--slide-padding-x, 4.5%);
  bottom: 2.2%;
  display: flex;
  align-items: center;
  justify-content: space-between;
  color: var(--atos-navy, #00005b);
  font-family: var(--atos-font, Arial, sans-serif);
  font-size: var(--footer-size, 11px);
  z-index: 10;
  pointer-events: none;
  box-sizing: border-box;
}

.footer-left {
  display: flex;
  gap: 0.6em;
  align-items: center;
  color: #3f4770;
  font-weight: 500;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  max-width: 75%;
}

.footer-divider {
  color: #b0b8d4;
  font-weight: 400;
}

.footer-right {
  display: flex;
  gap: 1.2em;
  align-items: center;
  flex-shrink: 0;
}

.footer-page-num {
  font-weight: 600;
  color: #555f8a;
  font-size: 11px;
}

.footer-logo {
  height: 18px;
  width: auto;
  object-fit: contain;
}
</style>
