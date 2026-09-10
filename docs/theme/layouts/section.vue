<script setup lang="ts">
import AtosSection from './atos-section.vue'
</script>

<template>
  <AtosSection v-bind="$attrs">
    <template v-for="(_, name) in $slots" #[name]="slotData">
      <slot :name="name" v-bind="slotData || {}" />
    </template>
  </AtosSection>
</template>
