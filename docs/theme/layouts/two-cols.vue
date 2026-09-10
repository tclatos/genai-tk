<script setup lang="ts">
import AtosTwoCols from './atos-two-cols.vue'
</script>

<template>
  <AtosTwoCols v-bind="$attrs">
    <template v-for="(_, name) in $slots" #[name]="slotData">
      <slot :name="name" v-bind="slotData || {}" />
    </template>
  </AtosTwoCols>
</template>
