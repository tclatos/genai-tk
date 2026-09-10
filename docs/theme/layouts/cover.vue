<script setup lang="ts">
import AtosCover from './atos-cover.vue'
</script>

<template>
  <AtosCover v-bind="$attrs">
    <template v-for="(_, name) in $slots" #[name]="slotData">
      <slot :name="name" v-bind="slotData || {}" />
    </template>
  </AtosCover>
</template>
