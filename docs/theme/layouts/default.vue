<script setup lang="ts">
import AtosDefault from './atos-default.vue'
</script>

<template>
  <AtosDefault v-bind="$attrs">
    <template v-for="(_, name) in $slots" #[name]="slotData">
      <slot :name="name" v-bind="slotData || {}" />
    </template>
  </AtosDefault>
</template>
