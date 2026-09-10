<script setup lang="ts">
import AtosImage from './atos-image.vue'
</script>

<template>
  <AtosImage v-bind="$attrs">
    <template v-for="(_, name) in $slots" #[name]="slotData">
      <slot :name="name" v-bind="slotData || {}" />
    </template>
  </AtosImage>
</template>
