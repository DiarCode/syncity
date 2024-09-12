import { createRouter, createWebHistory } from 'vue-router'
import HomePage from './modules/home/pages/HomePage.vue'

export const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  scrollBehavior: () => {
    return { top: 0 }
  },
  routes: [{ path: '/', component: HomePage }]
})
