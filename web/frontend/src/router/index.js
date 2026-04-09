import { createRouter, createWebHistory } from 'vue-router'

const routes = [
  { path: '/', name: 'Dashboard', component: () => import('../views/Dashboard.vue'), meta: { title: '系统概览' } },
  { path: '/data', name: 'DataAccess', component: () => import('../views/DataAccess.vue'), meta: { title: '数据接入' } },
  { path: '/visualization', name: 'Visualization', component: () => import('../views/Visualization.vue'), meta: { title: '数据处理可视化' } },
  { path: '/inference', name: 'Inference', component: () => import('../views/Inference.vue'), meta: { title: '模型推理计算' } },
  { path: '/models', name: 'Models', component: () => import('../views/Models.vue'), meta: { title: '模型算法管理' } },
  { path: '/lightweight', name: 'Lightweight', component: () => import('../views/Lightweight.vue'), meta: { title: '模型轻量化' } },
]

const router = createRouter({
  history: createWebHistory(),
  routes,
})

export default router
