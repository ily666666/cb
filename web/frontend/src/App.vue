<template>
  <div class="app-layout">
    <aside class="sidebar">
      <div class="sidebar-header">
        <h1>云边端协同框架</h1>
        <div class="subtitle">电磁数据处理平台</div>
      </div>
      <nav class="sidebar-nav">
        <router-link
          v-for="item in navItems" :key="item.path"
          :to="item.path"
          class="nav-item"
          active-class="active"
        >
          <el-icon><component :is="item.icon" /></el-icon>
          {{ item.label }}
        </router-link>
      </nav>
      <div style="padding: 16px 24px; font-size: 11px; color: var(--text-secondary); border-top: 1px solid rgba(255,255,255,0.06);">
        v1.0.0
      </div>
    </aside>
    <main class="main-content">
      <router-view v-slot="{ Component }">
        <transition name="fade" mode="out-in">
          <component :is="Component" />
        </transition>
      </router-view>
    </main>
  </div>
</template>

<script setup>
const navItems = [
  { path: '/', label: '系统概览', icon: 'Monitor' },
  { path: '/data', label: '数据接入', icon: 'FolderOpened' },
  { path: '/visualization', label: '数据处理可视化', icon: 'DataAnalysis' },
  { path: '/inference', label: '模型推理计算', icon: 'Cpu' },
  { path: '/models', label: '模型算法管理', icon: 'Setting' },
]
</script>
