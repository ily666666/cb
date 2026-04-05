<template>
  <div>
    <div class="page-header">
      <h2>数据处理可视化</h2>
      <p>查看任务执行结果、耗时分析图表、推理报告</p>
    </div>

    <!-- 任务选择 -->
    <div class="card" style="display: flex; align-items: center; gap: 16px;">
      <span style="font-size: 14px; white-space: nowrap;">选择任务:</span>
      <el-select v-model="selectedTask" placeholder="选择任务" style="width: 300px;" @change="loadVisualization" filterable>
        <el-option v-for="t in tasks" :key="t.task_id" :label="t.task_id" :value="t.task_id">
          <span>{{ t.task_id }}</span>
          <span style="float: right; color: var(--text-secondary); font-size: 12px;">{{ t.purpose }}</span>
        </el-option>
      </el-select>
      <el-button type="primary" @click="loadVisualization" :disabled="!selectedTask">
        <el-icon><Refresh /></el-icon> 刷新
      </el-button>
    </div>

    <template v-if="visData">
      <!-- 耗时统计 -->
      <div class="stats-grid">
        <div class="stat-card">
          <div class="stat-icon blue"><el-icon><Timer /></el-icon></div>
          <div class="stat-info">
            <div class="value">{{ visData.summary?.total?.toFixed(2) }}s</div>
            <div class="label">总耗时</div>
          </div>
        </div>
        <div class="stat-card">
          <div class="stat-icon green"><el-icon><Cpu /></el-icon></div>
          <div class="stat-info">
            <div class="value">{{ visData.summary?.total_inference?.toFixed(2) }}s</div>
            <div class="label">推理耗时</div>
          </div>
        </div>
        <div class="stat-card">
          <div class="stat-icon orange"><el-icon><Upload /></el-icon></div>
          <div class="stat-info">
            <div class="value">{{ visData.summary?.total_transfer?.toFixed(2) }}s</div>
            <div class="label">传输耗时</div>
          </div>
        </div>
        <div class="stat-card">
          <div class="stat-icon purple"><el-icon><Loading /></el-icon></div>
          <div class="stat-info">
            <div class="value">{{ visData.summary?.total_overhead?.toFixed(2) }}s</div>
            <div class="label">加载+热身</div>
          </div>
        </div>
      </div>

      <!-- 图表 -->
      <div class="grid-2">
        <div class="card">
          <div class="card-title"><el-icon><Histogram /></el-icon> 各步骤耗时分布</div>
          <div ref="barChartRef" class="chart-container"></div>
        </div>
        <div class="card">
          <div class="card-title"><el-icon><PieChart /></el-icon> 耗时占比</div>
          <div ref="pieChartRef" class="chart-container"></div>
        </div>
      </div>

      <!-- 报告 -->
      <div class="card" v-if="Object.keys(visData.reports || {}).length">
        <div class="card-title"><el-icon><Document /></el-icon> 执行报告</div>
        <el-tabs type="border-card" style="background: transparent;">
          <el-tab-pane v-for="(content, name) in visData.reports" :key="name" :label="name">
            <pre class="report-content">{{ content }}</pre>
          </el-tab-pane>
        </el-tabs>
      </div>
    </template>

    <el-empty v-else-if="selectedTask && !loading" description="该任务暂无可视化数据" />
  </div>
</template>

<script setup>
import { ref, nextTick, onMounted, onUnmounted } from 'vue'
import { taskApi, inferenceApi } from '../api'
import * as echarts from 'echarts'

const tasks = ref([])
const selectedTask = ref('')
const visData = ref(null)
const loading = ref(false)
const barChartRef = ref(null)
const pieChartRef = ref(null)
let barChart = null
let pieChart = null

async function loadVisualization() {
  if (!selectedTask.value) return
  loading.value = true
  try {
    visData.value = await inferenceApi.visualization(selectedTask.value)
    await nextTick()
    renderCharts()
  } catch (e) {
    visData.value = null
  } finally {
    loading.value = false
  }
}

function renderCharts() {
  if (!visData.value) return

  // 柱状图
  if (barChartRef.value) {
    barChart?.dispose()
    barChart = echarts.init(barChartRef.value, 'dark')
    const bar = visData.value.timing_bar
    const colors = ['#53a8ff', '#b388ff', '#e94560', '#f0c040', '#ff9a3e', '#00d4aa']
    const series = Object.entries(bar.series).map(([name, data], i) => ({
      name, type: 'bar', stack: 'total', data,
      itemStyle: { color: colors[i % colors.length] },
      emphasis: { focus: 'series' },
    }))
    barChart.setOption({
      backgroundColor: 'transparent',
      tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
      legend: { top: 0, textStyle: { color: '#a0a0a0', fontSize: 11 } },
      grid: { left: 60, right: 20, top: 40, bottom: 40 },
      xAxis: { type: 'category', data: bar.categories, axisLabel: { color: '#a0a0a0' } },
      yAxis: { type: 'value', name: '秒 (s)', axisLabel: { color: '#a0a0a0' }, nameTextStyle: { color: '#a0a0a0' } },
      series,
    })
  }

  // 饼图
  if (pieChartRef.value) {
    pieChart?.dispose()
    pieChart = echarts.init(pieChartRef.value, 'dark')
    const pie = visData.value.timing_pie
    pieChart.setOption({
      backgroundColor: 'transparent',
      tooltip: { trigger: 'item', formatter: '{b}: {c}s ({d}%)' },
      legend: { orient: 'vertical', right: 10, top: 'center', textStyle: { color: '#a0a0a0', fontSize: 11 } },
      series: [{
        type: 'pie', radius: ['40%', '70%'], center: ['40%', '50%'],
        data: pie.data.filter(d => d.value > 0),
        label: { color: '#e0e0e0', fontSize: 12 },
        itemStyle: { borderColor: '#16213e', borderWidth: 2 },
        emphasis: { itemStyle: { shadowBlur: 10, shadowOffsetX: 0, shadowColor: 'rgba(0, 0, 0, 0.5)' } },
      }],
      color: ['#53a8ff', '#b388ff', '#00d4aa', '#f0c040', '#e94560'],
    })
  }
}

function handleResize() {
  barChart?.resize()
  pieChart?.resize()
}

onMounted(async () => {
  window.addEventListener('resize', handleResize)
  try {
    const res = await taskApi.list()
    tasks.value = (res.tasks || []).filter(t => t.has_output)
  } catch (e) {
    console.error(e)
  }
})

onUnmounted(() => {
  window.removeEventListener('resize', handleResize)
  barChart?.dispose()
  pieChart?.dispose()
})
</script>

<style scoped>
.el-tabs {
  --el-bg-color: transparent;
  --el-fill-color-blank: rgba(255,255,255,0.03);
}
:deep(.el-tabs__content) {
  padding: 12px;
}
:deep(.el-tabs__header) {
  background: rgba(255,255,255,0.02);
}
</style>
