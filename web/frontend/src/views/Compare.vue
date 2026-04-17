<template>
  <div>
    <div class="page-header compact">
      <h2>对比分析</h2>
      <p>选择方案 → 运行 → 查看对比结果</p>
    </div>

    <!-- 对比配置 -->
    <div class="card compact">
      <div class="card-title"><el-icon><Setting /></el-icon> 选择对比方案</div>

      <div v-for="(group, ds) in tasksByDataset" :key="ds" class="dataset-group">
        <div class="dataset-group-label">{{ group[0]?.dataset_label || ds }}</div>
        <div class="task-grid">
          <div v-for="task in group" :key="task.task_id" class="task-card"
               :class="{ selected: selectedIds.includes(task.task_id) }"
               @click="toggleTask(task.task_id)">
            <el-checkbox :model-value="selectedIds.includes(task.task_id)" @click.stop
              @change="toggleTask(task.task_id)" size="small" />
            <div class="task-body">
              <div class="task-purpose">{{ task.purpose }}</div>
              <div class="task-id-text">{{ task.task_id }}</div>
            </div>
            <div class="task-actions">
              <el-button :icon="Setting" size="small" circle
                @click.stop="openConfig(task)" title="模拟参数" />
              <el-button :icon="CopyDocument" size="small" circle
                @click.stop="openClone(task)" title="复制为新方案" />
            </div>
          </div>
        </div>
      </div>

      <div class="action-bar">
        <div class="compare-dims">
          <span class="dim-label">对比维度：</span>
          <el-checkbox v-model="showTime" size="small">耗时</el-checkbox>
          <el-checkbox v-model="showAccuracy" size="small">准确率</el-checkbox>
        </div>
        <el-button type="primary" :disabled="selectedIds.length !== 2 || running" :loading="running"
          @click="runCompare">
          <el-icon><VideoPlay /></el-icon> 开始对比
        </el-button>
      </div>
    </div>

    <!-- 运行进度 -->
    <div class="card compact" v-if="runStates.length">
      <div class="progress-inline">
        <span v-for="rs in runStates" :key="rs.task_id" class="progress-item">
          {{ rs.label }}
          <el-tag size="small"
            :type="rs.status === 'success' ? 'success' : rs.status === 'running' ? 'warning' : 'info'">
            {{ rs.status === 'success' ? '完成' : '处理中...' }}
          </el-tag>
        </span>
      </div>
    </div>

    <!-- 对比结果图表 + 汇总表 -->
    <div class="result-row" v-if="results.length === 2">
      <div class="card compact result-chart" v-if="showTime">
        <div class="card-title"><el-icon><Timer /></el-icon> 耗时对比</div>
        <div ref="timeChartEl" class="chart-box"></div>
      </div>
      <div class="card compact result-chart" v-if="showAccuracy">
        <div class="card-title"><el-icon><DataAnalysis /></el-icon> 准确率对比</div>
        <div ref="accChartEl" class="chart-box"></div>
      </div>
      <div class="card compact result-table">
        <div class="card-title"><el-icon><Document /></el-icon> 汇总</div>
        <el-table :data="results" size="small" :row-class-name="rowClassName">
          <el-table-column prop="purpose" label="方案" min-width="100" show-overflow-tooltip />
          <el-table-column label="耗时(s)" width="80" align="center" v-if="showTime">
            <template #default="{ row }">
              <span :class="{ 'best-val': row.task_id === bestTimeId }">{{ row.total_time }}</span>
            </template>
          </el-table-column>
          <el-table-column label="准确率(%)" width="90" align="center" v-if="showAccuracy">
            <template #default="{ row }">
              <span :class="{ 'best-val': row.task_id === bestAccId }">{{ row.accuracy ?? '—' }}</span>
            </template>
          </el-table-column>
        </el-table>
        <div class="diff-summary">
          <div class="diff-item" v-if="showTime">
            <span class="diff-label">耗时差异</span>
            <span class="diff-val" :class="timeDiffClass">{{ timeDiffText }}</span>
          </div>
          <div class="diff-item" v-if="showAccuracy && accDiff != null">
            <span class="diff-label">准确率差值</span>
            <span class="diff-val" :class="accDiffClass">{{ accDiffText }}</span>
          </div>
        </div>
      </div>
    </div>

    <!-- 齿轮配置对话框 -->
    <el-dialog v-model="configVisible" title="模拟参数配置" width="580px" destroy-on-close>
      <el-form size="small" label-position="left" label-width="90px" style="margin-bottom: 16px;">
        <el-form-item label="方案名称">
          <el-input v-model="editLabel" placeholder="自定义名称，留空使用默认" clearable style="width: 320px;" />
        </el-form-item>
      </el-form>
      <div class="config-step" v-for="step in editSteps" :key="step.step_name">
        <div class="config-step-name">{{ step.step_name }}</div>
        <el-form :inline="true" size="small" label-position="left">
          <el-form-item label="数据量(MB)">
            <el-input-number v-model="step.display_config.data_size_mb" :min="0" :precision="1"
              :step="10" style="width: 140px;" />
          </el-form-item>
          <el-form-item label="耗时(秒)">
            <el-input-number v-model="step.display_config.time" :min="0" :precision="2"
              :step="1" style="width: 140px;" />
          </el-form-item>
          <el-form-item label="准确率(%)" v-if="step.display_config.accuracy != null">
            <el-input-number v-model="step.display_config.accuracy" :min="0" :max="100"
              :precision="2" :step="0.1" style="width: 140px;" />
          </el-form-item>
        </el-form>
      </div>
      <template #footer>
        <el-button @click="configVisible = false">取消</el-button>
        <el-button type="primary" :loading="saving" @click="saveConfig">保存</el-button>
      </template>
    </el-dialog>

    <!-- 复制方案对话框 -->
    <el-dialog v-model="cloneVisible" title="复制为新方案" width="480px" destroy-on-close>
      <el-form size="default" label-width="100px">
        <el-form-item label="源方案">
          <span style="font-size: 13px; color: var(--text-primary);">
            {{ cloneSource?.purpose }} ({{ cloneSource?.task_id }})
          </span>
        </el-form-item>
        <el-form-item label="新任务ID" required>
          <el-input v-model="cloneNewId" placeholder="例如 003b_edge_nofed_link11" />
          <div style="font-size: 11px; color: var(--text-secondary); margin-top: 4px;">
            建议包含数据集名称（link11/rml2016/radar/ratr）和用途标识
          </div>
        </el-form-item>
        <el-form-item label="方案名称" required>
          <el-input v-model="cloneLabel" placeholder="例如 边推理（非联邦）" />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="cloneVisible = false">取消</el-button>
        <el-button type="primary" :loading="cloning" @click="doClone">
          <el-icon><CopyDocument /></el-icon> 确定复制
        </el-button>
      </template>
    </el-dialog>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, onUnmounted, nextTick, watch } from 'vue'
import { compareApi } from '../api'
import { ElMessage } from 'element-plus'
import { CopyDocument } from '@element-plus/icons-vue'
import * as echarts from 'echarts'

const allTasks = ref([])
const selectedIds = ref([])
const showTime = ref(true)
const showAccuracy = ref(true)
const running = ref(false)
const runStates = ref([])
const results = ref([])

const configVisible = ref(false)
const editTask = ref(null)
const editLabel = ref('')
const editSteps = ref([])
const saving = ref(false)

const cloneVisible = ref(false)
const cloneSource = ref(null)
const cloneNewId = ref('')
const cloneLabel = ref('')
const cloning = ref(false)

const timeChartEl = ref(null)
const accChartEl = ref(null)
const chartInstances = []

const COLORS = ['#53a8ff', '#b388ff', '#00d4aa', '#ff9a3e', '#e94560', '#ffd93d']

const tasksByDataset = computed(() => {
  const groups = {}
  for (const t of allTasks.value) {
    const ds = t.dataset || 'other'
    if (!groups[ds]) groups[ds] = []
    groups[ds].push(t)
  }
  return groups
})

const bestTimeId = computed(() => {
  if (results.value.length < 2) return null
  return results.value.reduce((best, r) => r.total_time < best.total_time ? r : best).task_id
})

const bestAccId = computed(() => {
  if (results.value.length < 2) return null
  const withAcc = results.value.filter(r => r.accuracy != null)
  if (!withAcc.length) return null
  return withAcc.reduce((best, r) => r.accuracy > best.accuracy ? r : best).task_id
})

const timeDiff = computed(() => {
  if (results.value.length !== 2) return null
  const [a, b] = results.value
  if (!a.total_time || !b.total_time) return null
  const slower = Math.max(a.total_time, b.total_time)
  const faster = Math.min(a.total_time, b.total_time)
  return ((slower - faster) / slower * 100)
})
const timeDiffText = computed(() => {
  if (timeDiff.value == null) return '—'
  const [a, b] = results.value
  const fasterName = a.total_time <= b.total_time ? a.purpose : b.purpose
  return `${fasterName} 快 ${timeDiff.value.toFixed(1)}%`
})
const timeDiffClass = computed(() => 'diff-better')

const accDiff = computed(() => {
  if (results.value.length !== 2) return null
  const [a, b] = results.value
  if (a.accuracy == null || b.accuracy == null) return null
  return (b.accuracy - a.accuracy)
})
const accDiffText = computed(() => {
  if (accDiff.value == null) return '—'
  const v = accDiff.value
  const sign = v > 0 ? '+' : ''
  return `${sign}${v.toFixed(2)}%`
})
const accDiffClass = computed(() => {
  if (accDiff.value == null) return ''
  return accDiff.value >= 0 ? 'diff-better' : 'diff-worse'
})

function toggleTask(taskId) {
  const idx = selectedIds.value.indexOf(taskId)
  if (idx >= 0) {
    selectedIds.value.splice(idx, 1)
  } else if (selectedIds.value.length < 2) {
    selectedIds.value.push(taskId)
  }
}

function rowClassName({ row }) {
  if (row.task_id === bestTimeId.value || row.task_id === bestAccId.value) return 'best-row'
  return ''
}

/* ---- 齿轮配置 ---- */
function openConfig(task) {
  editTask.value = task
  editLabel.value = task.purpose || ''
  editSteps.value = JSON.parse(JSON.stringify(task.steps))
  configVisible.value = true
}

async function saveConfig() {
  saving.value = true
  try {
    if (editLabel.value !== editTask.value.purpose) {
      await compareApi.updateLabel(editTask.value.task_id, editLabel.value)
    }
    for (const step of editSteps.value) {
      await compareApi.updateStepConfig(editTask.value.task_id, step.step_name, step.display_config)
    }
    const idx = allTasks.value.findIndex(t => t.task_id === editTask.value.task_id)
    if (idx >= 0) {
      if (editLabel.value) allTasks.value[idx].purpose = editLabel.value
      allTasks.value[idx].steps = JSON.parse(JSON.stringify(editSteps.value))
    }
    configVisible.value = false
    ElMessage.success('参数已保存')
  } catch (e) {
    ElMessage.error('保存失败: ' + (e.message || e))
  } finally {
    saving.value = false
  }
}

/* ---- 复制方案 ---- */
function openClone(task) {
  cloneSource.value = task
  const num = allTasks.value.length + 1
  const prefix = String(num).padStart(3, '0')
  cloneNewId.value = `${prefix}_edge_nofed_${task.dataset}`
  cloneLabel.value = ''
  cloneVisible.value = true
}

async function doClone() {
  if (!cloneNewId.value.trim()) {
    ElMessage.warning('请输入新任务ID')
    return
  }
  if (!cloneLabel.value.trim()) {
    ElMessage.warning('请输入方案名称')
    return
  }
  cloning.value = true
  try {
    const res = await compareApi.clone(cloneSource.value.task_id, cloneNewId.value.trim(), cloneLabel.value.trim())
    if (res.status === 'error') {
      ElMessage.error(res.message)
      return
    }
    cloneVisible.value = false
    ElMessage.success('方案已复制')
    await loadTasks()
  } catch (e) {
    ElMessage.error('复制失败: ' + (e.message || e))
  } finally {
    cloning.value = false
  }
}

/* ---- 对比 ---- */
function sleep(ms) { return new Promise(r => setTimeout(r, ms)) }

async function runCompare() {
  const ids = [...selectedIds.value]
  if (ids.length < 2) return

  running.value = true
  results.value = []
  runStates.value = ids.map(id => {
    const t = allTasks.value.find(x => x.task_id === id)
    return { task_id: id, label: t?.purpose || id, status: 'running' }
  })

  try {
    const res = await compareApi.results(ids)
    for (const rs of runStates.value) {
      await sleep(300)
      rs.status = 'success'
    }
    results.value = res.results || []
    await nextTick()
    renderCharts()
  } catch (e) {
    ElMessage.error('获取对比结果失败')
  }

  running.value = false
}

/* ---- 图表 ---- */
function renderCharts() {
  chartInstances.forEach(c => c.dispose())
  chartInstances.length = 0
  if (results.value.length < 2) return

  const labels = results.value.map(r => r.purpose)
  const times = results.value.map(r => r.total_time)
  const accs = results.value.map(r => r.accuracy)
  const colors = results.value.map((_, i) => COLORS[i % COLORS.length])

  if (showTime.value && timeChartEl.value) {
    const chart = echarts.init(timeChartEl.value)
    chartInstances.push(chart)
    chart.setOption({
      tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
      grid: { left: 100, right: 50, top: 10, bottom: 20 },
      xAxis: { type: 'value', name: '秒', nameTextStyle: { color: '#aaa', fontSize: 11 }, axisLabel: { color: '#aaa', fontSize: 11 } },
      yAxis: { type: 'category', data: labels, inverse: true, axisLabel: { color: '#ccc', fontSize: 11 } },
      series: [{
        type: 'bar',
        data: times.map((v, i) => ({ value: v, itemStyle: { color: colors[i], borderRadius: [0, 4, 4, 0] } })),
        barMaxWidth: 26,
        label: { show: true, position: 'right', formatter: '{c}s', fontSize: 11, fontWeight: 600, color: '#ddd' },
      }],
    })
  }

  if (showAccuracy.value && accChartEl.value) {
    const validAccs = accs.filter(a => a != null)
    const minAcc = validAccs.length ? Math.floor(Math.min(...validAccs) - 2) : 0
    const chart = echarts.init(accChartEl.value)
    chartInstances.push(chart)
    chart.setOption({
      tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
      grid: { left: 100, right: 50, top: 10, bottom: 20 },
      xAxis: { type: 'value', name: '%', min: Math.max(0, minAcc), nameTextStyle: { color: '#aaa', fontSize: 11 }, axisLabel: { color: '#aaa', fontSize: 11 } },
      yAxis: { type: 'category', data: labels, inverse: true, axisLabel: { color: '#ccc', fontSize: 11 } },
      series: [{
        type: 'bar',
        data: accs.map((v, i) => ({ value: v, itemStyle: { color: colors[i], borderRadius: [0, 4, 4, 0] } })),
        barMaxWidth: 26,
        label: { show: true, position: 'right', formatter: '{c}%', fontSize: 11, fontWeight: 600, color: '#ddd' },
      }],
    })
  }
}

watch([showTime, showAccuracy], () => {
  if (results.value.length >= 2) {
    nextTick(() => renderCharts())
  }
})

async function loadTasks() {
  const res = await compareApi.tasks()
  allTasks.value = res.tasks || []
}

onMounted(async () => {
  try { await loadTasks() } catch (e) { console.error(e) }
})

onUnmounted(() => {
  chartInstances.forEach(c => c.dispose())
})
</script>

<style scoped>
.compact { padding: 12px 16px; }
.page-header.compact { margin-bottom: 8px; }
.page-header.compact h2 { margin-bottom: 2px; }
.page-header.compact p { font-size: 12px; }

.dataset-group {
  margin-bottom: 10px;
}
.dataset-group-label {
  font-size: 12px;
  font-weight: 600;
  color: var(--text-secondary);
  margin-bottom: 6px;
  padding-left: 2px;
}
.task-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
  gap: 8px;
}
.task-card {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 10px;
  border-radius: 8px;
  border: 1.5px solid rgba(255,255,255,0.06);
  background: rgba(255,255,255,0.02);
  cursor: pointer;
  transition: all 0.2s;
}
.task-card:hover {
  border-color: rgba(83,168,255,0.25);
  background: rgba(83,168,255,0.03);
}
.task-card.selected {
  border-color: #53a8ff;
  background: rgba(83,168,255,0.06);
}
.task-body {
  flex: 1;
  min-width: 0;
}
.task-purpose {
  font-size: 12px;
  font-weight: 600;
  color: var(--text-primary);
  line-height: 1.3;
}
.task-id-text {
  font-size: 10px;
  color: var(--text-secondary);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.task-actions {
  display: flex;
  gap: 2px;
  flex-shrink: 0;
}

.action-bar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-top: 10px;
  padding-top: 10px;
  border-top: 1px solid rgba(255,255,255,0.06);
}
.compare-dims {
  display: flex;
  align-items: center;
  gap: 10px;
}
.dim-label {
  font-size: 12px;
  color: var(--text-secondary);
}

.progress-inline {
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
}
.progress-item {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  font-size: 12px;
  color: var(--text-primary);
}

.result-row {
  display: flex;
  gap: 12px;
  margin-top: 12px;
}
.result-chart {
  flex: 1;
  min-width: 0;
}
.result-table {
  flex: 0 0 280px;
}

.chart-box {
  height: 200px;
  border-radius: 8px;
  border: 1px solid rgba(255,255,255,0.06);
  background: rgba(0,0,0,0.15);
}

.best-val {
  color: #00d4aa;
  font-weight: 700;
}
.diff-summary {
  margin-top: 10px;
  padding-top: 10px;
  border-top: 1px solid rgba(255,255,255,0.06);
  display: flex;
  flex-direction: column;
  gap: 6px;
}
.diff-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-size: 12px;
}
.diff-label {
  color: var(--text-secondary);
}
.diff-val {
  font-weight: 700;
  font-size: 13px;
}
.diff-better { color: #00d4aa; }
.diff-worse { color: #e94560; }
:deep(.best-row) td {
  background: rgba(0,212,170,0.04) !important;
}

.config-step {
  margin-bottom: 14px;
  padding-bottom: 10px;
  border-bottom: 1px solid rgba(255,255,255,0.06);
}
.config-step:last-child { border-bottom: none; margin-bottom: 0; padding-bottom: 0; }
.config-step-name {
  font-weight: 600;
  font-size: 12px;
  color: var(--text-primary);
  margin-bottom: 6px;
}
</style>
