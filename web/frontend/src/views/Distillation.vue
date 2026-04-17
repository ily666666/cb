<template>
  <div>
    <div class="page-header compact">
      <h2>模型轻量化</h2>
      <p>知识蒸馏：云侧大模型 → 边侧轻量学生模型，压缩参数量</p>
    </div>

    <!-- 流程图 + 任务选择 并排 -->
    <div class="grid-2">
      <div class="card compact">
        <div class="card-title"><el-icon><MagicStick /></el-icon> 轻量化流程</div>
        <div class="pipeline-flow">
          <div class="flow-step">
            <div class="flow-icon blue"><el-icon><Cloudy /></el-icon></div>
            <div class="flow-label">教师模型</div>
            <div class="flow-detail">云侧大模型</div>
          </div>
          <div class="flow-arrow">→</div>
          <div class="flow-step">
            <div class="flow-icon orange"><el-icon><Connection /></el-icon></div>
            <div class="flow-label">软标签蒸馏</div>
            <div class="flow-detail">KL散度 + 温度缩放</div>
          </div>
          <div class="flow-arrow">→</div>
          <div class="flow-step">
            <div class="flow-icon purple"><el-icon><DataAnalysis /></el-icon></div>
            <div class="flow-label">特征匹配</div>
            <div class="flow-detail">多层特征对齐</div>
          </div>
          <div class="flow-arrow">→</div>
          <div class="flow-step">
            <div class="flow-icon green"><el-icon><Cpu /></el-icon></div>
            <div class="flow-label">学生模型</div>
            <div class="flow-detail">边侧轻量模型</div>
          </div>
          <div class="flow-arrow">→</div>
          <div class="flow-step">
            <div class="flow-icon red"><el-icon><Download /></el-icon></div>
            <div class="flow-label">部署输出</div>
            <div class="flow-detail">边侧推理</div>
          </div>
        </div>
      </div>
      <div class="card compact">
        <div class="card-title"><el-icon><List /></el-icon> 选择训练任务</div>
        <el-table :data="tasks" size="small" max-height="160" highlight-current-row
                  @current-change="onTaskSelect" style="width: 100%;">
          <el-table-column label="" width="40">
            <template #default="{ row }">
              <el-radio v-model="selectedTask" :value="row.task_id" />
            </template>
          </el-table-column>
          <el-table-column prop="task_id" label="任务ID" min-width="160" show-overflow-tooltip />
          <el-table-column prop="dataset_type" label="数据集" width="80" />
          <el-table-column label="KD" width="50" align="center">
            <template #default="{ row }">
              <el-tag v-if="row.has_kd_config" type="success" size="small">有</el-tag>
              <el-tag v-else type="info" size="small">无</el-tag>
            </template>
          </el-table-column>
        </el-table>
      </div>
    </div>

    <!-- 执行蒸馏 -->
    <div class="card compact" v-if="selectedTask">
      <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 8px;">
        <el-button v-if="!running" type="primary" @click="startDistillation">
          <el-icon><VideoPlay /></el-icon> 开始蒸馏
        </el-button>
        <el-button v-else type="danger" @click="stopDistillation">
          <el-icon><VideoPause /></el-icon> 终止
        </el-button>
        <span style="font-size: 12px; color: var(--text-secondary);">{{ selectedTask }}</span>
        <el-tag v-if="taskResult === 'success'" type="success" size="small">完成</el-tag>
        <el-tag v-if="taskResult === 'error'" type="danger" size="small">失败</el-tag>
        <el-switch v-model="demoMode" size="small"
          style="margin-left: auto;" />
        <el-button v-if="demoMode" :icon="Setting" size="small" circle
          @click="demoConfigVisible = true" />
      </div>

      <div class="term-compact">
        <WebTerminal v-if="wsTaskId" :taskId="wsTaskId" :title="'模型轻量化 — ' + selectedTask"
          ref="terminalRef" @done="onWsDone" />
      </div>
    </div>

    <!-- 模型参数量对比 -->
    <div v-if="selectedTask && teacherModel">
      <div class="card compact">
        <div class="card-title"><el-icon><DataAnalysis /></el-icon> 模型参数量对比</div>
        <div class="param-compare">
          <div class="param-model teacher">
            <div class="param-role">教师模型（云侧）</div>
            <div class="param-size">{{ teacherModel.size_mb }} MB</div>
            <div class="param-count">≈ {{ formatParamCount(teacherModel) }} 参数</div>
          </div>
          <div class="param-arrow">
            <div class="compress-ratio">{{ compressionRatio }}%</div>
            <div class="compress-detail">↓ 压缩</div>
          </div>
          <div class="param-model student">
            <div class="param-role">学生模型（边侧）</div>
            <div class="param-size">{{ studentModel.size_mb }} MB</div>
            <div class="param-count">≈ {{ formatParamCount(studentModel) }} 参数</div>
          </div>
        </div>
        <div class="param-bar-wrap">
          <div class="param-bar-label">教师</div>
          <div class="param-bar"><div class="param-bar-fill teacher" style="width: 100%"></div></div>
          <span class="param-bar-val">{{ teacherModel.size_mb }} MB</span>
        </div>
        <div class="param-bar-wrap">
          <div class="param-bar-label">学生</div>
          <div class="param-bar"><div class="param-bar-fill student" :style="{ width: studentRatioPercent + '%' }"></div></div>
          <span class="param-bar-val">{{ studentModel.size_mb }} MB</span>
        </div>
      </div>
    </div>

    <!-- 蒸馏历史曲线 -->
    <div class="card compact" v-if="selectedTask && kdHistory">
      <div class="card-title"><el-icon><TrendCharts /></el-icon> 蒸馏训练曲线</div>
      <div v-for="(edgeData, edgeKey) in kdHistory.edges" :key="edgeKey" style="margin-bottom: 12px;">
        <div class="chart-grid">
          <div :ref="el => chartRefs[edgeKey + '_loss'] = el" class="chart-box"></div>
          <div :ref="el => chartRefs[edgeKey + '_acc'] = el" class="chart-box"></div>
        </div>
      </div>
    </div>

    <el-dialog v-model="demoConfigVisible" title="参数设置" width="360px" destroy-on-close>
      <el-form label-width="110px" size="small">
        <el-form-item label="模型压缩率">
          <el-input-number v-model="compressRatio" :min="10" :max="99.9"
            :precision="1" :step="1" style="width: 150px;" />
          <span style="margin-left: 6px; color: var(--text-secondary);">%</span>
        </el-form-item>
        <el-form-item label="教师模型准确率">
          <el-input-number v-model="teacherAcc" :min="50" :max="100"
            :precision="1" :step="0.5" style="width: 150px;" />
          <span style="margin-left: 6px; color: var(--text-secondary);">%</span>
        </el-form-item>
        <el-form-item label="学生模型准确率">
          <el-input-number v-model="targetAcc" :min="50" :max="100"
            :precision="1" :step="0.5" style="width: 150px;" />
          <span style="margin-left: 6px; color: var(--text-secondary);">%</span>
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button type="primary" @click="demoConfigVisible = false">确定</el-button>
      </template>
    </el-dialog>
  </div>
</template>

<script setup>
import { ref, computed, watch, onMounted, onUnmounted, nextTick } from 'vue'
import { distillationApi, taskApi } from '../api'
import { ElMessage } from 'element-plus'
import WebTerminal from '../components/WebTerminal.vue'
import { Setting } from '@element-plus/icons-vue'
import * as echarts from 'echarts'

const demoMode = ref(false)
const teacherAcc = ref(97.5)
const targetAcc = ref(96.5)
const compressRatio = ref(97.4)
const demoConfigVisible = ref(false)
const tasks = ref([])
const selectedTask = ref('')
const running = ref(false)
const taskResult = ref('')
const wsTaskId = ref('')
const terminalRef = ref(null)
const outputModels = ref([])
const kdHistory = ref(null)
const chartRefs = ref({})
const chartInstances = []

const teacherModel = computed(() => outputModels.value.find(m => m.role === 'teacher'))
const studentModel = computed(() => {
  const t = teacherModel.value
  if (!t || !t.size_mb) return null
  const ratio = compressRatio.value / 100
  const sMb = +(t.size_mb * (1 - ratio)).toFixed(2)
  const tpc = t.param_count || Math.round(t.size_mb * 1024 * 1024 / 4)
  const spc = Math.round(tpc * (1 - ratio))
  return { size_mb: sMb, param_count: spc }
})
const compressionRatio = computed(() => compressRatio.value.toFixed(1))
const studentRatioPercent = computed(() => Math.max(3, 100 - compressRatio.value))

function formatParamCount(model) {
  const pc = model.param_count || Math.round(model.size_mb / 4 * 1024 * 1024)
  if (pc >= 1_000_000) return (pc / 1_000_000).toFixed(2) + ' M'
  if (pc >= 1_000) return (pc / 1_000).toFixed(1) + ' K'
  return pc.toString()
}

function onTaskSelect(row) {
  if (row) selectedTask.value = row.task_id
}

async function loadTasks() {
  try {
    const res = await distillationApi.tasks()
    tasks.value = res.tasks || []
  } catch (e) {
    console.error(e)
  }
}

async function loadTaskDetails(taskId) {
  if (!taskId) return
  try {
    const [modelsRes, histRes] = await Promise.all([
      distillationApi.models(taskId),
      distillationApi.history(taskId),
    ])
    outputModels.value = modelsRes.models || []
    kdHistory.value = histRes.history
    if (kdHistory.value) {
      await nextTick()
      renderCharts()
    }
  } catch (e) {
    console.error(e)
  }
}

function renderCharts() {
  chartInstances.forEach(c => c.dispose())
  chartInstances.length = 0

  if (!kdHistory.value?.edges) return

  for (const [edgeKey, data] of Object.entries(kdHistory.value.edges)) {
    const lossEl = chartRefs.value[edgeKey + '_loss']
    const accEl = chartRefs.value[edgeKey + '_acc']

    if (lossEl) {
      const chart = echarts.init(lossEl)
      chartInstances.push(chart)
      const epochs = data.train_loss?.map((_, i) => i + 1) || []
      chart.setOption({
        title: { text: 'Loss 曲线', textStyle: { fontSize: 13, color: '#ccc' } },
        tooltip: { trigger: 'axis' },
        legend: { textStyle: { color: '#aaa' }, top: 24 },
        grid: { top: 60, bottom: 30, left: 50, right: 20 },
        xAxis: { type: 'category', data: epochs, name: 'Epoch' },
        yAxis: { type: 'value', name: 'Loss' },
        series: [
          data.train_loss && { name: 'Train Loss', type: 'line', data: data.train_loss, smooth: true },
          data.ce_loss && { name: 'CE Loss', type: 'line', data: data.ce_loss, smooth: true, lineStyle: { type: 'dashed' } },
          data.kd_loss && { name: 'KD Loss', type: 'line', data: data.kd_loss, smooth: true, lineStyle: { type: 'dashed' } },
        ].filter(Boolean),
      })
    }

    if (accEl) {
      const chart = echarts.init(accEl)
      chartInstances.push(chart)
      const epochs = data.train_acc?.map((_, i) => i + 1) || data.test_acc?.map((_, i) => i + 1) || []
      chart.setOption({
        title: { text: 'Accuracy 曲线', textStyle: { fontSize: 13, color: '#ccc' } },
        tooltip: { trigger: 'axis' },
        legend: { textStyle: { color: '#aaa' }, top: 24 },
        grid: { top: 60, bottom: 30, left: 50, right: 20 },
        xAxis: { type: 'category', data: epochs, name: 'Epoch' },
        yAxis: { type: 'value', name: 'Acc (%)' },
        series: [
          data.train_acc && { name: 'Train Acc', type: 'line', data: data.train_acc, smooth: true },
          data.test_acc && { name: 'Test Acc', type: 'line', data: data.test_acc, smooth: true },
        ].filter(Boolean),
      })
    }
  }
}

async function startDistillation() {
  taskResult.value = ''
  try {
    const res = await distillationApi.start(selectedTask.value, demoMode.value,
      demoMode.value ? targetAcc.value : null, demoMode.value ? teacherAcc.value : null)
    if (res.status === 'error') {
      ElMessage.error(res.message)
      return
    }
    running.value = true
    wsTaskId.value = selectedTask.value
    ElMessage.success('蒸馏任务已启动')
  } catch (e) {
    ElMessage.error('启动蒸馏失败: ' + (e.message || e))
  }
}

async function stopDistillation() {
  try {
    await distillationApi.stop(selectedTask.value)
    running.value = false
    wsTaskId.value = ''
    ElMessage.warning('蒸馏任务已终止')
  } catch (e) {
    ElMessage.error('终止失败: ' + (e.message || e))
  }
}

function onWsDone(status) {
  running.value = false
  taskResult.value = (status === 'success' || status?.status === 'success') ? 'success' : 'error'
  if (taskResult.value === 'success') {
    ElMessage.success('蒸馏完成')
  }
  loadTaskDetails(selectedTask.value)
}

watch(selectedTask, (taskId) => {
  wsTaskId.value = ''
  taskResult.value = ''
  running.value = false
  outputModels.value = []
  kdHistory.value = null
  loadTaskDetails(taskId)
})

onMounted(() => {
  loadTasks()
})

onUnmounted(() => {
  chartInstances.forEach(c => c.dispose())
  wsTaskId.value = ''
})
</script>

<style scoped>
.compact { padding: 12px 16px; margin-bottom: 10px; }
.page-header.compact { margin-bottom: 8px; }
.page-header.compact h2 { margin-bottom: 2px; }
.page-header.compact p { font-size: 12px; }
.grid-2 { gap: 10px; margin-bottom: 10px; }
.param-compare {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 16px;
  padding: 10px 0 12px;
}
.param-model {
  text-align: center;
  padding: 10px 16px;
  border-radius: 10px;
  min-width: 120px;
}
.param-model.teacher {
  background: rgba(83,168,255,0.08);
  border: 1px solid rgba(83,168,255,0.2);
}
.param-model.student {
  background: rgba(0,212,170,0.08);
  border: 1px solid rgba(0,212,170,0.2);
}
.param-role {
  font-size: 11px;
  color: var(--text-secondary);
  margin-bottom: 4px;
}
.param-size {
  font-size: 18px;
  font-weight: 700;
  color: var(--text-primary);
}
.param-count {
  font-size: 11px;
  color: var(--text-secondary);
  margin-top: 2px;
}
.param-arrow { text-align: center; }
.compress-ratio {
  font-size: 20px;
  font-weight: 700;
  color: #ff9a3e;
  line-height: 1.3;
}
.compress-detail {
  font-size: 11px;
  color: #ff9a3e;
}
.param-bar-wrap {
  display: flex;
  align-items: center;
  gap: 8px;
  margin: 4px 0;
  padding: 0 8px;
}
.param-bar-label {
  font-size: 11px;
  color: var(--text-secondary);
  min-width: 36px;
  text-align: right;
}
.param-bar {
  flex: 1;
  height: 14px;
  background: rgba(255,255,255,0.04);
  border-radius: 4px;
  overflow: hidden;
}
.param-bar-fill {
  height: 100%;
  border-radius: 4px;
  transition: width 0.6s ease;
}
.param-bar-fill.teacher { background: rgba(83,168,255,0.5); }
.param-bar-fill.student { background: rgba(0,212,170,0.5); }
.param-bar-val {
  font-size: 11px;
  color: var(--text-secondary);
  min-width: 50px;
}

.pipeline-flow {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 2px;
  padding: 10px 0;
  flex-wrap: wrap;
}
.flow-step {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 4px;
  min-width: 64px;
}
.flow-icon {
  width: 34px;
  height: 34px;
  border-radius: 10px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 16px;
}
.flow-icon.blue   { background: rgba(83,168,255,0.15); color: #53a8ff; }
.flow-icon.orange  { background: rgba(255,154,62,0.15); color: #ff9a3e; }
.flow-icon.purple  { background: rgba(179,136,255,0.15); color: #b388ff; }
.flow-icon.green   { background: rgba(0,212,170,0.15); color: #00d4aa; }
.flow-icon.red     { background: rgba(233,69,96,0.15); color: #e94560; }
.flow-label {
  font-size: 11px;
  font-weight: 600;
  color: var(--text-primary);
}
.flow-detail {
  font-size: 10px;
  color: var(--text-secondary);
}
.flow-arrow {
  font-size: 14px;
  color: rgba(255,255,255,0.15);
  margin: 0 1px;
}
.chart-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 10px;
}
.chart-box {
  height: 200px;
  border-radius: 8px;
  border: 1px solid rgba(255,255,255,0.06);
  background: rgba(0,0,0,0.15);
}
.term-compact { max-height: 200px; overflow: hidden; }
.term-compact :deep(.term-body) { max-height: 160px; }
</style>
