<template>
  <div>
    <div class="page-header">
      <h2>知识蒸馏</h2>
      <p>通过教师-学生模型蒸馏，将大模型知识迁移至轻量学生模型，实现边侧高效部署</p>
    </div>

    <!-- 蒸馏流程图 -->
    <div class="card">
      <div class="card-title"><el-icon><MagicStick /></el-icon> 蒸馏流程</div>
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

    <!-- 选择训练任务 -->
    <div class="card">
      <div class="card-title"><el-icon><List /></el-icon> 选择训练任务</div>
      <el-table :data="tasks" size="small" max-height="360" highlight-current-row
                @current-change="onTaskSelect" style="width: 100%;">
        <el-table-column label="" width="50">
          <template #default="{ row }">
            <el-radio v-model="selectedTask" :value="row.task_id" />
          </template>
        </el-table-column>
        <el-table-column prop="task_id" label="任务ID" min-width="180" show-overflow-tooltip />
        <el-table-column prop="dataset_type" label="数据集" width="100" />
        <el-table-column prop="teacher_model" label="教师模型" width="200" show-overflow-tooltip />
        <el-table-column prop="student_model" label="学生模型" width="200" show-overflow-tooltip />
        <el-table-column label="KD配置" width="80" align="center">
          <template #default="{ row }">
            <el-tag v-if="row.has_kd_config" type="success" size="small">有</el-tag>
            <el-tag v-else type="info" size="small">无</el-tag>
          </template>
        </el-table-column>
        <el-table-column label="教师权重" width="80" align="center">
          <template #default="{ row }">
            <el-tag v-if="row.has_teacher" type="success" size="small">有</el-tag>
            <el-tag v-else type="warning" size="small">需训练</el-tag>
          </template>
        </el-table-column>
      </el-table>
    </div>

    <!-- 执行蒸馏 -->
    <div class="card" v-if="selectedTask">
      <div style="display: flex; align-items: center; gap: 16px; margin-bottom: 12px;">
        <el-button v-if="!running" type="primary" size="large" @click="startDistillation">
          <el-icon><VideoPlay /></el-icon> 开始蒸馏
        </el-button>
        <el-button v-else type="danger" size="large" @click="stopDistillation">
          <el-icon><VideoPause /></el-icon> 终止蒸馏
        </el-button>
        <span style="font-size: 13px; color: var(--text-secondary);">
          任务: {{ selectedTask }}
        </span>
        <el-tag v-if="taskResult === 'success'" type="success">蒸馏完成</el-tag>
        <el-tag v-if="taskResult === 'error'" type="danger">蒸馏失败</el-tag>
      </div>

      <WebTerminal v-if="wsTaskId" :taskId="wsTaskId" :title="'知识蒸馏 — ' + selectedTask"
        ref="terminalRef" @done="onWsDone" />
    </div>

    <!-- 产出模型 -->
    <div class="card" v-if="selectedTask && outputModels.length">
      <div class="card-title"><el-icon><Files /></el-icon> 蒸馏产出模型</div>
      <el-table :data="outputModels" size="small" max-height="300">
        <el-table-column label="角色" width="80">
          <template #default="{ row }">
            <el-tag :type="row.role === 'teacher' ? 'primary' : row.role === 'student' ? 'success' : 'info'" size="small">
              {{ row.role === 'teacher' ? '教师' : row.role === 'student' ? '学生' : '其他' }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="name" label="模型文件" min-width="220" show-overflow-tooltip />
        <el-table-column prop="step" label="训练步骤" width="160" />
        <el-table-column prop="size_mb" label="大小(MB)" width="90" />
      </el-table>
    </div>

    <!-- 蒸馏历史曲线 -->
    <div class="card" v-if="selectedTask && kdHistory">
      <div class="card-title"><el-icon><TrendCharts /></el-icon> 蒸馏训练曲线</div>
      <div v-for="(edgeData, edgeKey) in kdHistory.edges" :key="edgeKey" style="margin-bottom: 24px;">
        <h4 style="font-size: 14px; margin-bottom: 12px; color: var(--text-primary);">{{ edgeKey }}</h4>
        <div class="chart-grid">
          <div :ref="el => chartRefs[edgeKey + '_loss'] = el" class="chart-box"></div>
          <div :ref="el => chartRefs[edgeKey + '_acc'] = el" class="chart-box"></div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, watch, onMounted, onUnmounted, nextTick } from 'vue'
import { distillationApi, taskApi } from '../api'
import { ElMessage } from 'element-plus'
import WebTerminal from '../components/WebTerminal.vue'
import * as echarts from 'echarts'

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
    const res = await distillationApi.start(selectedTask.value)
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
.pipeline-flow {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 4px;
  padding: 20px 0;
  flex-wrap: wrap;
}
.flow-step {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 6px;
  min-width: 80px;
}
.flow-icon {
  width: 44px;
  height: 44px;
  border-radius: 12px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 20px;
}
.flow-icon.blue   { background: rgba(83,168,255,0.15); color: #53a8ff; }
.flow-icon.orange  { background: rgba(255,154,62,0.15); color: #ff9a3e; }
.flow-icon.purple  { background: rgba(179,136,255,0.15); color: #b388ff; }
.flow-icon.green   { background: rgba(0,212,170,0.15); color: #00d4aa; }
.flow-icon.red     { background: rgba(233,69,96,0.15); color: #e94560; }
.flow-label {
  font-size: 12px;
  font-weight: 600;
  color: var(--text-primary);
}
.flow-detail {
  font-size: 11px;
  color: var(--text-secondary);
}
.flow-arrow {
  font-size: 18px;
  color: rgba(255,255,255,0.15);
  margin: 0 2px;
}
.chart-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 16px;
}
.chart-box {
  height: 280px;
  border-radius: 8px;
  border: 1px solid rgba(255,255,255,0.06);
  background: rgba(0,0,0,0.15);
}
</style>
