<template>
  <div>
    <div class="page-header">
      <h2>模型推理计算</h2>
      <p>执行推理/训练任务、监控运行状态、查看输出结果</p>
    </div>

    <div class="grid-2">
      <!-- 推理模式 -->
      <div class="card">
        <div class="card-title"><el-icon><Cpu /></el-icon> 推理模式</div>
        <div class="mode-list">
          <div v-for="(info, key) in inferenceModes" :key="key" class="mode-item"
               :class="{ active: resolvedMode === key }">
            <div class="mode-name">{{ info.label }}</div>
            <div class="mode-steps">
              <el-tag v-for="s in info.steps" :key="s" size="small" type="info" style="margin: 2px;">{{ s }}</el-tag>
            </div>
          </div>
        </div>
      </div>

      <!-- 训练模式 -->
      <div class="card">
        <div class="card-title"><el-icon><Opportunity /></el-icon> 训练模式</div>
        <div class="mode-list">
          <div v-for="(info, key) in trainModes" :key="key" class="mode-item"
               :class="{ active: resolvedMode === key, clickable: isTrainTask }"
               @click="isTrainTask && (trainMode = key)">
            <div class="mode-name">{{ info.label }}</div>
            <div class="mode-steps">
              <el-tag v-for="s in info.steps" :key="s" size="small" type="warning" style="margin: 2px;">{{ s }}</el-tag>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- 执行区域 -->
    <div class="card">
      <div class="card-title"><el-icon><VideoPlay /></el-icon> 执行任务</div>
      <div class="exec-row">
        <el-select v-model="selectedTask" placeholder="选择任务" style="width: 280px;" filterable>
          <el-option-group v-for="g in taskGroups" :key="g.label" :label="g.label">
            <el-option v-for="t in g.tasks" :key="t.task_id" :label="t.task_id" :value="t.task_id">
              <span>{{ t.task_id }}</span>
              <span style="float: right; color: var(--text-secondary); font-size: 12px;">{{ t.dataset }}</span>
            </el-option>
          </el-option-group>
        </el-select>

        <el-tag v-if="resolvedMode" type="primary">{{ resolvedModeLabel }}</el-tag>
        <span v-if="isTrainTask && !trainMode" style="color: var(--text-secondary); font-size: 13px;">← 请在上方选择训练模式</span>

        <el-button type="primary" @click="startTask" :disabled="!canStart" :loading="starting">
          <el-icon><CaretRight /></el-icon> 开始执行
        </el-button>
        <el-button type="danger" plain :disabled="!selectedTask" :loading="cleaning" @click="confirmClean">
          <el-icon><Delete /></el-icon> 清空结果
        </el-button>
      </div>
      <el-alert v-if="hasOldResult" type="warning" show-icon :closable="false" style="margin-top: 12px;">
        <template #title>
          该任务已有历史结果，建议执行前先
          <el-button type="danger" size="small" plain style="margin-left: 4px;" :loading="cleaning" @click="confirmClean">清空结果</el-button>
        </template>
      </el-alert>
    </div>

    <!-- 任务监控 -->
    <div class="card" v-if="activeTasks.length">
      <div class="card-title">
        <el-icon><Monitor /></el-icon> 任务监控
        <el-button size="small" style="margin-left: auto;" @click="loadActiveTasks">
          <el-icon><Refresh /></el-icon>
        </el-button>
      </div>
      <div class="task-tabs">
        <div v-for="t in activeTasks" :key="t.task_id" class="task-tab"
             :class="{ active: viewingTask === t.task_id }" @click="viewTask(t.task_id)">
          <el-tag :type="tagType(t.status)" size="small" style="margin-right: 6px;" effect="dark">
            {{ statusLabel(t.status) }}
          </el-tag>
          <span class="task-tab-name">{{ t.task_id }}</span>
          <span class="task-tab-time">{{ formatTime(t.started_at) }}</span>
          <span class="task-tab-actions" @click.stop>
            <el-button v-if="t.status === 'running'" type="warning" size="small" plain @click="stopTask(t.task_id)">
              <el-icon><VideoPause /></el-icon>
            </el-button>
            <el-button v-else type="danger" size="small" plain @click="confirmRemove(t.task_id)">
              <el-icon><Close /></el-icon>
            </el-button>
          </span>
        </div>
      </div>
    </div>

    <!-- 运行终端 -->
    <div v-if="viewingTask && (viewStatus || wsTaskId)">
      <div v-if="viewStatus?.command" style="margin-bottom: 8px;">
        <code style="font-size: 12px; color: var(--accent-blue);">$ {{ viewStatus.command }}</code>
      </div>
      <WebTerminal :taskId="wsTaskId" :title="viewingTask" @done="onWsDone" ref="terminalRef" />
    </div>

    <!-- 任务结果（执行过的 或 历史文件） -->
    <div class="card" v-if="displayResult">
      <div class="card-title"><el-icon><DataAnalysis /></el-icon> 任务结果 — {{ displayResultTask }}</div>

      <div v-if="displayResult.output_steps?.length" style="margin-bottom: 16px;">
        <h4 style="font-size: 14px; margin-bottom: 8px; color: var(--text-secondary);">输出步骤</h4>
        <el-table :data="displayResult.output_steps" size="small">
          <el-table-column prop="name" label="步骤" width="200" />
          <el-table-column label="输出文件">
            <template #default="{ row }">
              <el-tag v-for="f in row.files" :key="f.filename" size="small" style="margin: 2px;">
                {{ f.filename }} ({{ f.size_mb }}MB)
              </el-tag>
            </template>
          </el-table-column>
        </el-table>
      </div>

      <div v-if="Object.keys(displayResult.reports || {}).length">
        <h4 style="font-size: 14px; margin-bottom: 8px; color: var(--text-secondary);">结果报告</h4>
        <el-tabs type="border-card" style="background: transparent;">
          <el-tab-pane v-for="(content, name) in displayResult.reports" :key="name" :label="name">
            <pre class="report-content">{{ content }}</pre>
          </el-tab-pane>
        </el-tabs>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, watch, onMounted, onUnmounted } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { taskApi, inferenceApi } from '../api'
import WebTerminal from '../components/WebTerminal.vue'

const PURPOSE_MODE_MAP = {
  '协同推理': 'device_to_edge_to_cloud',
  '仅云推理': 'device_to_cloud',
  '仅边推理': 'device_to_edge',
}

const tasks = ref([])
const selectedTask = ref('')
const trainMode = ref('')
const starting = ref(false)
const cleaning = ref(false)
const inferenceModes = ref({})
const trainModes = ref({})

const activeTasks = ref([])
const viewingTask = ref('')
const viewStatus = ref(null)
const viewResult = ref(null)
const selectedResult = ref(null)
const wsTaskId = ref('')
const terminalRef = ref(null)

const taskGroups = computed(() => {
  const inferTasks = tasks.value.filter(t => t.purpose !== '训练')
  const trainTasks = tasks.value.filter(t => t.purpose === '训练')
  const groups = []
  if (inferTasks.length) groups.push({ label: '推理任务', tasks: inferTasks })
  if (trainTasks.length) groups.push({ label: '训练任务', tasks: trainTasks })
  return groups
})

const selectedTaskInfo = computed(() => tasks.value.find(t => t.task_id === selectedTask.value))
const isTrainTask = computed(() => selectedTaskInfo.value?.purpose === '训练')

const resolvedMode = computed(() => {
  if (!selectedTaskInfo.value) return ''
  if (isTrainTask.value) return trainMode.value
  return PURPOSE_MODE_MAP[selectedTaskInfo.value.purpose] || ''
})

const resolvedModeLabel = computed(() => {
  if (!resolvedMode.value) return ''
  const all = { ...inferenceModes.value, ...trainModes.value }
  return all[resolvedMode.value]?.label || resolvedMode.value
})

const resolvedSteps = computed(() => {
  if (!resolvedMode.value) return []
  const all = { ...inferenceModes.value, ...trainModes.value }
  return all[resolvedMode.value]?.steps || []
})

const canStart = computed(() => selectedTask.value && resolvedMode.value)

const hasOldResult = computed(() => {
  const info = selectedTaskInfo.value
  return info && (info.has_output || info.has_result)
})

const displayResult = computed(() => viewResult.value || selectedResult.value)
const displayResultTask = computed(() => {
  if (viewResult.value) return viewingTask.value
  if (selectedResult.value) return selectedTask.value
  return ''
})

function tagType(status) {
  if (status === 'running') return 'warning'
  if (status === 'success') return 'success'
  if (status === 'stopped') return 'info'
  if (status === 'error') return 'danger'
  return 'info'
}

function statusLabel(status) {
  const map = { running: '运行中', success: '完成', stopped: '已停止', error: '失败' }
  return map[status] || status
}


function formatTime(iso) {
  if (!iso) return ''
  return iso.replace('T', ' ').slice(0, 19)
}

watch(selectedTask, async (tid) => {
  trainMode.value = ''
  selectedResult.value = null
  viewingTask.value = ''
  viewStatus.value = null
  viewResult.value = null
  wsTaskId.value = ''

  if (!tid) return
  const info = tasks.value.find(t => t.task_id === tid)
  if (info && (info.has_output || info.has_result)) {
    try {
      selectedResult.value = await inferenceApi.result(tid).catch(() => null)
    } catch {}
  }
})

async function startTask() {
  if (!canStart.value) return
  starting.value = true
  try {
    await taskApi.run(selectedTask.value, { mode: resolvedMode.value })
    ElMessage.success('任务已启动')
    await loadActiveTasks()
    viewTask(selectedTask.value)
  } catch (e) {
    ElMessage.error(e.message)
  } finally {
    starting.value = false
  }
}

function onWsDone() {
  loadActiveTasks()
  if (viewingTask.value) {
    inferenceApi.result(viewingTask.value).then(r => { viewResult.value = r }).catch(() => {})
  }
}

async function confirmClean() {
  try {
    await ElMessageBox.confirm('确认清空该任务的 output 和 result？', '清空结果', { type: 'warning', confirmButtonText: '确认', cancelButtonText: '取消' })
    await cleanOutput()
  } catch {}
}

async function confirmRemove(taskId) {
  try {
    await ElMessageBox.confirm('移除该任务记录？', '移除', { type: 'warning', confirmButtonText: '确认', cancelButtonText: '取消' })
    await removeRecord(taskId)
  } catch {}
}

async function cleanOutput() {
  if (!selectedTask.value) return
  cleaning.value = true
  try {
    const res = await taskApi.clean(selectedTask.value)
    ElMessage.success(res.message || '已清空')
    selectedResult.value = null
    if (viewingTask.value === selectedTask.value) {
      viewStatus.value = null
      viewResult.value = null
    }
    const taskRes = await taskApi.list()
    tasks.value = taskRes.tasks || []
    await loadActiveTasks()
  } catch (e) {
    ElMessage.error(e.message)
  } finally {
    cleaning.value = false
  }
}

async function stopTask(taskId) {
  try {
    const res = await taskApi.stop(taskId)
    ElMessage.success(res.message || '已停止')
    await loadActiveTasks()
  } catch (e) {
    ElMessage.error(e.message)
  }
}

async function removeRecord(taskId) {
  try {
    const res = await taskApi.removeRecord(taskId)
    ElMessage.success(res.message || '已移除')
    if (viewingTask.value === taskId) {
      viewingTask.value = ''
      viewStatus.value = null
      viewResult.value = null
    }
    await loadActiveTasks()
  } catch (e) {
    ElMessage.error(e.message)
  }
}

async function loadActiveTasks() {
  try {
    const res = await taskApi.activeTasks()
    activeTasks.value = res.tasks || []
  } catch (e) {
    console.error(e)
  }
}

async function viewTask(taskId) {
  viewingTask.value = taskId
  wsTaskId.value = ''
  viewStatus.value = null
  viewResult.value = null

  try {
    const [status, result] = await Promise.all([
      taskApi.status(taskId),
      inferenceApi.result(taskId).catch(() => null),
    ])
    viewStatus.value = status?.status !== 'idle' ? status : null
    viewResult.value = result
    if (status?.status && status.status !== 'idle') {
      wsTaskId.value = taskId
    }
  } catch (e) {
    console.error(e)
  }
}

onMounted(async () => {
  try {
    const [taskRes, modesRes] = await Promise.all([
      taskApi.list(),
      inferenceApi.modes(),
    ])
    tasks.value = taskRes.tasks || []
    inferenceModes.value = modesRes.inference_modes || {}
    trainModes.value = modesRes.train_modes || {}

    await loadActiveTasks()
    const running = activeTasks.value.find(t => t.status === 'running')
    if (running) {
      viewTask(running.task_id)
    } else if (activeTasks.value.length) {
      viewTask(activeTasks.value[0].task_id)
    }
  } catch (e) {
    console.error(e)
  }
})

onUnmounted(() => { wsTaskId.value = '' })
</script>

<style scoped>
.mode-list {
  display: flex;
  flex-direction: column;
  gap: 8px;
}
.mode-item {
  padding: 12px 16px;
  border-radius: 8px;
  border: 1px solid rgba(255,255,255,0.06);
  transition: all 0.2s;
  opacity: 0.5;
}
.mode-item.active {
  border-color: var(--accent-blue);
  background: rgba(83,168,255,0.1);
  opacity: 1;
}
.mode-item.clickable {
  cursor: pointer;
  opacity: 0.8;
}
.mode-item.clickable:hover {
  border-color: rgba(83,168,255,0.3);
  background: rgba(83,168,255,0.04);
  opacity: 1;
}
.mode-name {
  font-weight: 600;
  font-size: 14px;
  color: #fff;
  margin-bottom: 6px;
}
.mode-steps {
  display: flex;
  flex-wrap: wrap;
  gap: 4px;
}
.exec-row {
  display: flex;
  gap: 12px;
  align-items: center;
  flex-wrap: wrap;
}
.task-tabs {
  display: flex;
  flex-direction: column;
  gap: 6px;
}
.task-tab {
  display: flex;
  align-items: center;
  padding: 10px 14px;
  border-radius: 8px;
  border: 1px solid rgba(255,255,255,0.06);
  cursor: pointer;
  transition: all 0.2s;
}
.task-tab:hover {
  border-color: rgba(83,168,255,0.3);
  background: rgba(83,168,255,0.04);
}
.task-tab.active {
  border-color: var(--accent-blue);
  background: rgba(83,168,255,0.1);
}
.task-tab-name {
  font-size: 13px;
  font-weight: 500;
  color: #fff;
}
.task-tab-time {
  margin-left: auto;
  font-size: 12px;
  color: var(--text-secondary);
}
.task-tab-actions {
  margin-left: 8px;
  display: flex;
  flex-shrink: 0;
}
:deep(.el-tabs__content) { padding: 12px; }
:deep(.el-tabs__header) { background: rgba(255,255,255,0.02); }
</style>
