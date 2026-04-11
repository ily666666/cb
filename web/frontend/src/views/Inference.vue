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
               :class="{ selected: selectedMode === key }" @click="selectedMode = key">
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
               :class="{ selected: selectedMode === key }" @click="selectedMode = key">
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
      <div style="display: flex; gap: 12px; align-items: center; flex-wrap: wrap;">
        <el-select v-model="selectedTask" placeholder="选择任务" style="width: 260px;" filterable>
          <el-option v-for="t in tasks" :key="t.task_id" :label="t.task_id" :value="t.task_id">
            <span>{{ t.task_id }}</span>
            <span style="float: right; color: var(--text-secondary); font-size: 12px;">{{ t.purpose }}</span>
          </el-option>
        </el-select>
        <el-tag v-if="selectedMode" type="primary">{{ selectedMode }}</el-tag>
        <el-button type="primary" @click="startTask" :disabled="!selectedTask || !selectedMode" :loading="starting">
          <el-icon><CaretRight /></el-icon> 开始执行
        </el-button>
        <el-button @click="refreshStatus" :disabled="!selectedTask">
          <el-icon><Refresh /></el-icon> 刷新状态
        </el-button>
      </div>
    </div>

    <!-- 运行状态 -->
    <div class="card" v-if="runStatus">
      <div class="card-title">
        <el-icon><Monitor /></el-icon> 运行状态
        <el-tag :type="statusTagType" style="margin-left: 8px;">{{ runStatus.status }}</el-tag>
        <span v-if="runStatus.started_at" style="margin-left: auto; font-size: 12px; color: var(--text-secondary);">
          启动于 {{ runStatus.started_at }}
        </span>
      </div>
      <div v-if="runStatus.command" style="margin-bottom: 12px;">
        <code style="font-size: 12px; color: var(--accent-blue);">{{ runStatus.command }}</code>
      </div>
      <div class="report-content" ref="logRef" style="max-height: 400px;">{{ (runStatus.output_lines || []).join('\n') }}</div>
    </div>

    <!-- 任务结果 -->
    <div class="card" v-if="taskResult">
      <div class="card-title"><el-icon><DataAnalysis /></el-icon> 任务结果</div>

      <div v-if="taskResult.output_steps?.length" style="margin-bottom: 16px;">
        <h4 style="font-size: 14px; margin-bottom: 8px; color: var(--text-secondary);">输出步骤</h4>
        <el-table :data="taskResult.output_steps" size="small">
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

      <div v-if="Object.keys(taskResult.reports || {}).length">
        <h4 style="font-size: 14px; margin-bottom: 8px; color: var(--text-secondary);">结果报告</h4>
        <el-tabs type="border-card" style="background: transparent;">
          <el-tab-pane v-for="(content, name) in taskResult.reports" :key="name" :label="name">
            <pre class="report-content">{{ content }}</pre>
          </el-tab-pane>
        </el-tabs>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, nextTick } from 'vue'
import { ElMessage } from 'element-plus'
import { taskApi, inferenceApi } from '../api'

const tasks = ref([])
const selectedTask = ref('')
const selectedMode = ref('')
const starting = ref(false)
const runStatus = ref(null)
const taskResult = ref(null)
const logRef = ref(null)
const inferenceModes = ref({})
const trainModes = ref({})
let pollTimer = null

const statusTagType = computed(() => {
  const s = runStatus.value?.status
  if (s === 'running') return 'warning'
  if (s === 'success') return 'success'
  if (s === 'error') return 'danger'
  return 'info'
})

async function startTask() {
  if (!selectedTask.value || !selectedMode.value) return
  starting.value = true
  try {
    await taskApi.run(selectedTask.value, { mode: selectedMode.value, summary: true })
    ElMessage.success('任务已启动')
    startPolling()
  } catch (e) {
    ElMessage.error(e.message)
  } finally {
    starting.value = false
  }
}

async function refreshStatus() {
  if (!selectedTask.value) return
  try {
    const [status, result] = await Promise.all([
      taskApi.status(selectedTask.value),
      inferenceApi.result(selectedTask.value).catch(() => null),
    ])
    runStatus.value = status
    taskResult.value = result
    await nextTick()
    if (logRef.value) logRef.value.scrollTop = logRef.value.scrollHeight
  } catch (e) {
    console.error(e)
  }
}

function startPolling() {
  stopPolling()
  pollTimer = setInterval(async () => {
    await refreshStatus()
    if (runStatus.value?.status && runStatus.value.status !== 'running') {
      stopPolling()
    }
  }, 2000)
}

function stopPolling() {
  if (pollTimer) {
    clearInterval(pollTimer)
    pollTimer = null
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
  } catch (e) {
    console.error(e)
  }
})
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
  cursor: pointer;
  transition: all 0.2s;
}
.mode-item:hover {
  border-color: rgba(83,168,255,0.3);
  background: rgba(83,168,255,0.04);
}
.mode-item.selected {
  border-color: var(--accent-blue);
  background: rgba(83,168,255,0.1);
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
:deep(.el-tabs__content) { padding: 12px; }
:deep(.el-tabs__header) { background: rgba(255,255,255,0.02); }
</style>
