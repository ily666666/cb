<template>
  <div>
    <div class="page-header">
      <h2>数据接入模块</h2>
      <p>管理电磁信号数据集、查看数据文件、配置任务输入参数</p>
    </div>

    <!-- 数据集概览 -->
    <div class="stats-grid">
      <div class="stat-card" v-for="ds in datasets" :key="ds.name"
           @click="selectDataset(ds.name)"
           :style="{ cursor: 'pointer', borderColor: selected === ds.name ? 'var(--accent-blue)' : '' }">
        <div class="stat-icon" :class="dsColors[ds.name] || 'blue'">
          <el-icon><Coin /></el-icon>
        </div>
        <div class="stat-info">
          <div class="value" style="font-size: 18px;">{{ ds.name }}</div>
          <div class="label">{{ ds.num_classes }} 类 / 长度 {{ ds.signal_length }} / {{ ds.data_file_count }} 文件</div>
          <div class="label">{{ ds.total_size_mb }} MB</div>
        </div>
      </div>
    </div>

    <div class="grid-2" v-if="detail">
      <!-- 原始数据文件 -->
      <div class="card">
        <div class="card-title"><el-icon><Document /></el-icon> 原始数据文件 ({{ detail.data_files?.length || 0 }})</div>
        <el-table :data="detail.data_files?.slice(0, 20)" size="small" max-height="360">
          <el-table-column prop="filename" label="文件名" />
          <el-table-column prop="size_mb" label="大小 (MB)" width="100" />
          <el-table-column label="操作" width="80">
            <template #default="{ row }">
              <el-button link type="primary" size="small" @click="previewFile(row.filename)">预览</el-button>
            </template>
          </el-table-column>
        </el-table>
        <div v-if="(detail.data_files?.length || 0) > 20" style="text-align: center; margin-top: 8px; font-size: 12px; color: var(--text-secondary);">
          仅显示前 20 个文件，共 {{ detail.data_files.length }} 个
        </div>
      </div>

      <!-- 训练划分数据 -->
      <div class="card">
        <div class="card-title"><el-icon><Files /></el-icon> 训练划分数据</div>
        <el-table :data="detail.split_files" size="small" max-height="360">
          <el-table-column prop="filename" label="文件名" />
          <el-table-column prop="size_mb" label="大小 (MB)" width="100" />
          <el-table-column label="操作" width="80">
            <template #default="{ row }">
              <el-button link type="primary" size="small" @click="previewSplitFile(row.filename)">预览</el-button>
            </template>
          </el-table-column>
        </el-table>
        <el-empty v-if="!detail.split_files?.length" description="暂无划分数据" />
      </div>
    </div>

    <!-- 任务配置管理 -->
    <div class="card">
      <div class="card-title">
        <el-icon><Setting /></el-icon> 任务配置管理
        <el-select v-model="selectedTask" placeholder="选择任务" size="small" style="margin-left: auto; width: 260px;"
                   @change="loadTaskConfigs">
          <el-option v-for="t in tasks" :key="t.task_id" :label="t.task_id" :value="t.task_id">
            <span>{{ t.task_id }}</span>
            <span style="float: right; color: var(--text-secondary); font-size: 12px;">{{ t.purpose }}</span>
          </el-option>
        </el-select>
      </div>

      <div v-if="taskConfigs.length">
        <el-collapse accordion v-model="activeConfig">
          <el-collapse-item v-for="cfg in taskConfigs" :key="cfg.filename" :title="cfg.filename" :name="cfg.filename">
            <Codemirror
              v-model="editBuffers[cfg.filename]"
              :extensions="cmExtensions"
              :style="{ fontSize: '13px', borderRadius: '8px', overflow: 'hidden' }"
              placeholder="JSON 配置内容"
              @ready="initEditBuffer(cfg)" />
            <div style="display: flex; gap: 8px; margin-top: 10px; justify-content: flex-end;">
              <el-button size="small" @click="formatJson(cfg.filename)">
                <el-icon><MagicStick /></el-icon> 格式化
              </el-button>
              <el-button size="small" @click="resetEdit(cfg)">
                <el-icon><RefreshLeft /></el-icon> 还原
              </el-button>
              <el-button type="primary" size="small" @click="saveConfig(cfg.filename)" :loading="saving">
                <el-icon><Check /></el-icon> 保存
              </el-button>
            </div>
          </el-collapse-item>
        </el-collapse>
      </div>
      <el-empty v-else-if="selectedTask" description="该任务无配置文件" />
    </div>

    <!-- 文件预览对话框 -->
    <el-dialog v-model="previewVisible" title="文件预览" width="500px">
      <div v-if="previewData">
        <p><strong>文件名:</strong> {{ previewData.filename }}</p>
        <p><strong>大小:</strong> {{ previewData.size_mb }} MB</p>
        <pre class="report-content" style="max-height: 300px;">{{ JSON.stringify(previewData, null, 2) }}</pre>
      </div>
    </el-dialog>
  </div>
</template>

<script setup>
import { ref, reactive, onMounted } from 'vue'
import { ElMessage } from 'element-plus'
import { Codemirror } from 'vue-codemirror'
import { json } from '@codemirror/lang-json'
import { oneDark } from '@codemirror/theme-one-dark'
import { dataApi, taskApi } from '../api'

const cmExtensions = [json(), oneDark]

const datasets = ref([])
const selected = ref('')
const detail = ref(null)
const tasks = ref([])
const selectedTask = ref('')
const taskConfigs = ref([])
const activeConfig = ref('')
const editBuffers = reactive({})
const saving = ref(false)
const previewVisible = ref(false)
const previewData = ref(null)
const dsColors = { link11: 'blue', rml2016: 'green', radar: 'orange', ratr: 'purple' }

async function selectDataset(name) {
  selected.value = name
  try {
    detail.value = await dataApi.datasetDetail(name)
  } catch (e) {
    console.error(e)
  }
}

async function previewFile(filename) {
  try {
    previewData.value = await dataApi.filePreview(selected.value, filename)
    previewVisible.value = true
  } catch (e) {
    console.error(e)
  }
}

async function previewSplitFile(filename) {
  try {
    previewData.value = await dataApi.filePreview(selected.value, filename)
    previewVisible.value = true
  } catch (e) {
    console.error(e)
  }
}

async function loadTaskConfigs() {
  if (!selectedTask.value) return
  try {
    const res = await dataApi.taskConfigs(selectedTask.value)
    taskConfigs.value = res.configs || []
    for (const cfg of taskConfigs.value) {
      editBuffers[cfg.filename] = JSON.stringify(cfg.content, null, 4)
    }
  } catch (e) {
    console.error(e)
  }
}

function initEditBuffer(cfg) {
  if (!editBuffers[cfg.filename]) {
    editBuffers[cfg.filename] = JSON.stringify(cfg.content, null, 4)
  }
}

function resetEdit(cfg) {
  editBuffers[cfg.filename] = JSON.stringify(cfg.content, null, 4)
  ElMessage.info('已还原')
}

function formatJson(filename) {
  try {
    const obj = JSON.parse(editBuffers[filename])
    editBuffers[filename] = JSON.stringify(obj, null, 4)
    ElMessage.success('已格式化')
  } catch (e) {
    ElMessage.error('JSON 格式错误: ' + e.message)
  }
}

async function saveConfig(filename) {
  let obj
  try {
    obj = JSON.parse(editBuffers[filename])
  } catch (e) {
    ElMessage.error('JSON 格式错误，无法保存: ' + e.message)
    return
  }
  saving.value = true
  try {
    await dataApi.saveConfig(selectedTask.value, filename, obj)
    ElMessage.success(filename + ' 已保存')
    const cfg = taskConfigs.value.find(c => c.filename === filename)
    if (cfg) cfg.content = obj
  } catch (e) {
    ElMessage.error('保存失败: ' + e.message)
  } finally {
    saving.value = false
  }
}

onMounted(async () => {
  try {
    const [dsRes, taskRes] = await Promise.all([
      dataApi.datasets(),
      taskApi.list(),
    ])
    datasets.value = dsRes.datasets || []
    tasks.value = taskRes.tasks || []
    if (datasets.value.length) selectDataset(datasets.value[0].name)
  } catch (e) {
    console.error(e)
  }
})
</script>
