<template>
  <div>
    <div class="page-header">
      <h2>模型轻量化</h2>
      <p>通过剪枝与量化技术压缩模型，适配边侧 / 端侧部署</p>
    </div>

    <!-- 压缩方法卡片 -->
    <div class="card">
      <div class="card-title"><el-icon><ScaleToOriginal /></el-icon> 选择压缩方案</div>
      <div class="method-grid">
        <div v-for="m in methods" :key="m.id" class="method-card"
             :class="{ active: selectedMethod === m.id }" @click="selectedMethod = m.id">
          <div class="method-header">
            <span class="method-name">{{ m.name }}</span>
          </div>
          <div class="method-tags">
            <el-tag size="small">{{ m.prune_type }}</el-tag>
            <el-tag size="small" type="warning">{{ m.quant_type }}</el-tag>
          </div>
          <div class="method-desc">{{ m.description }}</div>
        </div>
      </div>
    </div>

    <!-- 参数配置 -->
    <div class="grid-2" v-if="currentMethod">
      <div class="card">
        <div class="card-title"><el-icon><Operation /></el-icon> 压缩参数配置</div>
        <el-form label-width="120px" size="default" style="max-width: 480px;">
          <el-form-item label="源模型">
            <el-select v-model="selectedModel" placeholder="选择待压缩模型" filterable style="width: 100%;">
              <el-option v-for="m in models" :key="m.path" :value="m.path"
                :label="`${m.task_id} / ${m.step} / ${m.name}`">
                <div style="display:flex; justify-content:space-between; align-items:center; width:100%;">
                  <span>{{ m.name }}</span>
                  <span style="font-size:11px; color:var(--text-secondary);">{{ m.dataset_type }} · {{ m.size_mb }}MB · {{ m.task_id }}</span>
                </div>
              </el-option>
            </el-select>
            <div v-if="selectedModelInfo" style="margin-top: 6px; font-size: 12px; color: var(--text-secondary);">
              {{ selectedModelInfo.model_type }} · {{ selectedModelInfo.dataset_type }} · {{ selectedModelInfo.num_classes }} 类
            </div>
          </el-form-item>

          <el-form-item label="训练数据">
            <el-select v-model="selectedData" placeholder="选择数据集" filterable style="width: 100%;">
              <el-option v-for="d in filteredDatasets" :key="d.path" :value="d.path"
                :label="`${d.dataset_type} / ${d.name}`">
                <div style="display:flex; justify-content:space-between; align-items:center; width:100%;">
                  <span>{{ d.name }}</span>
                  <span style="font-size:11px; color:var(--text-secondary);">{{ d.dataset_type }} · {{ d.size_mb }}MB</span>
                </div>
              </el-option>
            </el-select>
          </el-form-item>

          <template v-for="(cfg, key) in currentMethod.params" :key="key">
            <el-form-item :label="cfg.label" v-if="cfg.options">
              <el-select v-model="paramValues[key]" style="width: 160px;">
                <el-option v-for="opt in cfg.options" :key="opt" :label="opt + ' bit'" :value="opt" />
              </el-select>
            </el-form-item>
            <el-form-item :label="cfg.label" v-else-if="cfg.min !== undefined && !Array.isArray(cfg.default)">
              <el-slider v-model="paramValues[key]" :min="cfg.min" :max="cfg.max"
                :step="cfg.step || 1" show-input :show-input-controls="false" style="padding-right: 12px;" />
            </el-form-item>
          </template>
        </el-form>
      </div>

      <!-- 方案流程图 -->
      <div class="card">
        <div class="card-title"><el-icon><Promotion /></el-icon> 压缩流程</div>
        <div class="pipeline-flow">
          <div class="flow-step">
            <div class="flow-icon blue"><el-icon><Cloudy /></el-icon></div>
            <div class="flow-label">加载原始模型</div>
            <div class="flow-detail">{{ selectedModelInfo?.name || '待选择' }}</div>
          </div>
          <div class="flow-arrow">→</div>
          <div class="flow-step">
            <div class="flow-icon orange"><el-icon><Scissor /></el-icon></div>
            <div class="flow-label">{{ selectedMethod?.includes('physical') ? '物理通道剪枝' : 'BN 软掩码剪枝' }}</div>
            <div class="flow-detail">移除 {{ ((paramValues.prune_ratio || 0.15) * 100).toFixed(0) }}% 通道</div>
          </div>
          <div class="flow-arrow">→</div>
          <div class="flow-step">
            <div class="flow-icon purple"><el-icon><Coin /></el-icon></div>
            <div class="flow-label">INT8 线性量化</div>
            <div class="flow-detail">{{ paramValues.num_bits || 8 }}-bit QAT 量化感知训练</div>
          </div>
          <div class="flow-arrow">→</div>
          <div class="flow-step">
            <div class="flow-icon green"><el-icon><CircleCheck /></el-icon></div>
            <div class="flow-label">QAT 微调</div>
            <div class="flow-detail">{{ paramValues.num_epochs || 15 }} Epochs</div>
          </div>
          <div class="flow-arrow">→</div>
          <div class="flow-step">
            <div class="flow-icon red"><el-icon><Download /></el-icon></div>
            <div class="flow-label">导出轻量模型</div>
            <div class="flow-detail">边侧部署</div>
          </div>
        </div>
      </div>
    </div>

    <!-- 执行压缩 -->
    <div class="card" v-if="currentMethod">
      <div style="display: flex; align-items: center; gap: 16px; margin-bottom: 12px;">
        <el-button v-if="!compressing" type="primary" size="large" :disabled="!selectedModel || !selectedData"
          @click="startCompress">
          <el-icon><VideoPlay /></el-icon> 开始压缩
        </el-button>
        <el-button v-else type="danger" size="large" @click="stopCompress">
          <el-icon><VideoPause /></el-icon> 终止压缩
        </el-button>
        <span v-if="!compressing && (!selectedModel || !selectedData)" style="font-size: 13px; color: var(--text-secondary);">请先选择源模型和训练数据</span>
        <span v-if="compressing" style="font-size: 13px; color: var(--text-secondary);">
          {{ taskStatus.method }} · 已运行 {{ Math.round(taskStatus.elapsed_s || 0) }}s
        </span>
        <el-tag v-if="compressResult === 'success'" type="success">压缩完成</el-tag>
        <el-tag v-if="compressResult === 'error'" type="danger">压缩失败</el-tag>
      </div>
      <WebTerminal v-if="showTerminal" ref="terminalRef" :pollFn="pollStatus"
        title="模型压缩" @done="onTermDone" />
    </div>

    <!-- 已有压缩模型 -->
    <div class="card" v-if="models.length">
      <div class="card-title"><el-icon><Files /></el-icon> 可压缩的模型列表</div>
      <el-table :data="models" size="small" max-height="400">
        <el-table-column label="角色" width="80">
          <template #default="{ row }">
            <el-tag :type="row.role === 'teacher' ? 'primary' : row.role === 'student' ? 'success' : 'info'" size="small">
              {{ row.role === 'teacher' ? '教师' : row.role === 'student' ? '学生' : '其他' }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="name" label="模型文件" min-width="180" show-overflow-tooltip />
        <el-table-column prop="model_type" label="模型架构" width="200" show-overflow-tooltip />
        <el-table-column prop="dataset_type" label="数据集" width="90" />
        <el-table-column prop="task_id" label="任务ID" width="150" />
        <el-table-column prop="step" label="训练步骤" width="150" />
        <el-table-column prop="size_mb" label="大小(MB)" width="80" />
      </el-table>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, inject, onMounted, onUnmounted, watch, nextTick } from 'vue'
import { lightweightApi } from '../api'
import { ElMessage } from 'element-plus'
import WebTerminal from '../components/WebTerminal.vue'

const demoMode = inject('demoMode', ref(false))

const methods = ref([])
const models = ref([])
const datasets = ref([])
const selectedMethod = ref('')
const selectedModel = ref('')
const selectedData = ref('')
const paramValues = ref({})
const compressing = ref(false)
const compressResult = ref('')
const taskStatus = ref({})
const showTerminal = ref(false)
const terminalRef = ref(null)

const currentMethod = computed(() => methods.value.find(m => m.id === selectedMethod.value))

const selectedModelInfo = computed(() => models.value.find(m => m.path === selectedModel.value))

const filteredDatasets = computed(() => {
  const info = selectedModelInfo.value
  if (!info?.dataset_type) return datasets.value
  return datasets.value.filter(d => d.dataset_type === info.dataset_type)
})

const pollStatus = () => lightweightApi.status()

async function startCompress() {
  compressResult.value = ''
  taskStatus.value = {}
  try {
    const params = { ...paramValues.value, data_path: selectedData.value }
    if (demoMode.value) params.fast_mode = true
    const res = await lightweightApi.run({
      method_id: selectedMethod.value,
      model_path: selectedModel.value,
      params,
    })
    if (res.status === 'error') {
      ElMessage.error(res.message)
      return
    }
    compressing.value = true
    showTerminal.value = true
    ElMessage.success('压缩任务已启动')
    await nextTick()
    terminalRef.value?.startPolling()
  } catch (e) {
    ElMessage.error('启动压缩失败: ' + (e.message || e))
  }
}

async function stopCompress() {
  try {
    await lightweightApi.stop()
    compressing.value = false
    compressResult.value = ''
    terminalRef.value?.stopPolling()
    ElMessage.warning('压缩任务已终止')
  } catch (e) {
    ElMessage.error('终止失败: ' + (e.message || e))
  }
}

function onTermDone(status) {
  compressing.value = false
  compressResult.value = 'success'
  ElMessage.success('压缩完成')
}

watch(selectedModel, () => {
  const matched = filteredDatasets.value
  if (matched.length) {
    const cloud = matched.find(d => d.name.includes('cloud'))
    selectedData.value = cloud ? cloud.path : matched[0].path
  } else {
    selectedData.value = ''
  }
})

watch(currentMethod, (m) => {
  if (!m) return
  const vals = {}
  for (const [key, cfg] of Object.entries(m.params)) {
    vals[key] = cfg.default
  }
  paramValues.value = vals
}, { immediate: true })

onMounted(async () => {
  try {
    const [methodRes, modelRes, dsRes] = await Promise.all([
      lightweightApi.methods(),
      lightweightApi.models(),
      lightweightApi.datasets(),
    ])
    methods.value = methodRes.methods || []
    models.value = modelRes.models || []
    datasets.value = dsRes.datasets || []
    if (methods.value.length) selectedMethod.value = methods.value[0].id

    const s = await lightweightApi.status()
    if (s.running) {
      compressing.value = true
      taskStatus.value = s
      showTerminal.value = true
      await nextTick()
      terminalRef.value?.startPolling()
    }
  } catch (e) {
    console.error(e)
  }
})

onUnmounted(() => terminalRef.value?.stopPolling())
</script>

<style scoped>
.method-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 12px;
}
.method-card {
  border: 1.5px solid rgba(255,255,255,0.06);
  border-radius: 10px;
  padding: 16px;
  cursor: pointer;
  transition: all 0.2s;
  background: rgba(255,255,255,0.02);
}
.method-card:hover {
  border-color: rgba(83,168,255,0.3);
  background: rgba(83,168,255,0.04);
}
.method-card.active {
  border-color: #53a8ff;
  background: rgba(83,168,255,0.08);
  box-shadow: 0 0 0 1px rgba(83,168,255,0.2);
}
.method-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 8px;
}
.method-name {
  font-size: 15px;
  font-weight: 600;
  color: var(--text-primary);
}
.method-tags {
  display: flex;
  gap: 6px;
  margin-bottom: 8px;
  flex-wrap: wrap;
}
.method-desc {
  font-size: 12px;
  color: var(--text-secondary);
  line-height: 1.6;
}

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
</style>
