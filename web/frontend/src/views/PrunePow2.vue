<template>
  <div>
    <div class="page-header">
      <h2>剪枝量化（2的幂次）</h2>
      <p>物理通道剪枝 + INQ 增量网络量化，权重量化为 2 的幂次值，乘法可用移位替代</p>
    </div>

    <!-- 方案说明卡片 -->
    <div class="card" v-if="method">
      <div class="card-title"><el-icon><Opportunity /></el-icon> 压缩方案</div>
      <div class="method-card active">
        <div class="method-header">
          <span class="method-name">{{ method.name }}</span>
        </div>
        <div class="method-tags">
          <el-tag size="small">{{ method.prune_type }}</el-tag>
          <el-tag size="small" type="success">{{ method.quant_type }}</el-tag>
        </div>
        <div class="method-desc">{{ method.description }}</div>
      </div>
    </div>

    <!-- 参数配置 -->
    <div class="grid-2" v-if="method">
      <div class="card">
        <div class="card-title"><el-icon><Operation /></el-icon> 压缩参数配置</div>
        <el-form label-width="140px" size="default" style="max-width: 500px;">
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

          <el-form-item label="剪枝比例">
            <el-slider v-model="paramValues.prune_ratio" :min="0.05" :max="0.5"
              :step="0.05" show-input :show-input-controls="false" style="padding-right: 12px;" />
          </el-form-item>

          <el-form-item label="量化位宽">
            <el-select v-model="paramValues.weight_bits" style="width: 160px;">
              <el-option v-for="opt in [4, 6, 8]" :key="opt" :label="opt + ' bit'" :value="opt" />
            </el-select>
          </el-form-item>

          <el-form-item label="每阶段轮数">
            <el-slider v-model="paramValues.epochs_per_step" :min="1" :max="20"
              :step="1" show-input :show-input-controls="false" style="padding-right: 12px;" />
          </el-form-item>

          <el-form-item label="训练批大小">
            <el-slider v-model="paramValues.batch_size" :min="8" :max="512"
              :step="8" show-input :show-input-controls="false" style="padding-right: 12px;" />
          </el-form-item>
        </el-form>
      </div>

      <!-- 压缩流程图 -->
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
            <div class="flow-label">物理通道剪枝</div>
            <div class="flow-detail">移除 {{ ((paramValues.prune_ratio || 0.15) * 100).toFixed(0) }}% 通道</div>
          </div>
          <div class="flow-arrow">→</div>
          <div class="flow-step">
            <div class="flow-icon purple"><el-icon><Coin /></el-icon></div>
            <div class="flow-label">INQ 2的幂次量化</div>
            <div class="flow-detail">{{ paramValues.weight_bits || 8 }}-bit · w = ±2ⁿ</div>
          </div>
          <div class="flow-arrow">→</div>
          <div class="flow-step">
            <div class="flow-icon green"><el-icon><CircleCheck /></el-icon></div>
            <div class="flow-label">增量微调</div>
            <div class="flow-detail">4 阶段 × {{ paramValues.epochs_per_step || 4 }} Epochs</div>
          </div>
          <div class="flow-arrow">→</div>
          <div class="flow-step">
            <div class="flow-icon red"><el-icon><Download /></el-icon></div>
            <div class="flow-label">导出轻量模型</div>
            <div class="flow-detail">硬件友好部署</div>
          </div>
        </div>

        <div class="inq-explain">
          <h4>INQ 量化原理</h4>
          <p>增量网络量化（INQ）将浮点权重逐步量化为 <strong>2 的幂次值</strong>（±2<sup>n</sup>），
          每个阶段固定一部分已量化权重、微调剩余权重，最终全网络量化。</p>
          <div class="inq-stages">
            <div class="stage" v-for="(pct, i) in [50, 75, 82, 100]" :key="i">
              <div class="stage-bar" :style="{ width: pct + '%' }"></div>
              <span class="stage-label">阶段{{ i+1 }}: {{ pct }}%</span>
            </div>
          </div>
          <p style="margin-top: 8px; font-size: 12px; color: var(--text-secondary);">
            硬件优势：乘法运算替换为移位操作（shift），计算效率大幅提升
          </p>
        </div>
      </div>
    </div>

    <!-- 执行压缩 -->
    <div class="card" v-if="method">
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
        title="INQ 2的幂次量化压缩" @done="onTermDone" />
    </div>

    <!-- 可压缩的模型列表 -->
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
import { ref, computed, onMounted, onUnmounted, watch, nextTick } from 'vue'
import { prunePow2Api } from '../api'
import { ElMessage } from 'element-plus'
import WebTerminal from '../components/WebTerminal.vue'

const method = ref(null)
const models = ref([])
const datasets = ref([])
const selectedModel = ref('')
const selectedData = ref('')
const paramValues = ref({
  prune_ratio: 0.15,
  weight_bits: 8,
  epochs_per_step: 4,
  batch_size: 64,
})
const compressing = ref(false)
const compressResult = ref('')
const taskStatus = ref({})
const showTerminal = ref(false)
const terminalRef = ref(null)

const selectedModelInfo = computed(() => models.value.find(m => m.path === selectedModel.value))

const filteredDatasets = computed(() => {
  const info = selectedModelInfo.value
  if (!info?.dataset_type) return datasets.value
  return datasets.value.filter(d => d.dataset_type === info.dataset_type)
})

const pollStatus = () => prunePow2Api.status()

async function startCompress() {
  compressResult.value = ''
  taskStatus.value = {}
  try {
    const res = await prunePow2Api.run({
      model_path: selectedModel.value,
      params: { ...paramValues.value, data_path: selectedData.value },
    })
    if (res.status === 'error') {
      ElMessage.error(res.message)
      return
    }
    compressing.value = true
    showTerminal.value = true
    ElMessage.success('INQ 压缩任务已启动')
    await nextTick()
    terminalRef.value?.startPolling()
  } catch (e) {
    ElMessage.error('启动压缩失败: ' + (e.message || e))
  }
}

async function stopCompress() {
  try {
    await prunePow2Api.stop()
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

onMounted(async () => {
  try {
    const [methodRes, modelRes, dsRes] = await Promise.all([
      prunePow2Api.method(),
      prunePow2Api.models(),
      prunePow2Api.datasets(),
    ])
    method.value = methodRes.method || null
    models.value = modelRes.models || []
    datasets.value = dsRes.datasets || []

    const s = await prunePow2Api.status()
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
.method-card {
  border: 1.5px solid rgba(83,168,255,0.4);
  border-radius: 10px;
  padding: 16px;
  background: rgba(83,168,255,0.06);
}
.method-card.active {
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

.inq-explain {
  margin-top: 20px;
  padding: 16px;
  border-radius: 8px;
  background: rgba(179,136,255,0.06);
  border: 1px solid rgba(179,136,255,0.15);
}
.inq-explain h4 {
  font-size: 13px;
  color: #b388ff;
  margin-bottom: 8px;
}
.inq-explain p {
  font-size: 12px;
  color: var(--text-secondary);
  line-height: 1.7;
}
.inq-stages {
  display: flex;
  flex-direction: column;
  gap: 6px;
  margin-top: 10px;
}
.stage {
  position: relative;
  height: 22px;
  background: rgba(255,255,255,0.04);
  border-radius: 4px;
  overflow: hidden;
}
.stage-bar {
  height: 100%;
  background: linear-gradient(90deg, rgba(179,136,255,0.3), rgba(179,136,255,0.15));
  border-radius: 4px;
  transition: width 0.5s;
}
.stage-label {
  position: absolute;
  top: 2px;
  left: 8px;
  font-size: 11px;
  color: var(--text-primary);
  font-weight: 500;
}
</style>
