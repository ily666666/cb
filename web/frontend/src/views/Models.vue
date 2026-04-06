<template>
  <div>
    <div class="page-header">
      <h2>模型算法管理</h2>
      <p>管理云侧/边侧模型文件、查看模型参数详情、数据集模型配置</p>
    </div>

    <div class="stats-grid">
      <div class="stat-card">
        <div class="stat-icon blue"><el-icon><Cloudy /></el-icon></div>
        <div class="stat-info">
          <div class="value">{{ modelData.total_cloud || 0 }}</div>
          <div class="label">云侧模型</div>
        </div>
      </div>
      <div class="stat-card">
        <div class="stat-icon green"><el-icon><Monitor /></el-icon></div>
        <div class="stat-info">
          <div class="value">{{ modelData.total_edge || 0 }}</div>
          <div class="label">边侧模型</div>
        </div>
      </div>
    </div>

    <!-- 数据集模型配置 -->
    <div class="card">
      <div class="card-title"><el-icon><Setting /></el-icon> 数据集模型配置</div>
      <el-table :data="configList" size="small">
        <el-table-column prop="dataset" label="数据集" width="90" />
        <el-table-column prop="num_classes" label="类别数" width="70" />
        <el-table-column prop="signal_length" label="信号长度" width="80" />
        <el-table-column label="云侧模型">
          <template #default="{ row }">
            <div style="display: flex; flex-wrap: wrap; gap: 3px;">
              <el-tag v-for="m in row.cloud_models" :key="m" type="primary" size="small"
                      style="white-space: normal; height: auto; line-height: 1.4;">{{ m }}</el-tag>
            </div>
          </template>
        </el-table-column>
        <el-table-column label="边侧模型">
          <template #default="{ row }">
            <div style="display: flex; flex-wrap: wrap; gap: 3px;">
              <el-tag v-for="m in row.edge_models" :key="m" type="success" size="small"
                      style="white-space: normal; height: auto; line-height: 1.4;">{{ m }}</el-tag>
            </div>
          </template>
        </el-table-column>
      </el-table>
    </div>

    <!-- 云侧模型 -->
    <div class="card">
      <div class="card-title"><el-icon><Cloudy /></el-icon> 云侧模型</div>
      <el-table :data="modelData.cloud_models" size="small" max-height="500">
        <el-table-column prop="name" label="模型文件" min-width="180" show-overflow-tooltip />
        <el-table-column prop="dataset" label="数据集" width="80">
          <template #default="{ row }">
            <el-tag size="small">{{ row.dataset }}</el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="size_mb" label="大小(MB)" width="80" />
        <el-table-column label="来源" width="160" show-overflow-tooltip>
          <template #default="{ row }">
            {{ row.source_task || '预置' }}
          </template>
        </el-table-column>
        <el-table-column label="操作" width="60">
          <template #default="{ row }">
            <el-button link type="primary" size="small" @click="showDetail(row)">详情</el-button>
          </template>
        </el-table-column>
      </el-table>
      <el-empty v-if="!modelData.cloud_models?.length" description="暂无云侧模型" />
    </div>

    <!-- 边侧模型 -->
    <div class="card">
      <div class="card-title"><el-icon><Monitor /></el-icon> 边侧模型</div>
      <el-table :data="modelData.edge_models" size="small" max-height="500">
        <el-table-column prop="name" label="模型文件" min-width="180" show-overflow-tooltip />
        <el-table-column prop="dataset" label="数据集" width="80">
          <template #default="{ row }">
            <el-tag size="small">{{ row.dataset }}</el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="size_mb" label="大小(MB)" width="80" />
        <el-table-column label="来源" width="160" show-overflow-tooltip>
          <template #default="{ row }">
            {{ row.source_task || '预置' }}
          </template>
        </el-table-column>
        <el-table-column label="操作" width="60">
          <template #default="{ row }">
            <el-button link type="primary" size="small" @click="showDetail(row)">详情</el-button>
          </template>
        </el-table-column>
      </el-table>
      <el-empty v-if="!modelData.edge_models?.length" description="暂无边侧模型" />
    </div>

    <!-- 模型详情对话框 -->
    <el-dialog v-model="detailVisible" title="模型详情" width="650px">
      <div v-if="detailData" style="color: var(--text-primary);">
        <el-descriptions :column="2" border size="small">
          <el-descriptions-item label="文件名">{{ detailData.name }}</el-descriptions-item>
          <el-descriptions-item label="大小">{{ detailData.size_mb }} MB</el-descriptions-item>
          <el-descriptions-item label="路径" :span="2">{{ detailData.path }}</el-descriptions-item>
          <el-descriptions-item v-if="detailData.total_params_m" label="参数量">
            {{ detailData.total_params_m }}M ({{ detailData.total_params?.toLocaleString() }})
          </el-descriptions-item>
          <el-descriptions-item v-if="detailData.num_layers" label="层数">{{ detailData.num_layers }}</el-descriptions-item>
        </el-descriptions>

        <div v-if="detailData.layers_summary?.length" style="margin-top: 16px;">
          <h4 style="font-size: 14px; margin-bottom: 8px;">层结构（前 20 层）</h4>
          <el-table :data="detailData.layers_summary" size="small" max-height="300">
            <el-table-column prop="name" label="层名" show-overflow-tooltip />
            <el-table-column label="形状" width="180">
              <template #default="{ row }">{{ row.shape?.join(' x ') }}</template>
            </el-table-column>
            <el-table-column label="参数" width="100">
              <template #default="{ row }">{{ row.params?.toLocaleString() }}</template>
            </el-table-column>
          </el-table>
        </div>

        <div v-if="detailData.load_error" style="margin-top: 12px;">
          <el-alert :title="detailData.load_error" type="warning" :closable="false" />
        </div>
      </div>
      <div v-else style="text-align: center; padding: 20px;">
        <el-icon class="is-loading" :size="32"><Loading /></el-icon>
      </div>
    </el-dialog>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import { modelApi } from '../api'

const modelData = ref({})
const modelConfig = ref({})
const detailVisible = ref(false)
const detailData = ref(null)

const configList = computed(() => {
  return Object.entries(modelConfig.value).map(([dataset, cfg]) => ({
    dataset,
    num_classes: cfg.num_classes,
    signal_length: cfg.signal_length,
    cloud_models: cfg.cloud_models || [cfg.cloud_model],
    edge_models: cfg.edge_models || [cfg.edge_model],
  }))
})

async function showDetail(model) {
  detailVisible.value = true
  detailData.value = null
  try {
    detailData.value = await modelApi.detail(model.path)
  } catch (e) {
    detailData.value = { name: model.name, path: model.path, load_error: e.message }
  }
}

onMounted(async () => {
  try {
    const [models, config] = await Promise.all([
      modelApi.list(),
      modelApi.config(),
    ])
    modelData.value = models
    modelConfig.value = config
  } catch (e) {
    console.error(e)
  }
})
</script>
