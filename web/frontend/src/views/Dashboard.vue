<template>
  <div>
    <div class="page-header">
      <h2>系统概览</h2>
      <p>面向典型电磁数据处理任务的云边端协同计算框架</p>
    </div>

    <div class="stats-grid">
      <div class="stat-card">
        <div class="stat-icon blue"><el-icon><FolderOpened /></el-icon></div>
        <div class="stat-info">
          <div class="value">{{ info.task_ids?.length || 0 }}</div>
          <div class="label">任务总数</div>
        </div>
      </div>
      <div class="stat-card">
        <div class="stat-icon green"><el-icon><Coin /></el-icon></div>
        <div class="stat-info">
          <div class="value">{{ info.datasets?.length || 0 }}</div>
          <div class="label">数据集</div>
        </div>
      </div>
      <div class="stat-card">
        <div class="stat-icon orange"><el-icon><Connection /></el-icon></div>
        <div class="stat-info">
          <div class="value">{{ Object.keys(info.pipeline_modes || {}).length }}</div>
          <div class="label">流水线模式</div>
        </div>
      </div>
      <div class="stat-card">
        <div class="stat-icon purple"><el-icon><Cpu /></el-icon></div>
        <div class="stat-info">
          <div class="value">{{ info.device || '-' }}</div>
          <div class="label">计算设备</div>
        </div>
      </div>
    </div>

    <!-- 架构图 -->
    <div class="card">
      <div class="card-title"><el-icon><Share /></el-icon> 系统架构</div>
      <div class="arch-diagram">
        <div class="arch-row">
          <div class="arch-node cloud-node">
            <div class="node-icon"><el-icon size="28"><Cloudy /></el-icon></div>
            <div class="node-title">云侧 Cloud</div>
            <div class="node-desc">complex_resnet50</div>
            <div class="node-tags">
              <span class="tag">教师模型推理</span>
              <span class="tag">联邦聚合</span>
              <span class="tag">预训练</span>
            </div>
          </div>
        </div>
        <div class="arch-connector">
          <div class="connector-line"></div>
          <div class="connector-label">低置信度数据 / 模型参数</div>
        </div>
        <div class="arch-row">
          <div class="arch-node edge-node">
            <div class="node-icon"><el-icon size="24"><Monitor /></el-icon></div>
            <div class="node-title">边侧 1</div>
            <div class="node-desc">real_resnet20</div>
            <div class="node-tags">
              <span class="tag">轻量推理</span>
              <span class="tag">知识蒸馏</span>
            </div>
          </div>
          <div class="arch-node edge-node">
            <div class="node-icon"><el-icon size="24"><Monitor /></el-icon></div>
            <div class="node-title">边侧 2</div>
            <div class="node-desc">real_resnet20</div>
            <div class="node-tags">
              <span class="tag">轻量推理</span>
              <span class="tag">联邦学习</span>
            </div>
          </div>
        </div>
        <div class="arch-connector">
          <div class="connector-line"></div>
          <div class="connector-label">原始信号数据</div>
        </div>
        <div class="arch-row">
          <div class="arch-node device-node" v-for="i in 4" :key="i">
            <div class="node-icon"><el-icon size="20"><Iphone /></el-icon></div>
            <div class="node-title">端 {{ i }}</div>
            <div class="node-desc">数据采集</div>
          </div>
        </div>
      </div>
    </div>

    <div class="grid-2">
      <!-- 数据集概览 -->
      <div class="card">
        <div class="card-title"><el-icon><Coin /></el-icon> 数据集配置</div>
        <el-table :data="datasetList" style="width: 100%" size="small">
          <el-table-column prop="name" label="数据集" width="100" />
          <el-table-column prop="num_classes" label="类别数" width="70" />
          <el-table-column prop="signal_length" label="信号长度" width="80" />
          <el-table-column prop="cloud_model" label="云侧模型">
            <template #default="{ row }">
              <el-tag size="small" type="primary">{{ row.cloud_model }}</el-tag>
            </template>
          </el-table-column>
        </el-table>
      </div>

      <!-- 流水线模式 -->
      <div class="card">
        <div class="card-title"><el-icon><Connection /></el-icon> 流水线模式</div>
        <el-table :data="pipelineList" style="width: 100%" size="small">
          <el-table-column prop="name" label="模式名" width="200" />
          <el-table-column label="步骤">
            <template #default="{ row }">
              <el-tag v-for="step in row.steps" :key="step" size="small" type="info" style="margin: 2px;">
                {{ step }}
              </el-tag>
            </template>
          </el-table-column>
        </el-table>
      </div>
    </div>

    <!-- 任务列表 -->
    <div class="card">
      <div class="card-title"><el-icon><List /></el-icon> 任务列表</div>
      <el-table :data="tasks" style="width: 100%" size="small">
        <el-table-column prop="task_id" label="任务ID" width="220" />
        <el-table-column prop="dataset" label="数据集" width="100">
          <template #default="{ row }">
            <el-tag v-if="row.dataset" size="small">{{ row.dataset }}</el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="purpose" label="用途" width="100" />
        <el-table-column label="状态">
          <template #default="{ row }">
            <el-tag v-if="row.has_result" type="success" size="small">有结果</el-tag>
            <el-tag v-else-if="row.has_output" type="warning" size="small">有输出</el-tag>
            <el-tag v-else type="info" size="small">待执行</el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="config_files" label="配置文件">
          <template #default="{ row }">
            <span style="font-size: 12px; color: var(--text-secondary);">
              {{ row.config_files?.length || 0 }} 个
            </span>
          </template>
        </el-table-column>
      </el-table>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import { systemApi, taskApi } from '../api'

const info = ref({})
const tasks = ref([])

const datasetList = computed(() => {
  const cfg = info.value.dataset_config || {}
  return Object.entries(cfg).map(([name, v]) => ({ name, ...v }))
})

const pipelineList = computed(() => {
  const modes = info.value.pipeline_modes || {}
  return Object.entries(modes).map(([name, steps]) => ({ name, steps }))
})

onMounted(async () => {
  try {
    const [sysInfo, taskData] = await Promise.all([
      systemApi.info(),
      taskApi.list(),
    ])
    info.value = sysInfo
    tasks.value = taskData.tasks || []
  } catch (e) {
    console.error('加载数据失败', e)
  }
})
</script>

<style scoped>
.arch-diagram {
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 20px 0;
}
.arch-row {
  display: flex;
  gap: 16px;
  justify-content: center;
  flex-wrap: wrap;
}
.arch-node {
  border-radius: 12px;
  padding: 16px 20px;
  text-align: center;
  min-width: 140px;
  border: 1px solid rgba(255,255,255,0.1);
}
.cloud-node { background: rgba(83,168,255,0.1); border-color: rgba(83,168,255,0.3); }
.edge-node { background: rgba(0,212,170,0.08); border-color: rgba(0,212,170,0.3); }
.device-node { background: rgba(179,136,255,0.08); border-color: rgba(179,136,255,0.3); min-width: 100px; }
.node-icon { margin-bottom: 8px; }
.node-title { font-weight: 600; font-size: 14px; color: #fff; }
.node-desc { font-size: 11px; color: var(--text-secondary); margin-top: 4px; }
.node-tags { margin-top: 8px; display: flex; gap: 4px; flex-wrap: wrap; justify-content: center; }
.node-tags .tag {
  font-size: 10px;
  padding: 2px 6px;
  border-radius: 4px;
  background: rgba(255,255,255,0.06);
  color: var(--text-secondary);
}
.arch-connector {
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 8px 0;
}
.connector-line {
  width: 2px;
  height: 24px;
  background: linear-gradient(180deg, rgba(83,168,255,0.4), rgba(0,212,170,0.4));
}
.connector-label {
  font-size: 11px;
  color: var(--text-secondary);
  margin-top: 4px;
}
</style>
