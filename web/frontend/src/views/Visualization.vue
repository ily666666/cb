<template>
  <div>
    <div class="page-header">
      <h2>数据处理可视化</h2>
      <p>查看信号波形、任务执行结果、耗时分析图表、训练曲线</p>
    </div>

    <!-- 信号波形查看 -->
    <div class="card">
      <div class="card-title"><el-icon><DataLine /></el-icon> 信号波形查看</div>
      <div style="display: flex; gap: 12px; align-items: center; flex-wrap: wrap; margin-bottom: 16px;">
        <el-select v-model="waveDs" placeholder="数据集" style="width: 140px;" @change="onDsChange">
          <el-option v-for="ds in datasets" :key="ds.name" :label="ds.name" :value="ds.name" />
        </el-select>
        <el-select v-model="waveFile" placeholder="数据文件" style="width: 220px;" filterable @change="onFileChange">
          <el-option v-for="f in dsFiles" :key="f.filename" :label="f.filename" :value="f.filename">
            <span>{{ f.filename }}</span>
            <span style="float: right; color: var(--text-secondary); font-size: 12px;">{{ f.size_mb }}MB</span>
          </el-option>
        </el-select>
        <el-input-number v-model="waveSampleIdx" :min="0" :max="Math.max(0, waveTotal - 1)" size="default" style="width: 140px;"
          :disabled="!waveTotal" controls-position="right" />
        <span style="font-size: 12px; color: var(--text-secondary);" v-if="waveTotal">
          / {{ waveTotal }} 条样本
        </span>
        <el-button type="primary" @click="loadWaveform" :loading="waveLoading" :disabled="!waveDs || !waveFile">
          <el-icon><View /></el-icon> 查看波形
        </el-button>
      </div>

      <div v-if="waveMeta && !waveData" style="margin-bottom: 12px; font-size: 13px; color: var(--text-secondary);">
        数据形状: <code>{{ waveMeta.shape?.join(' x ') }}</code>
        · 类型: <code>{{ waveMeta.dtype }}</code>
        · 信号长度: {{ waveMeta.signal_length }}
        · {{ waveMeta.has_iq ? 'I/Q 双通道' : '单通道' }}
      </div>
      <div v-if="waveData">
        <div style="margin-bottom: 12px; font-size: 13px; color: var(--text-secondary);">
          数据形状: <code>{{ waveData.shape?.join(' x ') }}</code>
          · 类型: <code>{{ waveData.dtype }}</code>
          · 信号长度: {{ waveData.signal_length }}
          · {{ waveData.has_iq ? 'I/Q 双通道' : '单通道' }}
        </div>
        <div v-for="(sample, idx) in waveData.samples" :key="sample.index" style="margin-bottom: 8px;">
          <div style="font-size: 12px; color: var(--text-secondary); margin-bottom: 4px;">
            样本 #{{ sample.index }}
            <el-tag size="small" style="margin-left: 6px;" v-if="sample.label_name">{{ sample.label_name }}</el-tag>
            <el-tag size="small" type="info" style="margin-left: 4px;" v-else-if="sample.label !== null">类别 {{ sample.label }}</el-tag>
          </div>
          <div :id="'wave-chart-' + idx" class="wave-chart"></div>
        </div>
      </div>
      <el-empty v-if="!waveData && !waveLoading && waveDs && waveFile" description="点击 查看波形 按钮加载数据" />
    </div>

    <!-- 任务选择 -->
    <div class="card" style="display: flex; align-items: center; gap: 16px;">
      <span style="font-size: 14px; white-space: nowrap;">选择任务:</span>
      <el-select v-model="selectedTask" placeholder="选择任务" style="width: 300px;" @change="loadVisualization" filterable>
        <el-option v-for="t in tasks" :key="t.task_id" :label="t.task_id" :value="t.task_id">
          <span>{{ t.task_id }}</span>
          <span style="float: right; color: var(--text-secondary); font-size: 12px;">{{ t.purpose }}</span>
        </el-option>
      </el-select>
      <el-tag v-if="visData" size="small" :type="visData.task_type === 'train' ? 'warning' : 'success'" style="margin-left: -8px;">
        {{ visData.task_type === 'train' ? '训练' : '推理' }}
      </el-tag>
      <el-button type="primary" @click="loadVisualization" :disabled="!selectedTask">
        <el-icon><Refresh /></el-icon> 刷新
      </el-button>
    </div>

    <!-- ==================== 推理任务可视化 ==================== -->
    <template v-if="visData && visData.task_type === 'inference'">
      <div class="stats-grid">
        <div class="stat-card">
          <div class="stat-icon blue"><el-icon><Timer /></el-icon></div>
          <div class="stat-info">
            <div class="value">{{ visData.summary?.total?.toFixed(2) }}s</div>
            <div class="label">总耗时</div>
          </div>
        </div>
        <div class="stat-card">
          <div class="stat-icon green"><el-icon><Cpu /></el-icon></div>
          <div class="stat-info">
            <div class="value">{{ visData.summary?.total_inference?.toFixed(2) }}s</div>
            <div class="label">推理耗时</div>
          </div>
        </div>
        <div class="stat-card" v-if="visData.summary?.has_transfer">
          <div class="stat-icon orange"><el-icon><Upload /></el-icon></div>
          <div class="stat-info">
            <div class="value">{{ visData.summary?.total_transfer?.toFixed(2) }}s</div>
            <div class="label">传输耗时</div>
          </div>
        </div>
        <div class="stat-card">
          <div class="stat-icon purple"><el-icon><Loading /></el-icon></div>
          <div class="stat-info">
            <div class="value">{{ visData.summary?.total_overhead?.toFixed(2) }}s</div>
            <div class="label">加载+热身</div>
          </div>
        </div>
      </div>

      <div class="grid-2">
        <div class="card">
          <div class="card-title"><el-icon><Histogram /></el-icon> 各步骤耗时分布</div>
          <div ref="barChartRef" class="chart-container"></div>
        </div>
        <div class="card">
          <div class="card-title"><el-icon><PieChart /></el-icon> 耗时占比</div>
          <div ref="pieChartRef" class="chart-container"></div>
        </div>
      </div>
    </template>

    <!-- ==================== 训练任务可视化 ==================== -->
    <template v-if="visData && visData.task_type === 'train'">
      <!-- 预训练 Loss/Acc 曲线 -->
      <template v-for="(hist, stepKey) in visData.histories" :key="stepKey">
        <div class="card" v-if="hist._type === 'pretrain'">
          <div class="card-title"><el-icon><TrendCharts /></el-icon> {{ hist._label }} — 训练曲线</div>
          <div class="grid-2">
            <div :ref="el => setTrainChartRef(stepKey + '_loss', el)" class="chart-container"></div>
            <div :ref="el => setTrainChartRef(stepKey + '_acc', el)" class="chart-container"></div>
          </div>
        </div>

        <div class="card" v-if="hist._type === 'federated'">
          <div class="card-title"><el-icon><TrendCharts /></el-icon> {{ hist._label }} — 准确率曲线</div>
          <div :ref="el => setTrainChartRef(stepKey + '_fed', el)" class="chart-container"></div>
        </div>

        <!-- 蒸馏过程可视化 -->
        <div class="card" v-if="hist._type === 'distillation'">
          <div class="card-title"><el-icon><Connection /></el-icon> {{ hist._label }}</div>
          <div v-if="hist.edges && Object.keys(hist.edges).length">
            <el-tabs type="border-card" style="background: transparent;" v-model="kdActiveTab">
              <el-tab-pane v-for="(edgeHist, edgeKey) in hist.edges" :key="edgeKey" :label="'边侧 ' + edgeKey.replace('edge_', '')" :name="edgeKey">
                <div style="margin-bottom: 12px; font-size: 13px; color: var(--text-secondary);">
                  蒸馏参数:
                  <el-tag size="small" style="margin-left: 4px;">α = {{ edgeHist.alpha }}</el-tag>
                  <el-tag size="small" type="warning" style="margin-left: 4px;">T = {{ edgeHist.temperature }}</el-tag>
                  <el-tag size="small" type="info" style="margin-left: 4px;">{{ edgeHist.train_acc?.length }} Epochs</el-tag>
                </div>
                <div class="grid-2">
                  <div :ref="el => setTrainChartRef(stepKey + '_' + edgeKey + '_loss', el)" class="chart-container"></div>
                  <div :ref="el => setTrainChartRef(stepKey + '_' + edgeKey + '_acc', el)" class="chart-container"></div>
                </div>
                <div class="grid-2">
                  <div :ref="el => setTrainChartRef(stepKey + '_' + edgeKey + '_guidance', el)" class="chart-container"></div>
                  <div :ref="el => setTrainChartRef(stepKey + '_' + edgeKey + '_conf', el)" class="chart-container"></div>
                </div>
                <div class="kd-explain">
                  <div class="kd-explain-title"><el-icon><InfoFilled /></el-icon> 指标说明</div>
                  <div class="kd-explain-grid">
                    <div class="kd-explain-item">
                      <span class="kd-dot" style="background:#e94560"></span>
                      <b>Loss 分解</b>：总损失 = (1-α)×CE + α×KD。CE Loss 衡量与真实标签的差距（硬标签），KD Loss 衡量与教师预测的差距（软标签）。
                    </div>
                    <div class="kd-explain-item">
                      <span class="kd-dot" style="background:#53a8ff"></span>
                      <b>准确率</b>：训练准确率（蓝色实线）和测试准确率（绿色虚线），反映学生模型在蒸馏过程中的学习效果。
                    </div>
                    <div class="kd-explain-item">
                      <span class="kd-dot" style="background:#ff9a3e"></span>
                      <b>教师指导强度</b>：KD Loss 在总 Loss 中的实际占比（α×KD / 总Loss）。虽然 α 是固定的，但两部分 Loss 的绝对值会变化，导致实际指导强度随训练动态波动。指导强度高说明教师影响大，低说明学生逐渐独立。
                    </div>
                    <div class="kd-explain-item">
                      <span class="kd-dot" style="background:#00d4aa"></span>
                      <b>学生置信度</b>：学生模型输出 softmax 后最大概率的平均值。置信度越高说明学生对预测越"确定"，通常随训练逐步提升。
                    </div>
                  </div>
                </div>
              </el-tab-pane>
            </el-tabs>
          </div>
        </div>
      </template>

      <div class="card" v-if="!Object.keys(visData.histories || {}).length">
        <el-empty description="该训练任务暂无训练历史数据（train_history.pkl）" />
      </div>
    </template>

    <!-- 执行报告（推理/训练通用） -->
    <div class="card" v-if="visData && Object.keys(visData.reports || {}).length">
      <div class="card-title"><el-icon><Document /></el-icon> 执行报告</div>
      <el-tabs type="border-card" style="background: transparent;">
        <el-tab-pane v-for="(content, name) in visData.reports" :key="name" :label="name">
          <pre class="report-content">{{ content }}</pre>
        </el-tab-pane>
      </el-tabs>
    </div>

    <el-empty v-else-if="selectedTask && !loading && !visData" description="该任务暂无可视化数据" />
  </div>
</template>

<script setup>
import { ref, nextTick, onMounted, onUnmounted, watch } from 'vue'
import { taskApi, inferenceApi, dataApi } from '../api'
import * as echarts from 'echarts'

const tasks = ref([])
const selectedTask = ref('')
const visData = ref(null)
const loading = ref(false)
const barChartRef = ref(null)
const pieChartRef = ref(null)
let barChart = null
let pieChart = null

const trainChartRefs = {}
let trainCharts = []
const kdActiveTab = ref('')

function setTrainChartRef(key, el) {
  if (el) trainChartRefs[key] = el
}

const datasets = ref([])
const waveDs = ref('')
const waveFile = ref('')
const waveSampleIdx = ref(0)
const waveLoading = ref(false)
const waveData = ref(null)
const waveTotal = ref(0)
const dsFiles = ref([])
const waveMeta = ref(null)
let waveCharts = []

function onDsChange() {
  waveFile.value = ''
  waveData.value = null
  waveTotal.value = 0
  waveMeta.value = null
  const ds = datasets.value.find(d => d.name === waveDs.value)
  dsFiles.value = ds?.data_files || []
  if (ds?.split_files?.length) {
    dsFiles.value = [...dsFiles.value, ...ds.split_files]
  }
}

async function onFileChange() {
  waveData.value = null
  waveTotal.value = 0
  waveMeta.value = null
  waveSampleIdx.value = 0
  if (!waveDs.value || !waveFile.value) return
  try {
    const info = await dataApi.filePreview(waveDs.value, waveFile.value)
    waveMeta.value = info
    const data = await dataApi.waveform(waveDs.value, waveFile.value, 0, 1)
    if (data && !data.error) {
      waveTotal.value = data.total_samples || 0
      waveMeta.value = { ...waveMeta.value, shape: data.shape, dtype: data.dtype, signal_length: data.signal_length, has_iq: data.has_iq }
    }
  } catch (e) {
    console.error(e)
  }
}

async function loadWaveform() {
  if (!waveDs.value || !waveFile.value) return
  waveLoading.value = true
  waveCharts.forEach(c => c.dispose())
  waveCharts = []
  try {
    const data = await dataApi.waveform(waveDs.value, waveFile.value, waveSampleIdx.value, 3)
    if (data.error) {
      waveData.value = null
      return
    }
    waveData.value = data
    waveTotal.value = data.total_samples || 0
    await nextTick()
    renderWaveforms()
  } catch (e) {
    waveData.value = null
  } finally {
    waveLoading.value = false
  }
}

function renderWaveforms() {
  if (!waveData.value?.samples) return
  waveCharts.forEach(c => c.dispose())
  waveCharts = []

  waveData.value.samples.forEach((sample, idx) => {
    const el = document.getElementById('wave-chart-' + idx)
    if (!el) return
    const chart = echarts.init(el, 'dark')
    waveCharts.push(chart)

    const xData = Array.from({ length: sample.I.length }, (_, i) => i)
    const series = [
      { name: 'I (同相)', type: 'line', data: sample.I, symbol: 'none', lineStyle: { width: 1.2 }, itemStyle: { color: '#53a8ff' } },
    ]
    if (sample.Q) {
      series.push(
        { name: 'Q (正交)', type: 'line', data: sample.Q, symbol: 'none', lineStyle: { width: 1.2 }, itemStyle: { color: '#00d4aa' } },
      )
    }

    chart.setOption({
      backgroundColor: 'transparent',
      tooltip: { trigger: 'axis' },
      legend: { top: 0, textStyle: { color: '#a0a0a0', fontSize: 11 } },
      grid: { left: 50, right: 20, top: 30, bottom: 30 },
      xAxis: { type: 'category', data: xData, axisLabel: { color: '#a0a0a0', fontSize: 10 }, name: '采样点', nameTextStyle: { color: '#666', fontSize: 10 } },
      yAxis: { type: 'value', axisLabel: { color: '#a0a0a0', fontSize: 10 }, splitLine: { lineStyle: { color: 'rgba(255,255,255,0.05)' } } },
      dataZoom: [{ type: 'inside' }, { type: 'slider', height: 16, bottom: 4 }],
      series,
    })
  })
}

async function loadVisualization() {
  if (!selectedTask.value) return
  loading.value = true
  try {
    visData.value = await inferenceApi.visualization(selectedTask.value)
    await nextTick()
    if (visData.value?.task_type === 'inference') {
      renderInferenceCharts()
    } else if (visData.value?.task_type === 'train') {
      await nextTick()
      renderTrainCharts()
    }
  } catch (e) {
    visData.value = null
  } finally {
    loading.value = false
  }
}

function renderInferenceCharts() {
  if (!visData.value || visData.value.task_type !== 'inference') return

  if (barChartRef.value) {
    barChart?.dispose()
    barChart = echarts.init(barChartRef.value, 'dark')
    const bar = visData.value.timing_bar
    const colors = ['#53a8ff', '#b388ff', '#e94560', '#f0c040', '#ff9a3e', '#00d4aa']
    const series = Object.entries(bar.series).map(([name, data], i) => ({
      name, type: 'bar', stack: 'total', data,
      itemStyle: { color: colors[i % colors.length] },
      emphasis: { focus: 'series' },
    }))
    barChart.setOption({
      backgroundColor: 'transparent',
      tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
      legend: { top: 0, textStyle: { color: '#a0a0a0', fontSize: 11 } },
      grid: { left: 60, right: 20, top: 40, bottom: 40 },
      xAxis: { type: 'category', data: bar.categories, axisLabel: { color: '#a0a0a0' } },
      yAxis: { type: 'value', name: '秒 (s)', axisLabel: { color: '#a0a0a0' }, nameTextStyle: { color: '#a0a0a0' } },
      series,
    })
  }

  if (pieChartRef.value) {
    pieChart?.dispose()
    pieChart = echarts.init(pieChartRef.value, 'dark')
    const pie = visData.value.timing_pie
    pieChart.setOption({
      backgroundColor: 'transparent',
      tooltip: { trigger: 'item', formatter: '{b}: {c}s ({d}%)' },
      legend: { orient: 'vertical', right: 10, top: 'center', textStyle: { color: '#a0a0a0', fontSize: 11 } },
      series: [{
        type: 'pie', radius: ['40%', '70%'], center: ['40%', '50%'],
        data: pie.data.filter(d => d.value > 0),
        label: { color: '#e0e0e0', fontSize: 12 },
        itemStyle: { borderColor: '#16213e', borderWidth: 2 },
        emphasis: { itemStyle: { shadowBlur: 10, shadowOffsetX: 0, shadowColor: 'rgba(0, 0, 0, 0.5)' } },
      }],
      color: ['#53a8ff', '#b388ff', '#00d4aa', '#f0c040', '#e94560'],
    })
  }
}

function renderTrainCharts() {
  if (!visData.value || visData.value.task_type !== 'train') return

  trainCharts.forEach(c => c.dispose())
  trainCharts = []

  const histories = visData.value.histories || {}
  for (const [stepKey, hist] of Object.entries(histories)) {
    if (hist._type === 'pretrain') {
      renderPretrainCharts(stepKey, hist)
    } else if (hist._type === 'federated') {
      renderFederatedChart(stepKey, hist)
    } else if (hist._type === 'distillation') {
      const firstEdge = Object.keys(hist.edges || {})[0]
      if (firstEdge && !kdActiveTab.value) kdActiveTab.value = firstEdge
      renderDistillationCharts(stepKey, hist)
    }
  }
}

function renderPretrainCharts(stepKey, hist) {
  const epochs = hist.train_loss?.map((_, i) => i + 1) || []

  const lossEl = trainChartRefs[stepKey + '_loss']
  if (lossEl) {
    const chart = echarts.init(lossEl, 'dark')
    trainCharts.push(chart)
    chart.setOption({
      backgroundColor: 'transparent',
      title: { text: 'Loss', left: 'center', top: 4, textStyle: { color: '#e0e0e0', fontSize: 14 } },
      tooltip: { trigger: 'axis' },
      legend: { top: 28, textStyle: { color: '#a0a0a0', fontSize: 11 } },
      grid: { left: 55, right: 20, top: 60, bottom: 35 },
      xAxis: { type: 'category', data: epochs, name: 'Epoch', axisLabel: { color: '#a0a0a0' }, nameTextStyle: { color: '#a0a0a0' } },
      yAxis: { type: 'value', name: 'Loss', axisLabel: { color: '#a0a0a0' }, nameTextStyle: { color: '#a0a0a0' }, splitLine: { lineStyle: { color: 'rgba(255,255,255,0.05)' } } },
      series: [
        { name: '训练 Loss', type: 'line', data: hist.train_loss, symbol: 'none', lineStyle: { width: 2 }, itemStyle: { color: '#e94560' } },
        { name: '验证 Loss', type: 'line', data: hist.val_loss, symbol: 'none', lineStyle: { width: 2, type: 'dashed' }, itemStyle: { color: '#f0c040' } },
      ],
    })
  }

  const accEl = trainChartRefs[stepKey + '_acc']
  if (accEl) {
    const chart = echarts.init(accEl, 'dark')
    trainCharts.push(chart)
    const toPercent = arr => arr?.map(v => +(v * 100).toFixed(2)) || []
    chart.setOption({
      backgroundColor: 'transparent',
      title: { text: '准确率', left: 'center', top: 4, textStyle: { color: '#e0e0e0', fontSize: 14 } },
      tooltip: { trigger: 'axis', valueFormatter: v => v + '%' },
      legend: { top: 28, textStyle: { color: '#a0a0a0', fontSize: 11 } },
      grid: { left: 55, right: 20, top: 60, bottom: 35 },
      xAxis: { type: 'category', data: epochs, name: 'Epoch', axisLabel: { color: '#a0a0a0' }, nameTextStyle: { color: '#a0a0a0' } },
      yAxis: { type: 'value', name: '%', min: v => Math.max(0, Math.floor(v.min - 5)), max: 100, axisLabel: { color: '#a0a0a0' }, nameTextStyle: { color: '#a0a0a0' }, splitLine: { lineStyle: { color: 'rgba(255,255,255,0.05)' } } },
      series: [
        { name: '训练准确率', type: 'line', data: toPercent(hist.train_acc), symbol: 'none', lineStyle: { width: 2 }, itemStyle: { color: '#53a8ff' } },
        { name: '验证准确率', type: 'line', data: toPercent(hist.val_acc), symbol: 'none', lineStyle: { width: 2, type: 'dashed' }, itemStyle: { color: '#00d4aa' } },
      ],
    })
  }
}

function renderFederatedChart(stepKey, hist) {
  const fedEl = trainChartRefs[stepKey + '_fed']
  if (!fedEl) return

  const chart = echarts.init(fedEl, 'dark')
  trainCharts.push(chart)

  const rounds = hist.round || []
  const avgAcc = (hist.avg_test_acc || []).map(v => +(v * 100).toFixed(2))
  const edgeAccs = hist.edge_test_accs || []

  const edgeColors = ['#53a8ff', '#00d4aa', '#b388ff', '#ff9a3e', '#e94560']
  const series = [
    { name: '平均准确率', type: 'line', data: avgAcc, symbol: 'circle', symbolSize: 6, lineStyle: { width: 2.5 }, itemStyle: { color: '#f0c040' } },
  ]

  if (edgeAccs.length > 0 && edgeAccs[0]) {
    const numEdges = Array.isArray(edgeAccs[0]) ? edgeAccs[0].length : 0
    for (let e = 0; e < numEdges; e++) {
      series.push({
        name: `边侧 ${e + 1}`,
        type: 'line',
        data: edgeAccs.map(arr => +(arr[e] * 100).toFixed(2)),
        symbol: 'none',
        lineStyle: { width: 1.5, type: 'dashed' },
        itemStyle: { color: edgeColors[e % edgeColors.length] },
      })
    }
  }

  chart.setOption({
    backgroundColor: 'transparent',
    tooltip: { trigger: 'axis', valueFormatter: v => v + '%' },
    legend: { top: 0, textStyle: { color: '#a0a0a0', fontSize: 11 } },
    grid: { left: 55, right: 20, top: 36, bottom: 35 },
    xAxis: { type: 'category', data: rounds, name: 'Round', axisLabel: { color: '#a0a0a0' }, nameTextStyle: { color: '#a0a0a0' } },
    yAxis: { type: 'value', name: '准确率 (%)', min: v => Math.max(0, Math.floor(v.min - 5)), max: 100, axisLabel: { color: '#a0a0a0' }, nameTextStyle: { color: '#a0a0a0' }, splitLine: { lineStyle: { color: 'rgba(255,255,255,0.05)' } } },
    dataZoom: rounds.length > 30 ? [{ type: 'inside' }, { type: 'slider', height: 16, bottom: 4 }] : [],
    series,
  })
}

function renderDistillationCharts(stepKey, hist) {
  for (const [edgeKey, eh] of Object.entries(hist.edges || {})) {
    if (edgeKey !== kdActiveTab.value) continue
    const epochs = eh.train_acc?.map((_, i) => i + 1) || []
    const prefix = stepKey + '_' + edgeKey
    const labelInterval = Math.max(0, Math.ceil(epochs.length / 10) - 1)
    const kdXAxis = { type: 'category', data: epochs, name: 'Epoch', axisLabel: { color: '#a0a0a0', interval: labelInterval }, nameTextStyle: { color: '#a0a0a0' } }

    function initKdChart(el) {
      const c = echarts.init(el, 'dark')
      c._kdChart = true
      trainCharts.push(c)
      return c
    }

    const lossEl = trainChartRefs[prefix + '_loss']
    if (lossEl) {
      const chart = initKdChart(lossEl)
      chart.setOption({
        backgroundColor: 'transparent',
        title: { text: 'Loss 分解', left: 'center', top: 4, textStyle: { color: '#e0e0e0', fontSize: 14 } },
        tooltip: { trigger: 'axis' },
        legend: { top: 28, textStyle: { color: '#a0a0a0', fontSize: 11 } },
        grid: { left: 55, right: 20, top: 60, bottom: 35 },
        xAxis: kdXAxis,
        yAxis: { type: 'value', name: 'Loss', axisLabel: { color: '#a0a0a0' }, nameTextStyle: { color: '#a0a0a0' }, splitLine: { lineStyle: { color: 'rgba(255,255,255,0.05)' } } },
        series: [
          { name: '总 Loss', type: 'line', data: eh.train_loss, symbol: 'none', lineStyle: { width: 2.5 }, itemStyle: { color: '#e94560' } },
          { name: 'CE Loss (硬标签)', type: 'line', data: eh.ce_loss, symbol: 'none', lineStyle: { width: 1.8, type: 'dashed' }, itemStyle: { color: '#f0c040' } },
          { name: 'KD Loss (软标签)', type: 'line', data: eh.kd_loss, symbol: 'none', lineStyle: { width: 1.8, type: 'dashed' }, itemStyle: { color: '#b388ff' } },
        ],
      })
    }

    const accEl = trainChartRefs[prefix + '_acc']
    if (accEl) {
      const chart = initKdChart(accEl)
      const toPercent = arr => arr?.map(v => +(v * 100).toFixed(2)) || []
      chart.setOption({
        backgroundColor: 'transparent',
        title: { text: '学生模型准确率', left: 'center', top: 4, textStyle: { color: '#e0e0e0', fontSize: 14 } },
        tooltip: { trigger: 'axis', valueFormatter: v => v + '%' },
        legend: { top: 28, textStyle: { color: '#a0a0a0', fontSize: 11 } },
        grid: { left: 55, right: 20, top: 60, bottom: 35 },
        xAxis: kdXAxis,
        yAxis: { type: 'value', name: '%', min: v => Math.max(0, Math.floor(v.min - 5)), max: 100, axisLabel: { color: '#a0a0a0' }, nameTextStyle: { color: '#a0a0a0' }, splitLine: { lineStyle: { color: 'rgba(255,255,255,0.05)' } } },
        series: [
          { name: '训练准确率', type: 'line', data: toPercent(eh.train_acc), symbol: 'none', lineStyle: { width: 2 }, itemStyle: { color: '#53a8ff' } },
          { name: '测试准确率', type: 'line', data: toPercent(eh.test_acc), symbol: 'none', lineStyle: { width: 2, type: 'dashed' }, itemStyle: { color: '#00d4aa' } },
        ],
      })
    }

    const guidanceEl = trainChartRefs[prefix + '_guidance']
    if (guidanceEl) {
      const chart = initKdChart(guidanceEl)
      const toPercent = arr => arr?.map(v => +(v * 100).toFixed(2)) || []
      chart.setOption({
        backgroundColor: 'transparent',
        title: { text: '教师指导强度', left: 'center', top: 4, textStyle: { color: '#e0e0e0', fontSize: 14 } },
        tooltip: {
          trigger: 'axis',
          formatter: params => {
            let s = `Epoch ${params[0].axisValue}<br/>`
            params.forEach(p => { s += `${p.marker} ${p.seriesName}: ${p.value}%<br/>` })
            s += `<span style="color:#666;font-size:11px">α=${eh.alpha}, T=${eh.temperature}</span>`
            return s
          },
        },
        legend: { top: 28, textStyle: { color: '#a0a0a0', fontSize: 11 } },
        grid: { left: 55, right: 20, top: 60, bottom: 35 },
        xAxis: kdXAxis,
        yAxis: { type: 'value', name: '%', min: 0, max: 100, axisLabel: { color: '#a0a0a0' }, nameTextStyle: { color: '#a0a0a0' }, splitLine: { lineStyle: { color: 'rgba(255,255,255,0.05)' } } },
        series: [
          {
            name: 'KD 贡献占比',
            type: 'line',
            data: toPercent(eh.guidance_intensity),
            symbol: 'circle', symbolSize: 4,
            lineStyle: { width: 2.5 },
            itemStyle: { color: '#ff9a3e' },
            areaStyle: { color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
              { offset: 0, color: 'rgba(255,154,62,0.3)' },
              { offset: 1, color: 'rgba(255,154,62,0.02)' },
            ]) },
          },
          {
            name: 'α 权重基线',
            type: 'line',
            data: epochs.map(() => +(eh.alpha * 100).toFixed(1)),
            symbol: 'none',
            lineStyle: { width: 1.5, type: 'dotted', color: '#666' },
            itemStyle: { color: '#666' },
          },
        ],
      })
    }

    const confEl = trainChartRefs[prefix + '_conf']
    if (confEl) {
      const chart = initKdChart(confEl)
      const toPercent = arr => arr?.map(v => +(v * 100).toFixed(2)) || []
      chart.setOption({
        backgroundColor: 'transparent',
        title: { text: '学生模型置信度', left: 'center', top: 4, textStyle: { color: '#e0e0e0', fontSize: 14 } },
        tooltip: { trigger: 'axis', valueFormatter: v => v + '%' },
        legend: { top: 28, textStyle: { color: '#a0a0a0', fontSize: 11 } },
        grid: { left: 55, right: 20, top: 60, bottom: 35 },
        xAxis: kdXAxis,
        yAxis: { type: 'value', name: '%', min: 0, max: 100, axisLabel: { color: '#a0a0a0' }, nameTextStyle: { color: '#a0a0a0' }, splitLine: { lineStyle: { color: 'rgba(255,255,255,0.05)' } } },
        series: [
          {
            name: '平均置信度',
            type: 'line',
            data: toPercent(eh.mean_confidence),
            symbol: 'circle', symbolSize: 4,
            lineStyle: { width: 2.5 },
            itemStyle: { color: '#00d4aa' },
            areaStyle: { color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
              { offset: 0, color: 'rgba(0,212,170,0.3)' },
              { offset: 1, color: 'rgba(0,212,170,0.02)' },
            ]) },
          },
        ],
      })
    }
  }
}

watch(kdActiveTab, async () => {
  await nextTick()
  trainCharts.filter(c => c._kdChart).forEach(c => { c.dispose() })
  trainCharts = trainCharts.filter(c => !c._kdChart)
  const histories = visData.value?.histories || {}
  for (const [stepKey, hist] of Object.entries(histories)) {
    if (hist._type === 'distillation') {
      renderDistillationCharts(stepKey, hist)
    }
  }
})

function handleResize() {
  barChart?.resize()
  pieChart?.resize()
  trainCharts.forEach(c => c.resize())
  waveCharts.forEach(c => c.resize())
}

onMounted(async () => {
  window.addEventListener('resize', handleResize)
  try {
    const [taskRes, dsRes] = await Promise.all([
      taskApi.list(),
      dataApi.datasets(),
    ])
    tasks.value = (taskRes.tasks || []).filter(t => t.has_output)
    datasets.value = dsRes.datasets || []
    if (datasets.value.length) {
      waveDs.value = datasets.value[0].name
      onDsChange()
    }
  } catch (e) {
    console.error(e)
  }
})

onUnmounted(() => {
  window.removeEventListener('resize', handleResize)
  barChart?.dispose()
  pieChart?.dispose()
  trainCharts.forEach(c => c.dispose())
  waveCharts.forEach(c => c.dispose())
})
</script>

<style scoped>
.wave-chart {
  width: 100%;
  height: 220px;
}
.el-tabs {
  --el-bg-color: transparent;
  --el-fill-color-blank: rgba(255,255,255,0.03);
}
:deep(.el-tabs__content) {
  padding: 12px;
}
:deep(.el-tabs__header) {
  background: rgba(255,255,255,0.02);
}
.kd-explain {
  margin-top: 16px;
  padding: 14px 18px;
  border-radius: 8px;
  background: rgba(255, 255, 255, 0.03);
  border: 1px solid rgba(255, 255, 255, 0.06);
}
.kd-explain-title {
  font-size: 14px;
  font-weight: 600;
  color: #c0c0c0;
  margin-bottom: 10px;
  display: flex;
  align-items: center;
  gap: 6px;
}
.kd-explain-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 10px 20px;
}
.kd-explain-item {
  font-size: 12.5px;
  color: #909399;
  line-height: 1.7;
}
.kd-dot {
  display: inline-block;
  width: 8px;
  height: 8px;
  border-radius: 50%;
  margin-right: 5px;
  vertical-align: middle;
}
</style>
