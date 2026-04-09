<template>
  <div class="web-terminal" ref="termRef">
    <div class="term-header">
      <span class="term-dot red"></span>
      <span class="term-dot yellow"></span>
      <span class="term-dot green"></span>
      <span class="term-title">{{ title }}</span>
      <span v-if="connected" class="term-status live">LIVE</span>
      <span v-else-if="done" class="term-status done">DONE</span>
    </div>
    <div class="term-body" ref="bodyRef">
      <div v-for="(line, i) in lines" :key="i" class="term-line" v-html="ansiToHtml(line)"></div>
      <div v-if="connected" class="term-cursor">_</div>
    </div>
  </div>
</template>

<script setup>
import { ref, watch, nextTick, onUnmounted } from 'vue'

const props = defineProps({
  taskId: { type: String, default: '' },
  title: { type: String, default: 'Terminal' },
  pollFn: { type: Function, default: null },
  pollInterval: { type: Number, default: 2000 },
})

const emit = defineEmits(['done'])

const lines = ref([])
const connected = ref(false)
const done = ref(false)
const termRef = ref(null)
const bodyRef = ref(null)
let ws = null

function ansiToHtml(text) {
  let html = text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')

  html = html
    .replace(/\x1b\[1m/g, '<b>')
    .replace(/\x1b\[0m/g, '</b></span>')
    .replace(/\x1b\[32m/g, '<span style="color:#4ec9b0">')
    .replace(/\x1b\[33m/g, '<span style="color:#dcdcaa">')
    .replace(/\x1b\[31m/g, '<span style="color:#f44747">')
    .replace(/\x1b\[34m/g, '<span style="color:#569cd6">')
    .replace(/\x1b\[36m/g, '<span style="color:#9cdcfe">')
    .replace(/\x1b\[35m/g, '<span style="color:#c586c0">')
    .replace(/\x1b\[\d+m/g, '')

  if (text.startsWith('[错误]') || text.startsWith('[失败]') || text.startsWith('[异常]')) {
    return `<span style="color:#f44747">${html}</span>`
  }
  if (text.startsWith('===') || text.startsWith('---')) {
    return `<span style="color:#569cd6">${html}</span>`
  }
  if (text.match(/^\[步骤/)) {
    return `<span style="color:#4ec9b0;font-weight:600">${html}</span>`
  }
  if (text.match(/^\[配置\]|^\[信息\]/)) {
    return `<span style="color:#9cdcfe">${html}</span>`
  }
  if (text.match(/^\[成功\]|^\[完成\]/)) {
    return `<span style="color:#6a9955">${html}</span>`
  }
  return html
}

function scrollToBottom() {
  nextTick(() => {
    if (bodyRef.value) {
      bodyRef.value.scrollTop = bodyRef.value.scrollHeight
    }
  })
}

let pollTimer = null

function connect(taskId) {
  disconnect()
  if (!taskId) return

  lines.value = []
  done.value = false

  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:'
  ws = new WebSocket(`${proto}//${location.host}/ws/tasks/${taskId}/output`)

  ws.onopen = () => { connected.value = true }

  ws.onmessage = (e) => {
    try {
      const msg = JSON.parse(e.data)
      if (msg.type === 'line') {
        lines.value.push(msg.data)
        scrollToBottom()
      } else if (msg.type === 'done') {
        done.value = true
        connected.value = false
        emit('done', msg.status)
      } else if (msg.type === 'error') {
        lines.value.push(`[错误] ${msg.data}`)
        done.value = true
        connected.value = false
      }
    } catch {}
  }

  ws.onclose = () => { connected.value = false }
  ws.onerror = () => { connected.value = false }
}

function startPolling() {
  stopPolling()
  lines.value = []
  done.value = false
  connected.value = true
  pollTimer = setInterval(async () => {
    if (!props.pollFn) return
    try {
      const s = await props.pollFn()
      if (s.log) {
        lines.value = s.log.split('\n')
        scrollToBottom()
      }
      if (!s.running) {
        done.value = true
        connected.value = false
        stopPolling()
        emit('done', s)
      }
    } catch (e) {
      console.error(e)
    }
  }, props.pollInterval)
}

function stopPolling() {
  if (pollTimer) { clearInterval(pollTimer); pollTimer = null }
}

function disconnect() {
  if (ws) {
    ws.close()
    ws = null
  }
  stopPolling()
  connected.value = false
}

function reset() {
  lines.value = []
  done.value = false
  connected.value = false
}

watch(() => props.taskId, (id) => {
  if (id) connect(id)
  else if (!props.pollFn) disconnect()
}, { immediate: true })

onUnmounted(() => disconnect())

defineExpose({ connect, disconnect, startPolling, stopPolling, reset })
</script>

<style scoped>
.web-terminal {
  background: #1a1a2e;
  border-radius: 8px;
  border: 1px solid rgba(255,255,255,0.08);
  overflow: hidden;
  font-family: 'Cascadia Code', 'Fira Code', 'JetBrains Mono', 'Consolas', monospace;
}
.term-header {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 8px 12px;
  background: #16162a;
  border-bottom: 1px solid rgba(255,255,255,0.06);
}
.term-dot {
  width: 10px;
  height: 10px;
  border-radius: 50%;
}
.term-dot.red { background: #ff5f57; }
.term-dot.yellow { background: #febc2e; }
.term-dot.green { background: #28c840; }
.term-title {
  font-size: 12px;
  color: rgba(255,255,255,0.5);
  margin-left: 8px;
}
.term-status {
  margin-left: auto;
  font-size: 11px;
  font-weight: 700;
  padding: 2px 8px;
  border-radius: 4px;
  letter-spacing: 0.5px;
}
.term-status.live {
  color: #28c840;
  background: rgba(40,200,64,0.15);
  animation: pulse 1.5s ease-in-out infinite;
}
.term-status.done {
  color: #569cd6;
  background: rgba(86,156,214,0.15);
}
@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.5; }
}
.term-body {
  padding: 12px 14px;
  max-height: 450px;
  overflow-y: auto;
  font-size: 13px;
  line-height: 1.6;
  color: #d4d4d4;
}
.term-body::-webkit-scrollbar {
  width: 6px;
}
.term-body::-webkit-scrollbar-thumb {
  background: rgba(255,255,255,0.15);
  border-radius: 3px;
}
.term-line {
  white-space: pre-wrap;
  word-break: break-all;
  min-height: 1em;
}
.term-cursor {
  display: inline-block;
  color: #28c840;
  animation: blink 1s step-end infinite;
}
@keyframes blink {
  0%, 100% { opacity: 1; }
  50% { opacity: 0; }
}
</style>
