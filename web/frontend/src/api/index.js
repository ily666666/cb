import axios from 'axios'

const api = axios.create({
  baseURL: '/api',
  timeout: 60000,
})

api.interceptors.response.use(
  res => res.data,
  err => {
    const msg = err.response?.data?.detail || err.message
    return Promise.reject(new Error(msg))
  }
)

export const systemApi = {
  health: () => api.get('/health'),
  info: () => api.get('/system/info'),
}

export const taskApi = {
  list: () => api.get('/tasks/'),
  modes: () => api.get('/tasks/modes'),
  detail: (id) => api.get(`/tasks/${id}`),
  timing: (id) => api.get(`/tasks/${id}/timing`),
  run: (id, params) => api.post(`/tasks/${id}/run`, null, { params }),
  clean: (id) => api.delete(`/tasks/${id}/output`),
  stop: (id) => api.post(`/tasks/${id}/stop`),
  removeRecord: (id) => api.delete(`/tasks/${id}/record`),
  status: (id) => api.get(`/tasks/${id}/status`),
  activeTasks: () => api.get('/tasks/active'),
}

export const dataApi = {
  datasets: () => api.get('/data/datasets'),
  datasetDetail: (name) => api.get(`/data/datasets/${name}`),
  filePreview: (ds, file) => api.get(`/data/datasets/${ds}/files/${file}/preview`),
  waveform: (ds, file, start = 0, count = 5) => api.get(`/data/datasets/${ds}/files/${file}/waveform`, { params: { sample_idx: start, max_samples: count } }),
  taskConfigs: (id) => api.get(`/data/tasks/${id}/configs`),
  saveConfig: (id, filename, content) => api.put(`/data/tasks/${id}/configs/${filename}`, content),
}

export const inferenceApi = {
  modes: () => api.get('/inference/modes'),
  start: (id, mode) => api.post(`/inference/${id}/start`, null, { params: { mode } }),
  result: (id) => api.get(`/inference/${id}/result`),
  visualization: (id) => api.get(`/inference/${id}/visualization`),
  report: (id, name) => api.get(`/inference/${id}/report/${name}`),
}

export const modelApi = {
  list: () => api.get('/models/'),
  config: () => api.get('/models/config'),
  detail: (path) => api.get('/models/detail', { params: { path } }),
  remove: (path) => api.delete('/models/', { params: { path } }),
}

export const lightweightApi = {
  methods: () => api.get('/lightweight/methods'),
  models: () => api.get('/lightweight/models'),
  datasets: () => api.get('/lightweight/datasets'),
  history: () => api.get('/lightweight/history'),
  run: (data) => api.post('/lightweight/run', data),
  stop: () => api.post('/lightweight/stop'),
  status: () => api.get('/lightweight/status'),
}

export const distillationApi = {
  tasks: () => api.get('/distillation/tasks'),
  start: (taskId, fastMode, accuracy, teacherAccuracy) => {
    const params = {}
    if (fastMode) params.fast_mode = true
    if (accuracy != null) params.accuracy = accuracy
    if (teacherAccuracy != null) params.teacher_accuracy = teacherAccuracy
    return api.post(`/distillation/${taskId}/start`, null, { params })
  },
  status: (taskId) => api.get(`/distillation/${taskId}/status`),
  stop: (taskId) => api.post(`/distillation/${taskId}/stop`),
  history: (taskId) => api.get(`/distillation/${taskId}/history`),
  models: (taskId) => api.get(`/distillation/${taskId}/models`),
}

export const compareApi = {
  tasks: () => api.get('/compare/tasks'),
  updateLabel: (taskId, label) => api.put(`/compare/tasks/${taskId}/label`, { label }),
  updateSummary: (taskId, data) => api.put(`/compare/tasks/${taskId}/summary`, data),
  updateStepConfig: (taskId, stepName, config) => api.put(`/compare/tasks/${taskId}/steps/${stepName}/config`, config),
  clone: (sourceTaskId, newTaskId, label) => api.post('/compare/clone', { source_task_id: sourceTaskId, new_task_id: newTaskId, label }),
  deleteTask: (taskId) => api.delete(`/compare/tasks/${taskId}`),
  results: (taskIds) => api.get('/compare/results', { params: { task_ids: taskIds.join(',') } }),
}

export const prunePow2Api = {
  method: () => api.get('/prune-pow2/method'),
  models: () => api.get('/prune-pow2/models'),
  datasets: () => api.get('/prune-pow2/datasets'),
  history: () => api.get('/prune-pow2/history'),
  run: (data) => api.post('/prune-pow2/run', data),
  stop: () => api.post('/prune-pow2/stop'),
  status: () => api.get('/prune-pow2/status'),
}
