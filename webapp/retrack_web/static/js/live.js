// Live tracking view (UC-13, FR-21): polls the structured result and draws the camera annotations and the
// bird's eye view. Viewing never affects processing; if this page is closed the session keeps running.
(() => {
  'use strict'

  const root = document.getElementById('live')
  if (!root) return
  const url = root.dataset.url
  const POLL_MS = 1000
  const $ = id => document.getElementById(id)

  const RUNTIME = {
    Initializing: ['info', 'hourglass-split', 'Initializing'],
    INITIALIZING: ['info', 'hourglass-split', 'Initializing'],
    WAITING_FOR_INPUT: ['secondary', 'hourglass', 'Waiting for input'],
    PROCESSING: ['primary', 'cpu', 'Processing'],
    PAUSED_MISSING_CAMERA: ['warning', 'pause-circle', 'Paused: missing camera'],
    STOPPING: ['info', 'stop-circle', 'Stopping'],
    FINALIZED: ['light', 'flag', 'Finalized']
  }
  const STATUS = { Initializing: 'info', Running: 'primary', Completed: 'success', Failed: 'danger' }

  const colour = id => `hsl(${(id * 137.508) % 360}, 75%, 48%)`
  const esc = text => String(text).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c])

  const tokenToggle = $('toggleToken')
  if (tokenToggle) {
    tokenToggle.addEventListener('click', () => {
      const input = $('ingestToken')
      input.type = input.type === 'password' ? 'text' : 'password'
    })
  }

  function setStatus (data) {
    const s = data.session
    $('statusBadge').innerHTML = `<span class="badge rounded-pill text-bg-${STATUS[s.status] || 'secondary'}">${esc(s.status)}</span>`
    const r = RUNTIME[s.runtimeStatus] || ['secondary', 'circle', s.runtimeStatus]
    $('runtimeBadge').innerHTML = `<span class="badge text-bg-${r[0]}${r[0] === 'light' ? ' border' : ''}"><i class="bi bi-${r[1]} me-1"></i>${esc(r[2])}</span>`

    // Distinct states (NFR-01): waiting, paused input fault, processing, failure, finished
    let level = 'secondary'
    if (s.status === 'Failed') level = 'danger'
    else if (s.status === 'Completed') level = 'success'
    else if (s.runtimeStatus === 'PAUSED_MISSING_CAMERA') level = 'warning'
    else if (s.runtimeStatus === 'PROCESSING') level = 'primary'
    else if (s.runtimeStatus === 'STOPPING' || s.status === 'Initializing') level = 'info'
    $('statusAlert').className = `alert alert-${level} d-flex align-items-center`
    $('statusMessage').textContent = s.statusMessage || ''
    $('statusSpinner').hidden = s.runtimeStatus === 'FINALIZED' || s.runtimeStatus === 'PAUSED_MISSING_CAMERA'

    $('framesProcessed').textContent = s.framesProcessed
    $('framesRejected').textContent = s.framesRejected
    const stopForm = $('stopForm')
    if (stopForm) stopForm.hidden = !(s.status === 'Running' || s.status === 'Initializing') || s.stopRequested

    if (s.lastInputFault) {
      $('faultCard').hidden = false
      const f = s.lastInputFault
      $('faultText').innerHTML = `<span class="badge text-bg-warning me-1">${esc(f.kind)}</span>` +
        `${f.frameIndex !== null ? 'Frame ' + esc(f.frameIndex) + ': ' : ''}${esc(f.reason)}` +
        `<div class="text-body-secondary mt-1 mono">${esc(f.at)}</div>`
    }

    if (s.summary) {
      $('finalSummary').hidden = false
      const items = [
        ['Frames processed', s.summary.framesProcessed],
        ['Unique tracks', s.summary.uniqueTracks],
        ['Mean occupancy', Number(s.summary.meanOccupancy).toFixed(3)],
        ['Peak occupancy', s.summary.peakOccupancy]
      ]
      $('summaryValues').innerHTML = items.map(([k, v]) =>
        `<div class="col"><div class="text-body-secondary small text-uppercase">${k}</div><div class="fs-4 fw-semibold">${esc(v)}</div></div>`).join('')
    }
  }

  function setFrame (data) {
    const frame = data.frameSet
    const results = data.results
    $('frameIndex').textContent = frame ? frame.frameIndex : '—'
    $('frameTime').textContent = frame ? frame.sourceTime : '—'
    if (!results) return
    $('occupancy').textContent = results.occupancy
    const rows = results.tracks.map(t =>
      `<tr><td><span class="badge" style="background:${colour(t.trackId)}">${t.trackId}</span></td>` +
      `<td class="text-end mono">${t.groundPosition.x.toFixed(1)}</td><td class="text-end mono">${t.groundPosition.y.toFixed(1)}</td>` +
      `<td class="text-end mono">${t.confidence.toFixed(2)}</td></tr>`)
    $('trackRows').innerHTML = rows.length ? rows.join('')
      : '<tr><td colspan="4" class="small"><i class="bi bi-check-circle text-success me-1"></i>Valid frame set with zero pedestrians.</td></tr>'
  }

  function drawBev (data) {
    const canvas = $('bev-canvas')
    const ctx = canvas.getContext('2d')
    const W = canvas.width
    const H = canvas.height
    const pad = 30
    const [x0, x1] = data.scene.groundXRangeCm
    const [y0, y1] = data.scene.groundYRangeCm
    // world y (36 m) runs left to right, world x (12 m) top to bottom
    const sx = (W - 2 * pad) / (y1 - y0)
    const sy = (H - 2 * pad) / (x1 - x0)
    const px = p => [pad + (p[1] - y0) * sx, pad + (p[0] - x0) * sy]
    const styles = getComputedStyle(document.documentElement)
    const grid = styles.getPropertyValue('--bs-border-color') || '#ccc'
    const text = styles.getPropertyValue('--bs-secondary-color') || '#666'

    ctx.clearRect(0, 0, W, H)
    ctx.strokeStyle = grid
    ctx.lineWidth = 1
    ctx.font = '11px system-ui, sans-serif'
    ctx.fillStyle = text
    for (let y = Math.ceil(y0 / 100) * 100; y <= y1; y += 100) {
      const [cx] = px([x0, y])
      ctx.globalAlpha = y % 500 === 0 ? 0.9 : 0.35
      ctx.beginPath(); ctx.moveTo(cx, pad); ctx.lineTo(cx, H - pad); ctx.stroke()
      if (y % 500 === 0) ctx.fillText(`${y / 100} m`, cx - 10, H - pad + 16)
    }
    for (let x = Math.ceil(x0 / 100) * 100; x <= x1; x += 100) {
      const [, cy] = px([x, y0])
      ctx.globalAlpha = x % 300 === 0 ? 0.9 : 0.35
      ctx.beginPath(); ctx.moveTo(pad, cy); ctx.lineTo(W - pad, cy); ctx.stroke()
      if (x % 300 === 0) ctx.fillText(`${x / 100} m`, 2, cy + 4)
    }
    ctx.globalAlpha = 1
    ctx.strokeRect(pad, pad, W - 2 * pad, H - 2 * pad)
    ctx.fillText('y →', W - pad - 20, pad - 8)
    ctx.fillText('x ↓', 4, pad - 8)

    if (!data.results) {
      ctx.fillText('No processed frame set yet', W / 2 - 70, H / 2)
      return
    }
    for (const traj of data.results.trajectories) {
      if (traj.points.length < 2) continue
      ctx.strokeStyle = colour(traj.trackId)
      ctx.globalAlpha = 0.7
      ctx.lineWidth = 2
      ctx.beginPath()
      traj.points.forEach((p, i) => { const [cx, cy] = px(p); i ? ctx.lineTo(cx, cy) : ctx.moveTo(cx, cy) })
      ctx.stroke()
    }
    ctx.globalAlpha = 1
    for (const t of data.results.tracks) {
      const [cx, cy] = px([t.groundPosition.x, t.groundPosition.y])
      ctx.fillStyle = colour(t.trackId)
      ctx.beginPath(); ctx.arc(cx, cy, 6, 0, 2 * Math.PI); ctx.fill()
      ctx.fillStyle = text
      ctx.fillText(String(t.trackId), cx + 8, cy - 6)
    }
  }

  function drawCameras (data) {
    const display = data.display || {}
    const images = display.cameraImages || {}
    const results = data.results
    const inSync = results && data.frameSet && display.frameIndex === data.frameSet.frameIndex
    document.querySelectorAll('.camera-tile').forEach(tile => {
      const cam = tile.dataset.camera
      const img = tile.querySelector('img')
      const canvas = tile.querySelector('canvas')
      const ctx = canvas.getContext('2d')
      ctx.clearRect(0, 0, canvas.width, canvas.height)
      if (images[cam]) {
        img.src = images[cam]
        img.hidden = false
        tile.querySelector('.camera-empty').hidden = true
      }
      if (!inSync) return
      const scale = canvas.width / 1920
      ctx.font = 'bold 12px system-ui, sans-serif'
      for (const t of results.tracks) {
        const p = t.cameraPixels[cam]
        if (!p) continue
        const x = p.u * scale
        const y = p.v * scale
        ctx.fillStyle = colour(t.trackId)
        ctx.strokeStyle = '#fff'
        ctx.lineWidth = 1.5
        ctx.beginPath(); ctx.arc(x, y, 4, 0, 2 * Math.PI); ctx.fill(); ctx.stroke()
        const label = String(t.trackId)
        const w = ctx.measureText(label).width + 6
        ctx.fillRect(x - w / 2, y - 22, w, 15)
        ctx.fillStyle = '#fff'
        ctx.fillText(label, x - w / 2 + 3, y - 10)
      }
    })
  }

  let finished = false
  async function poll () {
    try {
      const response = await fetch(`${url}?images=1`, { headers: { Accept: 'application/json' }, cache: 'no-store' })
      if (response.status === 401) { window.location.reload(); return }
      if (!response.ok) throw new Error(`HTTP ${response.status}`)
      const data = await response.json()
      setStatus(data)
      setFrame(data)
      drawBev(data)
      drawCameras(data)
      finished = data.session.runtimeStatus === 'FINALIZED'
    } catch (err) {
      $('statusAlert').className = 'alert alert-danger d-flex align-items-center'
      $('statusMessage').textContent = `The live view could not be updated (${err.message}). Retrying…`
    }
    if (!finished) setTimeout(poll, POLL_MS)
  }
  poll()
})()
