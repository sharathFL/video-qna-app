/**
 * Video inference UI — API calls and DOM updates.
 * Edit behavior here; edit styles in styles.css.
 */

const videoSelect = document.getElementById('videoSelect');
const runBtn = document.getElementById('runBtn');
const result = document.getElementById('result');
const videoContainer = document.getElementById('videoContainer');
const resultVideo = document.getElementById('resultVideo');
const overlayGT = document.getElementById('overlayGT');
const overlayPred = document.getElementById('overlayPred');
const currentFrameLabel = document.getElementById('currentFrameLabel');
const resPath = document.getElementById('resPath');
const resGT = document.getElementById('resGT');
const resPred = document.getElementById('resPred');
const resCorrect = document.getElementById('resCorrect');
const voteSAFE = document.getElementById('voteSAFE');
const voteUNSAFE = document.getElementById('voteUNSAFE');
const frameDist = document.getElementById('frameDist');
const frameList = document.getElementById('frameList');

let lastFrameDistribution = [];
let lastGroundTruth = null;

function getFrameAtTime(timeSec) {
  if (!lastFrameDistribution.length) return null;
  let best = lastFrameDistribution[0];
  for (const f of lastFrameDistribution) {
    if (f.time_sec <= timeSec) best = f;
    else break;
  }
  return best;
}

function frameLabel(f) {
  if (!f) return 'Frame: —';
  var t = 'Frame ' + f.frame_index + ' (' + f.time_sec + 's): ' + (f.prediction || '—');
  if (f.reason && f.reason.trim()) t += ' — ' + f.reason.trim();
  return t;
}

function setOverlayForFrame(overlayPredEl, currentFrameLabelEl, f) {
  const pred = f ? (f.prediction || '—') : '—';
  overlayPredEl.textContent = pred;
  var correct = lastGroundTruth != null && pred === lastGroundTruth;
  overlayPredEl.className = 'badge frame-pred ' + (correct ? 'badge-pred-correct' : 'badge-pred-wrong');
  currentFrameLabelEl.textContent = frameLabel(f);
}

// Load video list
fetch('/videos')
  .then(function (r) { return r.json(); })
  .then(function (data) {
    videoSelect.innerHTML = '<option value="">-- choose a video --</option>';
    (data.videos || []).forEach(function (v) {
      const opt = document.createElement('option');
      opt.value = v.path;
      opt.textContent = v.path + ' (' + v.label + ')';
      videoSelect.appendChild(opt);
    });
    runBtn.disabled = false;
  });

// Full test: run all test videos, show progress and metrics
(function () {
  var runFullTestBtn = document.getElementById('runFullTestBtn');
  var fullTestProgress = document.getElementById('fullTestProgress');
  var fullTestStatus = document.getElementById('fullTestStatus');
  var fullTestProgressFill = document.getElementById('fullTestProgressFill');
  var fullTestMetrics = document.getElementById('fullTestMetrics');
  var metricAccuracy = document.getElementById('metricAccuracy');
  var metricCorrect = document.getElementById('metricCorrect');
  var metricTotal = document.getElementById('metricTotal');

  if (!runFullTestBtn) return;

  var pollInterval = null;

  function stopPolling() {
    if (pollInterval) {
      clearInterval(pollInterval);
      pollInterval = null;
    }
  }

  function updateProgress(current, total) {
    fullTestStatus.textContent = total ? current + ' / ' + total + ' videos' : 'Starting…';
    var pct = total ? Math.round((100 * current) / total) : 0;
    fullTestProgressFill.style.width = pct + '%';
  }

  function pollStatus() {
    fetch('/test-status')
      .then(function (r) { return r.json(); })
      .then(function (data) {
        updateProgress(data.current || 0, data.total || 0);
        if (data.error) {
          stopPolling();
          fullTestStatus.textContent = 'Error: ' + data.error;
          runFullTestBtn.disabled = false;
          return;
        }
        if (data.done) {
          stopPolling();
          fullTestStatus.textContent = 'Done.';
          runFullTestBtn.disabled = false;
          fullTestMetrics.classList.remove('hidden');
          metricAccuracy.textContent = data.accuracy != null ? data.accuracy : '—';
          metricCorrect.textContent = data.correct != null ? data.correct : '—';
          metricTotal.textContent = data.total != null ? data.total : '—';
        }
      })
      .catch(function (err) {
        stopPolling();
        fullTestStatus.textContent = 'Request failed: ' + (err.message || 'unknown');
        runFullTestBtn.disabled = false;
      });
  }

  runFullTestBtn.addEventListener('click', function () {
    if (runFullTestBtn.disabled) return;
    runFullTestBtn.disabled = true;
    fullTestMetrics.classList.add('hidden');
    fullTestProgress.classList.remove('hidden');
    updateProgress(0, 0);

    fetch('/run-test')
      .then(function (r) { return r.json(); })
      .then(function (data) {
        if (data.error && data.error !== 'Test already in progress') {
          fullTestStatus.textContent = data.error;
          runFullTestBtn.disabled = false;
          return;
        }
        pollStatus();
        pollInterval = setInterval(pollStatus, 1000);
      })
      .catch(function (err) {
        fullTestStatus.textContent = 'Failed to start: ' + (err.message || 'unknown');
        runFullTestBtn.disabled = false;
      });
  });
})();

runBtn.addEventListener('click', function () {
  const path = videoSelect.value;
  if (!path) return;
  runBtn.disabled = true;
  frameDist.innerHTML = '';
  frameList.innerHTML = 'Loading...';
  videoContainer.style.display = 'none';

  fetch('/infer-video?path=' + encodeURIComponent(path))
    .then(function (r) { return r.json(); })
    .then(function (data) {
      runBtn.disabled = false;
      if (data.error) {
        result.classList.add('show');
        resPath.textContent = path;
        resGT.textContent = '-';
        resPred.textContent = data.error;
        resCorrect.textContent = '';
        voteSAFE.textContent = 'SAFE: 0';
        voteUNSAFE.textContent = 'UNSAFE: 0';
        frameDist.innerHTML = '';
        frameList.innerHTML = '';
        lastFrameDistribution = [];
        return;
      }
      result.classList.add('show');
      lastFrameDistribution = data.frame_distribution || [];
      lastGroundTruth = data.ground_truth || null;
      resPath.textContent = data.video_path || path;
      resGT.textContent = data.ground_truth || '-';
      resPred.textContent = data.prediction || '-';
      resPred.className = (data.prediction === 'SAFE') ? 'SAFE' : (data.prediction === 'UNSAFE') ? 'UNSAFE' : 'other';
      resCorrect.textContent = data.correct !== undefined ? (data.correct ? ' ✓' : ' ✗') : '';
      if (data.multiclass && data.frame_votes && Object.keys(data.frame_votes).length > 2) {
        var parts = Object.entries(data.frame_votes).filter(function (e) { return e[1] > 0; }).map(function (e) { return e[0] + ': ' + e[1]; });
        voteSAFE.textContent = 'Votes: ' + (parts.length ? parts.join(' | ') : '—');
        voteUNSAFE.textContent = '';
      } else {
        voteSAFE.textContent = 'SAFE: ' + (data.frame_votes && data.frame_votes.SAFE != null ? data.frame_votes.SAFE : 0);
        voteUNSAFE.textContent = 'UNSAFE: ' + (data.frame_votes && data.frame_votes.UNSAFE != null ? data.frame_votes.UNSAFE : 0);
      }

      frameDist.innerHTML = '';
      lastFrameDistribution.forEach(function (f) {
        const span = document.createElement('span');
        span.className = 'frame ' + (f.prediction === 'SAFE' ? 'SAFE' : f.prediction === 'UNSAFE' ? 'UNSAFE' : 'other');
        span.title = frameLabel(f);
        frameDist.appendChild(span);
      });
      frameList.innerHTML = '';
      lastFrameDistribution.forEach(function (f) {
        const div = document.createElement('div');
        div.className = 'frame-item';
        const predLine = document.createElement('div');
        predLine.className = 'frame-pred-text';
        predLine.textContent = 'Frame ' + f.frame_index + ' (' + f.time_sec + 's): ' + (f.prediction || '—');
        div.appendChild(predLine);
        if (f.reason && f.reason.trim()) {
          const reasonLine = document.createElement('div');
          reasonLine.className = 'frame-reason';
          reasonLine.textContent = f.reason.trim();
          div.appendChild(reasonLine);
        }
        frameList.appendChild(div);
      });

      videoContainer.style.display = 'inline-block';
      resultVideo.src = '/video?path=' + encodeURIComponent(data.video_path || path);
      overlayGT.textContent = 'GT: ' + (data.ground_truth || '—');
      overlayGT.className = 'badge badge-gt';
      var overlayVideoPred = document.getElementById('overlayVideoPred');
      overlayVideoPred.textContent = 'Video: ' + (data.prediction || '—');
      overlayVideoPred.className = 'badge final-badge ' + (data.correct ? 'badge-pred-correct' : 'badge-pred-wrong');
      setOverlayForFrame(overlayPred, currentFrameLabel, lastFrameDistribution[0] || null);
      resultVideo.ontimeupdate = function () {
        var t = Math.floor(resultVideo.currentTime);
        setOverlayForFrame(overlayPred, currentFrameLabel, getFrameAtTime(t));
      };
    })
    .catch(function (err) {
      runBtn.disabled = false;
      result.classList.add('show');
      resPred.textContent = err.message || 'Request failed';
      frameList.innerHTML = '';
      videoContainer.style.display = 'none';
      lastFrameDistribution = [];
    });
});
