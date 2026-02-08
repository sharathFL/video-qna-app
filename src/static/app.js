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
  overlayPredEl.className = 'badge frame-pred ' + (pred === 'SAFE' ? 'SAFE' : pred === 'UNSAFE' ? 'UNSAFE' : '');
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
      resPath.textContent = data.video_path || path;
      resGT.textContent = data.ground_truth || '-';
      resPred.textContent = data.prediction || '-';
      resPred.className = (data.prediction === 'SAFE') ? 'SAFE' : 'UNSAFE';
      resCorrect.textContent = data.correct !== undefined ? (data.correct ? ' ✓' : ' ✗') : '';
      voteSAFE.textContent = 'SAFE: ' + (data.frame_votes && data.frame_votes.SAFE != null ? data.frame_votes.SAFE : 0);
      voteUNSAFE.textContent = 'UNSAFE: ' + (data.frame_votes && data.frame_votes.UNSAFE != null ? data.frame_votes.UNSAFE : 0);

      frameDist.innerHTML = '';
      lastFrameDistribution.forEach(function (f) {
        const span = document.createElement('span');
        span.className = 'frame ' + (f.prediction === 'SAFE' || f.prediction === 'UNSAFE' ? f.prediction : 'other');
        span.title = frameLabel(f);
        frameDist.appendChild(span);
      });
      frameList.innerHTML = '';
      lastFrameDistribution.forEach(function (f) {
        const div = document.createElement('div');
        div.textContent = frameLabel(f);
        frameList.appendChild(div);
      });

      videoContainer.style.display = 'inline-block';
      resultVideo.src = '/video?path=' + encodeURIComponent(data.video_path || path);
      overlayGT.textContent = 'GT: ' + (data.ground_truth || '—');
      overlayGT.className = 'badge ' + (data.ground_truth === 'SAFE' ? 'SAFE' : 'UNSAFE');
      var overlayVideoPred = document.getElementById('overlayVideoPred');
      overlayVideoPred.textContent = 'Video: ' + (data.prediction || '—');
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
