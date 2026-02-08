/**
 * YouTube inference UI: loading overlay until model ready, then embed video, run inference, show result.
 */

(function () {
  var loadOverlay = document.getElementById('loadOverlay');
  var mainContent = document.getElementById('mainContent');
  var loadMessage = document.getElementById('loadMessage');
  var loadPct = document.getElementById('loadPct');
  var loadProgressFill = document.getElementById('loadProgressFill');
  var loadStats = document.getElementById('loadStats');
  var statGpu = document.getElementById('statGpu');
  var statGpuMem = document.getElementById('statGpuMem');
  var statCpu = document.getElementById('statCpu');
  var statRam = document.getElementById('statRam');
  var loadError = document.getElementById('loadError');

  var ytUrl = document.getElementById('ytUrl');
  var embedBtn = document.getElementById('embedBtn');
  var runInferBtn = document.getElementById('runInferBtn');
  var embedWrap = document.getElementById('embedWrap');
  var embedPlaceholder = document.getElementById('embedPlaceholder');
  var ytEmbed = document.getElementById('ytEmbed');
  var resultSection = document.getElementById('resultSection');
  var resultStatus = document.getElementById('resultStatus');
  var resultContent = document.getElementById('resultContent');
  var resPred = document.getElementById('resPred');
  var voteSAFE = document.getElementById('voteSAFE');
  var voteUNSAFE = document.getElementById('voteUNSAFE');
  var totalFrames = document.getElementById('totalFrames');
  var frameDist = document.getElementById('frameDist');
  var frameList = document.getElementById('frameList');

  function formatStat(s) {
    if (s == null || s === undefined) return '—';
    return String(s);
  }

  function pollLoadStatus() {
    fetch('/load-status')
      .then(function (r) { return r.json(); })
      .then(function (d) {
        loadMessage.textContent = d.message || 'Loading…';
        loadPct.textContent = (d.progress_pct != null ? d.progress_pct : 0) + '%';
        loadProgressFill.style.width = (d.progress_pct != null ? d.progress_pct : 0) + '%';

        statGpu.textContent = d.gpu_util_pct != null ? d.gpu_util_pct + '% util' : (d.gpu_temp_c != null ? d.gpu_temp_c + '°C' : '—');
        statGpuMem.textContent = (d.gpu_mem_used_mb != null && d.gpu_mem_total_mb != null)
          ? (d.gpu_mem_used_mb + ' / ' + d.gpu_mem_total_mb + ' MB') : '—';
        statCpu.textContent = d.cpu_pct != null ? d.cpu_pct + '%' : '—';
        if (d.ram_used_mb != null && d.ram_total_mb != null) {
          statRam.textContent = d.ram_used_mb + ' / ' + d.ram_total_mb + ' MB';
        } else {
          statRam.textContent = '—';
        }

        if (d.error) {
          loadError.textContent = d.error;
          loadError.classList.remove('hidden');
          loadOverlay.classList.remove('hidden');
          mainContent.classList.add('hidden');
          return;
        }
        if (d.loaded) {
          loadOverlay.classList.add('hidden');
          mainContent.classList.remove('hidden');
          return;
        }
        setTimeout(pollLoadStatus, 500);
      })
      .catch(function (err) {
        loadMessage.textContent = 'Waiting for server…';
        setTimeout(pollLoadStatus, 1000);
      });
  }

  pollLoadStatus();

  function getVideoId(url) {
    if (!url || !url.trim()) return null;
    var m = url.match(/(?:v=|\.be\/|embed\/)([a-zA-Z0-9_-]{11})/);
    return m ? m[1] : null;
  }

  function embedVideo() {
    var url = (ytUrl.value || '').trim();
    var id = getVideoId(url);
    if (!id) {
      embedPlaceholder.textContent = 'Enter a valid YouTube URL and click "Embed video".';
      embedWrap.classList.remove('has-embed');
      ytEmbed.src = '';
      return;
    }
    ytEmbed.src = 'https://www.youtube.com/embed/' + id;
    embedWrap.classList.add('has-embed');
    embedPlaceholder.textContent = '';
  }

  function frameLabel(f) {
    if (!f) return 'Frame: —';
    var t = 'Frame ' + f.frame_index + ' (' + f.time_sec + 's): ' + (f.prediction || '—');
    if (f.reason && f.reason.trim()) t += ' — ' + f.reason.trim();
    return t;
  }

  embedBtn.addEventListener('click', embedVideo);

  runInferBtn.addEventListener('click', function () {
    var url = (ytUrl.value || '').trim();
    if (!url) {
      resultStatus.textContent = 'Enter a YouTube URL first.';
      resultContent.classList.add('hidden');
      return;
    }
    if (!getVideoId(url)) {
      resultStatus.textContent = 'Invalid YouTube URL.';
      resultContent.classList.add('hidden');
      return;
    }
    runInferBtn.disabled = true;
    resultStatus.textContent = 'Running inference (streaming video, no download)… This may take a minute.';
    resultContent.classList.add('hidden');

    fetch('/infer-youtube?url=' + encodeURIComponent(url))
      .then(function (r) { return r.json(); })
      .then(function (data) {
        runInferBtn.disabled = false;
        if (data.error) {
          resultStatus.textContent = 'Error: ' + data.error;
          resultContent.classList.add('hidden');
          return;
        }
        resultStatus.textContent = 'Done.';
        resultContent.classList.remove('hidden');
        resPred.textContent = data.prediction || '—';
        resPred.className = 'badge ' + (data.prediction === 'SAFE' ? 'SAFE' : data.prediction === 'UNSAFE' ? 'UNSAFE' : '');
        voteSAFE.textContent = 'SAFE: ' + (data.frame_votes && data.frame_votes.SAFE != null ? data.frame_votes.SAFE : 0);
        voteUNSAFE.textContent = 'UNSAFE: ' + (data.frame_votes && data.frame_votes.UNSAFE != null ? data.frame_votes.UNSAFE : 0);
        totalFrames.textContent = data.total_frames != null ? data.total_frames : 0;

        frameDist.innerHTML = '';
        (data.frame_distribution || []).forEach(function (f) {
          var span = document.createElement('span');
          span.className = 'frame ' + (f.prediction === 'SAFE' || f.prediction === 'UNSAFE' ? f.prediction : 'other');
          span.title = frameLabel(f);
          frameDist.appendChild(span);
        });
        frameList.innerHTML = '';
        (data.frame_distribution || []).forEach(function (f) {
          var div = document.createElement('div');
          div.textContent = frameLabel(f);
          frameList.appendChild(div);
        });
      })
      .catch(function (err) {
        runInferBtn.disabled = false;
        resultStatus.textContent = 'Request failed: ' + (err.message || 'unknown');
        resultContent.classList.add('hidden');
      });
  });
})();
