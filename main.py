"""
main.py
-------
Intelligent Face Tracker — single entry point.

Runs Flask dashboard + video/RTSP processing pipeline together.
User selects source from the web UI at runtime.

Usage:
  python main.py              # local VSCode: dashboard at localhost:5000
  python main.py --colab      # Colab: creates public Cloudflare URL
  python main.py --source x   # auto-start with specific file or RTSP URL
"""

import sys, os, argparse, logging, threading, time, subprocess, re, shutil, cv2
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from config_loader import Config
from event_logger import setup_logging
from pipeline import FaceTrackerPipeline
from flask import Flask, render_template_string, send_file, \
                  request, Response, jsonify, redirect
import sqlite3
from datetime import datetime

# ------------------------------------------------------------------ #
# Globals — initialised at startup, used by Flask routes              #
# ------------------------------------------------------------------ #

_DB_PATH = "face_tracker.db"
_CFG     = None          # Config object — set in _init_globals()
_PORT    = 5000

def _init_globals(config_path="config/config.json", db_path=None, port=5000):
    """
    Called once at startup (from main() or from Colab cell).
    Sets all globals so Flask routes can use them immediately.
    """
    global _CFG, _DB_PATH, _PORT
    _CFG     = Config(config_path=config_path)
    _DB_PATH = db_path or _CFG.database.get("path", "face_tracker.db")
    _PORT    = port
    setup_logging(
        log_file=_CFG.logging.get("log_file", "logs/events.log"),
        log_level=_CFG.logging.get("log_level", "INFO"),
    )
    # Ensure log dirs exist
    for d in ["logs/entries", "logs/exits", "logs/registered"]:
        os.makedirs(d, exist_ok=True)


# ------------------------------------------------------------------ #
# Shared processing state                                              #
# ------------------------------------------------------------------ #

class ProcessingState:
    def __init__(self):
        self.lock          = threading.Lock()
        self._source       = ""
        self._source_type  = ""
        self._frame_count  = 0
        self._unique_count = 0
        self._running      = False
        self._status       = "idle"   # idle|running|paused|stopped|error
        self._error_msg    = ""
        self.request_stop  = threading.Event()
        self.request_pause = threading.Event()  # set=paused, clear=running

    def update(self, **kw):
        with self.lock:
            for k, v in kw.items():
                setattr(self, f"_{k}", v)

    def set_source(self, source, source_type):
        with self.lock:
            self._source      = source
            self._source_type = source_type

    def snapshot(self):
        with self.lock:
            return dict(
                source       = self._source,
                source_type  = self._source_type,
                frame_count  = self._frame_count,
                unique_count = self._unique_count,
                running      = self._running,
                paused       = self.request_pause.is_set(),
                status       = self._status,
                error        = self._error_msg,
            )


STATE = ProcessingState()


# ------------------------------------------------------------------ #
# Processing worker                                                    #
# ------------------------------------------------------------------ #

def _run_pipeline(source: str):
    """
    Worker that runs in a background thread.
    Uses the global _CFG — must be set before calling.
    """
    logger      = logging.getLogger("worker")
    is_live     = source.startswith(("rtsp://","rtmp://","http://","https://"))
    source_type = "RTSP stream" if is_live else "Video file"

    STATE.set_source(source, source_type)
    STATE.update(running=True, status="running", frame_count=0, error="")
    STATE.request_stop.clear()
    STATE.request_pause.clear()

    logger.info(f"Pipeline starting — {source_type}: {source}")

    # RTSP pre-flight check
    if is_live:
        cap_test = cv2.VideoCapture(source)
        cap_test.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 8000)
        if not cap_test.isOpened():
            cap_test.release()
            msg = (
                f"Cannot connect to RTSP stream: {source}\n"
                "Check: IP address, port, username, password, camera power."
            )
            logger.error(msg)
            STATE.update(running=False, status="error", error=msg)
            return
        ret, _ = cap_test.read()
        cap_test.release()
        if not ret:
            msg = f"Connected but could not read frame from: {source}"
            logger.error(msg)
            STATE.update(running=False, status="error", error=msg)
            return
        logger.info("RTSP pre-flight passed.")

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        msg = f"Cannot open source: {source}"
        logger.error(msg)
        STATE.update(running=False, status="error", error=msg)
        return

    fps          = cap.get(cv2.CAP_PROP_FPS) or 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.info(
        f"Opened — FPS:{fps:.1f}  "
        f"frames:{'live' if is_live else total_frames}"
    )

    # Build pipeline using the global _CFG
    try:
        pipeline = FaceTrackerPipeline(_CFG)
    except Exception as ex:
        msg = f"Pipeline init failed: {ex}"
        logger.error(msg, exc_info=True)
        STATE.update(running=False, status="error", error=msg)
        cap.release()
        return

    frame_num            = 0
    consecutive_failures = 0
    MAX_FAILURES         = 20 if is_live else 10

    try:
        while not STATE.request_stop.is_set():
            ret, frame = cap.read()

            if not ret:
                consecutive_failures += 1
                if is_live:
                    if consecutive_failures <= 5:
                        time.sleep(0.1)
                        continue
                    logger.warning("Stream dropped — reconnecting...")
                    cap.release()
                    time.sleep(3)
                    cap = cv2.VideoCapture(source)
                    if not cap.isOpened():
                        logger.error("Reconnect failed.")
                        break
                    consecutive_failures = 0
                    logger.info("Reconnected.")
                    continue
                else:
                    if consecutive_failures >= MAX_FAILURES:
                        logger.info("End of video file.")
                        break
                    continue

            consecutive_failures = 0

            # Pause handling — sleep here, keeps capture alive
            while STATE.request_pause.is_set():
                if STATE.request_stop.is_set():
                    break
                time.sleep(0.15)

            if STATE.request_stop.is_set():
                break

            frame_num += 1
            pipeline.process_frame(frame)

            # Update shared state every 10 frames
            if frame_num % 10 == 0:
                STATE.update(
                    frame_count  = frame_num,
                    unique_count = pipeline.get_unique_visitor_count(),
                )

            if frame_num % 60 == 0:
                logger.info(
                    f"Frame {frame_num} | "
                    f"unique: {pipeline.get_unique_visitor_count()}"
                )

    except Exception as ex:
        logger.error(f"Processing error: {ex}", exc_info=True)
        STATE.update(status="error", error=str(ex))

    finally:
        pipeline.flush_remaining_tracks()
        cap.release()
        unique = pipeline.get_unique_visitor_count()
        STATE.update(
            running      = False,
            status       = "stopped",
            unique_count = unique,
            frame_count  = frame_num,
        )
        logger.info(f"Done — frames:{frame_num}  unique:{unique}")


# ------------------------------------------------------------------ #
# Flask app                                                            #
# ------------------------------------------------------------------ #

flask_app = Flask(__name__)

TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Face Tracker Dashboard</title>
  <meta http-equiv="refresh" content="4;url=/?tab={{ tab }}">
  <style>
    *,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
    body{font-family:'Courier New',monospace;background:#0a0a0a;color:#e0e0e0;min-height:100vh}
    a{color:inherit;text-decoration:none}
    header{background:#111;border-bottom:2px solid #00d4aa44;padding:.9rem 1.5rem;
           display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:.5rem}
    header h1{color:#00d4aa;font-size:1.15rem;letter-spacing:2px}
    .hbadge{font-size:.65rem;color:#555;background:#0d0d0d;
            border:1px solid #1a1a1a;padding:2px 8px;border-radius:3px}
    .ctrl{background:#0d1a10;border-bottom:1px solid #00d4aa22;padding:.9rem 1.5rem}
    .ctrl-title{color:#00d4aa88;font-size:.72rem;text-transform:uppercase;
                letter-spacing:1px;margin-bottom:.7rem}
    .ctrl-row{display:flex;flex-wrap:wrap;gap:.6rem;align-items:flex-end}
    .ctrl select,.ctrl input[type=text]{background:#111;border:1px solid #1e3a2a;
      color:#ccc;padding:.42rem .7rem;border-radius:3px;font-family:inherit;font-size:.78rem}
    .ctrl select{min-width:140px}
    .ctrl input[type=text]{flex:1;min-width:260px}
    .ctrl input[type=text]:focus{outline:none;border-color:#00d4aa88}
    .btn{padding:.42rem 1.1rem;border-radius:3px;border:none;cursor:pointer;
         font-family:inherit;font-size:.78rem;font-weight:bold;white-space:nowrap}
    .btn-start{background:#00d4aa;color:#000}.btn-start:hover{background:#00ffcc}
    .btn-stop{background:#c0392b;color:#fff}.btn-stop:hover{background:#e74c3c}
    .btn-pause{background:#e67e22;color:#fff}.btn-pause:hover{background:#f39c12}
    .btn-resume{background:#27ae60;color:#fff}.btn-resume:hover{background:#2ecc71}
    .btn-clear{background:#1a0a0a;color:#c0392b;border:1px solid #c0392b55}
    .btn-clear:hover{background:#2e0a0a;color:#e74c3c}
    .btn-row{display:flex;flex-wrap:wrap;gap:.5rem;align-items:center;margin-top:.55rem}
    .src-display{opacity:.6;flex:1;min-width:260px;background:#111;border:1px solid #1e3a2a;
                 color:#ccc;padding:.42rem .7rem;border-radius:3px;font-family:inherit;
                 font-size:.78rem}
    .status-wrap{margin-top:.6rem;font-size:.72rem}
    .s-idle{color:#444}
    .live-dot{display:inline-block;width:7px;height:7px;background:#00d4aa;
              border-radius:50%;margin-right:5px;animation:pulse 1.4s ease-in-out infinite}
    @keyframes pulse{0%,100%{opacity:1}50%{opacity:.2}}
    .banner{border-radius:5px;padding:.8rem 1.1rem;margin-top:.7rem;
            display:flex;align-items:flex-start;gap:.8rem}
    .banner-ok{background:#0a2e1a;border:1px solid #00d4aa44}
    .banner-err{background:#2e0a0a;border:1px solid #c0392b44}
    .banner-pause{background:#1a1000;border:1px solid #e67e2244}
    .b-icon{font-size:1.3rem;line-height:1;flex-shrink:0}
    .b-body{font-size:.74rem;line-height:1.75}
    .b-title{font-weight:bold;font-size:.8rem;display:block;margin-bottom:.15rem}
    .ok{color:#00d4aa} .err{color:#e74c3c;white-space:pre-wrap} .warn{color:#e67e22}
    .stats{display:flex;background:#0d0d0d;border-bottom:1px solid #1a1a1a}
    .stat{flex:1;padding:1rem;text-align:center;border-right:1px solid #1a1a1a}
    .stat:last-child{border-right:none}
    .stat .n{font-size:2rem;font-weight:bold;color:#00d4aa;display:block;line-height:1}
    .stat .l{font-size:.6rem;color:#555;text-transform:uppercase;
             letter-spacing:1.5px;margin-top:4px;display:block}
    .tabs{display:flex;background:#0d0d0d;border-bottom:1px solid #1a1a1a;padding:0 1.5rem}
    .tab{padding:.7rem 1.1rem;font-size:.78rem;color:#555;border-bottom:2px solid transparent}
    .tab:hover{color:#aaa}.tab.active{color:#00d4aa;border-bottom-color:#00d4aa}
    .content{padding:1.2rem 1.5rem}
    table{width:100%;border-collapse:collapse;font-size:.78rem}
    thead th{font-size:.6rem;text-transform:uppercase;letter-spacing:1px;color:#444;
             padding:6px 10px;text-align:left;border-bottom:1px solid #1a1a1a}
    tbody td{padding:7px 10px;border-bottom:1px solid #111;vertical-align:middle}
    tbody tr:hover td{background:#0f0f0f}
    .crop{width:52px;height:52px;object-fit:cover;border-radius:4px;
          border:1px solid #1e1e1e;display:block;background:#111}
    .noimg{width:52px;height:52px;background:#111;border:1px solid #1a1a1a;
           border-radius:4px;display:flex;align-items:center;justify-content:center;
           font-size:.5rem;color:#2a2a2a}
    .fid{font-size:.68rem;color:#666;word-break:break-all;max-width:220px}
    .ts{font-size:.68rem;color:#444;white-space:nowrap}
    .be{display:inline-block;padding:2px 7px;border-radius:3px;
        font-size:.65rem;font-weight:bold}
    .be-entry{background:#0a2e1a;color:#00d4aa;border:1px solid #00d4aa33}
    .be-exit{background:#2e0a0a;color:#ff5555;border:1px solid #ff555533}
    .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(110px,1fr));gap:.75rem}
    .card{background:#111;border:1px solid #1a1a1a;border-radius:5px;
          padding:.6rem;text-align:center}
    .card:hover{border-color:#00d4aa33}
    .card img{width:72px;height:72px;object-fit:cover;border-radius:4px;
              display:block;margin:0 auto 4px;background:#0a0a0a}
    .card .cid{font-size:.55rem;color:#444;word-break:break-all}
    .card .cts{font-size:.6rem;color:#555;margin-top:2px}
    .empty{padding:3rem;text-align:center;color:#222;font-size:.82rem}
    footer{padding:.9rem 1.5rem;font-size:.62rem;color:#2a2a2a;
           border-top:1px solid #111;margin-top:2rem}
    footer a{color:#333}
    code{background:#111;padding:1px 4px;border-radius:3px;font-size:.73rem}
    .val-err{color:#e74c3c;font-size:.7rem;margin-top:.3rem;display:none}
  </style>
</head>
<body>
<header>
  <h1>&#9632; Face Tracker — Live Dashboard</h1>
  <span class="hbadge">Auto-refresh 4s &nbsp;|&nbsp; {{ now }}</span>
</header>

<div class="ctrl">
  <div class="ctrl-title">&#9654; Processing Control</div>

  {% if not proc.running %}
  <!-- SOURCE SELECTION FORM -->
  <form method="POST" action="/start" onsubmit="return validateForm()">
    <div class="ctrl-row">
      <select name="mode" id="modesel" onchange="onModeChange()">
        <option value="file">Video File</option>
        <option value="rtsp">RTSP Camera</option>
      </select>
      <input type="text" name="source" id="srcinput"
             placeholder="video_sample1.mp4"
             value="{{ proc.source if proc.source and proc.status != 'error' else '' }}">
      <button class="btn btn-start" type="submit">&#9654; Start</button>
    </div>
    <div class="val-err" id="valerr"></div>
  </form>
  <!-- CLEAR DATA (only when not running) -->
  <form method="POST" action="/clear"
        onsubmit="return confirm('Delete ALL data (database + logs)? Cannot be undone.')">
    <div class="btn-row">
      <button class="btn btn-clear" type="submit">&#128465; Clear All Data</button>
      <span style="font-size:.62rem;color:#333">
        Wipes database &amp; log images — use before a fresh run
      </span>
    </div>
  </form>

  {% else %}
  <!-- RUNNING / PAUSED CONTROLS -->
  <div class="ctrl-row">
    <select disabled class="src-display" style="min-width:140px;opacity:.5">
      <option>{{ proc.source_type }}</option>
    </select>
    <input type="text" disabled class="src-display" value="{{ proc.source }}">
  </div>
  <div class="btn-row">
    {% if proc.paused %}
    <form method="POST" action="/resume">
      <button class="btn btn-resume" type="submit">&#9654; Resume</button>
    </form>
    {% else %}
    <form method="POST" action="/pause">
      <button class="btn btn-pause" type="submit">&#9646;&#9646; Pause</button>
    </form>
    {% endif %}
    <form method="POST" action="/stop">
      <button class="btn btn-stop" type="submit">&#9632; Stop</button>
    </form>
  </div>
  {% endif %}

  <!-- STATUS -->
  <div class="status-wrap">
    {% if proc.running and not proc.paused %}
      <span class="ok">
        <span class="live-dot"></span>
        LIVE &mdash; {{ proc.source_type }} &nbsp;|&nbsp;
        Frame {{ proc.frame_count }} &nbsp;|&nbsp;
        {{ proc.unique_count }} unique visitor{{ 's' if proc.unique_count != 1 else '' }}
      </span>

    {% elif proc.running and proc.paused %}
      <div class="banner banner-pause">
        <span class="b-icon warn">&#9646;&#9646;</span>
        <div class="b-body">
          <span class="b-title warn">Paused</span>
          <span class="warn">
            Frame {{ proc.frame_count }} &nbsp;|&nbsp;
            {{ proc.unique_count }} unique visitor{{ 's' if proc.unique_count != 1 else '' }}
            &nbsp;&mdash;&nbsp; click Resume to continue
          </span>
        </div>
      </div>

    {% elif proc.status == 'stopped' %}
      <div class="banner banner-ok">
        <span class="b-icon ok">&#10003;</span>
        <div class="b-body">
          <span class="b-title ok">Processing Complete</span>
          <span>
            Source: <code>{{ proc.source }}</code><br>
            Frames processed: <strong class="ok">{{ proc.frame_count }}</strong><br>
            Unique visitors: <strong class="ok">
              {{ proc.unique_count }} visitor{{ 's' if proc.unique_count != 1 else '' }}
            </strong><br>
            <span style="color:#446655;font-size:.67rem">
              Results saved to database and logs/. Select a new source to run again.
            </span>
          </span>
        </div>
      </div>

    {% elif proc.status == 'error' %}
      <div class="banner banner-err">
        <span class="b-icon err">&#9888;</span>
        <div class="b-body err">{{ proc.error }}</div>
      </div>

    {% else %}
      <span class="s-idle">
        &#9656; Enter a video filename or RTSP URL above and click Start.
      </span>
    {% endif %}
  </div>
</div>

<div class="stats">
  <div class="stat"><span class="n">{{ s.unique_visitors }}</span><span class="l">Unique Visitors</span></div>
  <div class="stat"><span class="n">{{ s.total_entries }}</span><span class="l">Total Entries</span></div>
  <div class="stat"><span class="n">{{ s.total_exits }}</span><span class="l">Total Exits</span></div>
  <div class="stat"><span class="n">{{ s.registered_faces }}</span><span class="l">Registered Faces</span></div>
</div>

<div class="tabs">
  <a class="tab {% if tab=='entries'    %}active{% endif %}" href="/?tab=entries">
    Entries ({{ s.total_entries }})</a>
  <a class="tab {% if tab=='exits'      %}active{% endif %}" href="/?tab=exits">
    Exits ({{ s.total_exits }})</a>
  <a class="tab {% if tab=='registered' %}active{% endif %}" href="/?tab=registered">
    Registered ({{ s.registered_faces }})</a>
  <a class="tab {% if tab=='all'        %}active{% endif %}" href="/?tab=all">
    All Events ({{ s.total_entries + s.total_exits }})</a>
</div>

<div class="content">
  {% if tab == 'entries' %}
    {% if rows %}
    <table><thead><tr><th>Crop</th><th>Face ID</th><th>Timestamp</th></tr></thead><tbody>
      {% for e in rows %}<tr>
        <td>{% if e.image_path %}<img class="crop" src="/img?p={{ e.image_path }}" alt="">
            {% else %}<div class="noimg">no img</div>{% endif %}</td>
        <td class="fid">{{ e.face_id }}</td>
        <td class="ts">{{ e.timestamp }}</td>
      </tr>{% endfor %}
    </tbody></table>
    {% else %}<div class="empty">No entry events yet. Enter a source and click Start.</div>
    {% endif %}

  {% elif tab == 'exits' %}
    {% if rows %}
    <table><thead><tr><th>Crop</th><th>Face ID</th><th>Timestamp</th></tr></thead><tbody>
      {% for e in rows %}<tr>
        <td>{% if e.image_path %}<img class="crop" src="/img?p={{ e.image_path }}" alt="">
            {% else %}<div class="noimg">no img</div>{% endif %}</td>
        <td class="fid">{{ e.face_id }}</td>
        <td class="ts">{{ e.timestamp }}</td>
      </tr>{% endfor %}
    </tbody></table>
    {% else %}<div class="empty">No exit events yet.</div>{% endif %}

  {% elif tab == 'registered' %}
    {% if rows %}
    <div class="grid">
      {% for f in rows %}<div class="card">
        {% if f.crop_path %}
          <img src="/img?p={{ f.crop_path }}" alt="" onerror="this.style.display='none'">
        {% else %}
          <div style="width:72px;height:72px;background:#0a0a0a;border-radius:4px;
                      margin:0 auto 4px"></div>
        {% endif %}
        <div class="cid">{{ f.face_id[-14:] }}</div>
        <div class="cts">{{ f.first_seen[11:19] if f.first_seen else '' }}</div>
      </div>{% endfor %}
    </div>
    {% else %}<div class="empty">No registered faces yet.</div>{% endif %}

  {% elif tab == 'all' %}
    {% if rows %}
    <table><thead><tr>
      <th>#</th><th>Crop</th><th>Face ID</th><th>Event</th><th>Timestamp</th>
    </tr></thead><tbody>
      {% for e in rows %}<tr>
        <td class="ts">{{ e.id }}</td>
        <td>{% if e.image_path %}<img class="crop" src="/img?p={{ e.image_path }}" alt="">
            {% else %}<div class="noimg">no img</div>{% endif %}</td>
        <td class="fid">{{ e.face_id }}</td>
        <td><span class="be be-{{ e.event_type }}">{{ e.event_type.upper() }}</span></td>
        <td class="ts">{{ e.timestamp }}</td>
      </tr>{% endfor %}
    </tbody></table>
    {% else %}<div class="empty">No events yet.</div>{% endif %}
  {% endif %}
</div>

<footer>
  This project is a part of a hackathon run by
  <a href="https://katomaran.com" target="_blank">https://katomaran.com</a>
  &nbsp;|&nbsp; DB: {{ db_path }}
</footer>

<script>
function onModeChange() {
  var mode = document.getElementById('modesel').value;
  var inp  = document.getElementById('srcinput');
  if (mode === 'rtsp') {
    inp.placeholder = 'rtsp://username:password@camera_ip:554/stream';
    if (!inp.value.startsWith('rtsp://')) inp.value = '';
  } else {
    inp.placeholder = 'video_sample1.mp4';
    if (inp.value.startsWith('rtsp://')) inp.value = '';
  }
}
function validateForm() {
  var mode = document.getElementById('modesel').value;
  var val  = document.getElementById('srcinput').value.trim();
  var err  = document.getElementById('valerr');
  if (!val) {
    err.textContent = 'Please enter a filename or URL.';
    err.style.display = 'block'; return false;
  }
  if (val === 'rtsp://username:password@camera_ip:554/stream') {
    err.textContent = 'Replace the placeholder with your actual RTSP URL.';
    err.style.display = 'block'; return false;
  }
  if (mode === 'rtsp' && !val.startsWith('rtsp://') && !val.startsWith('rtmp://')) {
    err.textContent = 'RTSP URL must start with rtsp://';
    err.style.display = 'block'; return false;
  }
  err.style.display = 'none'; return true;
}
</script>
</body></html>"""


# ------------------------------------------------------------------ #
# DB helpers                                                           #
# ------------------------------------------------------------------ #

def _load_db(tab):
    empty = {"unique_visitors":0,"total_entries":0,"total_exits":0,"registered_faces":0}
    if not os.path.exists(_DB_PATH):
        return empty, []
    try:
        conn = sqlite3.connect(_DB_PATH, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        v = conn.execute(
            "SELECT unique_visitors FROM visitor_stats WHERE id=1"
        ).fetchone()
        stats = {
            "unique_visitors":  v["unique_visitors"] if v else 0,
            "total_entries":    conn.execute("SELECT COUNT(*) FROM events WHERE event_type='entry'").fetchone()[0],
            "total_exits":      conn.execute("SELECT COUNT(*) FROM events WHERE event_type='exit'").fetchone()[0],
            "registered_faces": conn.execute("SELECT COUNT(*) FROM faces").fetchone()[0],
        }
        SQL = {
            "entries":    "SELECT id,face_id,event_type,timestamp,image_path FROM events WHERE event_type='entry' ORDER BY timestamp DESC LIMIT 300",
            "exits":      "SELECT id,face_id,event_type,timestamp,image_path FROM events WHERE event_type='exit' ORDER BY timestamp DESC LIMIT 300",
            "registered": "SELECT face_id,first_seen,crop_path FROM faces ORDER BY first_seen DESC LIMIT 500",
            "all":        "SELECT id,face_id,event_type,timestamp,image_path FROM events ORDER BY timestamp DESC LIMIT 500",
        }
        rows = [dict(r) for r in conn.execute(SQL.get(tab, SQL["all"])).fetchall()]
        conn.close()
        return stats, rows
    except Exception as ex:
        logging.getLogger("db").warning(f"DB read error: {ex}")
        return empty, []


# ------------------------------------------------------------------ #
# Flask routes                                                         #
# ------------------------------------------------------------------ #

@flask_app.route("/")
def index():
    tab = request.args.get("tab","entries")
    if tab not in ("entries","exits","registered","all"):
        tab = "entries"
    stats, rows = _load_db(tab)
    return render_template_string(
        TEMPLATE, tab=tab, s=stats, rows=rows,
        proc=STATE.snapshot(), db_path=_DB_PATH,
        now=datetime.now().strftime("%H:%M:%S"),
        request=request,
    )


@flask_app.route("/start", methods=["POST"])
def start_processing():
    if STATE.snapshot()["running"]:
        return redirect("/?tab=entries")

    if _CFG is None:
        STATE.update(status="error",
                     error="Config not loaded. Restart the application.")
        return redirect("/?tab=entries")

    mode   = request.form.get("mode","file")
    source = request.form.get("source","").strip()

    if not source:
        STATE.update(status="error", error="No source provided.")
        return redirect("/?tab=entries")

    if source == "rtsp://username:password@camera_ip:554/stream":
        STATE.update(status="error",
                     error="Please replace the placeholder with your actual RTSP URL.")
        return redirect("/?tab=entries")

    if mode == "rtsp" and not source.startswith(("rtsp://","rtmp://")):
        source = "rtsp://" + source

    threading.Thread(target=_run_pipeline, args=(source,), daemon=True).start()
    return redirect("/?tab=entries")


@flask_app.route("/stop", methods=["POST"])
def stop_processing():
    STATE.request_pause.clear()
    STATE.request_stop.set()
    return redirect("/?tab=entries")


@flask_app.route("/pause", methods=["POST"])
def pause_processing():
    if STATE.snapshot()["running"]:
        STATE.request_pause.set()
        STATE.update(status="paused")
    return redirect("/?tab=entries")


@flask_app.route("/resume", methods=["POST"])
def resume_processing():
    if STATE.snapshot()["running"]:
        STATE.request_pause.clear()
        STATE.update(status="running")
    return redirect("/?tab=entries")


@flask_app.route("/clear", methods=["POST"])
def clear_data():
    if STATE.snapshot()["running"]:
        STATE.request_pause.clear()
        STATE.request_stop.set()
        time.sleep(2)

    if os.path.exists(_DB_PATH):
        os.remove(_DB_PATH)
    for folder in ["logs/entries","logs/exits","logs/registered"]:
        if os.path.exists(folder):
            shutil.rmtree(folder)
        os.makedirs(folder, exist_ok=True)
    log_file = "logs/events.log"
    if os.path.exists(log_file):
        open(log_file,"w").close()

    STATE.update(status="idle", running=False,
                 frame_count=0, unique_count=0, error="")
    STATE.request_stop.clear()
    logging.getLogger("clear").info("All data cleared.")
    return redirect("/?tab=entries")


@flask_app.route("/img")
def serve_img():
    path = request.args.get("p","").strip()
    if not path:
        return _grey()
    abs_path = os.path.abspath(path)
    logs_dir = os.path.abspath("logs")
    if not abs_path.startswith(logs_dir) or not os.path.isfile(abs_path):
        return _grey()
    return send_file(abs_path, mimetype="image/jpeg")


@flask_app.route("/api/status")
def api_status():
    stats, _ = _load_db("entries")
    d = STATE.snapshot()
    d.update(stats)
    return jsonify(d)


def _grey():
    data = bytes([
        0xff,0xd8,0xff,0xe0,0x00,0x10,0x4a,0x46,0x49,0x46,0x00,0x01,0x01,0x00,
        0x00,0x01,0x00,0x01,0x00,0x00,0xff,0xdb,0x00,0x43,0x00,0x08,0x06,0x06,
        0x07,0x06,0x05,0x08,0x07,0x07,0x07,0x09,0x09,0x08,0x0a,0x0c,0x14,0x0d,
        0x0c,0x0b,0x0b,0x0c,0x19,0x12,0x13,0x0f,0x14,0x1d,0x1a,0x1f,0x1e,0x1d,
        0x1a,0x1c,0x1c,0x20,0x24,0x2e,0x27,0x20,0x22,0x2c,0x23,0x1c,0x1c,0x28,
        0x37,0x29,0x2c,0x30,0x31,0x34,0x34,0x34,0x1f,0x27,0x39,0x3d,0x38,0x32,
        0x3c,0x2e,0x33,0x34,0x32,0xff,0xc0,0x00,0x0b,0x08,0x00,0x01,0x00,0x01,
        0x01,0x01,0x11,0x00,0xff,0xc4,0x00,0x1f,0x00,0x00,0x01,0x05,0x01,0x01,
        0x01,0x01,0x01,0x01,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x01,0x02,
        0x03,0x04,0x05,0x06,0x07,0x08,0x09,0x0a,0x0b,0xff,0xda,0x00,0x08,0x01,
        0x01,0x00,0x00,0x3f,0x00,0xfb,0x00,0xff,0xd9
    ])
    return Response(data, mimetype="image/jpeg")


# ------------------------------------------------------------------ #
# Flask + tunnel launchers                                             #
# ------------------------------------------------------------------ #

def _start_flask(port: int):
    flask_app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)


def _cloudflare_tunnel(port: int) -> str:
    try:
        if subprocess.run(["which","cloudflared"], capture_output=True).returncode != 0:
            print("Installing cloudflared...")
            subprocess.run(
                "wget -q https://github.com/cloudflare/cloudflared/releases/latest/"
                "download/cloudflared-linux-amd64 -O /usr/local/bin/cloudflared && "
                "chmod +x /usr/local/bin/cloudflared",
                shell=True, check=True
            )
        proc = subprocess.Popen(
            ["/usr/local/bin/cloudflared","tunnel","--url",f"http://localhost:{port}"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        pat      = re.compile(r"https://[a-z0-9\-]+\.trycloudflare\.com")
        deadline = time.time() + 45
        while time.time() < deadline:
            line = proc.stdout.readline()
            if not line:
                time.sleep(0.2)
                continue
            m = pat.search(line)
            if m:
                return m.group(0)
    except Exception as ex:
        print(f"Tunnel error: {ex}")
    return ""


# ------------------------------------------------------------------ #
# CLI entry point                                                      #
# ------------------------------------------------------------------ #

def parse_args():
    p = argparse.ArgumentParser(description="Face Tracker")
    p.add_argument("--config",   default="config/config.json")
    p.add_argument("--source",   default=None,
                   help="File path or RTSP URL to auto-start")
    p.add_argument("--rtsp",     action="store_true")
    p.add_argument("--headless", action="store_true")
    p.add_argument("--port",     type=int, default=5000)
    p.add_argument("--colab",    action="store_true",
                   help="Create Cloudflare public tunnel")
    p.add_argument("--no-dashboard", action="store_true",
                   help="Skip starting Flask")
    return p.parse_args()


def main():
    args = parse_args()

    # Initialise globals (config, DB path, logging)
    _init_globals(
        config_path=args.config,
        port=args.port
    )
    logger = logging.getLogger("main")

    # Start Flask
    if not args.no_dashboard:
        ft = threading.Thread(target=_start_flask, args=(args.port,), daemon=True)
        ft.start()
        time.sleep(2)
        logger.info(f"Dashboard on port {args.port}")

    # Cloudflare tunnel
    if args.colab:
        url = _cloudflare_tunnel(args.port)
        if url:
            print(f"\n{'='*62}")
            print("  Face Tracker Dashboard — OPEN IN YOUR BROWSER:")
            print(f"  {url}")
            print(f"\n  {url}/?tab=entries")
            print(f"  {url}/?tab=registered")
            print(f"  {url}/?tab=all")
            print("\n  Use the control panel to Start / Pause / Stop.")
            print(f"{'='*62}\n")
        else:
            print("\nTunnel failed — use Colab Ports tab: add port 5000\n")

    # Auto-start if source given via CLI
    source = None
    if args.source:
        source = args.source
    elif args.rtsp and _CFG:
        source = _CFG.get("rtsp_url","")
    elif not args.colab and _CFG:
        source = _CFG.video_source

    if source:
        pt = threading.Thread(target=_run_pipeline, args=(source,), daemon=False)
        pt.start()
        try:
            pt.join()
        except KeyboardInterrupt:
            STATE.request_stop.set()
            pt.join(timeout=5)
        snap = STATE.snapshot()
        print(f"\n{'='*50}")
        print(f"  Unique visitors  : {snap['unique_count']}")
        print(f"  Frames processed : {snap['frame_count']}")
        print(f"  Dashboard        : http://localhost:{args.port}")
        print(f"{'='*50}\n")
    else:
        # Colab mode — keep alive, user picks source via web UI
        try:
            while True: time.sleep(30)
        except KeyboardInterrupt:
            logger.info("Stopped.")


if __name__ == "__main__":
    main()
