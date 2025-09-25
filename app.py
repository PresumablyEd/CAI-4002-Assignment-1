# app.py — 8-Puzzle UI (image upload at top, manual+autoplay in sidebar, no fallback)
from typing import List, Tuple, Optional
import random, json, time
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

Grid = Tuple[int, ...]  # 9 ints in row-major; 0 = blank

# Try to import your teammate's solver. If missing, disable Solve.
HAS_SOLVER = True
try:
    from solver_impl import solve_puzzle  # teammate must provide this file+function
except Exception:
    HAS_SOLVER = False
    solve_puzzle = None  # type: ignore

# ---------- puzzle basics ----------
GOAL: Grid = (1, 2, 3, 4, 5, 6, 7, 8, 0)
ADJ = {
    0:(1,3), 1:(0,2,4), 2:(1,5),
    3:(0,4,6), 4:(1,3,5,7), 5:(2,4,8),
    6:(3,7), 7:(4,6,8), 8:(5,7)
}

def neighbors(g: Grid):
    z = g.index(0)
    for j in ADJ[z]:
        arr = list(g); arr[z], arr[j] = arr[j], arr[z]
        yield tuple(arr)

def shuffle_via_legal_moves(steps: int = 40) -> Grid:
    """Start at GOAL, do legal moves → guaranteed solvable."""
    g: Grid = GOAL
    prev: Optional[Grid] = None
    for _ in range(steps):
        opts = list(neighbors(g))
        if prev in opts and len(opts) > 1:
            opts = [x for x in opts if x != prev]  # avoid undo
        prev, g = g, random.choice(opts)
    return g

# ---------- image helpers ----------
def _measure_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> Tuple[int, int]:
    """Robust across Pillow versions: try textbbox -> font.getbbox -> font.getsize -> fallback."""
    try:
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        return right - left, bottom - top
    except Exception:
        pass
    try:
        left, top, right, bottom = font.getbbox(text)
        return right - left, bottom - top
    except Exception:
        pass
    try:
        return font.getsize(text)
    except Exception:
        pass
    return max(8, 8 * len(text)), 12

def slice_image_to_tiles(img: Image.Image, side: int = 600) -> List[Image.Image]:
    img = img.convert("RGB")
    w, h = img.size
    s = min(w, h)
    img = img.crop(((w-s)//2, (h-s)//2, (w+s)//2, (h+s)//2)).resize((side, side))
    tiles: List[Image.Image] = []
    step = side // 3
    for r in range(3):
        for c in range(3):
            tiles.append(img.crop((c*step, r*step, (c+1)*step, (r+1)*step)))
    return tiles  # map tile numbers 1..8 -> tiles[val-1]; 0 = blank

def render_grid(state: Grid, tiles: Optional[List[Image.Image]], side: int = 600) -> Image.Image:
    canvas = Image.new("RGB", (side, side), (245, 245, 245))
    step = side // 3
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for i, val in enumerate(state):
        r, c = divmod(i, 3)
        x0, y0 = c*step, r*step
        draw.rectangle([x0, y0, x0+step, y0+step], fill=(245,245,245))
        if val != 0:
            if tiles:
                canvas.paste(tiles[val-1], (x0, y0))
            else:
                text = str(val)
                tw, th = _measure_text(draw, text, font)
                draw.text((x0 + (step - tw)//2, y0 + (step - th)//2),
                          text, fill=(30,30,30), font=font)
    for k in (step, 2*step):
        draw.line([(k, 0), (k, side)], width=3, fill=(30,30,30))
        draw.line([(0, k), (side, k)], width=3, fill=(30,30,30))
    return canvas

def _rerun():
    # compatible with older/newer Streamlit versions
    try:
        st.rerun()
    except Exception:
        try:
            st.experimental_rerun()  # type: ignore
        except Exception:
            pass

# ---------- Streamlit UI ----------
st.set_page_config(page_title="8-Puzzle", layout="centered")
st.title("8-Puzzle")
st.caption("Upload an image → Shuffle → Solve (A* by teammate) → Step-by-step.")

# Session state (persists across reruns)
ss = st.session_state
if "state" not in ss: ss.state = GOAL
if "solution" not in ss: ss.solution = []         # List[Grid]
if "step" not in ss: ss.step = 0
if "tiles" not in ss: ss.tiles = None             # List[Image.Image] | None
if "moves_made" not in ss: ss.moves_made = 0
if "start_time" not in ss: ss.start_time = time.time()
if "last_tick" not in ss: ss.last_tick = 0.0

# ===== TOP: Image upload (first thing the user sees) =====
st.subheader("1) Choose an image (optional)")
uploaded_main = st.file_uploader("Upload an image (JPG/PNG) for the puzzle tiles", type=["jpg","jpeg","png"])
if uploaded_main:
    try:
        ss.tiles = slice_image_to_tiles(Image.open(uploaded_main))
        st.success("Image sliced into 9 tiles.")
    except Exception as e:
        ss.tiles = None
        st.error(f"Couldn’t process image: {e}")

st.divider()

# ----- SIDEBAR: controls -----
st.sidebar.header("Controls")

# Shuffle intensity
shuffle_steps = st.sidebar.slider("Shuffle moves", 10, 100, 40, 5)

# Manual controls (move the blank)
st.sidebar.subheader("Manual moves")
def _try_move(dr: int, dc: int):
    g = list(ss.state)
    z = g.index(0)
    r, c = divmod(z, 3)
    nr, nc = r + dr, c + dc
    if 0 <= nr < 3 and 0 <= nc < 3:
        ni = nr*3 + nc
        g[z], g[ni] = g[ni], g[z]
        ss.state = tuple(g)
        # user moved manually -> clear any existing solution
        ss.solution, ss.step = [], 0
        ss.moves_made += 1

if st.sidebar.button("⬆️ Up"):    _try_move(-1, 0)
if st.sidebar.button("⬅️ Left"):  _try_move(0, -1)
if st.sidebar.button("➡️ Right"): _try_move(0,  1)
if st.sidebar.button("⬇️ Down"):  _try_move(1,  0)

# Autoplay controls
st.sidebar.subheader("Autoplay solution")
auto_play = st.sidebar.checkbox("Enable autoplay", value=False)
auto_speed_ms = st.sidebar.slider("Speed (ms/step)", 50, 1500, 300, 50)

st.sidebar.markdown("---")
st.sidebar.caption("Solve enables when `solver_impl.py` with `solve_puzzle(start)` is present.")

# ----- MAIN: top buttons -----
col1, col2, col3 = st.columns(3)

if col1.button("Shuffle"):
    ss.state = shuffle_via_legal_moves(shuffle_steps)
    ss.solution, ss.step = [], 0
    ss.moves_made = 0
    ss.start_time = time.time()

if col2.button("Solve", disabled=not HAS_SOLVER):
    if not HAS_SOLVER:
        st.warning("Solver not found. Ask your teammate to add solver_impl.py.")
    else:
        try:
            path = solve_puzzle(ss.state)  # expects List[Grid]
            if not isinstance(path, list) or not path:
                raise ValueError("Solver must return a non-empty list of states.")
            norm: List[Grid] = []
            for s in path:
                if hasattr(s, "tolist"):  # numpy array
                    s = s.tolist()
                t = tuple(int(x) for x in list(s))
                if len(t) != 9:
                    raise ValueError("Each state must have 9 integers (0..8).")
                norm.append(t)
            if norm[0] != ss.state:
                norm = [ss.state] + norm
            ss.solution = norm
            ss.step = 0
            ss.moves_made = 0  # solving resets manual counter
            ss.start_time = time.time()
        except Exception as e:
            ss.solution = []
            st.error(f"Solver error: {e}")

if col3.button("Reset"):
    ss.state, ss.solution, ss.step = GOAL, [], 0
    ss.moves_made = 0
    ss.start_time = time.time()

# ----- Playback + stats -----
sol = ss.solution
elapsed = time.time() - ss.get("start_time", time.time())
st.caption(f"Manual moves: {ss.moves_made}  •  Elapsed: {elapsed:0.1f}s")

if sol and len(sol) > 1:
    last_idx = len(sol) - 1
    if ss.step < 0: ss.step = 0
    if ss.step > last_idx: ss.step = last_idx

    # Basic playback
    b1, b2, b3 = st.columns(3)
    st.write(f"Solution length: **{last_idx}** moves")
    if b1.button("Prev", disabled=ss.step <= 0):
        ss.step = max(0, ss.step - 1)
    if b2.button("Next", disabled=ss.step >= last_idx):
        ss.step = min(last_idx, ss.step + 1)
    if b3.button("End"):
        ss.step = last_idx

    # Autoplay tick
    if auto_play and ss.step < last_idx:
        now_ms = time.time() * 1000.0
        if now_ms - ss.last_tick >= auto_speed_ms:
            ss.step = min(last_idx, ss.step + 1)
            ss.last_tick = now_ms
            _rerun()

# ----- Single image render -----
display_state = sol[ss.step] if (sol and len(sol) > 1) else ss.state
caption = f"Step {ss.step}/{len(sol)-1}" if (sol and len(sol) > 1) else "Current puzzle"
st.image(render_grid(display_state, ss.tiles), caption=caption, use_column_width=True)

# ----- Download solution as JSON -----
if sol and len(sol) > 1:
    data = {"path": [list(s) for s in sol]}
    st.download_button("Download solution (JSON)",
                       data=json.dumps(data, indent=2),
                       file_name="solution.json",
                       mime="application/json")

# Tip when solver missing
if not HAS_SOLVER:
    st.caption("Solve will enable when your teammate adds solver_impl.py with solve_puzzle(start).")
