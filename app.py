# app.py — 8-Puzzle UI (image upload at top, manual/auto play in sidebar)
from typing import List, Tuple, Optional
import random, json, time
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

Grid = Tuple[int, ...]  # 9 cells, row-major; 0 = blank

# Load solvers from solver_impl.py if available
HAS_SOLVER = True
try:
    from solver_impl import solve_puzzle, solve_puzzle_bfs, solve_puzzle_dfs
except Exception:
    HAS_SOLVER = False
    solve_puzzle = None  # type: ignore
    solve_puzzle_bfs = None  # type: ignore
    solve_puzzle_dfs = None  # type: ignore

#  puzzle basics 
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
    """Shuffle by valid moves from GOAL (always solvable)."""
    g: Grid = GOAL
    prev: Optional[Grid] = None
    for _ in range(steps):
        opts = list(neighbors(g))
        if prev in opts and len(opts) > 1:
            opts = [x for x in opts if x != prev]  # avoid immediate undo
        prev, g = g, random.choice(opts)
    return g

#  image helpers 
def _measure_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> Tuple[int, int]:
    """Text size with fallbacks (Pillow versions differ)."""
    try:
        l, t, r, b = draw.textbbox((0, 0), text, font=font)
        return r - l, b - t
    except Exception:
        pass
    try:
        l, t, r, b = font.getbbox(text)
        return r - l, b - t
    except Exception:
        pass
    try:
        return font.getsize(text)
    except Exception:
        pass
    return max(8, 8 * len(text)), 12

def slice_image_to_tiles(img: Image.Image, side: int = 600) -> List[Image.Image]:
    """Square-crop, resize, split into 3×3 tiles."""
    img = img.convert("RGB")
    w, h = img.size
    s = min(w, h)
    img = img.crop(((w-s)//2, (h-s)//2, (w+s)//2, (h+s)//2)).resize((side, side))
    tiles: List[Image.Image] = []
    step = side // 3
    for r in range(3):
        for c in range(3):
            tiles.append(img.crop((c*step, r*step, (c+1)*step, (r+1)*step)))
    return tiles  # 1..8 map to tiles[val-1]; 0 = blank

def render_grid(state: Grid, tiles: Optional[List[Image.Image]], side: int = 600) -> Image.Image:
    """Draw the board for a given state."""
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
    """Force a rerun (handles old/new Streamlit APIs)."""
    try:
        st.rerun()
    except Exception:
        try:
            st.experimental_rerun()  # type: ignore
        except Exception:
            pass

#  Streamlit UI 
st.set_page_config(page_title="8-Puzzle", layout="centered")
st.title("8-Puzzle")
st.caption("Upload an image → Shuffle → Solve (A* / BFS / DFS) → Step-by-step.")

# Session state
ss = st.session_state
if "state" not in ss: ss.state = GOAL
if "solution" not in ss: ss.solution = []         # List[Grid]
if "step" not in ss: ss.step = 0
if "tiles" not in ss: ss.tiles = None             # List[Image.Image] | None
if "moves_made" not in ss: ss.moves_made = 0
if "start_time" not in ss: ss.start_time = time.time()
if "last_tick" not in ss: ss.last_tick = 0.0

#  TOP: Image upload 
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

#  SIDEBAR: controls 
st.sidebar.header("Controls")

solver_choice = st.sidebar.selectbox("Solver Algorithm", ["A*", "BFS", "DFS"], index=0)
shuffle_steps = st.sidebar.slider("Shuffle moves", 10, 100, 40, 5)

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
        ss.solution, ss.step = [], 0
        ss.moves_made += 1
        ss.last_tick = 0.0

if st.sidebar.button("⬆️ Up"):    _try_move(-1, 0)
if st.sidebar.button("⬅️ Left"):  _try_move(0, -1)
if st.sidebar.button("➡️ Right"): _try_move(0,  1)
if st.sidebar.button("⬇️ Down"):  _try_move(1,  0)

st.sidebar.subheader("Autoplay solution")
auto_play = st.sidebar.checkbox("Enable autoplay", value=False, key="autoplay")
auto_speed_ms = st.sidebar.slider("Speed (ms/step)", 100, 1500, 300, 50)

st.sidebar.markdown("---")
st.sidebar.caption("Solve enables when `solver_impl.py` provides A* / BFS / DFS functions.")

#  MAIN: top buttons 
col1, col2, col3 = st.columns(3)

if col1.button("Shuffle"):
    ss.state = shuffle_via_legal_moves(shuffle_steps)
    ss.solution, ss.step = [], 0
    ss.moves_made = 0
    ss.start_time = time.time()
    ss.last_tick = 0.0

if col2.button("Solve", disabled=not HAS_SOLVER):
    if not HAS_SOLVER:
        st.warning("Solver not found. Ask your teammate to add solver_impl.py.")
    else:
        try:
            if solver_choice == "A*":
                path = solve_puzzle(ss.state)
            elif solver_choice == "BFS":
                path = solve_puzzle_bfs(ss.state)
            else:
                path = solve_puzzle_dfs(ss.state)

            if not isinstance(path, list) or not path:
                raise ValueError("Solver must return a non-empty list of states.")
            norm: List[Grid] = []
            for s in path:
                if hasattr(s, "tolist"):
                    s = s.tolist()
                t = tuple(int(x) for x in list(s))
                if len(t) != 9:
                    raise ValueError("Each state must have 9 integers (0..8).")
                norm.append(t)
            if norm[0] != ss.state:
                norm = [ss.state] + norm
            ss.solution = norm
            ss.step = 0
            ss.moves_made = 0
            ss.start_time = time.time()
            ss.last_tick = 0.0
        except Exception as e:
            ss.solution = []
            st.error(f"Solver error: {e}")

if col3.button("Reset"):
    ss.state, ss.solution, ss.step = GOAL, [], 0
    ss.moves_made = 0
    ss.start_time = time.time()
    ss.last_tick = 0.0

#  Playback + stats 
sol = ss.solution
elapsed = time.time() - ss.get("start_time", time.time())
st.caption(f"Manual moves: {ss.moves_made}  •  Elapsed: {elapsed:0.1f}s")

# Clamp indices and compute last_idx
if sol and len(sol) > 1:
    last_idx = len(sol) - 1
    if ss.step < 0: ss.step = 0
    if ss.step > last_idx: ss.step = last_idx

    # Controls above the image
    st.write(f"Solution length: **{last_idx}** moves")
    c1, c2, c3 = st.columns(3)
    if c1.button("⬅️ Prev", disabled=ss.step <= 0):
        ss.step = max(0, ss.step - 1)
        ss.last_tick = 0.0
        _rerun()
    if c2.button("➡️ Next", disabled=ss.step >= last_idx):
        ss.step = min(last_idx, ss.step + 1)
        ss.last_tick = 0.0
        _rerun()
    if c3.button("⏩ End", disabled=ss.step >= last_idx):
        ss.step = last_idx
        ss.last_tick = 0.0
        _rerun()

    display_state = sol[ss.step]
    caption = f"Step {ss.step}/{last_idx}"
else:
    last_idx = 0
    display_state = ss.state
    caption = "Current puzzle"

#  Show current frame (image) 
st.image(render_grid(display_state, ss.tiles), caption=caption, use_column_width=True)

#  Autoplay tick (render, then schedule next step) 
if sol and len(sol) > 1 and auto_play and ss.step < last_idx:
    now_ms = time.time() * 1000.0
    if ss.last_tick == 0.0:
        ss.last_tick = now_ms
    due_ms = ss.last_tick + auto_speed_ms
    remaining_ms = max(0.0, due_ms - now_ms)

    if remaining_ms > 0:
        time.sleep(remaining_ms / 1000.0)

    ss.step = min(last_idx, ss.step + 1)
    ss.last_tick = time.time() * 1000.0
    _rerun()
    st.stop()

#  Download solution as JSON 
if sol and len(sol) > 1:
    data = {"path": [list(s) for s in sol]}
    st.download_button("Download solution (JSON)",
                       data=json.dumps(data, indent=2),
                       file_name="solution.json",
                       mime="application/json")

# Solver tip
if not HAS_SOLVER:
    st.caption("Solve will enable when your teammate adds solver_impl.py with solve_puzzle(start).")
