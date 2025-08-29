from __future__ import annotations
import random, json, re
from typing import List, Tuple, Dict

import pandas as pd
import streamlit as st

st.set_page_config(page_title="목우회 월례회 대진표", layout="wide")

# ---------- 타입 ----------
Game = Tuple[Tuple[int, int], Tuple[int, int]]  # ((A1,A2),(B1,B2)) — 0-index

# ---------- 시드형(공정) 대진 ----------
def seeded_order(n: int) -> List[int]:
    # [1, n, 2, n-1, 3, n-2, ...]  (지난 대회 순위 기반 번호)
    out = []
    for i in range(1, n // 2 + 1):
        out.extend([i, n - i + 1])
    return out

def base_pairs_for_8x4() -> List[Game]:
    # 8명/1인 4게임에 맞는 안정 스케줄(인덱스 0..7 기준)
    return [
        ((0, 1), (2, 3)),
        ((4, 5), (6, 7)),
        ((0, 2), (4, 6)),
        ((1, 3), (5, 7)),
        ((0, 3), (5, 6)),
        ((1, 2), (4, 7)),
        ((0, 4), (3, 7)),
        ((1, 5), (2, 6)),
    ]

def generate_seeded_schedule(n: int, gpp: int) -> tuple[list[Game], list[str]]:
    """8명/4게임일 때 공정 대진(시드형) 생성 + 숫자코드(예: 18:27)"""
    assert n == 8 and gpp == 4
    seeds = seeded_order(n)  # 1-based 번호
    sched = []
    codes = []
    for (i1, i2), (j1, j2) in base_pairs_for_8x4():
        a = (seeds[i1] - 1, seeds[i2] - 1)
        b = (seeds[j1] - 1, seeds[j2] - 1)
        sched.append((a, b))
        A = tuple(sorted([seeds[i1], seeds[i2]]))
        B = tuple(sorted([seeds[j1], seeds[j2]]))
        codes.append(f"{A[0]}{A[1]}:{B[0]}{B[1]}")
    return sched, codes

def schedule_from_vs_codes(vs_codes: list[str], n_players: int) -> list[Game]:
    """VS 코드(예: '18:27' 또는 '1,10:2,9') → 스케줄(0-index)"""
    out: list[Game] = []
    for code in vs_codes:
        s = re.sub(r"\s", "", str(code))
        if ":" not in s:
            out.append(((0,0),(0,0))); continue
        L, R = s.split(":", 1)

        def parse_team(t: str):
            m = re.match(r"^(\d+)[,/_-]?(\d+)$", t)
            if not m:
                if len(t) == 2 and t.isdigit():   # '18' → (1,8)
                    return (int(t[0])-1, int(t[1])-1)
                return None
            return (int(m.group(1))-1, int(m.group(2))-1)

        A = parse_team(L); B = parse_team(R)
        if not A or not B: out.append(((0,0),(0,0))); continue
        a1,a2 = A; b1,b2 = B
        if any(x<0 or x>=n_players for x in [a1,a2,b1,b2]):
            out.append(((0,0),(0,0))); continue
        out.append(((a1,a2),(b1,b2)))
    return out

# ---------- 통계 ----------
def compute_tables(schedule: list[Game], scores: list[tuple[int|None,int|None]],
                   names: list[str], win_target: int):
    n = len(names)
    stats = {i: {"이름": names[i], "경기수": 0, "승수": 0, "득점": 0, "실점": 0} for i in range(n)}
    rounds_by_player: Dict[int, list[int]] = {i: [] for i in range(n)}  # 1-based 경기번호

    for idx, ((a1,a2),(b1,b2)) in enumerate(schedule, start=1):
        for p in [a1,a2,b1,b2]:
            rounds_by_player[p].append(idx)
        sA, sB = scores[idx-1]
        if sA is None or sB is None: continue
        for p in [a1,a2,b1,b2]:
            stats[p]["경기수"] += 1
        stats[a1]["득점"] += sA; stats[a2]["득점"] += sA
        stats[b1]["득점"] += sB; stats[b2]["득점"] += sB
        stats[a1]["실점"] += sB; stats[a2]["실점"] += sB
        stats[b1]["실점"] += sA; stats[b2]["실점"] += sA
        if sA == win_target and sB < win_target:
            stats[a1]["승수"] += 1; stats[a2]["승수"] += 1
        elif sB == win_target and sA < win_target:
            stats[b1]["승수"] += 1; stats[b2]["승수"] += 1

    rank_df = pd.DataFrame(stats).T
    rank_df["득실차"] = rank_df["득점"] - rank_df["실점"]
    # 동률 정렬: 득실차 → 승수 → 득점 → 실점(적을수록 유리)
    rank_df = rank_df.sort_values(by=["득실차","승수","득점","실점"], ascending=[False,False,False,True])
    rank_df.insert(0, "순위", range(1, len(rank_df)+1))
    return rank_df, rounds_by_player

# ========================= 사이드바 =========================
with st.sidebar:
    st.header("대회 설정")
    n_players = st.number_input("참가 인원(짝수)", min_value=8, max_value=16, value=8, step=2)
    default_gpp = n_players // 2  # 8명 → 1인 4게임
    games_per_player = st.slider("1인당 경기 수", min_value=max(2, n_players//4),
                                 max_value=n_players-1, value=default_gpp)
    win_target = st.number_input("게임 종료 점수(예: 6)", min_value=4, max_value=8, value=6)
    gen = st.button("대진표 생성", type="primary")

if gen:
    if n_players == 8 and games_per_player == 4:
        schedule, vs_codes = generate_seeded_schedule(n_players, games_per_player)
    else:
        # 8/4가 아니면 균등-랜덤(단순)으로 생성
        rnd = random.Random(42)
        need = [games_per_player]*n_players
        schedule = []
        while sum(need) > 0:
            cand = [i for i in range(n_players) if need[i] > 0]
            rnd.shuffle(cand)
            if len(cand) < 4: break
            a,b,c,d = cand[:4]
            schedule.append(((a,b),(c,d)))
            for p in [a,b,c,d]: need[p]-=1
        vs_codes = [f"{min(a1+1,a2+1)}{max(a1+1,a2+1)}:{min(b1+1,b2+1)}{max(b1+1,b2+1)}"
                    for ((a1,a2),(b1,b2)) in schedule]
    st.session_state["schedule"] = schedule
    st.session_state["scores"] = [(None,None) for _ in range(len(schedule))]
    st.session_state["names"] = [f"플레이어 {i+1}" for i in range(n_players)]
    st.session_state["vs_codes"] = vs_codes

st.title("목우회 월례회 대진표")
if "names" not in st.session_state:
    st.info("좌측에서 인원을 설정하고 **대진표 생성**을 눌러 시작해줘.")
    st.stop()

# ========================= 1) 선수 명단 입력 =========================
st.subheader("선수 명단 입력")
names = st.session_state["names"]
cols = st.columns(4)
for i in range(len(names)):
    with cols[i % 4]:
        names[i] = st.text_input(f"번호 {i+1}", value=names[i], key=f"name_{i}")
st.session_state["names"] = names

st.divider()

schedule: list[Game] = st.session_state["schedule"]
scores: list[tuple[int|None,int|None]] = st.session_state["scores"]
vs_codes: list[str] = st.session_state.get("vs_codes", [])

# ========================= 2) 대진표(숫자 + 팀 이름) =========================
st.subheader("대진표 (숫자 + 팀 이름)")
edit_mode = st.checkbox("대진표 숫자 수정 모드", value=False,
                        help="예: 18:27 (8명) / 1,10:2,9 (10명 이상)")

if edit_mode:
    new_codes = []
    st.caption("숫자를 바꾸면 스케줄이 즉시 반영돼. (유효하지 않은 코드는 무시)")
    for i in range(len(schedule)):
        new_codes.append(st.text_input(f"게임{i+1} VS", value=vs_codes[i] if i < len(vs_codes) else "",
                                       key=f"code_{i}"))
    new_sched = schedule_from_vs_codes(new_codes, len(names))
    if any(g != ((0,0),(0,0)) for g in new_sched):
        schedule = new_sched
        vs_codes = new_codes
        st.session_state["schedule"] = schedule
        st.session_state["vs_codes"] = vs_codes

# 표시용 표
rows = []
for i, ((a1,a2),(b1,b2)) in enumerate(schedule, start=1):
    code = vs_codes[i-1] if i-1 < len(vs_codes) else f"{min(a1+1,a2+1)}{max(a1+1,a2+1)}:{min(b1+1,b2+1)}{max(b1+1,b2+1)}"
    rows.append({
        "게임": f"게임{i}",
        "VS": code,
        "A팀": f"{names[a1]} & {names[a2]}",
        "B팀": f"{names[b1]} & {names[b2]}",
    })
st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

st.divider()

# ========================= 3) 대진표 & 점수입력 =========================
st.subheader("대진표 & 점수입력")
hdr = st.columns([1.1, 3, 3, 1, 1])
for c, t in zip(hdr, ["구분","A팀 player","B팀 player","A팀 점수","B팀 점수"]):
    c.markdown(f"**{t}**")

for idx, ((a1,a2),(b1,b2)) in enumerate(schedule, start=1):
    c = st.columns([1.1, 3, 3, 1, 1])
    c[0].write(f"게임{idx}")
    c[1].write(f"{names[a1]}, {names[a2]}")
    c[2].write(f"{names[b1]}, {names[b2]}")
    a_init, b_init = scores[idx-1]
    a_sc = c[3].number_input(f"A{idx}", min_value=0, max_value=win_target,
                             value=int(a_init) if a_init is not None else 0, key=f"g{idx}_A")
    b_sc = c[4].number_input(f"B{idx}", min_value=0, max_value=win_target,
                             value=int(b_init) if b_init is not None else 0, key=f"g{idx}_B")
    if (a_sc == win_target and b_sc < win_target) or (b_sc == win_target and a_sc < win_target):
        scores[idx-1] = (a_sc, b_sc)
    else:
        scores[idx-1] = (None, None)

st.session_state["scores"] = scores

st.divider()

# ========================= 4) 개인 기록(1R~4R) + 최종 결과(포디움) =========================
rank_df, rounds_by_player = compute_tables(schedule, scores, names, win_target)

def round_cell_text(i: int, r: int) -> str:
    lst = rounds_by_player[i]
    if r > len(lst): return ":"
    g = lst[r-1]
    (x1,x2),(y1,y2) = schedule[g-1]
    sA,sB = scores[g-1]
    if sA is None or sB is None: return ":"
    return f"{sA}:{sB}" if i in (x1,x2) else f"{sB}:{sA}"

ordered = rank_df.sort_values("순위").copy()
# 포디움 카드
st.markdown("### 최종 결과")
top3 = ordered.head(3)
col1,col2,col3 = st.columns(3)
cards = [(col1,"🥇", "#fff3b0"), (col2,"🥈","#e5e7eb"), (col3,"🥉","#f5e1c8")]
for (col, medal, bg), (_, row) in zip(cards, top3.iterrows()):
    col.markdown(
        f"""
        <div style="padding:14px;border-radius:14px;background:{bg};">
          <div style="font-size:22px">{medal} {int(row['순위'])}위 — <b>{row['이름']}</b></div>
          <div style="margin-top:6px;">승수 {int(row['승수'])} · 득점 {int(row['득점'])} · 실점 {int(row['실점'])} · 득실차 <b>{int(row['득실차'])}</b></div>
        </div>
        """,
        unsafe_allow_html=True
    )

# 개인 기록표(구분 제거, 1위부터)
rows2 = []
for _, r in ordered.iterrows():
    i = names.index(r["이름"])
    rows2.append({
        "이름": r["이름"],
        "1R": round_cell_text(i,1),
        "2R": round_cell_text(i,2),
        "3R": round_cell_text(i,3),
        "4R": round_cell_text(i,4),
        "승수": int(r["승수"]),
        "득점": int(r["득점"]),
        "실점": int(r["실점"]),
        "득실차": int(r["득실차"]),
        "순위": int(r["순위"]),
    })
table_df = pd.DataFrame(rows2, columns=["이름","1R","2R","3R","4R","승수","득점","실점","득실차","순위"])

medal = {1:"🥇", 2:"🥈", 3:"🥉"}
table_df["순위"] = table_df["순위"].apply(lambda x: f"{medal.get(x,'')} {x}".strip())

def highlight_top3(row):
    try:
        raw = int(row["순위"].split()[-1])
    except Exception:
        raw = 99
    if raw == 1: return ["background-color:#fff3b0;font-weight:700" for _ in row]
    if raw == 2: return ["background-color:#e5e7eb;font-weight:600" for _ in row]
    if raw == 3: return ["background-color:#f5e1c8;font-weight:600" for _ in row]
    return [""]*len(row)

st.markdown("### 개인 경기 기록 (1R~4R) 및 순위")
st.dataframe(table_df.style.apply(highlight_top3, axis=1), use_container_width=True, hide_index=True)

# ---------- 내보내기/복원 ----------
with st.expander("CSV 내보내기 / 상태 백업·복원"):
    # 대진표(숫자+이름)
    export_vs = pd.DataFrame([{
        "게임": f"게임{i+1}",
        "VS": (st.session_state.get('vs_codes', []) or
               [f"{min(a1+1,a2+1)}{max(a1+1,a2+1)}:{min(b1+1,b2+1)}{max(b1+1,b2+1)}"
                for ((a1,a2),(b1,b2)) in schedule])[i],
        "A팀": f"{names[a1]} & {names[a2]}",
        "B팀": f"{names[b1]} & {names[b2]}",
    } for i, ((a1,a2),(b1,b2)) in enumerate(schedule)])
    st.download_button("대진표(숫자+이름) CSV", export_vs.to_csv(index=False).encode("utf-8-sig"),
                       file_name="vs_numeric_named.csv")

    export_input = pd.DataFrame([{
        "구분": f"게임{i+1}",
        "A팀 player": f"{names[a1]}, {names[a2]}",
        "B팀 player": f"{names[b1]}, {names[b2]}",
        "A팀 점수": scores[i][0] if scores[i][0] is not None else "",
        "B팀 점수": scores[i][1] if scores[i][1] is not None else "",
    } for i, ((a1,a2),(b1,b2)) in enumerate(schedule)])
    st.download_button("대진표&점수입력 CSV", export_input.to_csv(index=False).encode("utf-8-sig"),
                       file_name="vs_with_scores.csv")

    st.download_button("개인 기록/순위 CSV", table_df.to_csv(index=False).encode("utf-8-sig"),
                       file_name="personal_ranking.csv")

    state_blob = json.dumps({
        "names": names, "schedule": schedule, "scores": scores,
        "vs_codes": st.session_state.get("vs_codes", []),
        "meta": {"win_target": win_target}
    }, ensure_ascii=False)
    st.download_button("상태 백업(JSON)", state_blob.encode("utf-8"), file_name="mokwoo_state.json")

    up = st.file_uploader("상태 복원(JSON)", type=["json"])
    if up is not None:
        data = json.loads(up.read().decode("utf-8"))
        st.session_state["names"] = data.get("names", names)
        st.session_state["schedule"] = [tuple(map(tuple, g)) for g in data.get("schedule", schedule)]
        st.session_state["scores"] = [tuple(s) if s is not None else (None, None) for s in data.get("scores", scores)]
        st.session_state["vs_codes"] = data.get("vs_codes", st.session_state.get("vs_codes", []))
        st.rerun()
