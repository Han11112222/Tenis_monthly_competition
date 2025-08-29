from __future__ import annotations
import itertools, random, json, re
from collections import defaultdict
from typing import List, Tuple, Dict

import pandas as pd
import streamlit as st

st.set_page_config(page_title="목우회 월례회 대진표", layout="wide")

# ---------- 타입 ----------
Game = Tuple[Tuple[int, int], Tuple[int, int]]  # ((A1,A2),(B1,B2)) — 0-index player id


# ---------- 유틸 ----------
def seeded_order(n: int) -> List[int]:
    """시드형 순서: [1,n,2,n-1,3,n-2,...] (1-based 숫자 의미는 '지난 경기 순위')"""
    out = []
    for i in range(1, n // 2 + 1):
        out.extend([i, n - i + 1])
    return out


def base_pairs_for_8x4() -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """인덱스(0..7) 기준 기본 8게임(각자 4회) — 안정 스케줄."""
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


def generate_seeded_schedule(n: int, gpp: int) -> Tuple[List[Game], List[str]]:
    """
    '공정 대진' 기본안:
      1) 시드형 순서(seeded_order)를 만들고
      2) 8x4 기본 페어링을 그 순서에 매핑 → 예: 18:27, 36:45, 12:34 ...
    return: (schedule[0-index], vs_codes like '18:27')
    """
    assert n == 8 and gpp == 4, "현재 공정 대진 기본안은 8명/4게임에 최적화"
    seeds = seeded_order(n)             # [1,8,2,7,3,6,4,5]
    idx_pairs = base_pairs_for_8x4()
    schedule: List[Game] = []
    vs_codes: List[str] = []
    for (i1, i2), (j1, j2) in idx_pairs:
        a = (seeds[i1] - 1, seeds[i2] - 1)   # 실제 0-index 플레이어
        b = (seeds[j1] - 1, seeds[j2] - 1)
        schedule.append((a, b))
        # 숫자코드는 1-based 그대로, 각 팀은 오름차순 표기
        A = tuple(sorted([seeds[i1], seeds[i2]]))
        B = tuple(sorted([seeds[j1], seeds[j2]]))
        vs_codes.append(f"{A[0]}{A[1]}:{B[0]}{B[1]}")
    return schedule, vs_codes


def schedule_from_vs_codes(vs_codes: List[str], n_players: int) -> List[Game]:
    """
    VS 코드(예: '18:27')를 스케줄(0-index)로 변환.
    * 현재는 1~9만 안전하게 파싱(8명 사용 가정). 10 이상을 쓰려면 '1,10:2,9' 꼴 권장.
    """
    schedule: List[Game] = []
    for code in vs_codes:
        s = re.sub(r"\s", "", str(code))
        if ":" not in s:
            schedule.append(((0, 0), (0, 0)))
            continue
        left, right = s.split(":", 1)

        def parse_team(team: str) -> Tuple[int, int] | None:
            # 우선 콤마/슬래시 지원: "1,8" or "1/8"
            m = re.match(r"^(\d+)[,/_-]?(\d+)$", team)
            if not m:
                # 콤마가 없고 한 자리씩이라면(8명 가정) "18" -> (1,8)
                if len(team) == 2 and team.isdigit():
                    a, b = int(team[0]), int(team[1])
                    return (a - 1, b - 1)
                return None
            a, b = int(m.group(1)), int(m.group(2))
            return (a - 1, b - 1)

        A = parse_team(left)
        B = parse_team(right)
        if not A or not B:
            schedule.append(((0, 0), (0, 0)))
            continue
        a1, a2 = A
        b1, b2 = B
        # 범위 체크
        ok = all(0 <= x < n_players for x in [a1, a2, b1, b2])
        schedule.append(((a1, a2), (b1, b2)) if ok else ((0, 0), (0, 0)))
    return schedule


def compute_tables(schedule: List[Game], scores, names: List[str], win_target: int):
    """개인 누적과 라운드별 표시용 인덱스를 만들어 반환."""
    n = len(names)
    stats = {i: {"이름": names[i], "경기수": 0, "승수": 0, "득점": 0, "실점": 0} for i in range(n)}
    rounds_by_player: Dict[int, list] = {i: [] for i in range(n)}  # 1-based 경기 번호

    for idx, ((a1, a2), (b1, b2)) in enumerate(schedule, start=1):
        for p in [a1, a2, b1, b2]:
            rounds_by_player[p].append(idx)

        sA, sB = scores[idx - 1]
        if sA is None or sB is None:
            continue

        for p in [a1, a2, b1, b2]:
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
    rank_df = rank_df.sort_values(by=["득실차", "승수", "득점", "실점"], ascending=[False, False, False, True])
    rank_df.insert(0, "순위", range(1, len(rank_df) + 1))
    return rank_df, rounds_by_player


# ========================= 사이드바 =========================
with st.sidebar:
    st.header("대회 설정")
    n_players = st.number_input("참가 인원(짝수)", min_value=8, max_value=16, value=8, step=2)
    default_gpp = n_players // 2           # 8명 → 1인 4게임
    games_per_player = st.slider("1인당 경기 수", min_value=max(2, n_players // 4),
                                 max_value=n_players - 1, value=default_gpp)
    win_target = st.number_input("게임 종료 점수(예: 6)", min_value=4, max_value=8, value=6)
    seed = st.number_input("스케줄 시드", min_value=0, max_value=99999, value=22)
    use_seeded = st.checkbox("공정 대진(시드형: 1↔n, 2↔n-1 ...)", value=True)
    gen = st.button("대진표 생성", type="primary")

# 초기 생성
if gen:
    if use_seeded and n_players == 8 and games_per_player == 4:
        schedule, vs_codes = generate_seeded_schedule(n_players, games_per_player)
    else:
        # 시드형 밖의 경우엔, 기본 랜덤-균형 로직 대신 간단한 라운드로빈 생성(균등 출전)로 폴백
        rnd = random.Random(int(seed))
        need = [games_per_player] * n_players
        schedule = []
        while sum(need) > 0:
            cand = [i for i in range(n_players) if need[i] > 0]
            rnd.shuffle(cand)
            if len(cand) < 4: break
            a, b, c, d = cand[:4]
            schedule.append(((a, b), (c, d)))
            for p in [a, b, c, d]: need[p] -= 1
        # 숫자코드는 1-based 번호로 표기
        vs_codes = [f"{min(a1+1,a2+1)}{max(a1+1,a2+1)}:{min(b1+1,b2+1)}{max(b1+1,b2+1)}"
                    for ((a1,a2),(b1,b2)) in schedule]
    st.session_state["schedule"] = schedule
    st.session_state["scores"] = [(None, None) for _ in range(len(schedule))]
    st.session_state["names"] = [f"플레이어 {i+1}" for i in range(n_players)]
    st.session_state["vs_codes"] = vs_codes

st.title("목우회 월례회 대진표")
if "names" not in st.session_state:
    st.info("좌측에서 인원을 설정하고 **대진표 생성**을 눌러 시작해줘.")
    st.stop()

# ========================= ① 선수 명단 입력 =========================
st.subheader("선수 명단 입력")
names = st.session_state["names"]
cols = st.columns(4)
for i in range(len(names)):
    with cols[i % 4]:
        names[i] = st.text_input(f"번호 {i+1}", value=names[i], key=f"name_{i}")
st.session_state["names"] = names

st.divider()

schedule: List[Game] = st.session_state["schedule"]
scores = st.session_state["scores"]
vs_codes: List[str] = st.session_state.get("vs_codes", [])

# ==================== ② 대진표(숫자+이름) — 수정 모드 지원 ====================
st.subheader("대진표 (숫자 + 팀 이름)")
edit_mode = st.checkbox("대진표 숫자 수정 모드", value=False, help="예: 18:27, 36:45 … (8명일 때 권장)")

# 수정 UI
if edit_mode:
    new_codes = []
    st.caption("※ 8명일 때는 '18:27'처럼 두 자리씩, 10명 이상이면 '1,10:2,9'처럼 콤마로 입력해줘.")
    for i in range(len(schedule)):
        code_val = st.text_input(f"게임{i+1} VS", value=vs_codes[i] if i < len(vs_codes) else "", key=f"vscode_{i}")
        new_codes.append(code_val)
    # 코드 → 스케줄 반영
    new_schedule = schedule_from_vs_codes(new_codes, len(names))
    # 무효(0,0)-(0,0) 줄이 있으면 유지, 아니면 교체
    if any(g != ((0, 0), (0, 0)) for g in new_schedule):
        schedule = new_schedule
        st.session_state["schedule"] = schedule
        st.session_state["vs_codes"] = new_codes
        vs_codes = new_codes

# 표시용 표(읽기 전용)
rows = []
for i, ((a1, a2), (b1, b2)) in enumerate(schedule, start=1):
    # VS 코드는 입력된 값을, 없으면 현재 스케줄로 생성해 보여줌
    code = vs_codes[i-1] if i-1 < len(vs_codes) else f"{min(a1+1,a2+1)}{max(a1+1,a2+1)}:{min(b1+1,b2+1)}{max(b1+1,b2+1)}"
    rows.append({
        "게임": f"게임{i}",
        "VS": code,
        "A팀": f"{names[a1]} & {names[a2]}",
        "B팀": f"{names[b1]} & {names[b2]}",
    })
st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

st.divider()

# ==================== ③ 대진표 & 점수입력 ====================
st.subheader("대진표 & 점수입력")
# 헤더
h = st.columns([1.1, 3, 3, 1, 1])
for t, lab in enumerate(["구분", "A팀 player", "B팀 player", "A팀 점수", "B팀 점수"]):
    h[t].markdown(f"**{lab}**")

for idx, ((a1, a2), (b1, b2)) in enumerate(schedule, start=1):
    c = st.columns([1.1, 3, 3, 1, 1])
    c[0].write(f"게임{idx}")
    c[1].write(f"{names[a1]}, {names[a2]}")
    c[2].write(f"{names[b1]}, {names[b2]}")
    a_init, b_init = scores[idx - 1]
    a_sc = c[3].number_input(f"A{idx}", min_value=0, max_value=int(win_target),
                             value=int(a_init) if a_init is not None else 0, key=f"g{idx}_A")
    b_sc = c[4].number_input(f"B{idx}", min_value=0, max_value=int(win_target),
                             value=int(b_init) if b_init is not None else 0, key=f"g{idx}_B")
    if (a_sc == win_target and b_sc < win_target) or (b_sc == win_target and a_sc < win_target):
        scores[idx - 1] = (a_sc, b_sc)
    else:
        scores[idx - 1] = (None, None)

st.session_state["scores"] = scores

st.divider()

# ==================== ④ 개인 기록(1R~4R) 및 순위 (1위부터, TOP3 강조) ====================
rank_df, rounds_by_player = compute_tables(schedule, scores, names, win_target)

def round_cell_text(player_idx: int, r: int) -> str:
    lst = rounds_by_player[player_idx]
    if r > len(lst): return ":"
    g_idx = lst[r-1]  # 1-based
    (x1, x2), (y1, y2) = schedule[g_idx-1]
    sA, sB = scores[g_idx-1]
    if sA is None or sB is None:
        return ":"
    return f"{sA}:{sB}" if player_idx in (x1, x2) else f"{sB}:{sA}"

ordered = rank_df.sort_values("순위").copy()
rows = []
for _, r in ordered.iterrows():
    i = names.index(r["이름"])
    rows.append({
        "이름": r["이름"],
        "1R": round_cell_text(i, 1),
        "2R": round_cell_text(i, 2),
        "3R": round_cell_text(i, 3),
        "4R": round_cell_text(i, 4),
        "승수": int(r["승수"]),
        "득점": int(r["득점"]),
        "실점": int(r["실점"]),
        "득실차": int(r["득실차"]),
        "순위": int(r["순위"]),
    })
table_df = pd.DataFrame(rows, columns=["이름","1R","2R","3R","4R","승수","득점","실점","득실차","순위"])

# TOP3 꾸미기
medal = {1:"🥇", 2:"🥈", 3:"🥉"}
table_df["순위"] = table_df["순위"].apply(lambda x: f"{medal.get(x,'')} {x}".strip())

def highlight_top3(row):
    try:
        raw_rank = int(row["순위"].split()[-1])
    except Exception:
        raw_rank = 99
    if raw_rank == 1:
        return ["background-color: #fff3b0; font-weight: 700" for _ in row]
    if raw_rank == 2:
        return ["background-color: #e5e7eb; font-weight: 600" for _ in row]
    if raw_rank == 3:
        return ["background-color: #f5e1c8; font-weight: 600" for _ in row]
    return [""] * len(row)

st.subheader("개인 경기 기록 (1R~4R) 및 순위")
st.dataframe(table_df.style.apply(highlight_top3, axis=1), use_container_width=True, hide_index=True)

# ---------- 내보내기/복원 ----------
with st.expander("CSV 내보내기 / 상태 백업·복원"):
    # 대진표(숫자+이름)
    st.download_button(
        "대진표(숫자+이름) CSV",
        pd.DataFrame(rows := [{
            "게임": f"게임{i+1}",
            "VS": (st.session_state.get('vs_codes', []) or
                   [f"{min(a1+1,a2+1)}{max(a1+1,a2+1)}:{min(b1+1,b2+1)}{max(b1+1,b2+1)}" for ((a1,a2),(b1,b2)) in schedule])[i],
            "A팀": f"{names[a1]} & {names[a2]}",
            "B팀": f"{names[b1]} & {names[b2]}",
        } for i, ((a1,a2),(b1,b2)) in enumerate(schedule)]).to_csv(index=False).encode("utf-8-sig"),
        file_name="vs_numeric_named.csv"
    )
    # 점수 포함 표
    st.download_button(
        "대진표&점수입력 CSV",
        pd.DataFrame([{
            "구분": f"게임{i+1}",
            "A팀 player": f"{names[a1]}, {names[a2]}",
            "B팀 player": f"{names[b1]}, {names[b2]}",
            "A팀 점수": scores[i][0] if scores[i][0] is not None else "",
            "B팀 점수": scores[i][1] if scores[i][1] is not None else "",
        } for i, ((a1,a2),(b1,b2)) in enumerate(schedule)]).to_csv(index=False).encode("utf-8-sig"),
        file_name="vs_with_scores.csv"
    )
    st.download_button("개인 기록/순위 CSV", table_df.to_csv(index=False).encode("utf-8-sig"),
                       file_name="personal_ranking.csv")

    state_blob = json.dumps({
        "names": names,
        "schedule": schedule,
        "scores": scores,
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
