from __future__ import annotations
import itertools, random, json
from collections import defaultdict
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="목우회 월례회 대진표", layout="wide")

# =============================
# 타입/유틸
# =============================
Game = Tuple[Tuple[int, int], Tuple[int, int]]  # ((A1,A2),(B1,B2)) — 모두 0-index player id

def pairing_penalty(team: Tuple[int, int], partner_counts: Dict[frozenset, int]) -> int:
    return partner_counts[frozenset(team)]

def choose_best_pairing(four: Tuple[int, int, int, int], partner_counts) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    a, b, c, d = four
    candidates = [((a, b), (c, d)), ((a, c), (b, d)), ((a, d), (b, c))]
    best = None
    best_score = 10**9
    for t1, t2 in candidates:
        score = pairing_penalty(tuple(sorted(t1)), partner_counts) + pairing_penalty(tuple(sorted(t2)), partner_counts)
        if score < best_score:
            best_score = score
            best = (tuple(sorted(t1)), tuple(sorted(t2)))
    assert best is not None
    return best

def generate_schedule(n_players: int, games_per_player: int, seed: int | None = None) -> List[Game]:
    """파트너 중복 최소화 대진 생성. 실패 시 8명/4게임 폴백 제공."""
    rnd = random.Random(seed)
    total_games = (n_players * games_per_player) // 4

    best_sched: List[Game] | None = None
    best_cost = 10**9

    for _ in range(500):
        need = [games_per_player] * n_players
        partner_counts: Dict[frozenset, int] = defaultdict(int)
        sched: List[Game] = []
        success = True

        for _ in range(total_games):
            cand = [i for i in range(n_players) if need[i] > 0]
            if len(cand) < 4:
                success = False
                break
            cand.sort(key=lambda x: (need[x], rnd.random()), reverse=True)
            pool = cand[: min(8, len(cand))]

            picked_four = None
            best_local = None
            best_local_pen = 10**9

            for four in itertools.combinations(pool, 4):
                t1, t2 = choose_best_pairing(four, partner_counts)
                pen = pairing_penalty(t1, partner_counts) + pairing_penalty(t2, partner_counts)
                if partner_counts[frozenset(t1)] >= 1: pen += 5
                if partner_counts[frozenset(t2)] >= 1: pen += 5
                pen -= sum(need[i] for i in four) * 0.05
                if pen < best_local_pen:
                    best_local_pen = pen
                    picked_four = four
                    best_local = (t1, t2)

            if picked_four is None:
                success = False
                break

            sched.append(best_local)  # type: ignore
            for p in picked_four:
                need[p] -= 1
            partner_counts[frozenset(best_local[0])] += 1  # type: ignore
            partner_counts[frozenset(best_local[1])] += 1  # type: ignore

        if success and all(x == 0 for x in need):
            cost = sum(partner_counts.values())
            if cost < best_cost:
                best_cost = cost
                best_sched = sched
                if best_cost <= 0:
                    break

    if best_sched is not None:
        return best_sched

    # 폴백 (8명/4게임) — 총 8게임
    if n_players == 8 and games_per_player == 4:
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

    # 마지막 보루 — 균등 출전 랜덤
    need = [games_per_player] * n_players
    sched: List[Game] = []
    while sum(need) > 0:
        cand = [i for i in range(n_players) if need[i] > 0]
        rnd.shuffle(cand)
        four = cand[:4]
        if len(four) < 4: break
        a, b, c, d = four
        sched.append(((a, b), (c, d)))
        for p in four: need[p] -= 1
    return sched

def compute_tables(schedule: List[Game], scores: List[Tuple[int | None, int | None]], names: List[str], win_target: int):
    n = len(names)
    stats = {i: {"이름": names[i], "경기수": 0, "승수": 0, "득점": 0, "실점": 0} for i in range(n)}
    rounds_by_player: Dict[int, list] = {i: [] for i in range(n)}  # 각 선수의 출전 게임 번호(1-based)

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

# =============================
# 사이드바
# =============================
with st.sidebar:
    st.header("대회 설정")
    n_players = st.number_input("참가 인원(짝수)", min_value=8, max_value=16, value=8, step=2)  # 사진 기준 8명 기본
    default_gpp = n_players // 2  # 8명 → 1인 4게임
    games_per_player = st.slider("1인당 경기 수", min_value=max(2, n_players // 4), max_value=n_players - 1, value=default_gpp)
    win_target = st.number_input("게임 종료 점수(예: 6)", min_value=4, max_value=8, value=6)
    seed = st.number_input("스케줄 시드", min_value=0, max_value=99999, value=22)
    gen = st.button("대진표 생성", type="primary")

if gen:
    schedule = generate_schedule(n_players, games_per_player, seed=int(seed))
    st.session_state["schedule"] = schedule
    st.session_state["scores"] = [(None, None) for _ in range(len(schedule))]
    st.session_state["names"] = [f"플레이어 {i+1}" for i in range(n_players)]

st.title("목우회 월례회 대진표")
if "names" not in st.session_state:
    st.info("좌측에서 인원을 설정하고 **대진표 생성**을 눌러 시작해줘.")
    st.stop()

# =============================
# ① 선수 명단 입력
# =============================
st.subheader("선수 명단 입력")
names = st.session_state["names"]
cols = st.columns(4)
for i in range(len(names)):
    with cols[i % 4]:
        names[i] = st.text_input(f"번호 {i+1}", value=names[i], key=f"name_{i}")
st.session_state["names"] = names

st.divider()

schedule: List[Game] = st.session_state["schedule"]
scores: List[Tuple[int | None, int | None]] = st.session_state["scores"]

# =============================
# ② 대진표(숫자만)
# =============================
st.subheader("대진표 (숫자)")
vs_simple = [{"게임": f"게임{i+1}", "VS": f"{a1+1}{a2+1} : {b1+1}{b2+1}"} for i, ((a1,a2),(b1,b2)) in enumerate(schedule)]
st.dataframe(pd.DataFrame(vs_simple), hide_index=True, use_container_width=True)

st.divider()

# =============================
# ③ 대진표 & 점수입력 (이름 포함 표)
# =============================
st.subheader("대진표 & 점수입력")
# 헤더
h = st.columns([1.1, 3, 3, 1, 1])
h[0].markdown("**구분**")
h[1].markdown("**A팀 player**")
h[2].markdown("**B팀 player**")
h[3].markdown("**A팀 점수**")
h[4].markdown("**B팀 점수**")

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

# =============================
# ④ 개인 기록(1R~4R) 및 순위 (구분 제거, 1위부터, TOP3 강조)
# =============================
rank_df, rounds_by_player = compute_tables(schedule, scores, names, win_target)

# 개인 라운드 텍스트
def round_cell_text(player_idx: int, r: int) -> str:
    lst = rounds_by_player[player_idx]
    if r > len(lst): return ":"
    g_idx = lst[r-1]  # 1-based
    (a1, a2), (b1, b2) = schedule[g_idx-1]
    sA, sB = scores[g_idx-1]
    if sA is None or sB is None:
        return ":"
    return f"{sA}:{sB}" if player_idx in (a1, a2) else f"{sB}:{sA}"

# 순위 오름차순으로 재정렬 후, 구분 제거
ordered = rank_df.sort_values("순위").copy()
rows = []
for _, r in ordered.iterrows():
    i = names.index(r["이름"])  # player index
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

# TOP3 꾸미기: 메달 + 행 하이라이트
medal = {1:"🥇", 2:"🥈", 3:"🥉"}
table_df["순위"] = table_df["순위"].apply(lambda x: f"{medal.get(x,'')} {x}".strip())

def highlight_top3(row):
    raw_rank = int(row["순위"].split()[-1])  # "🥇 1" → 1
    if raw_rank == 1:
        return ["background-color: #fff3b0; font-weight: 700" for _ in row]
    if raw_rank == 2:
        return ["background-color: #e5e7eb; font-weight: 600" for _ in row]
    if raw_rank == 3:
        return ["background-color: #f5e1c8; font-weight: 600" for _ in row]
    return [""] * len(row)

st.subheader("개인 경기 기록 (1R~4R) 및 순위")
st.dataframe(table_df.style.apply(highlight_top3, axis=1), use_container_width=True, hide_index=True)

# 다운로드/백업
with st.expander("CSV 내보내기 / 상태 백업·복원"):
    st.download_button("대진표(숫자) CSV",
                       pd.DataFrame(vs_simple).to_csv(index=False).encode("utf-8-sig"),
                       file_name="vs_numeric.csv")
    st.download_button("대진표&점수입력 CSV",
                       pd.DataFrame([{
                           "구분": f"게임{i+1}",
                           "A팀 player": f"{names[a1]}, {names[a2]}",
                           "B팀 player": f"{names[b1]}, {names[b2]}",
                           "A팀 점수": scores[i][0] if scores[i][0] is not None else "",
                           "B팀 점수": scores[i][1] if scores[i][1] is not None else "",
                       } for i, ((a1,a2),(b1,b2)) in enumerate(schedule)]).to_csv(index=False).encode("utf-8-sig"),
                       file_name="vs_with_scores.csv")
    st.download_button("개인 기록/순위 CSV",
                       table_df.to_csv(index=False).encode("utf-8-sig"),
                       file_name="personal_ranking.csv")
    state_blob = json.dumps({"names": names, "schedule": schedule, "scores": scores, "meta": {"win_target": win_target}}, ensure_ascii=False)
    st.download_button("상태 백업(JSON)", state_blob.encode("utf-8"), file_name="mokwoo_state.json")
    up = st.file_uploader("상태 복원(JSON)", type=["json"])
    if up is not None:
        data = json.loads(up.read().decode("utf-8"))
        st.session_state["names"] = data.get("names", names)
        st.session_state["schedule"] = [tuple(map(tuple, g)) for g in data.get("schedule", schedule)]
        st.session_state["scores"] = [tuple(s) if s is not None else (None, None) for s in data.get("scores", scores)]
        st.rerun()
