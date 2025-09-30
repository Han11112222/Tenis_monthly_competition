from __future__ import annotations
import random, json, re
from typing import List, Tuple, Dict

import pandas as pd
import streamlit as st

st.set_page_config(page_title="목우회 월례회 대진표", layout="wide")

# ---------- 타입 ----------
Game = Tuple[Tuple[int, int], Tuple[int, int]]  # ((A1,A2),(B1,B2)) — 0-index

# ---------- 공통 유틸 ----------
def seeded_order(n: int) -> List[int]:
    out = []
    for i in range(1, n // 2 + 1):
        out.extend([i, n - i + 1])
    return out

def base_pairs_for_8x4() -> List[Game]:
    # 8명/1인 4게임일 때의 안정 스케줄(인덱스 0..7)
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
    """개인전 8명 기준 공정 대진 + gpp(최소3)만큼 잘라서 사용"""
    assert n == 8 and gpp >= 3
    seeds = seeded_order(n)
    sched, codes = [], []
    for (i1, i2), (j1, j2) in base_pairs_for_8x4():
        a = (seeds[i1]-1, seeds[i2]-1)
        b = (seeds[j1]-1, seeds[j2]-1)
        sched.append((a, b))
        A = tuple(sorted([seeds[i1], seeds[i2]]))
        B = tuple(sorted([seeds[j1], seeds[j2]]))
        codes.append(f"{A[0]}{A[1]}:{B[0]}{B[1]}")
    need_games = gpp * 2  # 1R=2경기(개인 8명 기준)
    return sched[:need_games], codes[:need_games]

def schedule_from_vs_codes(vs_codes: list[str], n_players: int) -> list[Game]:
    """VS 코드('18:27' 또는 '1,10:2,9') → 스케줄(0-index)"""
    out: list[Game] = []
    for code in vs_codes:
        s = re.sub(r"\s", "", str(code))
        if ":" not in s:
            out.append(((0,0),(0,0))); continue
        L, R = s.split(":", 1)

        def parse_team(t: str):
            m = re.match(r"^(\d+)[,/_-]?(\d+)$", t)
            if not m:
                if len(t) == 2 and t.isdigit():
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

def make_vs_codes(schedule: list[Game]) -> list[str]:
    return [
        f"{min(a1+1,a2+1)}{max(a1+1,a2+1)}:{min(b1+1,b2+1)}{max(b1+1,b2+1)}"
        for (a1,a2),(b1,b2) in schedule
    ]

def parse_pairs_line(s: str) -> list[tuple[int,int]]:
    """
    '1-2 3-4 5-6 7-8' 또는 '1-2,3-4,5-6,7-8' → [(0,1),(2,3),(4,5),(6,7)]
    """
    s = s.strip()
    if not s: return []
    toks = re.split(r"[,\s]+", s)
    pairs = []
    for tok in toks:
        if not tok: continue
        m = re.match(r"^(\d+)\s*[-:_/&]\s*(\d+)$", tok)
        if not m: return []
        a, b = int(m.group(1))-1, int(m.group(2))-1
        if a == b: return []
        pairs.append(tuple(sorted((a,b))))
    flat = [x for p in pairs for x in p]
    if len(set(flat)) != len(flat): return []
    return pairs

def parse_numbers_line(s: str) -> list[int]:
    """'1 2 3 4 9,10,11,12' → [1,2,3,4,9,10,11,12]"""
    if not s.strip(): return []
    toks = re.split(r"[,\s]+", s.strip())
    out = []
    for t in toks:
        if t.isdigit(): out.append(int(t))
    return out

# ---------- 팀전 스케줄 유틸 ----------
def latin_cross_rounds(blue_pairs: list[tuple[int,int]],
                       white_pairs: list[tuple[int,int]],
                       rounds: int) -> list[Game]:
    """
    k=쌍 수(팀인원/2). round r에서 blue[i] vs white[(i+r)%k]
    rounds는 1~k 범위(최대 k라운드)
    """
    k = len(blue_pairs)
    rounds = max(1, min(rounds, k))
    sched: list[Game] = []
    for r in range(rounds):
        for i in range(k):
            sched.append((blue_pairs[i], white_pairs[(i+r) % k]))
    return sched  # 길이 = rounds * k

def pair_in_team_random(team: list[int], rng: random.Random) -> list[tuple[int,int]]:
    """팀 내부 랜덤 페어링(재현 가능한 시드)"""
    arr = team[:]
    rng.shuffle(arr)
    pairs = []
    for i in range(0, len(arr), 2):
        a,b = sorted((arr[i], arr[i+1]))
        pairs.append((a,b))
    pairs.sort(key=lambda x: (x[0], x[1]))
    return pairs

# ---------- 집계(개인 / 페어) ----------
def compute_tables_individual(schedule: list[Game], scores: list[tuple[int|None,int|None]],
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
    rank_df = rank_df.sort_values(by=["득실차","승수","득점","실점"], ascending=[False,False,False,True])
    rank_df.insert(0, "순위", range(1, len(rank_df)+1))
    return rank_df, rounds_by_player

def compute_tables_pair(schedule: list[Game], scores: list[tuple[int|None,int|None]],
                        pair_labels: Dict[tuple,int],  # pair -> 0(청) / 1(백)
                        names: list[str], win_target: int):
    """
    페어 단위 집계. pair_labels: {(a,b): team_id} (a<b)
    반환: pair_df(팀, 페어(tuple), 표시명, 승/득/실/득실차, 팀내순위)
    """
    pair_keys = list(pair_labels.keys())
    # 초기화
    stats = {
        p: {"팀": "청" if pair_labels[p]==0 else "백",
            "페어": p,
            "표시명": "",
            "경기수": 0, "승수": 0, "득점": 0, "실점": 0}
        for p in pair_keys
    }
    for p in stats:
        a,b = p
        prefix = "청" if pair_labels[p]==0 else "백"
        stats[p]["표시명"] = f"{prefix}({a+1},{b+1}) · {names[a]} & {names[b]}"

    # 경기 반영
    for idx, ((a1,a2),(b1,b2)) in enumerate(schedule, start=1):
        A = tuple(sorted((a1,a2)))
        B = tuple(sorted((b1,b2)))
        if A not in stats or B not in stats:
            continue  # 변동 모드 대비
        sA, sB = scores[idx-1]
        if sA is None or sB is None:
            continue
        for K,sc_for,sc_against in [(A,sA,sB),(B,sB,sA)]:
            stats[K]["경기수"] += 1
            stats[K]["득점"] += sc_for
            stats[K]["실점"] += sc_against
        if sA == win_target and sB < win_target:
            stats[A]["승수"] += 1
        elif sB == win_target and sA < win_target:
            stats[B]["승수"] += 1

    pair_df = pd.DataFrame(stats).T
    pair_df["득실차"] = pair_df["득점"] - pair_df["실점"]

    # 팀 먼저, 그다음 팀 내부 정렬(득실차↓, 승수↓, 득점↓, 실점↑)
    pair_df = pair_df.sort_values(
        by=["팀","득실차","승수","득점","실점"],
        ascending=[True, False, False, False, True]
    ).copy()

    # 팀 내 순위 = 현재 정렬 순서 기반 누적 카운트 + 1 (pandas 에러 수정)
    pair_df["팀내순위"] = pair_df.groupby("팀").cumcount() + 1

    for col in ["경기수","승수","득점","실점","득실차","팀내순위"]:
        pair_df[col] = pair_df[col].astype(int)

    return pair_df

# ========================= 사이드바 =========================
with st.sidebar:
    st.header("⚙️ 대회 설정")
    mode = st.radio("복식 모드 선택", ["각자복식(개인)", "팀전 · 파트너 고정", "팀전 · 파트너 변동"])

    if "팀전" in mode:
        # 팀전도 참가 인원 조정 가능: 8~32명, 4의 배수
        n_players = st.number_input("참가 인원(팀전, 4의 배수)", min_value=8, max_value=32, value=16, step=4)
        team_size = n_players // 2
        games_per_player = st.slider("1인당 경기 수(최소 3)", min_value=3, max_value=max(3, team_size//2), value=4)
    else:
        n_players = st.number_input("참가 인원(짝수)", min_value=8, max_value=16, value=8, step=2)
        default_gpp = 4 if n_players == 8 else max(3, n_players // 4)
        games_per_player = st.slider("1인당 경기 수(최소 3)", min_value=3, max_value=n_players-1, value=default_gpp)

    win_target = st.number_input("게임 종료 점수(예: 6)", min_value=4, max_value=8, value=6)

    # -------- 팀전 전용: 한 줄 입력 --------
    blue_line, white_pairs_line, blue_pairs_line = "", "", ""
    if "팀전" in mode:
        st.subheader("🧩 팀 구성(한 줄 입력)")
        st.caption("청팀 번호를 한 줄로 입력(쉼표/공백 구분). 나머지는 자동으로 백팀.")
        default_blue = " ".join(str(i) for i in range(1, (n_players//2)+1))
        blue_line = st.text_input("청팀 번호 입력 예) 1 2 3 4 9 10 11 12", value=default_blue)
        # 자동 백팀 미리보기
        blue_sel = sorted(set(parse_numbers_line(blue_line)))
        blue_sel = [x for x in blue_sel if 1 <= x <= n_players]
        auto_white = [i for i in range(1, n_players+1) if i not in blue_sel]
        st.text(f"백팀 자동배정: {auto_white}")
        if len(blue_sel) != n_players//2:
            st.warning(f"청팀 인원은 정확히 {n_players//2}명을 선택해야 해.")

        if mode == "팀전 · 파트너 고정":
            st.subheader("🔗 파트너 고정 입력(한 줄)")
            st.caption("형식: 1-2 3-4 ... / 팀 내부에서만 쌍을 구성")
            # 기본값: 팀 내부에서 (1-2)(3-4)…
            default_bp = " ".join(f"{i}-{i+1}" for i in range(1, n_players//2, 2))
            default_wp = " ".join(f"{i}-{i+1}" for i in range(n_players//2+1, n_players, 2))
            blue_pairs_line  = st.text_input("청팀 파트너", value=default_bp)
            white_pairs_line = st.text_input("백팀 파트너", value=default_wp)

    gen = st.button("🏁 대진표 생성", type="primary")

# ========================= 대진 생성 =========================
if gen:
    # 이름 초기화
    st.session_state["names"] = [f"플레이어 {i+1}" for i in range(n_players)]
    st.session_state["team_mode"] = "팀전" in mode
    st.session_state["win_target"] = win_target

    if mode == "각자복식(개인)":
        if n_players == 8:
            schedule, vs_codes = generate_seeded_schedule(n_players, games_per_player)
        else:
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
            vs_codes = make_vs_codes(schedule)
        st.session_state["schedule"] = schedule
        st.session_state["vs_codes"] = vs_codes
        st.session_state["scores"] = [(None,None) for _ in schedule]
        st.session_state["pair_info"] = None
        st.session_state["finals"] = {"bronze": (None,None), "final": (None,None)}

    else:
        # 팀 준비
        team_size = n_players // 2
        if n_players % 4 != 0:
            st.error("팀전은 전체 인원이 4의 배수여야 해(팀당 짝수로 페어링).")
            st.stop()
        blue_sel = sorted(set(parse_numbers_line(blue_line)))
        blue_sel = [x for x in blue_sel if 1 <= x <= n_players]
        if len(blue_sel) != team_size:
            st.error(f"청팀을 정확히 {team_size}명 입력해줘."); st.stop()
        blue_team = sorted([x-1 for x in blue_sel])
        white_team = sorted([i for i in range(n_players) if (i+1) not in blue_sel])

        # 라벨: index -> '청k'/'백k'
        label_map: Dict[int,str] = {}
        for idx,p in enumerate(blue_team, start=1): label_map[p] = f"청{idx}"
        for idx,p in enumerate(white_team, start=1): label_map[p] = f"백{idx}"
        st.session_state["team_labels"] = label_map

        k = team_size // 2  # 팀 내 쌍 수

        if mode == "팀전 · 파트너 고정":
            bp = parse_pairs_line(blue_pairs_line or "")
            wp = parse_pairs_line(white_pairs_line or "")
            # 유효성: 팀 내부 페어, k쌍
            def valid_pairs(pairs: list[tuple[int,int]], team: list[int]) -> bool:
                if len(pairs) != k: return False
                ts = set(team)
                return all(a in ts and b in ts for a,b in pairs)
            if not (valid_pairs(bp, blue_team) and valid_pairs(wp, white_team)):
                st.error(f"파트너 고정은 팀 내부에서 정확히 {k}쌍을 지정해야 해."); st.stop()

            rounds = min(games_per_player, k)
            schedule = latin_cross_rounds(bp, wp, rounds)
            vs_codes = make_vs_codes(schedule)

            pair_labels = {}
            for a,b in bp: pair_labels[tuple(sorted((a,b)))] = 0  # 청
            for a,b in wp: pair_labels[tuple(sorted((a,b)))] = 1  # 백
            st.session_state["pair_info"] = {
                "mode": "fixed",
                "blue_pairs": bp,
                "white_pairs": wp,
                "pair_labels": pair_labels,
            }

        else:  # 팀전 · 파트너 변동
            rounds = min(games_per_player, k)
            rngB = random.Random(20250930); rngW = random.Random(20250930 + 1)
            schedule: list[Game] = []
            for r in range(rounds):
                bp = pair_in_team_random(blue_team, rngB)
                wp = pair_in_team_random(white_team, rngW)
                for i in range(k):
                    schedule.append((bp[i], wp[i]))
            vs_codes = make_vs_codes(schedule)
            st.session_state["pair_info"] = None  # 변동은 페어 고정X, 결승 비적용

        st.session_state["schedule"] = schedule
        st.session_state["vs_codes"] = vs_codes
        st.session_state["scores"] = [(None,None) for _ in schedule]
        st.session_state["finals"] = {"bronze": (None,None), "final": (None,None)}

# ========================= 본문 =========================
st.title("🎾 목우회 월례회 대진표")

if "names" not in st.session_state:
    st.info("좌측에서 모드/인원 설정 후 **대진표 생성**을 눌러 시작해줘.")
    st.stop()

# ---------- 1) 선수 명단 입력 ----------
st.subheader("🧑‍🤝‍🧑 선수 명단 입력")
names = st.session_state["names"]
team_labels: Dict[int,str] = st.session_state.get("team_labels", {})
team_mode = st.session_state.get("team_mode", False)

cols = st.columns(4)
for i in range(len(names)):
    with cols[i % 4]:
        label = f"번호 {i+1}"
        if team_mode and i in team_labels:
            tl = team_labels[i]          # '청1' / '백2'
            team_word = "청팀" if tl.startswith("청") else "백팀"
            label = f"번호 {i+1} ({team_word}{tl[1:]})"
        names[i] = st.text_input(label, value=names[i], key=f"name_{i}")
st.session_state["names"] = names

st.divider()

schedule: list[Game] = st.session_state.get("schedule", [])
scores: list[tuple[int|None,int|None]] = st.session_state.get("scores", [])
vs_codes: list[str] = st.session_state.get("vs_codes", [])
pair_info = st.session_state.get("pair_info", None)
win_target = st.session_state.get("win_target", 6)

def label_name(idx: int) -> str:
    if team_mode and idx in team_labels:
        tl = team_labels[idx]  # '청1' 등
        team_word = "청팀" if tl.startswith("청") else "백팀"
        return f"{team_word}{tl[1:]} · {names[idx]}"
    return names[idx]

# ---------- 2) 대진표(숫자 + 팀 이름) ----------
st.subheader("📋 대진표 (숫자 + 팀 이름)")
edit_mode = st.checkbox("대진표 숫자 수정 모드", value=False,
                        help="예: 18:27 (8명) / 1,10:2,9 (10명 이상)")

if edit_mode and schedule:
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
if schedule:
    rows = []
    for i, ((a1,a2),(b1,b2)) in enumerate(schedule, start=1):
        code = vs_codes[i-1] if i-1 < len(vs_codes) else f"{min(a1+1,a2+1)}{max(a1+1,a2+1)}:{min(b1+1,b2+1)}{max(b1+1,b2+1)}"
        rows.append({
            "게임": f"게임{i}",
            "VS": code,
            "A팀": f"{label_name(a1)} & {label_name(a2)}",
            "B팀": f"{label_name(b1)} & {label_name(b2)}",
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
else:
    st.info("대진표가 아직 없어. 좌측에서 생성해줘.")

st.divider()

# ---------- 3) 대진표 & 점수입력 ----------
if schedule:
    st.subheader("✍️ 대진표 & 점수입력")
    hdr = st.columns([1.1, 3, 3, 1, 1])
    for c, t in zip(hdr, ["구분","A팀 player","B팀 player","A팀 점수","B팀 점수"]):
        c.markdown(f"**{t}**")

    for idx, ((a1,a2),(b1,b2)) in enumerate(schedule, start=1):
        c = st.columns([1.1, 3, 3, 1, 1])
        c[0].write(f"게임{idx}")
        c[1].write(f"{label_name(a1)}, {label_name(a2)}")
        c[2].write(f"{label_name(b1)}, {label_name(b2)}")
        a_init, b_init = scores[idx-1] if idx-1 < len(scores) else (None, None)
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

# ---------- 4) 순위 섹션 (개인/페어) + 결승/3위전 ----------
if schedule:
    if pair_info and pair_info.get("mode") == "fixed":
        st.subheader("🥨 페어 기록 · 순위 (파트너 고정)")
        pair_df = compute_tables_pair(schedule, scores, pair_info["pair_labels"], names, win_target)

        # 팀별 1~2위 추출(결승/3위전용)
        by_team = {
            "청": pair_df[pair_df["팀"]=="청"].sort_values(by=["득실차","승수","득점","실점"], ascending=[False,False,False,True]),
            "백": pair_df[pair_df["팀"]=="백"].sort_values(by=["득실차","승수","득점","실점"], ascending=[False,False,False,True]),
        }
        blue_top2 = by_team["청"].head(2)
        white_top2 = by_team["백"].head(2)

        # 포디움 카드(예선 기준)
        col1,col2,col3 = st.columns(3)
        podium = pair_df.sort_values(by=["득실차","승수","득점","실점"], ascending=[False,False,False,True]).head(3)
        cards = [(col1,"🥇","#fff3b0"), (col2,"🥈","#e5e7eb"), (col3,"🥉","#f5e1c8")]
        for (col, medal, bg), (_, row) in zip(cards, podium.iterrows()):
            col.markdown(
                f"""
                <div style="padding:14px;border-radius:14px;background:{bg};">
                  <div style="font-size:22px">{medal} <b>{row['표시명']}</b></div>
                  <div style="margin-top:6px;">승수 {int(row['승수'])} · 득점 {int(row['득점'])} · 실점 {int(row['실점'])} · 득실차 <b>{int(row['득실차'])}</b></div>
                </div>
                """,
                unsafe_allow_html=True
            )

        disp = pair_df[["팀","표시명","경기수","승수","득점","실점","득실차"]].copy()
        disp.insert(0, "순위", range(1, len(disp)+1))
        st.dataframe(disp, use_container_width=True, hide_index=True)

        st.divider()
        # ---------- 결승 / 3위전 ----------
        st.subheader("🏟️ 결승전 / 준결승전 (페어 기준)")

        def pair_to_label(p: tuple[int,int]) -> str:
            a,b = p
            prefix = "청팀" if pair_info["pair_labels"][tuple(sorted(p))]==0 else "백팀"
            return f"{prefix} ({a+1},{b+1}) · {names[a]} & {names[b]}"

        finals_state = st.session_state.get("finals", {"bronze": (None,None), "final": (None,None)})

        # 결승 참가자
        fin_A = fin_B = None
        if len(blue_top2)>=1 and len(white_top2)>=1:
            fin_A = tuple(blue_top2.iloc[0]["페어"])
            fin_B = tuple(white_top2.iloc[0]["페어"])
            st.markdown(f"**결승** — {pair_to_label(fin_A)}  vs  {pair_to_label(fin_B)}")
            c1, c2 = st.columns(2)
            fa = c1.number_input("결승 · A팀 점수", min_value=0, max_value=win_target,
                                 value=int(finals_state["final"][0]) if finals_state["final"][0] is not None else 0, key="final_A")
            fb = c2.number_input("결승 · B팀 점수", min_value=0, max_value=win_target,
                                 value=int(finals_state["final"][1]) if finals_state["final"][1] is not None else 0, key="final_B")
            finals_state["final"] = (fa, fb)

        # 3위전 참가자
        br_A = br_B = None
        if len(blue_top2)>=2 and len(white_top2)>=2:
            br_A = tuple(blue_top2.iloc[1]["페어"])
            br_B = tuple(white_top2.iloc[1]["페어"])
            st.markdown(f"**3위전** — {pair_to_label(br_A)}  vs  {pair_to_label(br_B)}")
            c3, c4 = st.columns(2)
            ba = c3.number_input("3위전 · A팀 점수", min_value=0, max_value=win_target,
                                 value=int(finals_state["bronze"][0]) if finals_state["bronze"][0] is not None else 0, key="bronze_A")
            bb = c4.number_input("3위전 · B팀 점수", min_value=0, max_value=win_target,
                                 value=int(finals_state["bronze"][1]) if finals_state["bronze"][1] is not None else 0, key="bronze_B")
            finals_state["bronze"] = (ba, bb)

        st.session_state["finals"] = finals_state

        # ---------- 최종 시상(결승/3위전 결과로 확정) ----------
        def winner_loser(scA, scB, A_pair, B_pair):
            # win_target 먼저 도달하고 상대는 미만
            if None in (scA, scB) or A_pair is None or B_pair is None:
                return None, None
            if (scA == win_target and scB < win_target):
                return A_pair, B_pair
            if (scB == win_target and scA < win_target):
                return B_pair, A_pair
            return None, None

        champions, runners = winner_loser(finals_state["final"][0], finals_state["final"][1], fin_A, fin_B)
        third, fourth       = winner_loser(finals_state["bronze"][0], finals_state["bronze"][1], br_A, br_B)

        def pair_badge(p):
            if not p: return "-"
            a,b = p
            prefix = "청팀" if pair_info["pair_labels"][tuple(sorted(p))] == 0 else "백팀"
            return f"{prefix} ({a+1},{b+1}) · {names[a]} & {names[b]}"

        st.divider()
        st.subheader("🏅 최종 시상")

        # 우승 히어로 배너
        if champions:
            st.balloons()
            a,b = champions
            prefix = "청팀" if pair_info["pair_labels"][tuple(sorted(champions))]==0 else "백팀"
            html = f"""
            <div style="padding:26px;border-radius:22px;background:linear-gradient(135deg,#ffd700 0%,#ffb700 35%,#ff8a00 100%);
                        color:#1f2937; box-shadow:0 10px 28px rgba(0,0,0,.18); margin-bottom:14px;">
              <div style="font-size:36px;line-height:1.15; font-weight:800;">🎉 최종 우승</div>
              <div style="font-size:22px;margin-top:8px;"><b>{prefix}</b> — ({a+1},{b+1}) · {names[a]} &amp; {names[b]}</div>
              <div style="margin-top:6px;font-size:14px;opacity:.9">결승 스코어: {finals_state['final'][0]} : {finals_state['final'][1]}</div>
            </div>
            """
            st.markdown(html, unsafe_allow_html=True)
        else:
            st.info("최종 우승: -")

        # 나머지 시상 라인(고정 표기)
        c1,c2,c3 = st.columns(3)
        c1.write(f"**준우승 🥈**: {pair_badge(runners)}")
        c2.write(f"**3위팀 🥉**: {pair_badge(third)}")
        c3.write(f"**4위팀**: {pair_badge(fourth)}")

    else:
        # 개인 집계(개인전 / 팀전-변동)
        st.subheader("🏆 개인 기록 · 순위")
        rank_df, rounds_by_player = compute_tables_individual(schedule, scores, names, win_target)

        # 포디움
        ordered = rank_df.sort_values("순위").copy()
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

        # 라운드 스코어 보드
        def round_cell_text(i: int, r: int) -> str:
            lst = rounds_by_player[i]
            if r > len(lst): return ":"
            g = lst[r-1]
            (x1,x2),(y1,y2) = schedule[g-1]
            sA,sB = scores[g-1]
            if sA is None or sB is None: return ":"
            return f"{sA}:{sB}" if i in (x1,x2) else f"{sB}:{sA}"

        rows2 = []
        max_round = max((len(v) for v in rounds_by_player.values()), default=4)
        for _, r in ordered.iterrows():
            i = names.index(r["이름"])
            row = {"이름": r["이름"]}
            for rr in range(1, max_round+1):
                row[f"{rr}R"] = round_cell_text(i, rr)
            row.update({
                "승수": int(r["승수"]),
                "득점": int(r["득점"]),
                "실점": int(r["실점"]),
                "득실차": int(r["득실차"]),
                "순위": int(r["순위"]),
            })
            rows2.append(row)
        cols_order = ["이름"] + [f"{rr}R" for rr in range(1, max_round+1)] + ["승수","득점","실점","득실차","순위"]
        table_df = pd.DataFrame(rows2, columns=cols_order)

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

        st.dataframe(table_df.style.apply(highlight_top3, axis=1), use_container_width=True, hide_index=True)

# ---------- 내보내기/복원 ----------
with st.expander("💾 CSV 내보내기 / 상태 백업·복원"):
    if schedule:
        export_vs = pd.DataFrame([{
            "게임": f"게임{i+1}",
            "VS": (st.session_state.get('vs_codes', []) or
                   [f"{min(a1+1,a2+1)}{max(a1+1,a2+1)}:{min(b1+1,b2+1)}{max(b1+1,b2+1)}"
                    for ((a1,a2),(b1,b2)) in schedule])[i],
            "A팀": f"{label_name(a1)} & {label_name(a2)}",
            "B팀": f"{label_name(b1)} & {label_name(b2)}",
        } for i, ((a1,a2),(b1,b2)) in enumerate(schedule)])
        st.download_button("대진표(숫자+이름) CSV", export_vs.to_csv(index=False).encode("utf-8-sig"),
                           file_name="vs_numeric_named.csv")

        export_input = pd.DataFrame([{
            "구분": f"게임{i+1}",
            "A팀 player": f"{label_name(a1)}, {label_name(a2)}",
            "B팀 player": f"{label_name(b1)}, {label_name(b2)}",
            "A팀 점수": scores[i][0] if scores[i][0] is not None else "",
            "B팀 점수": scores[i][1] if scores[i][1] is not None else "",
        } for i, ((a1,a2),(b1,b2)) in enumerate(schedule)])
        st.download_button("대진표&점수입력 CSV", export_input.to_csv(index=False).encode("utf-8-sig"),
                           file_name="vs_with_scores.csv")

    state_blob = json.dumps({
        "names": st.session_state.get("names", []),
        "schedule": st.session_state.get("schedule", []),
        "scores": st.session_state.get("scores", []),
        "vs_codes": st.session_state.get("vs_codes", []),
        "meta": {"win_target": st.session_state.get("win_target"),
                 "pair_info_mode": st.session_state.get("pair_info", {}).get("mode","")},
        "finals": st.session_state.get("finals", {"bronze": (None,None), "final": (None,None)}),
    }, ensure_ascii=False)
    st.download_button("상태 백업(JSON)", state_blob.encode("utf-8"), file_name="mokwoo_state.json")

    up = st.file_uploader("상태 복원(JSON)", type=["json"])
    if up is not None:
        data = json.loads(up.read().decode("utf-8"))
        st.session_state["names"] = data.get("names", st.session_state.get("names", []))
        st.session_state["schedule"] = [tuple(map(tuple, g)) for g in data.get("schedule", st.session_state.get("schedule", []))]
        st.session_state["scores"] = [tuple(s) if s is not None else (None, None) for s in data.get("scores", st.session_state.get("scores", []))]
        st.session_state["vs_codes"] = data.get("vs_codes", st.session_state.get("vs_codes", []))
        st.session_state["finals"] = data.get("finals", st.session_state.get("finals", {"bronze": (None,None), "final": (None,None)}))
        st.rerun()
