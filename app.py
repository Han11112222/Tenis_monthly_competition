from __future__ import annotations
import random, json, re
from typing import List, Tuple, Dict

import pandas as pd
import streamlit as st

st.set_page_config(page_title="목우회 월례회 대진표", layout="wide")

# ---------- 타입 ----------
Game = Tuple[Tuple[int, int], Tuple[int, int]]  # ((A1,A2),(B1,B2)) — 0-index

# ---------- 유틸 ----------
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

# ---------- 팀전(16인) 전용 유틸 ----------
def parse_pair_string(s: str) -> list[tuple[int,int]]:
    """
    입력 예시: '1-2,3-4,5-6,7-8'
    반환: [(0,1),(2,3),(4,5),(6,7)]
    """
    s = s.strip()
    if not s:
        return []
    pairs = []
    for tok in s.split(","):
        tok = tok.strip()
        m = re.match(r"^(\d+)\s*[-:_/]\s*(\d+)$", tok)
        if not m:
            return []
        a, b = int(m.group(1))-1, int(m.group(2))-1
        if a == b or a < 0 or b < 0:
            return []
        pairs.append(tuple(sorted((a,b))))
    # 중복/교집합 체크
    flat = [x for p in pairs for x in p]
    if len(set(flat)) != len(flat):
        return []
    return pairs

def latin_square_cross_rounds(blue_pairs: list[tuple[int,int]],
                              white_pairs: list[tuple[int,int]]) -> list[Game]:
    """
    4라운드: round r에서 blue[i] vs white[(i+r)%4]
    반환은 Game 리스트(라운드 순으로 4경기씩)
    """
    sched: list[Game] = []
    for r in range(4):
        for i in range(4):
            b = blue_pairs[i]
            w = white_pairs[(i + r) % 4]
            sched.append((b, w))
    return sched

def make_vs_codes(schedule: list[Game]) -> list[str]:
    codes = []
    for (a1,a2),(b1,b2) in schedule:
        A = tuple(sorted((a1+1,a2+1)))
        B = tuple(sorted((b1+1,b2+1)))
        codes.append(f"{A[0]}{A[1]}:{B[0]}{B[1]}")
    return codes

def pair_in_teams(players: list[int], rng: random.Random) -> list[tuple[int,int]]:
    """
    주어진 팀 플레이어 리스트(0-index)를 랜덤/시드로 안정적으로 4쌍으로 나눔
    """
    arr = players[:]
    rng.shuffle(arr)
    pairs = []
    for i in range(0, 8, 2):
        a, b = sorted((arr[i], arr[i+1]))
        pairs.append((a, b))
    # 번호 오름차순 기준 가독성 정렬
    pairs.sort(key=lambda x: (x[0], x[1]))
    return pairs

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
    mode = st.radio("복식 모드 선택", ["각자복식(개인)", "팀전 · 파트너 고정", "팀전 · 파트너 변동"])
    # 기본 인원 설정(개인전은 8~16, 팀전은 16 고정)
    if "팀전" in mode:
        st.caption("팀전은 16명(청8 · 백8)만 지원")
        n_players = 16
        st.number_input("참가 인원(팀전은 16 고정)", min_value=16, max_value=16, value=16, step=0, key="n_players_locked", disabled=True)
        games_per_player = st.slider("1인당 경기 수(권장 4)", min_value=4, max_value=6, value=4)
    else:
        n_players = st.number_input("참가 인원(짝수)", min_value=8, max_value=16, value=8, step=2)
        default_gpp = n_players // 2  # 8명 → 1인 4게임
        games_per_player = st.slider("1인당 경기 수", min_value=max(2, n_players//4),
                                     max_value=n_players-1, value=default_gpp)
    win_target = st.number_input("게임 종료 점수(예: 6)", min_value=4, max_value=8, value=6)

    # -------- 팀전 전용 입력 --------
    blue_sel, white_sel = None, None
    blue_fixed_str, white_fixed_str = None, None
    if "팀전" in mode:
        st.subheader("팀 구성")
        default_blue = list(range(1,9))
        default_white = list(range(9,17))
        blue_sel = st.multiselect("청팀(번호 8명 선택)", options=list(range(1,17)), default=default_blue)
        # 남는 인원은 자동으로 백팀
        auto_white = [i for i in range(1,17) if i not in blue_sel]
        if len(blue_sel) != 8 or len(auto_white) != 8:
            st.warning("청팀 8명을 정확히 선택해줘. 나머지 8명은 자동으로 백팀.")
        st.text(f"백팀 자동배정: {auto_white}")

        if mode == "팀전 · 파트너 고정":
            st.subheader("파트너 고정 입력")
            blue_fixed_str = st.text_input("청팀 파트너 (예: 1-2,3-4,5-6,7-8)", value="1-2,3-4,5-6,7-8")
            white_fixed_str = st.text_input("백팀 파트너 (예: 9-10,11-12,13-14,15-16)", value="9-10,11-12,13-14,15-16")

    gen = st.button("대진표 생성", type="primary")

# ========================= 대진 생성 =========================
if gen:
    # 초기 이름/상태 준비
    if "names" not in st.session_state or len(st.session_state.get("names", [])) != n_players:
        st.session_state["names"] = [f"플레이어 {i+1}" for i in range(n_players)]

    # --- 개인 각자복식 ---
    if mode == "각자복식(개인)":
        if n_players == 8 and games_per_player == 4:
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
            vs_codes = [f"{min(a1+1,a2+1)}{max(a1+1,a2+1)}:{min(b1+1,b2+1)}{max(b1+1,b2+1)}"
                        for ((a1,a2),(b1,b2)) in schedule]

    # --- 팀전: 공통 준비 ---
    elif "팀전" in mode:
        if n_players != 16:
            st.error("팀전은 16명만 지원해.")
            st.stop()
        if blue_sel is None:
            st.error("청팀을 먼저 선택해줘.")
            st.stop()
        blue_team = sorted([x-1 for x in blue_sel])
        white_team = sorted([x for x in range(16) if x not in blue_team])

        # (이름 배열이 팀 선택에 따라 섞이지 않아도 번호 기반 매칭이므로 문제 없음)

        # --- 팀전 · 파트너 고정 ---
        if mode == "팀전 · 파트너 고정":
            bp = parse_pair_string(blue_fixed_str or "")
            wp = parse_pair_string(white_fixed_str or "")
            # 유효성: 각 팀 4쌍, 같은 팀 내부 인원만
            def valid_pairs(pairs: list[tuple[int,int]], team: list[int]) -> bool:
                if len(pairs) != 4: return False
                team_set = set(team)
                return all(a in team_set and b in team_set for a,b in pairs)
            if not (valid_pairs(bp, blue_team) and valid_pairs(wp, white_team)):
                st.error("파트너 고정 입력이 올바르지 않아. 팀 내부 4쌍씩 정확히 지정해줘.")
                st.stop()

            # 4라운드 라틴스퀘어 매칭
            schedule = latin_square_cross_rounds(bp, wp)
            vs_codes = make_vs_codes(schedule)

        # --- 팀전 · 파트너 변동 ---
        else:
            rounds = max(4, min(6, games_per_player))  # 보통 4라운드 권장
            rngB = random.Random(20250930)  # 재현 가능한 시드
            rngW = random.Random(20250930 + 1)
            schedule: list[Game] = []
            for r in range(rounds):
                bp = pair_in_teams(blue_team, rngB)
                wp = pair_in_teams(white_team, rngW)
                # 같은 슬롯끼리 대결(동시에 4코트)
                for i in range(4):
                    schedule.append((bp[i], wp[i]))
            vs_codes = make_vs_codes(schedule)

    # 세션 저장
    st.session_state["schedule"] = schedule
    st.session_state["scores"] = [(None,None) for _ in range(len(schedule))]
    st.session_state["vs_codes"] = vs_codes

# ========================= 페이지 본문 =========================
st.title("목우회 월례회 대진표")
if "names" not in st.session_state:
    st.info("좌측에서 모드/인원을 설정하고 **대진표 생성**을 눌러 시작해줘.")
    st.stop()

# ---------- 1) 선수 명단 입력 ----------
st.subheader("선수 명단 입력")
names = st.session_state["names"]
cols = st.columns(4)
for i in range(len(names)):
    with cols[i % 4]:
        names[i] = st.text_input(f"번호 {i+1}", value=names[i], key=f"name_{i}")
st.session_state["names"] = names

st.divider()

schedule: list[Game] = st.session_state.get("schedule", [])
scores: list[tuple[int|None,int|None]] = st.session_state.get("scores", [])
vs_codes: list[str] = st.session_state.get("vs_codes", [])

# ---------- 2) 대진표(숫자 + 팀 이름) ----------
st.subheader("대진표 (숫자 + 팀 이름)")
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
            "A팀": f"{names[a1]} & {names[a2]}",
            "B팀": f"{names[b1]} & {names[b2]}",
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
else:
    st.info("대진표가 아직 없어. 좌측에서 생성해줘.")

st.divider()

# ---------- 3) 대진표 & 점수입력 ----------
if schedule:
    st.subheader("대진표 & 점수입력")
    hdr = st.columns([1.1, 3, 3, 1, 1])
    for c, t in zip(hdr, ["구분","A팀 player","B팀 player","A팀 점수","B팀 점수"]):
        c.markdown(f"**{t}**")

    for idx, ((a1,a2),(b1,b2)) in enumerate(schedule, start=1):
        c = st.columns([1.1, 3, 3, 1, 1])
        c[0].write(f"게임{idx}")
        c[1].write(f"{names[a1]}, {names[a2]}")
        c[2].write(f"{names[b1]}, {names[b2]}")
        a_init, b_init = scores[idx-1] if idx-1 < len(scores) else (None, None)
        a_sc = c[3].number_input(f"A{idx}", min_value=0, max_value=win_target,
                                 value=int(a_init) if a_init is not None else 0, key=f"g{idx}_A")
        b_sc = c[4].number_input(f"B{idx}", min_value=0, max_value=win_target,
                                 value=int(b_init) if b_init is not None else 0, key=f"g{idx}_B")
        if (a_sc == win_target and b_sc < win_target) or (b_sc == win_target and a_sc < win_target):
            if idx-1 < len(scores):
                scores[idx-1] = (a_sc, b_sc)
        else:
            if idx-1 < len(scores):
                scores[idx-1] = (None, None)

    st.session_state["scores"] = scores

st.divider()

# ---------- 4) 개인 기록 + 포디움 ----------
if schedule:
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

    # 개인 기록표
    rows2 = []
    # 라운드 최대치 추출(개인전/팀전 라운드수가 다를 수 있음)
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

    st.markdown("### 개인 경기 기록 및 순위")
    st.dataframe(table_df.style.apply(highlight_top3, axis=1), use_container_width=True, hide_index=True)

# ---------- 내보내기/복원 ----------
with st.expander("CSV 내보내기 / 상태 백업·복원"):
    if schedule:
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

    # 개인 순위는 schedule 없어도 빈 테이블 방지
    if schedule:
        # 위에서 만든 table_df 재사용
        st.download_button("개인 기록/순위 CSV", table_df.to_csv(index=False).encode("utf-8-sig"),
                           file_name="personal_ranking.csv")

    state_blob = json.dumps({
        "names": st.session_state.get("names", []),
        "schedule": st.session_state.get("schedule", []),
        "scores": st.session_state.get("scores", []),
        "vs_codes": st.session_state.get("vs_codes", []),
        "meta": {"win_target": st.session_state.get("win_target")},
    }, ensure_ascii=False)
    st.download_button("상태 백업(JSON)", state_blob.encode("utf-8"), file_name="mokwoo_state.json")

    up = st.file_uploader("상태 복원(JSON)", type=["json"])
    if up is not None:
        data = json.loads(up.read().decode("utf-8"))
        st.session_state["names"] = data.get("names", st.session_state.get("names", []))
        st.session_state["schedule"] = [tuple(map(tuple, g)) for g in data.get("schedule", st.session_state.get("schedule", []))]
        st.session_state["scores"] = [tuple(s) if s is not None else (None, None) for s in data.get("scores", st.session_state.get("scores", []))]
        st.session_state["vs_codes"] = data.get("vs_codes", st.session_state.get("vs_codes", []))
        st.rerun()
