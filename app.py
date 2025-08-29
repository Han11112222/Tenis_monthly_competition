# 🟢 목우회 월례회 대진표

Streamlit으로 월례대회 *복식 개인전* 대진표 생성 + 결과 입력 + 자동 순위 산출까지 한 번에.

* 좌측: 참가 인원/옵션 설정 → **대진표 생성**
* 우측: 번호 순서대로 **선수 이름 입력**
* 중앙: 경기별 점수 입력(예: 6:5) → 실시간 **개인 누적 성적·순위** 계산
* 기본 규칙(예시): 8명일 때 총 **8게임**, 1인당 **4게임**. 승점=득점, 실점=실점, 득실차=득점−실점. 동률 시 (1)승수, (2)득점, (3)실점(적을수록 유리) 순으로 정렬.

---

## 1) 로컬 실행 방법

```bash
# 가상환경 권장 (선택)
python -m venv .venv && source .venv/bin/activate  # (Windows는 .venv\Scripts\activate)

# 저장소 클론 (또는 새 repo에 파일 2개 저장)
git clone <YOUR_REPO_URL>
cd <YOUR_REPO_DIR>

pip install -r requirements.txt
streamlit run app.py
```

## 2) Streamlit Community Cloud 배포

1. GitHub 새 저장소에 아래 **app.py**, **requirements.txt** 그대로 업로드
2. Streamlit Community Cloud에서 **New app** → 본 repo 선택 → `app.py` 지정 → Deploy

---

## requirements.txt

```
streamlit>=1.38
pandas>=2.2
numpy>=1.26
```

---

## app.py

```python
# 제목: 목우회 월례회 대진표
# 기능: 참가 인원 설정 → 파트너가 계속 바뀌는 복식 대진 자동 생성 → 경기 결과 입력 → 개인 누적 성적/순위 산출

from __future__ import annotations
import itertools, random, json
from collections import defaultdict
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="목우회 월례회 대진표", layout="wide")

# =============================
# 유틸
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
    """그리디+재시도 방식으로 파트너 중복을 최소화하며 대진을 생성.
    필요 시 n=8,g=4 전용 폴백 스케줄 제공.
    """
    rnd = random.Random(seed)
    total_games = (n_players * games_per_player) // 4

    # 여러 번 시도해 가장 좋은 스케줄 채택
    best_sched: List[Game] | None = None
    best_cost = 10**9

    for attempt in range(500):
        need = [games_per_player] * n_players
        partner_counts: Dict[frozenset, int] = defaultdict(int)
        sched: List[Game] = []
        success = True

        for g in range(total_games):
            cand = [i for i in range(n_players) if need[i] > 0]
            if len(cand) < 4:
                success = False
                break
            # 필요 경기 수가 많은 사람 우선 + 약간 랜덤성
            cand.sort(key=lambda x: (need[x], rnd.random()), reverse=True)
            pool = cand[: min(8, len(cand))]

            picked_four = None
            best_local = None
            best_local_pen = 10**9

            for four in itertools.combinations(pool, 4):
                t1, t2 = choose_best_pairing(four, partner_counts)
                pen = pairing_penalty(t1, partner_counts) + pairing_penalty(t2, partner_counts)
                # 파트너 중복 강한 페널티
                if partner_counts[frozenset(t1)] >= 1:
                    pen += 5
                if partner_counts[frozenset(t2)] >= 1:
                    pen += 5
                # 필요 경기 수 균형
                pen -= sum(need[i] for i in four) * 0.05
                if pen < best_local_pen:
                    best_local_pen = pen
                    picked_four = four
                    best_local = (t1, t2)

            if picked_four is None:
                success = False
                break

            # 반영
            sched.append(best_local)  # type: ignore
            for p in picked_four:
                need[p] -= 1
            partner_counts[frozenset(best_local[0])] += 1  # type: ignore
            partner_counts[frozenset(best_local[1])] += 1  # type: ignore

        if success and all(x == 0 for x in need):
            cost = sum(partner_counts.values())  # 파트너 중복 총합이 작을수록 좋음
            if cost < best_cost:
                best_cost = cost
                best_sched = sched
                # 일단 충분히 좋으면 종료
                if best_cost <= total_games * 0:  # 이상적(모든 경기에서 파트너 중복 0)은 현실적으로 힘듦
                    break

    if best_sched is not None:
        return best_sched

    # ---- 폴백 (n=8, g=4) — 안정 스케줄 ----
    if n_players == 8 and games_per_player == 4:
        # 1~8 → 0~7 인덱스로 변환
        fallback = [
            ((0, 1), (2, 3)),
            ((4, 5), (6, 7)),
            ((0, 2), (4, 6)),
            ((1, 3), (5, 7)),
            ((0, 3), (5, 6)),
            ((1, 2), (4, 7)),
            ((0, 4), (3, 7)),
            ((1, 5), (2, 6)),
        ]
        return fallback

    # 마지막 보루 — 단순 랜덤 (균등 출전만 보장)
    rnd = random.Random(seed)
    need = [games_per_player] * n_players
    sched: List[Game] = []
    while sum(need) > 0:
        cand = [i for i in range(n_players) if need[i] > 0]
        rnd.shuffle(cand)
        four = cand[:4]
        if len(four) < 4:
            break
        a, b, c, d = four
        sched.append(((a, b), (c, d)))
        for p in four:
            need[p] -= 1
    return sched


def compute_tables(schedule: List[Game], scores: List[Tuple[int | None, int | None]], names: List[str], win_target: int):
    n = len(names)
    stats = {
        i: {
            "이름": names[i],
            "경기수": 0,
            "승수": 0,
            "득점": 0,
            "실점": 0,
        }
        for i in range(n)
    }

    rows = []
    for idx, ((a1, a2), (b1, b2)) in enumerate(schedule, start=1):
        sA, sB = scores[idx - 1]
        rows.append({
            "게임": idx,
            "A팀": f"{names[a1]} & {names[a2]}",
            "B팀": f"{names[b1]} & {names[b2]}",
            "득점(A)": sA if sA is not None else "",
            "득점(B)": sB if sB is not None else "",
        })
        # 통계 집계
        if sA is None or sB is None:
            continue
        for p in [a1, a2, b1, b2]:
            stats[p]["경기수"] += 1
        # 득실 처리
        stats[a1]["득점"] += sA
        stats[a2]["득점"] += sA
        stats[b1]["득점"] += sB
        stats[b2]["득점"] += sB
        stats[a1]["실점"] += sB
        stats[a2]["실점"] += sB
        stats[b1]["실점"] += sA
        stats[b2]["실점"] += sA
        # 승수
        if sA == win_target and sB < win_target:
            stats[a1]["승수"] += 1
            stats[a2]["승수"] += 1
        elif sB == win_target and sA < win_target:
            stats[b1]["승수"] += 1
            stats[b2]["승수"] += 1

    sched_df = pd.DataFrame(rows)

    rank_df = pd.DataFrame(stats).T
    rank_df["득실차"] = rank_df["득점"] - rank_df["실점"]
    # 동률 규칙: 득실차 → 승수 → 득점 → 실점(적은 사람이 우선)
    rank_df = rank_df.sort_values(by=["득실차", "승수", "득점", "실점"], ascending=[False, False, False, True])
    rank_df.insert(0, "순위", range(1, len(rank_df) + 1))

    return sched_df, rank_df


# =============================
# UI
# =============================
st.title("목우회 월례회 대진표")

left, right = st.columns([1, 1])

with left:
    st.subheader("① 참가 설정")
    n_players = st.number_input("참가 인원(짝수)", min_value=4, max_value=16, value=8, step=2)
    default_gpp = n_players // 2  # 예: 8명 → 1인 4게임
    games_per_player = st.slider("1인당 경기 수", min_value=max(2, n_players // 4), max_value=n_players - 1, value=default_gpp)
    win_target = st.number_input("게임 종료 점수 (예: 6)", min_value=4, max_value=8, value=6)
    seed = st.number_input("스케줄 시드(재현용)", min_value=0, max_value=99999, value=22)

    if st.button("대진표 생성", type="primary"):
        schedule = generate_schedule(n_players, games_per_player, seed=int(seed))
        st.session_state["schedule"] = schedule
        st.session_state["scores"] = [(None, None) for _ in range(len(schedule))]
        st.session_state["names"] = [f"플레이어 {i+1}" for i in range(n_players)]

    st.markdown("""
    **도움말**  
    • 8명 기준: 총 8게임, 1인당 4게임.  
    • 스케줄은 파트너 중복을 최소화하도록 자동 생성.  
    • 동일 조건일 때 시드를 바꾸면 다른 스케줄 생성.
    """)

with right:
    st.subheader("② 선수 이름 입력")
    if "names" not in st.session_state:
        st.info("좌측에서 먼저 **대진표 생성**을 눌러줘.")
    else:
        names = st.session_state["names"]
        for i in range(len(names)):
            names[i] = st.text_input(f"번호 {i+1}", value=names[i])
        st.session_state["names"] = names

st.divider()

if "schedule" in st.session_state:
    schedule: List[Game] = st.session_state["schedule"]
    names: List[str] = st.session_state["names"]
    scores: List[Tuple[int | None, int | None]] = st.session_state["scores"]

    st.subheader("③ 경기 결과 입력")
    tbl_rows = []
    for idx, ((a1, a2), (b1, b2)) in enumerate(schedule, start=1):
        c1, c2, c3, c4, c5 = st.columns([2, 3, 1, 1, 1])
        with c1:
            st.markdown(f"**게임 {idx}**")
        with c2:
            st.write(f"A팀: {names[a1]} & {names[a2]}  |  B팀: {names[b1]} & {names[b2]}")
        a_init, b_init = scores[idx - 1]
        with c3:
            a_sc = st.number_input(f"A{idx}", min_value=0, max_value=int(win_target), value=int(a_init) if a_init is not None else 0, key=f"A{idx}")
        with c4:
            st.markdown(":vs:")
        with c5:
            b_sc = st.number_input(f"B{idx}", min_value=0, max_value=int(win_target), value=int(b_init) if b_init is not None else 0, key=f"B{idx}")
        # 유효성: 한쪽이 win_target이어야 확정으로 간주
        if (a_sc == win_target and b_sc < win_target) or (b_sc == win_target and a_sc < win_target):
            scores[idx - 1] = (a_sc, b_sc)
        else:
            scores[idx - 1] = (None, None)

    st.session_state["scores"] = scores

    # 결과표
    sched_df, rank_df = compute_tables(schedule, scores, names, win_target)

    st.subheader("④ 대진표")
    st.dataframe(sched_df, use_container_width=True, hide_index=True)

    st.subheader("⑤ 개인 순위")
    st.dataframe(rank_df, use_container_width=True, hide_index=True)

    # 다운로드
    exp = st.expander("CSV 내보내기/불러오기")
    with exp:
        csv_sched = sched_df.to_csv(index=False).encode("utf-8-sig")
        csv_rank = rank_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button("대진표 CSV 다운로드", csv_sched, file_name="schedule.csv")
        st.download_button("순위표 CSV 다운로드", csv_rank, file_name="ranking.csv")

        state_blob = json.dumps({
            "names": names,
            "schedule": schedule,
            "scores": scores,
            "meta": {"win_target": win_target},
        }, ensure_ascii=False)
        st.download_button("상태 백업(JSON)", state_blob.encode("utf-8"), file_name="mokwoo_state.json")

        up = st.file_uploader("상태 복원(JSON)", type=["json"])
        if up is not None:
            data = json.loads(up.read().decode("utf-8"))
            st.session_state["names"] = data.get("names", names)
            st.session_state["schedule"] = [tuple(map(tuple, g)) for g in data.get("schedule", schedule)]
            st.session_state["scores"] = [tuple(s) if s is not None else (None, None) for s in data.get("scores", scores)]
            st.rerun()
else:
    st.info("좌측에서 **대진표 생성** 후 진행해줘.")
```

---

## 메모

* 파트너 중복 최소화 그리디 알고리즘 + 재시도. 실패 시 8명/4게임 폴백 스케줄 포함.
* 정렬 기준은 일반적인 경기 운영 관행에 맞춰 \*득실차 → 승수 → 득점 → 실점(적을수록 우선)\*으로 구성. 필요하면 코드 내 정렬 키만 바꾸면 됨.
* 8명이 아닐 때도 작동하도록 **1인당 경기 수**를 노출해 유연하게 운영.
