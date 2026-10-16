# -*- coding: utf-8 -*-
"""
compute_cloze_surprisal_ettin_v1_word.py
=========================================================================
Ettin 짝 모델(인코더 = MLM, 디코더 = CLM)로 수능 빈칸 선지의 surprisal을 계산한다.
150m·1b를 포함해 Ettin의 모든 크기(17m / 32m / 68m / 150m / 400m / 1b)를 쓸 수 있다.

■ 무엇을 하는가 (한 줄 요약)
  선지 ①~⑤를 빈칸에 하나씩 넣어 보고, 모델이 그 선지를 얼마나 '뜻밖'으로 느끼는지
  (= surprisal, 단위 bits) 계산해 엑셀로 저장한다. 가장 덜 뜻밖인 선지 = 모델의 선택.

■ 두 가지 채점 방식 (모델 종류를 config에서 자동 감지)
  인코더(MLM) : BERT v5·DeBERTa v1과 같은 규칙 = Salazar et al.(2020) PLL
                선지 토큰마다 사본을 만들어 한 조각씩 [MASK]로 가리고,
                가린 자리에서 원래 조각이 나올 로그확률을 읽는다.
                입력 = [CLS] 지문 전체(선지 포함) [SEP]   → 앞뒤 문맥을 모두 본다
  디코더(CLM) : GPT-2 v4·GPT-2 XL v5와 같은 규칙
                위치 p 조각의 로그확률 = 위치 p-1에서 낸 '다음 조각' 예측 분포에서 읽는다.
                입력 = [BOS] 빈칸 앞 지문 + 선지          → 앞 문맥만 본다

■ 두 가지 집계 방식 (--agg)
  token : 선지 조각 로그확률의 단순 평균            Σℓ ÷ T
  word  : 공백 단어마다 먼저 평균 → 단어끼리 다시 평균 (단어 수준 2단 집계)

■ 왜 한 스크립트인가
  Ettin 인코더·디코더는 토크나이저가 같다. 선지 찾기·단어 묶기 함수를 두 모델이
  공유하므로 같은 선지는 두 모델에서 같은 조각 수(T)·단어 수(W)를 갖는다.
  → 두 결과표의 ntok·nword 열이 완전히 같아야 정상이다.
  → 두 모델의 차이는 '어텐션 방향 + 학습 목표'만 남는다 (Ettin 설계 취지).

■ 기존 스크립트와 다른 점
  (1) 토큰화 : 완성 문장을 한 번에 토큰화하고 오프셋으로 선지·단어를 찾는다 (GPT-2 v4 규칙)
  (2) BOS    : 디코더는 Ettin 토크나이저 규칙(add_bos_token=True)대로 맨 앞에 [CLS]를 붙인다
               (GPT-2 v4는 붙이지 않았다). 결과표 'BOS' 열에 기록한다.
  (3) 길이   : 최대 위치 7,999 → 수능 지문은 잘리지 않는다
               (넘으면 인코더는 선지 중심 앞·뒤 절단, 디코더는 왼쪽 절단)
  (4) LM 헤드: 필요한 위치에만 적용한다 (값은 전체 계산과 같음 → 셀프테스트 C)
  (5) 지원 안 함 : encoder-from-decoder(MNTP) 변형 (가린 단어를 '앞 위치'에서 예측하는 방식)

실행 예 (PowerShell, conda env bert312)
---------------------------------------
  pip install -U transformers        # 처음 한 번 (디코더 'modernbert-decoder' 형식 인식에 필요)

  ── 모델 고르는 법 : --arch (enc|dec) + --size (17m|32m|68m|150m|400m|1b)
     기본값 = --arch enc --size 150m   →  jhu-clsp/ettin-encoder-150m
     출력 파일 이름은 자동 : ettin_{arch}{size}_agg_{agg}.xlsx

  # 0) 구현 검증 (모델마다 한 번)
  python compute_cloze_surprisal_ettin_v1_word.py --arch enc --size 150m --selftest
  python compute_cloze_surprisal_ettin_v1_word.py --arch dec --size 150m --selftest
  python compute_cloze_surprisal_ettin_v1_word.py --arch enc --size 1b --selftest
  python compute_cloze_surprisal_ettin_v1_word.py --arch dec --size 1b --selftest

  # 1) 150m 짝 — 인코더·디코더 × token·word  (출력: ettin_enc150m_agg_token.xlsx 등)
  python compute_cloze_surprisal_ettin_v1_word.py --arch enc --size 150m --agg token --input "20260729_csat_data_original.xlsx"
  python compute_cloze_surprisal_ettin_v1_word.py --arch enc --size 150m --agg word  --input "20260729_csat_data_original.xlsx"
  python compute_cloze_surprisal_ettin_v1_word.py --arch dec --size 150m --agg token --input "20260729_csat_data_original.xlsx"
  python compute_cloze_surprisal_ettin_v1_word.py --arch dec --size 150m --agg word  --input "20260729_csat_data_original.xlsx"

  # 2) 1b 짝 — 위와 같고 --size 1b 만 다름  (출력: ettin_enc1b_agg_token.xlsx 등)
  python compute_cloze_surprisal_ettin_v1_word.py --arch enc --size 1b --agg token --input "20260729_csat_data_original.xlsx"
  python compute_cloze_surprisal_ettin_v1_word.py --arch enc --size 1b --agg word  --input "20260729_csat_data_original.xlsx"
  python compute_cloze_surprisal_ettin_v1_word.py --arch dec --size 1b --agg token --input "20260729_csat_data_original.xlsx"
  python compute_cloze_surprisal_ettin_v1_word.py --arch dec --size 1b --agg word  --input "20260729_csat_data_original.xlsx"

  # 확인: 같은 크기의 인코더·디코더 결과표에서 ntok1~5, nword1~5 열이 완전히 같아야 정상
  # 직접 지정도 가능: --model jhu-clsp/ettin-encoder-400m  (--arch·--size보다 우선)
  # GPU 메모리 부족(OOM) 시: 인코더는 --mask-batch 1 (값에는 영향 없음)
  #   그래도 부족하면 --precision-check 10 --dtype fp16 으로 오차 확인 후 --dtype fp16
"""

# ──────────────────────────────────────────────────────────────────────────
# 라이브러리 불러오기
# ──────────────────────────────────────────────────────────────────────────
import argparse   # 명령줄 옵션(--model, --agg 등)을 읽는 표준 라이브러리
import math       # log 등 수학 함수
import re         # 정규표현식: 문자열에서 패턴(①, 숫자, 공백 단어 등) 찾기
import sys        # 오류 메시지 출력·프로그램 종료

import numpy as np    # 숫자 배열 계산 (평균, argmin 등)
import pandas as pd   # 엑셀 읽기·쓰기
import torch          # 딥러닝 계산 (모델 실행)
from transformers import (AutoConfig,             # 모델 설정 파일(config.json) 읽기
                          AutoModelForCausalLM,   # 디코더(다음 단어 예측) 모델 불러오기
                          AutoModelForMaskedLM,   # 인코더(빈칸 예측) 모델 불러오기
                          AutoTokenizer)          # 토크나이저(문장 → 조각 번호) 불러오기

LN2 = math.log(2.0)                                     # 자연로그 → bits 변환용 (bits = nat ÷ ln2)
CIRCLED = {"①": 1, "②": 2, "③": 3, "④": 4, "⑤": 5}    # 원문자 → 숫자
DTYPES = {"fp32": torch.float32,                        # 기본 정밀도 (기존 모델들과 같음)
          "fp16": torch.float16,                        # 메모리 절약용 (오차 확인 후 사용)
          "bf16": torch.bfloat16}
ETTIN_SIZES = ["17m", "32m", "68m", "150m", "400m", "1b"]   # Ettin이 제공하는 크기


# ══════════════════════════════════════════════════════════════════════════
# 0) 모델 이름 정하기 — --arch·--size → 'jhu-clsp/ettin-encoder-150m' 같은 이름
# ══════════════════════════════════════════════════════════════════════════
def resolve_model_name(arch, size, model=None):
    """--model을 직접 줬으면 그것을 쓰고, 아니면 arch·size로 이름을 만든다."""
    if model:                                            # 직접 지정이 최우선
        return model
    kind = "encoder" if arch == "enc" else "decoder"     # enc → encoder, dec → decoder
    return f"jhu-clsp/ettin-{kind}-{size}"


def default_output_name(model_name, agg):
    """출력 파일 이름 자동 생성: ettin_enc150m_agg_word.xlsx 처럼"""
    m = re.search(r"ettin-(encoder|decoder)-(\w+)$", model_name)
    if m:                                                # Ettin 공식 이름이면 짧게
        tag = ("enc" if m.group(1) == "encoder" else "dec") + m.group(2)
    else:                                                # 그 밖의 이름(로컬 폴더 등)은 끝부분 사용
        tag = re.sub(r"[^\w\-]", "_", model_name.rstrip("/\\").split("/")[-1].split("\\")[-1])
    return f"ettin_{tag}_agg_{agg}.xlsx"


def default_mask_batch(model_name):
    """인코더 마스킹 사본을 한 번에 몇 개씩 계산할지 (값은 같고 속도·메모리만 다름)"""
    return 4 if re.search(r"-1b$", model_name) else 16   # 1b는 크니까 4, 나머지는 16


# ══════════════════════════════════════════════════════════════════════════
# 1) 엑셀에서 지문·선지 찾기 (BERT v5 / GPT-2 v4와 동일)
# ══════════════════════════════════════════════════════════════════════════
def find_col(df, *subs):
    """열 이름에 subs가 모두 들어 있는 첫 열을 찾는다. 예: find_col(df, '정답', '숫자')"""
    for c in df.columns:
        if all(s in str(c) for s in subs):
            return c
    return None


def detect_text_col(df):
    """'[MASK]'가 가장 많이 들어 있는 열 = 빈칸 지문 열"""
    best, col = 0, None
    for c in df.columns:
        cnt = df[c].astype(str).str.contains("[MASK]", regex=False, na=False).sum()
        if cnt > best:
            best, col = cnt, c
    return col


def detect_choice_cols(df):
    """열 이름에 '선지'나 '선택지'가 있는 열을 번호순으로 최대 5개"""
    cols = [c for c in df.columns if ("선택지" in str(c)) or ("선지" in str(c))]
    return sorted(cols, key=lambda c: int(m.group(1)) if (m := re.search(r"(\d)", str(c))) else 99)[:5]


def parse_choices(row, choice_cols):
    """한 문항의 선지 5개를 [①, ②, ③, ④, ⑤] 순서의 문자열 목록으로 만든다.
    한 칸에 '①... ②...'처럼 여러 선지가 붙어 있어도 원문자 기준으로 나눈다."""
    slots = {}
    for ci, col in enumerate(choice_cols, start=1):
        c = row[col]
        if c is None or str(c) == "nan":                 # 빈 칸은 건너뜀
            continue
        s = str(c).strip()
        ms = [(m.start(), CIRCLED[m.group()]) for m in re.finditer(r"[①②③④⑤]", s)]
        if not ms:                                       # 원문자가 없으면 열 순서 = 선지 번호
            slots.setdefault(ci, s)
            continue
        for i, (pos, num) in enumerate(ms):              # 원문자 사이 글자를 각 선지로
            end = ms[i + 1][0] if i + 1 < len(ms) else len(s)
            slots.setdefault(num, s[pos + 1:end].strip())
    return [slots.get(i, "").strip() for i in range(1, 6)]


def fit_window(prefix_ids, choice_ids, suffix_ids, max_len):
    """인코더용: 길이가 넘치면 선지를 가운데 두고 앞·뒤를 절반씩 자른다 (BERT v5와 같은 규칙).
    Ettin은 최대 7,997조각까지 들어가므로 수능 지문에서는 실제로 자르는 일이 없다."""
    if len(prefix_ids) + len(choice_ids) + len(suffix_ids) <= max_len:
        return prefix_ids, suffix_ids                    # 안 넘치면 그대로
    budget = max_len - len(choice_ids)                   # 선지를 넣고 남는 자리
    if budget <= 0:
        return [], []
    half = budget // 2
    return (prefix_ids[-(budget - half):] if budget - half > 0 else [],   # 앞 문맥: 선지에 가까운 쪽 보존
            suffix_ids[:half] if half > 0 else [])                       # 뒤 문맥: 선지에 가까운 쪽 보존


# ══════════════════════════════════════════════════════════════════════════
# 2) 모델 불러오기 — 종류 자동 감지 + 가중치 누락 검사
# ══════════════════════════════════════════════════════════════════════════
def detect_objective(config, name):
    """config를 보고 'mlm'(인코더) 또는 'clm'(디코더)를 돌려준다."""
    # MNTP 변형은 가린 단어를 '앞 위치'에서 예측 → 이 스크립트의 PLL 규칙과 맞지 않아 거부
    if getattr(config, "masked_prediction", False) or re.search(r"enc(oder)?-from-dec", name):
        sys.exit("[error] encoder-from-decoder(MNTP) 변형은 v1이 지원하지 않습니다. "
                 "가린 단어를 '앞 위치'에서 예측하는 방식이라 PLL 규칙으로 읽으면 틀린 값이 나옵니다.")
    archs = " ".join(getattr(config, "architectures", None) or [])
    if getattr(config, "is_causal", False) or "CausalLM" in archs \
            or config.model_type.endswith("decoder"):     # 디코더 표시가 하나라도 있으면 CLM
        return "clm"
    return "mlm"


class EttinScorer:
    """모델 + 토크나이저 + 'LM 헤드' 함수를 한 상자에 담은 클래스.
    sc = EttinScorer(...) 로 만든 뒤 sc.model, sc.tok, sc.objective 등으로 꺼내 쓴다."""

    def __init__(self, name, device, dtype, objective=None, verbose=True):
        self.name, self.device = name, device
        config = AutoConfig.from_pretrained(name)                 # config.json 읽기
        self.objective = objective or detect_objective(config, name)
        cls = AutoModelForCausalLM if self.objective == "clm" else AutoModelForMaskedLM
        if hasattr(config, "reference_compile"):                  # 옛 판본: Windows에서 torch.compile 끔
            config.reference_compile = False
        kw = dict(config=config, output_loading_info=True)        # 불러온 뒤 '누락된 가중치' 목록도 받기
        try:
            try:
                model, info = cls.from_pretrained(name, dtype=dtype, **kw)        # 최신 transformers
            except TypeError as e:
                if "dtype" not in str(e):
                    raise
                model, info = cls.from_pretrained(name, torch_dtype=dtype, **kw)  # 옛 transformers
        except (KeyError, ValueError) as e:
            sys.exit(f"[error] 모델 형식을 인식하지 못했습니다 ({e}). pip install -U transformers")

        # ★ 가중치 누락 검사: 누락된 가중치는 '무작위 값'으로 채워지므로 결과가 엉터리가 된다.
        #   (DeBERTa에서 실제로 있었던 문제. 입력 임베딩과 공유되는 출력층은 누락이 아니므로 제외)
        tied = set(getattr(model, "_tied_weights_keys", None) or [])
        missing = [k for k in info.get("missing_keys", []) if k not in tied]
        if missing:
            sys.exit(f"[error] 가중치 누락(무작위 초기화됨): {missing[:10]}")
        self.model = model.to(device).eval()                      # eval(): 드롭아웃 끄기 (결정적 계산)

        # LM 헤드 = 은닉 벡터 → 어휘 점수로 바꾸는 마지막 층. 인코더는 head, 디코더는 lm_head
        # (그 뒤에 공통으로 model.decoder 선형층이 붙는다)
        self.head = model.head if self.objective == "mlm" else model.lm_head

        self.tok = AutoTokenizer.from_pretrained(name)
        if not self.tok.is_fast:                                  # 오프셋(글자 위치) 정보는 fast 토크나이저만 제공
            sys.exit("[error] 오프셋 매핑에는 fast tokenizer가 필요합니다.")

        # 디코더 BOS 자동 판단: 토크나이저가 기본으로 맨 앞에 붙이는 특수 토큰을 찾아 그대로 쓴다
        with_sp = self.tok("a")["input_ids"]                         # 특수 토큰 포함
        without = self.tok("a", add_special_tokens=False)["input_ids"]  # 특수 토큰 없이
        k = with_sp.index(without[0]) if without and without[0] in with_sp else 0
        self.bos = with_sp[:k] if self.objective == "clm" else []     # 예: [CLS] 한 개
        self.max_pos = getattr(config, "max_position_embeddings", 512)  # Ettin = 7,999

        if verbose:
            unexpected = sorted(info.get("unexpected_keys", []))
            print(f"[check] 목표={self.objective.upper()} / 가중치 누락 0개 ✅ / 예상 밖 키 "
                  f"{len(unexpected)}개 {unexpected[:3]} / 어휘 {model.config.vocab_size} / "
                  f"최대 위치 {self.max_pos} / BOS={self.tok.convert_ids_to_tokens(self.bos) or '없음'}",
                  file=sys.stderr)

    @torch.no_grad()                                    # 학습이 아니므로 기울기 계산 끔 (메모리 절약)
    def hidden(self, ids):
        """조각 번호 → 마지막 층 은닉 벡터 [배치, 길이, 은닉차원]"""
        return self.model.model(input_ids=ids).last_hidden_state

    @torch.no_grad()
    def logits_from_hidden(self, h):
        """은닉 벡터 → 어휘 50,368개 각각의 점수(logits). 필요한 위치의 벡터만 넣는다."""
        return self.model.decoder(self.head(h))


# ══════════════════════════════════════════════════════════════════════════
# 3) 토큰화 + 공백 단어 묶기  (★ 인코더·디코더 공유 ★)
# ══════════════════════════════════════════════════════════════════════════
def tokenize_with_choice(tok, prefix, choice, suffix):
    """빈칸에 선지를 넣은 완성 문장을 한 번에 토큰화해서 네 가지를 돌려준다.
        prefix_ids : 선지 앞 지문의 조각 번호
        choice_ids : 선지의 조각 번호
        suffix_ids : 선지 뒤 지문의 조각 번호
        groups     : 선지 조각을 공백 단어별로 묶은 목록
    예) 선지 "can't circumvent it" → 조각 [Ġcan, 't, Ġcircumvent, Ġit] → groups [[0,1],[2],[3]]"""
    choice = choice.strip()
    full = prefix + choice + suffix                     # 완성 문장 (원문 띄어쓰기 그대로)
    cs, ce = len(prefix), len(prefix) + len(choice)     # 선지가 차지하는 글자 범위 [cs, ce)
    enc = tok(full, add_special_tokens=False, return_offsets_mapping=True)
    ids, offs = enc["input_ids"], enc["offset_mapping"]  # offs[i] = i번째 조각의 (시작 글자, 끝 글자)

    # 선지 조각 = 글자 범위가 선지 범위와 겹치는 조각
    pos = [i for i, (s, e) in enumerate(offs) if e > s and s < ce and e > cs]
    if not pos:
        return ids, [], [], []
    assert pos == list(range(pos[0], pos[-1] + 1)), "선지 토큰이 연속되지 않음"

    # 선지 안의 공백 단어 글자 범위. 예) "keep it" → [(cs, cs+4), (cs+5, cs+7)]
    spans = [(cs + m.start(), cs + m.end()) for m in re.finditer(r"\S+", choice)]

    def word_of(s, e):
        """조각 (s, e)가 가장 많이 겹치는 단어 번호"""
        ov = [min(e, we) - max(s, ws) for ws, we in spans]
        return int(np.argmax(ov))

    wids = [word_of(*offs[p]) for p in pos]                                       # 조각마다 단어 번호
    groups = [[k for k, w in enumerate(wids) if w == g] for g in sorted(set(wids))]  # 단어별로 묶기
    return ids[:pos[0]], ids[pos[0]:pos[-1] + 1], ids[pos[-1] + 1:], groups


# ══════════════════════════════════════════════════════════════════════════
# 4) 선지 조각별 로그확률 — 인코더(PLL) / 디코더(CLM)
# ══════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def piece_logprobs_mlm(sc, pre, ch, suf, mask_batch):
    """인코더 PLL: 선지 조각 n개마다 사본을 만들어 i번째 조각만 [MASK]로 가리고,
    가린 자리에서 원래 조각이 나올 로그확률을 읽는다. 반환: 조각별 로그확률 n개"""
    tok = sc.tok
    pre, suf = fit_window(pre, ch, suf, sc.max_pos - 2)              # [CLS]·[SEP] 두 자리 빼고
    base = [tok.cls_token_id] + pre + ch + suf + [tok.sep_token_id]  # [CLS] 지문 [SEP]
    start = 1 + len(pre)                                             # 선지 첫 조각의 위치
    n, out = len(ch), []
    for b0 in range(0, n, mask_batch):                               # 사본을 mask_batch개씩 묶어 계산
        rows = list(range(b0, min(n, b0 + mask_batch)))
        batch = torch.tensor([base] * len(rows), device=sc.device)   # 같은 문장 사본 여러 줄
        for r, i in enumerate(rows):
            batch[r, start + i] = tok.mask_token_id                  # r번째 사본은 i번째 조각만 가림
        h = sc.hidden(batch)
        sel = h[torch.arange(len(rows), device=sc.device),          # 각 사본에서 '가린 자리' 벡터만 꺼냄
                torch.tensor([start + i for i in rows], device=sc.device)]
        lp = torch.log_softmax(sc.logits_from_hidden(sel).float(), dim=-1)  # 점수 → 로그확률
        out += [lp[r, ch[i]].item() for r, i in enumerate(rows)]    # 원래 조각의 로그확률
    return out


@torch.no_grad()
def piece_logprobs_clm(sc, pre, ch):
    """디코더: [BOS]+선지 앞 지문+선지를 한 번 계산하고,
    선지의 i번째 조각 로그확률을 '바로 앞 위치'의 다음 조각 예측에서 읽는다.
    반환: (조각별 로그확률, 측정한 조각 번호 목록)"""
    ids = sc.bos + pre + ch                      # 선지 뒤 지문은 넣지 않음 (앞만 보는 모델이라 값이 같음)
    first = len(sc.bos) + len(pre)               # 선지 첫 조각의 위치
    keep = list(range(len(ch)))
    if first == 0:                               # BOS 없음 + 선지가 맨 앞 → 첫 조각은 측정 불가 (v4와 같음)
        keep = keep[1:]
    if len(ids) > sc.max_pos:                    # 너무 길면 왼쪽(지문 앞부분)을 자른다 (v4와 같은 규칙)
        cut = len(ids) - sc.max_pos
        ids, first = ids[cut:], first - cut
    if not keep:
        return [], keep
    h = sc.hidden(torch.tensor([ids], device=sc.device))[0]
    prev = torch.tensor([first + i - 1 for i in keep], device=sc.device)   # 한 칸 앞 위치들
    lp = torch.log_softmax(sc.logits_from_hidden(h.index_select(0, prev)).float(), dim=-1)
    return [lp[k, ch[i]].item() for k, i in enumerate(keep)], keep


def choice_score(sc, prefix, choice, suffix, agg="word", mask_batch=16):
    """선지 하나의 점수(평균 로그확률, nat)와 조각 수 T, 단어 수 W를 돌려준다.
    점수가 클수록(0에 가까울수록) 모델이 자연스럽다고 본 선지다."""
    pre, ch, suf, groups = tokenize_with_choice(sc.tok, prefix, choice, suffix)
    if not ch:                                                    # 빈 선지
        return float("-inf"), 0, 0
    if sc.objective == "mlm":
        lps = piece_logprobs_mlm(sc, pre, ch, suf, mask_batch)
    else:
        lps, keep = piece_logprobs_clm(sc, pre, ch)
        if not lps:
            return float("-inf"), 0, 0
        groups = [[keep.index(j) for j in g if j in keep] for g in groups]   # 측정 못 한 조각 제외
        groups = [g for g in groups if g]
    T, W = len(lps), len(groups)
    if agg == "token":
        score = sum(lps) / T                                                 # 조각 평균
    elif agg == "word":
        score = sum(sum(lps[j] for j in g) / len(g) for g in groups) / W     # 단어 내 평균 → 단어 간 평균
    else:
        raise ValueError(agg)
    return score, T, W


def softmax_pct(mean_lps, temperature=1.0):
    """선지 5개의 점수를 합이 100%인 백분율로 바꾼다 (학생 선택률과 비교용)."""
    x = np.array(mean_lps, float) / temperature
    w = np.exp(x - x.max())                                       # 큰 값 빼기 = 넘침 방지 (결과 동일)
    return (w / w.sum() * 100).tolist()


# ══════════════════════════════════════════════════════════════════════════
# 5) 구현 검증 (--selftest) — 본 실행 전 '계산기 고장 점검'
# ══════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def hand_check(sc, prefix, ch, suffix):
    """손 검산: 메모리 절약 없이 model(...).logits 전체를 계산해서 word 점수를 구한다.
    함수(choice_score)의 값과 같아야 한다 → 테스트 C"""
    pre, ids, suf, groups = tokenize_with_choice(sc.tok, prefix, ch, suffix)
    if sc.objective == "mlm":
        base = [sc.tok.cls_token_id] + pre + ids + suf + [sc.tok.sep_token_id]
        st = 1 + len(pre)
        batch = torch.tensor([base] * len(ids), device=sc.device)
        for i in range(len(ids)):
            batch[i, st + i] = sc.tok.mask_token_id
        full = sc.model(input_ids=batch).logits.float().log_softmax(-1)   # 모든 위치·모든 어휘
        pieces = [full[i, st + i, ids[i]].item() for i in range(len(ids))]
    else:
        seq = sc.bos + pre + ids + suf
        st = len(sc.bos) + len(pre)
        full = sc.model(input_ids=torch.tensor([seq], device=sc.device)).logits[0].float().log_softmax(-1)
        pieces = [full[st + i - 1, ids[i]].item() for i in range(len(ids))]
    return float(np.mean([np.mean([pieces[j] for j in g]) for g in groups]))


def run_selftest(sc, dtype_name):
    ok = True
    prefix, suffix = "I can't ", " it."
    # GPU fp32에서는 배치 모양에 따라 1e-5 수준의 누적오차가 생긴다 (DeBERTa 실측 2.6e-5)
    tol = 1e-4 if dtype_name == "fp32" else 5e-2
    print(f"\n[정보] 모델={sc.name}  목표={sc.objective.upper()}  "
          f"BOS={sc.tok.convert_ids_to_tokens(sc.bos) or '없음'}  "
          f"[MASK] id={sc.tok.mask_token_id}  [CLS]/[SEP] id={sc.tok.cls_token_id}/{sc.tok.sep_token_id}")

    # A: 단어가 하나면 '조각 평균'과 '단어 2단 평균'은 수학적으로 같아야 한다
    print("\n[테스트 A] 단일 단어 → token = word")
    for ch in ["circumvent", "dishonesty", "secrecy"]:
        a, T, W = choice_score(sc, prefix, ch, suffix, "token")
        b, _, _ = choice_score(sc, prefix, ch, suffix, "word")
        same = abs(a - b) < 1e-9
        ok &= same
        print(f"  '{ch}' T={T} W={W} token={a:+.6f} word={b:+.6f} {'✅' if same else '❌'}")

    # B: 단어 묶기가 공백 기준 단어 수와 맞는가
    print("\n[테스트 B] 단어 그룹 (공백 기준)")
    for ch in ["the circumvent", "can't circumvent it", "keep it far enough away"]:
        _, ids, _, groups = tokenize_with_choice(sc.tok, prefix, ch, suffix)
        passed = len(groups) == len(ch.split())
        ok &= passed
        print(f"  '{ch}' 조각={sc.tok.convert_ids_to_tokens(ids)} 그룹={groups} W={len(groups)} "
              f"(공백 단어 {len(ch.split())}) {'✅' if passed else '❌'}")

    # C: 메모리 절약 경로와 전체 계산 경로가 같은 값을 내는가
    print("\n[테스트 C] 손 검산 — 전체 logits 경로 vs 함수(선택 위치 헤드 경로)")
    for ch in ["keep it far enough away", "can't circumvent it"]:
        hand = hand_check(sc, prefix, ch, suffix)
        b, _, _ = choice_score(sc, prefix, ch, suffix, "word", mask_batch=2)
        same = abs(hand - b) < tol
        ok &= same
        print(f"  '{ch}' 손검산={hand:+.6f} 함수={b:+.6f} |차|={abs(hand-b):.2e} "
              f"(허용 {tol:g}) {'✅' if same else '❌'}")

    # D: 진짜 학습된 가중치인가 (무작위 가중치면 여기서 걸린다)
    print("\n[테스트 D] 사전학습 지식 점검 — 진짜 학습된 가중치인가")
    if sc.objective == "mlm":
        text = f"The capital of France is {sc.tok.mask_token}."    # [MASK]가 앞 공백을 흡수(lstrip)
        enc = sc.tok(text, return_tensors="pt").to(sc.device)
        mpos = (enc["input_ids"][0] == sc.tok.mask_token_id).nonzero()[0, 0]
        h = sc.hidden(enc["input_ids"])[0, mpos]
    else:
        text = "The capital of France is"
        ids = sc.bos + sc.tok(text, add_special_tokens=False)["input_ids"]
        h = sc.hidden(torch.tensor([ids], device=sc.device))[0, -1]
    top = sc.logits_from_hidden(h[None])[0].topk(5).indices.tolist()
    top5 = [sc.tok.decode([i]).strip() for i in top]
    passed = "Paris" in top5
    ok &= passed
    shown = text if sc.objective == "mlm" else text + " ___"
    print(f"  '{shown}' 상위 5 = {top5} {'✅' if passed else '❌ (가중치가 무작위일 가능성)'}")
    for pre_, good, bad, suf_ in [("She drank a cup of hot ", "coffee", "gravel", " this morning."),
                                  ("The children were ", "playing", "evaporate", " in the park.")]:
        g, _, _ = choice_score(sc, pre_, good, suf_, "word")
        bb, _, _ = choice_score(sc, pre_, bad, suf_, "word")
        passed = g > bb                                     # 자연스러운 쪽 점수가 더 높아야
        ok &= passed
        print(f"  {good}={-g/LN2:.2f} bits vs {bad}={-bb/LN2:.2f} bits {'✅' if passed else '❌'}")

    print("\n" + ("✅ 모든 검증 통과" if ok else "❌ 검증 실패") + "\n")
    return ok


# ══════════════════════════════════════════════════════════════════════════
# 6) 문항 읽기·채점·저장
# ══════════════════════════════════════════════════════════════════════════
def iter_items(path, limit=0):
    """엑셀의 모든 시트를 돌며 문항을 하나씩 내보낸다 (generator).
    내보내는 것: (시트명, 행, 열 이름표, 빈칸 앞 지문, 빈칸 뒤 지문, 선지 5개)"""
    xls = pd.ExcelFile(path)
    n = 0
    for sheet in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet)
        text_col, choice_cols = detect_text_col(df), detect_choice_cols(df)
        if text_col is None or len(choice_cols) < 5:
            print(f"[warn] '{sheet}' 시트 건너뜀", file=sys.stderr)
            continue
        cols = {"gold": find_col(df, "정답", "숫자") or find_col(df, "정답"),
                "grade": find_col(df, "학년"), "year": find_col(df, "연도"),
                "month": find_col(df, "월"),
                "qno": find_col(df, "문항번호") or find_col(df, "문항", "번호"),
                "err": find_col(df, "오답")}
        for _, row in df.dropna(subset=[text_col]).iterrows():
            text = str(row[text_col])
            if text.count("[MASK]") != 1:                # 빈칸이 정확히 하나인 문항만
                continue
            prefix, suffix = text.split("[MASK]")        # 빈칸 앞 / 뒤로 나누기
            yield sheet, row, cols, prefix, suffix, parse_choices(row, choice_cols)
            n += 1
            if limit and n >= limit:                     # --limit N: 앞 N문항만 (시험용)
                return


def score_item(sc, prefix, suffix, choices, agg, mb):
    """한 문항의 선지 5개를 모두 채점 → (로그확률, surprisal bits, 조각 수, 단어 수) 각 5개"""
    lps, surps, Ts, Ws = [], [], [], []
    for ch in choices:
        lp, T, W = choice_score(sc, prefix, ch, suffix, agg, mb)
        lps.append(lp)
        surps.append(-lp / LN2 if T > 0 else float("inf"))   # surprisal(bits) = −로그확률 ÷ ln2
        Ts.append(T)
        Ws.append(W)
    return lps, surps, Ts, Ws


def main():
    # ── 명령줄 옵션 ──
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", help="수능 문항 엑셀 파일")
    ap.add_argument("--arch", default="enc", choices=["enc", "dec"],
                    help="enc = 인코더(MLM), dec = 디코더(CLM). 기본 enc")
    ap.add_argument("--size", default="150m", choices=ETTIN_SIZES,
                    help="Ettin 크기. 기본 150m")
    ap.add_argument("--model", default=None,
                    help="모델 이름·폴더를 직접 지정 (주면 --arch·--size보다 우선)")
    ap.add_argument("--output", default=None,
                    help="결과 엑셀 이름. 비우면 ettin_{arch}{size}_agg_{agg}.xlsx")
    ap.add_argument("--objective", choices=["mlm", "clm"], default=None,
                    help="보통 비워 둠 (config로 자동 감지)")
    ap.add_argument("--temperature", type=float, default=1.0, help="백분율 변환 온도 (기본 1)")
    ap.add_argument("--limit", type=int, default=0, help="앞 N문항만 처리 (0 = 전부)")
    ap.add_argument("--agg", default="word", choices=["token", "word"], help="집계 방식")
    ap.add_argument("--dtype", default="fp32", choices=list(DTYPES), help="계산 정밀도")
    ap.add_argument("--mask-batch", type=int, default=None,
                    help="인코더 사본 묶음 크기. 비우면 1b=4, 그 밖=16 (값에는 영향 없음)")
    ap.add_argument("--selftest", action="store_true", help="구현 검증만 하고 끝냄")
    ap.add_argument("--precision-check", type=int, default=0, metavar="N",
                    help="앞 N문항을 지정 dtype·GPU와 fp32·CPU로 각각 계산해 차이 보고")
    args = ap.parse_args()

    # ── 모델 이름·출력 이름·사본 묶음 크기 정하기 ──
    model_name = resolve_model_name(args.arch, args.size, args.model)
    output = args.output or default_output_name(model_name, args.agg)
    mask_batch = args.mask_batch or default_mask_batch(model_name)

    # ── 계산 장치: GPU가 있으면 GPU ──
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu" and args.dtype != "fp32":
        print("[warn] CPU에서는 fp32로 강제합니다.", file=sys.stderr)
        args.dtype = "fp32"
    print(f"[info] device={device}, model={model_name}, dtype={args.dtype}, agg={args.agg}, "
          f"mask_batch={mask_batch}", file=sys.stderr)

    sc = EttinScorer(model_name, device, DTYPES[args.dtype], args.objective)
    if args.model is None and (sc.objective == "mlm") != (args.arch == "enc"):   # 이름과 실제 종류가 다르면 경고
        print(f"[warn] --arch {args.arch} 인데 모델은 {sc.objective.upper()} 입니다.", file=sys.stderr)

    # ── 셀프테스트만 하고 끝내기 ──
    if args.selftest:
        sys.exit(0 if run_selftest(sc, args.dtype) else 1)
    if not args.input:
        ap.error("--input 이 필요합니다 (또는 --selftest)")

    # ── 정밀도 점검만 하고 끝내기 (fp16을 쓰기 전에) ──
    if args.precision_check:
        print(f"[precision] 앞 {args.precision_check}문항: {args.dtype}@{device} vs fp32@cpu",
              file=sys.stderr)
        ref = EttinScorer(model_name, "cpu", torch.float32, sc.objective, verbose=False)
        diffs, agree = [], 0
        items = list(iter_items(args.input, args.precision_check))
        for _, _, _, pre, suf, chs in items:
            _, s1, _, _ = score_item(sc, pre, suf, chs, args.agg, mask_batch)
            _, s2, _, _ = score_item(ref, pre, suf, chs, args.agg, mask_batch)
            diffs += [abs(a - b) for a, b in zip(s1, s2)]
            agree += int(np.argmin(s1) == np.argmin(s2))
        print(f"[precision] 최대 |Δsurprisal| = {max(diffs):.5f} bits, 평균 = {np.mean(diffs):.5f} bits, "
              f"모델 선택 일치 {agree}/{len(items)}", file=sys.stderr)
        sys.exit(0)

    # ── 본 실행: 문항마다 채점해서 한 줄씩 기록 ──
    records = []
    for k, (sheet, row, c, pre, suf, choices) in enumerate(iter_items(args.input, args.limit), 1):
        lps, surps, Ts, Ws = score_item(sc, pre, suf, choices, args.agg, mask_batch)
        pct = softmax_pct(lps, args.temperature)
        pick = int(np.argmin(surps)) + 1                 # surprisal이 가장 낮은 선지 = 모델 선택
        gold = row[c["gold"]] if c["gold"] else None
        try:
            gold = int(gold)
        except (ValueError, TypeError):
            gold = None
        rec = {"학년": row[c["grade"]] if c["grade"] else sheet,
               "연도": row[c["year"]] if c["year"] else None,
               "월": row[c["month"]] if c["month"] else None,
               "문항번호": row[c["qno"]] if c["qno"] else None,
               "오답률": row[c["err"]] if c["err"] else None,
               "정답숫자": gold, "모델선택": pick,
               "정답여부": (pick == gold) if gold else None,
               "집계방식": args.agg, "단어경계": "whitespace",
               "모델": model_name, "목표": sc.objective.upper(),
               "BOS": "yes" if sc.bos else "no", "dtype": args.dtype}
        for i in range(5):                               # 선지별 열: 선지, surprisal, 백분율, 조각 수, 단어 수
            rec[f"선지{i+1}"] = choices[i]
            rec[f"surprisal{i+1}"] = round(surps[i], 4)
            rec[f"pct{i+1}"] = round(pct[i], 2)
            rec[f"ntok{i+1}"] = Ts[i]
            rec[f"nword{i+1}"] = Ws[i]
        if gold and 1 <= gold <= 5:
            rec["정답백분율"] = round(pct[gold - 1], 2)
        records.append(rec)
        if k % 25 == 0:
            print(f"[info] processed {k} items", file=sys.stderr)

    # ── 엑셀로 저장 + 정확도 출력 ──
    out = pd.DataFrame(records)
    out.to_excel(output, index=False)
    valid = out.dropna(subset=["정답여부"])
    if len(valid):
        print(f"[result] N={len(valid)} 정확도={valid['정답여부'].mean():.1%}", file=sys.stderr)
    print(f"[done] saved -> {output}", file=sys.stderr)


if __name__ == "__main__":   # 이 파일을 직접 실행했을 때만 main() 실행
    main()
