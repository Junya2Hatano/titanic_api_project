from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
import os, uuid

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression

# 追加インポート
from typing import List, Union, Optional
import json
from datetime import datetime
from typing import Dict, Any, List
from typing import Literal
from sklearn.preprocessing import MinMaxScaler

# 例: pydantic 入力（既存 SessionIn に use を足せるなら推奨）
from typing import Optional, Literal
from fastapi.responses import JSONResponse, StreamingResponse
# フォント設定（imports の直後、最初に一度だけ）
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

FONT_PATH = os.path.join("fonts", "IPAexGothic.ttf")  # 配置したパス
if os.path.exists(FONT_PATH):
    font_manager.fontManager.addfont(FONT_PATH)
    plt.rcParams["font.family"] = "IPAexGothic"
# マイナス記号の文字化け回避
plt.rcParams["axes.unicode_minus"] = False

# -------------------------
# App & CORS
# -------------------------
app = FastAPI(title="ML No-Code Prototype API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

STATIC_DIR = "static"
os.makedirs(STATIC_DIR, exist_ok=True)
# 先頭付近に追加
TARGET_EN = "Survived"
TARGET_JA = "死亡したかどうか"

EN2JA = {
    "PassengerId": "乗客番号",
    "Survived":    TARGET_JA,
    "Pclass":      "チケットのクラス",
    "Sex":         "性別",
    "Age":         "年齢",
    "SibSp":       "兄弟・姉妹・夫/妻の人数",
    "Parch":       "親や子どもの人数",
    "Fare":        "料金",
    "Embarked":    "乗った場所",
}
# -------------------------
# セッション毎のインメモリ状態
# -------------------------
class SessionState:
    def __init__(self):
        self.raw_df: Optional[pd.DataFrame] = None
        self.work_df: Optional[pd.DataFrame] = None
        self.features: List[str] = []
        self.target: str = TARGET_JA
        self.split: Dict[str, Any] = {}
        self.model: Optional[Any] = None
        self.model_name: Optional[str] = None
        self.split_method: str = "holdout"  # "holdout" or "cv"
        self.cv_n_splits: int = 5
        self.test_size: float = 0.2
        self.random_state: int = 42
        self.stratify: bool = True
        self.encoders: Dict[str, Any] = {}
        self.scalers: Dict[str, Any] = {}
        self.encode_logs: List[Dict[str, Any]] = []  # ←追加
        self.df_version: int = 0  # ← データ更新ごとに +1
        self.df_updated_at: str = datetime.utcnow().isoformat() + "Z"
        self.last_pred_df: Optional[pd.DataFrame] = None
        self.last_pred_csv_path: Optional[str] = None
SESSIONS: Dict[str, SessionState] = {}

def get_state(session_id: str) -> SessionState:
    if session_id not in SESSIONS:
        SESSIONS[session_id] = SessionState()
    return SESSIONS[session_id]

def _append_encode_log(state, column, method, detail=None):
    log = {
        "time": datetime.utcnow().isoformat() + "Z",
        "column": str(column),
        "method": str(method),  # "label" | "onehot" | "frequency"
        "detail": detail or {}  # 追加情報（作成列やクラス一覧など）
    }
    state.encode_logs.append(log)
    # 必要なら上限（例: 最新100件）で丸める
    state.encode_logs = state.encode_logs[-100:]
    return log
def _normalize_to_ja(df: pd.DataFrame) -> pd.DataFrame:
    to_rename = {k:v for k,v in EN2JA.items() if k in df.columns and v not in df.columns}
    if to_rename:
        df = df.rename(columns=to_rename)
    if TARGET_JA not in df.columns:
        df[TARGET_JA] = pd.NA
    return df
KAGGLE_COLS = [
    "乗客番号","死亡したかどうか","チケットのクラス","Name","性別","年齢",
    "兄弟・姉妹・夫/妻の人数","親や子どもの人数","Ticket","料金","Cabin","乗った場所"
]

def _ensure_kaggle_order(df: pd.DataFrame) -> pd.DataFrame:
    df = _normalize_to_ja(df)
    for c in KAGGLE_COLS:
        if c not in df.columns:
            df[c] = pd.NA
    return df[KAGGLE_COLS]

def _ensure_target(df: pd.DataFrame, state: SessionState):
    if state.target in df.columns:
        return
    if TARGET_EN in df.columns and state.target == TARGET_JA:
        df.rename(columns={TARGET_EN: TARGET_JA}, inplace=True)
        return
    raise HTTPException(400, f"目的変数 {state.target} が見つかりません。現列: {list(df.columns)}")
@app.get("/preprocess/encode_logs")
def get_encode_logs(session_id: str):
    state = get_state(session_id or "")
    return {"logs": state.encode_logs}

# -------------------------
# Pydantic Models
# -------------------------
# 追加
class SessIn(BaseModel):
    session_id: str

@app.post("/ping_light")
def ping_light(body: SessIn):
    return {"ok": True, "session_id": body.session_id, "ts": datetime.utcnow().isoformat() + "Z"}

class SessionIn(BaseModel):
    session_id: Optional[str] = None
    use: Optional[Literal["train", "test", "both"]] = "train"

class ColIn(SessionIn):
    column: str

class PlotIn(SessionIn):
    plot_type: str  # "count" | "pie" | "heatmap" | "pclass_by_target" | "scatter" など
    column: Optional[str] = None   # count/pie用
    x: Optional[str] = None        # scatter用
    y: Optional[str] = None        # scatter用
    hue: Optional[str] = None      # 散布図の色分け列（任意）
    alpha: Optional[float] = 0.7   # 散布図の透過（任意）
    size: Optional[int] = 25       # 散布図の点サイズ（任意）
    sample: Optional[int] = None   # サンプリング数（任意・表示軽量化））
    top_n: Optional[int] = None    # カテゴリ多すぎ対策（任意）
    bins: Optional[int] = None     # 数値列用のビン数 (例: 10)
    bin_strategy: Optional[str] = "width"  # "width" | "quantile"
    include_na: Optional[bool] = False  # 欠損を1カテゴリとして数えるか

# ▼ 追加 or 置き換え
class ImputeIn(SessionIn):
    column: str
    # mean, median, mode, ffill, bfill, constant
    strategy: str = "mean"
    constant: Optional[float] = None  # strategy=="constant" のとき使用

class EncodeIn(SessionIn):
    column: str
    # label, onehot, frequency
    method: str = "label"

class ScaleOneIn(SessionIn):
    column: str                 # ← 1列だけ
    method: str = "standard"    # "standard" or "minmax"

class FeatureAddIn(BaseModel):
    session_id: str
    new_col: str
    # A, B は「列」か「定数」のどちらか
    a_kind: Literal["col", "const"]
    b_kind: Literal["col", "const"]
    a_col: Optional[str] = None
    b_col: Optional[str] = None
    a_const: Optional[float] = None
    b_const: Optional[float] = None
    op: Literal["add", "sub", "mul", "div"]  # ＋ − × ÷

def _operand_to_series(kind: str, col: Optional[str], const: Optional[float],
                       df: pd.DataFrame) -> pd.Series:
    if kind == "col":
        if not col or col not in df.columns:
            raise HTTPException(400, f"列が正しく選べていません: {col}")
        # 数値化（文字や欠損は 0 に）
        return pd.to_numeric(df[col], errors="coerce").fillna(0)
    elif kind == "const":
        if const is None:
            raise HTTPException(400, "数値が入力されていません。")
        return pd.Series([const] * len(df), index=df.index)
    else:
        raise HTTPException(400, f"未知のkind: {kind}")

def _normalize_columns(body: FeatureAddIn, df: pd.DataFrame) -> List[str]:
    # 優先順: columns(list/str) → columns_csv
    raw = body.columns if body.columns is not None else body.columns_csv
    cols: List[str] = []
    if raw is None:
        return cols
    if isinstance(raw, list):
        cols = raw
    else:
        s = str(raw).strip()
        # ["Age","Fare"] のようなJSON文字列も受け取れるように
        if s.startswith('['):
            try:
                cols = json.loads(s)
            except Exception:
                # JSONとして壊れていたらカンマ区切りで分解
                cols = [c.strip(" '\"") for c in s.split(',') if c.strip()]
        else:
            # ただの "Age,Fare" もOK
            cols = [c.strip(" '\"") for c in s.split(',') if c.strip()]

    # 実在列だけに絞る & 重複除去
    seen = set()
    cols = [c for c in cols if c in df.columns and not (c in seen or seen.add(c))]
    return cols

# 既存の _safe_cols を使う想定
def _normalize_list_any(raw) -> List[str]:
    """list, JSON文字列, CSV どれでもリスト化。全角/半角の引用符も除去。"""
    if raw is None:
        return []
    if isinstance(raw, list):
        vals = [str(x) for x in raw]
    else:
        s = str(raw).strip()
        if not s:
            return []
        if s.startswith('['):
            try:
                vals = json.loads(s)
                if not isinstance(vals, list):
                    vals = [s]
            except Exception:
                vals = [s]
        else:
            vals = [p for p in s.split(',') if p]

    # 余計な引用符や全角カギ括弧を除去
    strip_chars = ''.join(['"', "'", '「', '」', '『', '』', '［', '］', '（', '）'])
    cleaned = []
    for v in vals:
        v = str(v).strip()
        v = v.strip(strip_chars)
        cleaned.append(v)
    return [c for c in cleaned if c]


# 入力モデルを柔軟化
class FeatureSetIn(BaseModel):
    session_id: str
    features: Optional[Union[List[str], str]] = None
    features_csv: Optional[str] = None  # 追加：CSV用

from pydantic import Field

# 追加: ホールドアウト用入力
class HoldoutIn(SessionIn):
    test_size: Optional[float] = Field(default=None, ge=0.05, le=0.9)  # 5%〜90%の範囲に制限
    stratify: Optional[bool] = True
    random_state: Optional[int] = 42

class TrainIn(SessionIn):
    model_name: str              # "decision_tree" | "logistic_regression" | "linear_regression"
    scale_numeric: bool = False
from pydantic import BaseModel, Field
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

class DecisionTreeTrainEvalIn(SessionIn):
    # どちらも未指定(None)なら scikit-learn の既定値を使います
    max_depth: Optional[int] = Field(default=None, ge=1)
    min_samples_leaf: Optional[int] = Field(default=None, ge=1)
# -------------------------
# ユーティリティ
# -------------------------
def _new_png_path(prefix: str) -> str:
    return os.path.join(STATIC_DIR, f"{prefix}_{uuid.uuid4().hex[:8]}.png")



def _safe_cols(df: pd.DataFrame, cols: List[str]) -> List[str]:
    return [c for c in cols if c in df.columns]

# -------------------------
# 1) データ読み込み
# -------------------------
from fastapi.responses import JSONResponse
import io

def _df_to_records(df: pd.DataFrame, offset: int, limit: int):
    out = df.iloc[offset: offset+limit].copy()
    for c in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[c]):
            out[c] = out[c].astype(str)
    return out.replace({np.nan: None}).to_dict(orient="records")
class PredictTestRunIn(SessionIn):
    max_depth: Optional[int] = Field(default=None, ge=1)
    min_samples_leaf: Optional[int] = Field(default=None, ge=1)
    max_rows: Optional[int] = None  # HTML 表示の行数（任意）


@app.get("/data/table_html")
def data_table_html(
    session_id: str,
    which: str = Query("work", regex="^(raw|work)$"),
    max_rows: Optional[int] = None,
    version_pass: Optional[int] = None,
):
    state = get_state(session_id or "")
    df = state.work_df if which == "work" else state.raw_df
    if df is None:
        raise HTTPException(400, "データが未ロードです。")

    _df = df.copy()
    if max_rows and max_rows > 0:
        _df = _df.head(int(max_rows))


    base_html = _df.to_html(index=False, classes="df-table", border=0, escape=False, na_rep="")

    styled_html = f"""
    <style>
    .df-wrap {{
      max-height: 520px;
      overflow-y: auto;
      border: 1px solid #e5e7eb;
      border-radius: 10px;
      box-shadow: 0 1px 3px rgba(0,0,0,.06);
      background: white;
    }}
    .df-table {{
      width: 100%;
      border-collapse: collapse;
      font-family: system-ui, -apple-system, Segoe UI, Roboto, "Helvetica Neue", Arial;
      font-size: 14px;
    }}
    .df-table thead th {{
      position: sticky; top: 0;
      background: #f8fafc;
      border-bottom: 1px solid #e5e7eb;
      text-align: left;
      padding: 10px 12px;
      z-index: 1;
    }}
    .df-table tbody td {{
      border-bottom: 1px solid #f1f5f9;
      padding: 8px 12px;
      white-space: nowrap;
      text-overflow: ellipsis;
      overflow: hidden;
    }}
    .df-table tbody tr:nth-child(even) {{ background: #fcfcfd; }}
    .df-table tbody tr:hover {{ background: #f5f7fb; }}
    </style>
    <div class="df-wrap">{base_html}</div>
    """

    return {
        "html": styled_html,
        "rows": int(len(_df)),
        "total": int(len(df)),
        "df_version": state.df_version,
        "df_updated_at": state.df_updated_at,
        "which": which,
    }

@app.get("/data/preview")
def data_preview(session_id: str, which: str = Query("work", regex="^(raw|work)$"), n:int=20):
    state = get_state(session_id or "")
    df = state.work_df if which == "work" else state.raw_df
    if df is None:
        raise HTTPException(400, "データが未ロードです。")
    n = max(1, min(n, 200))
    head = df.head(n).replace({np.nan: None}).to_dict(orient="records")
    return {
        "ok": True,
        "which": which,
        "cols": df.columns.tolist(),
        "rows": head,
        "df_version": state.df_version,
        "df_updated_at": state.df_updated_at,
    }

@app.get("/data/csv")
def data_csv(session_id: str, which: str = Query("work", regex="^(raw|work)$")):
    state = get_state(session_id or "")
    df = state.work_df if which == "work" else state.raw_df
    if df is None:
        raise HTTPException(400, "データが未ロードです。")
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    filename = f"{which}_df_v{state.df_version}.csv"
    return StreamingResponse(
        buf, media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'}
    )

@app.post("/data/reset_work")
def data_reset_work(body: SessionIn):
    state = get_state(body.session_id or "")
    if state.raw_df is None:
        raise HTTPException(400, "raw_df がありません。/dataset/load を先に。")
    state.work_df = state.raw_df.copy()
    state.df_version += 1
    state.df_updated_at = datetime.utcnow().isoformat() + "Z"
    return {"ok": True, "message": "work_df を raw_df から再作成", "df_version": state.df_version}

# これをファイル上部に追加（_ensure_kaggle_order より前）
KAGGLE_COLS = [
    "乗客番号","死亡したかどうか","チケットのクラス","Name","性別","年齢",
    "兄弟・姉妹・夫/妻の人数","親や子どもの人数","Ticket","料金","Cabin","乗った場所"
]



EXCLUDE_COLS = ["Name", "Cabin", "Ticket"]
KEEP_ORDER   = [
    "乗客番号", "死亡したかどうか", "チケットのクラス", "性別", "年齢",
    "兄弟・姉妹・夫/妻の人数", "親や子どもの人数", "料金", "乗った場所"
]

@app.post("/dataset/load")
def load_dataset(body: SessionIn):
    session_id = (body.session_id or uuid.uuid4().hex)
    use = getattr(body, "use", "train") or "train"
    state = get_state(session_id)

    # ★ UIで IsTrain を残したい場合だけ True に。既定は False（=安全側）
    include_is_train_flag = False

    base = os.path.join("data", "titanic")
    train_path = os.path.join(base, "train.csv")
    test_path  = os.path.join(base, "test.csv")

    def _read_csv(path: str) -> pd.DataFrame:
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        return pd.read_csv(path)

    try:
        if use == "train":
            df = _ensure_kaggle_order(_read_csv(train_path))
        elif use == "test":
            df = _ensure_kaggle_order(_read_csv(test_path))
            # ★ test はターゲット欠損を明示（nullable int で保持）
            df[TARGET_JA] = pd.Series([pd.NA]*len(df), dtype="Int64")
        elif use == "both":
            df_tr = _ensure_kaggle_order(_read_csv(train_path)).copy()
            df_te = _ensure_kaggle_order(_read_csv(test_path)).copy()
            # ★ ターゲットの型を train: Int64 / test: NA で揃える
            if df_tr[TARGET_JA].dtype != "Int64":
                df_tr[TARGET_JA] = pd.to_numeric(df_tr[TARGET_JA], errors="coerce").astype("Int64")
            df_te[TARGET_JA] = pd.Series([pd.NA]*len(df_te), dtype="Int64")
            # ★ フラグは“付けても”良いが既定で後段で落とす
            df_tr["IsTrain"] = True
            df_te["IsTrain"] = False
            df = pd.concat([df_tr, df_te], axis=0, ignore_index=True)
        else:
            raise HTTPException(400, f"未知のuse: {use}")

    except FileNotFoundError:
        titanic = sns.load_dataset("titanic")
        rename_map = {
            "survived": "死亡したかどうか", "pclass": "チケットのクラス", "sex": "性別",
            "age": "年齢", "sibsp": "兄弟・姉妹・夫/妻の人数", "parch": "親や子どもの人数",
            "fare": "料金", "embarked": "乗った場所"
        }
        df_all = titanic.rename(columns=rename_map)
        if "乗客番号" not in df_all.columns:
            df_all["乗客番号"] = np.arange(1, len(df_all) + 1)
        df_all = _ensure_kaggle_order(df_all).drop(columns=EXCLUDE_COLS, errors="ignore")
        df_all = df_all[[c for c in KEEP_ORDER if c in df_all.columns]]

        if use == "train":
            df = df_all.copy()
            if df[TARGET_JA].dtype != "Int64":
                df[TARGET_JA] = pd.to_numeric(df[TARGET_JA], errors="coerce").astype("Int64")

        elif use == "test":
            df = df_all.copy()
            df[TARGET_JA] = pd.Series([pd.NA]*len(df), dtype="Int64")

        elif use == "both":
            n_test = min(50, max(1, len(df_all) // 5))
            df_tr = df_all.iloc[:-n_test].copy()
            df_te = df_all.iloc[-n_test:].copy()
            # ★ 型を揃える（train: Int64 / test: NA）
            df_tr[TARGET_JA] = pd.to_numeric(df_tr[TARGET_JA], errors="coerce").astype("Int64")
            df_te[TARGET_JA] = pd.Series([pd.NA]*len(df_te), dtype="Int64")
            df_tr["IsTrain"] = True
            df_te["IsTrain"] = False
            df = pd.concat([df_tr, df_te], ignore_index=True)
        else:
            raise HTTPException(400, f"未知のuse: {use}")

    # ---- 不要列を落として並べ替え（堅牢化）----
    df = df.drop(columns=EXCLUDE_COLS, errors="ignore")

    # ★ 既定で IsTrain を落とす（UIで必要なら include_is_train_flag を True に）
    if not include_is_train_flag and "IsTrain" in df.columns:
        df = df.drop(columns=["IsTrain"], errors="ignore")

    # ★ 並び替え：定義順 + 残り
    head_cols = [c for c in KEEP_ORDER if c in df.columns]
    tail_cols = [c for c in df.columns if c not in head_cols]
    df = df[head_cols + tail_cols]

    # 状態に保存
    state.raw_df  = df.copy()
    state.work_df = df.copy()
    state.df_version += 1
    state.df_updated_at = datetime.utcnow().isoformat() + "Z"

    # 返却情報を少しリッチに（both のときだけ train/test 件数を足す）
    n_train = n_test = None
    if include_is_train_flag and "IsTrain" in df.columns:
        n_train = int((df["IsTrain"] == True).sum())
        n_test  = int((df["IsTrain"] == False).sum())

    return {
        "session_id": session_id,
        "use": use,
        "rows": int(len(df)),
        "cols": list(df.columns),
        "head": df.head(5).to_dict(orient="records"),
        "n_train": n_train, "n_test": n_test,
        "df_version": state.df_version,
        "df_updated_at": state.df_updated_at
    }

@app.post("/eda/describe_html")
def eda_describe_html(body: SessionIn):
    state = get_state(body.session_id or "")
    if state.work_df is None:
        raise HTTPException(400, "データが未ロードです。")

    try:
        desc = state.work_df.describe(include="all", datetime_is_numeric=True)
    except TypeError:
        desc = state.work_df.describe(include="all")

    desc = desc.round(2).rename(index={
        "count":"データの数","mean":"平均","std":"ばらつきの大きさ","min":"最小値",
        "25%":"下から4分の1の値","50%":"中央値","75%":"上から4分の1の値","max":"最大値",
        "unique":"ちがう種類の数","top":"最頻値","freq":"最頻値の出た回数"
    })
    # 表だけ生成（欠損表示は空）
    base_html = desc.to_html(index=True, classes="df-table table table-striped",
                             border=0, na_rep="")

    # 横スクロール用のラッパーとCSSを付ける
    min_width_px = 140 * len(desc.columns)  # 列数に応じて最小幅を確保
    styled_html = f"""
    <style>
      .df-wrap {{
        width: 100%;
        overflow-x: auto;      /* ← 横スクロール */
        -webkit-overflow-scrolling: touch; /* スマホでなめらかに */
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        background: #fff;
      }}
      .df-table {{
        margin: 0;
        border-collapse: collapse;
        table-layout: auto;    /* 内容に応じて列幅 */
        min-width: {min_width_px}px;  /* 画面より広くできる */
        font-size: 14px;
        white-space: nowrap;    /* 1セルを折り返さない（必要なら削除） */
      }}
      .df-table thead th {{
        position: sticky; top: 0;
        background: #f8fafc;
        z-index: 1;
      }}
      .df-table td, .df-table th {{ padding: 8px 12px; }}
    </style>
    <div class="df-wrap">{base_html}</div>
    """

    return JSONResponse({"html": styled_html})
    # ★ ここだけ変更：dtype を触らずに NaN の表示だけ空文字に
    html = desc.to_html(classes="table table-striped", border=0, na_rep="")
    return JSONResponse({"html": html})

# -------------------------
# 2) EDA：describe
# -------------------------


# -------------------------
# 3) EDA：欠損確認
# -------------------------
# ---- 追加：欠損テーブル（HTML） ----
@app.post("/eda/missing_html")
def eda_missing_html(body: SessionIn):
    state = get_state(body.session_id or "")
    if state.work_df is None:
        raise HTTPException(400, "データが未ロードです。")

    df = state.work_df
    miss = df.isna().sum().astype(int)
    rate = (miss / len(df) * 100).round(2)

    out = pd.DataFrame({
        "特徴量": miss.index,
        "データが存在していない数（欠損値）": miss.values,
        "欠損値の割合(%)": rate.values
    }).sort_values(["データが存在していない数（欠損値）", "欠損値の割合(%)"], ascending=False).reset_index(drop=True)

    # ---- オプション：棒グラフバーをHTMLで埋め込む（見た目UP） ----
    out["欠損値の割合(bar)"] = out["欠損値の割合(%)"].apply(
        lambda r: f'<div class="bar"><div class="bar-fill" style="width:{r}%;"></div></div><span class="num">{r:.2f}%</span>'
    )
    # ↑ 棒グラフ＋数値を同じセルに描画（escape=False で埋め込み）

    html = out[["特徴量", "データが存在していない数（欠損値）", "欠損値の割合(bar)"]].to_html(
        index=False, classes="table miss-table", border=0, escape=False
    )
    return JSONResponse({"html": html})



# -------------------------
# 4) EDA：プロット生成
# -------------------------
def _bin_series_for_plot(s: pd.Series, bins: int | None, strat: str | None):
    """数値Seriesをbinsと戦略でビニング。失敗時は等幅にフォールバック"""
    if bins is None or bins <= 1:
        bins = 10
    strat = (strat or "width").lower()
    try:
        if strat == "quantile":
            b = pd.qcut(s, q=bins, duplicates="drop")
        else:
            b = pd.cut(s, bins=bins, include_lowest=True)
    except Exception:
        b = pd.cut(s, bins=bins, include_lowest=True)
    return b.astype("category")
@app.post("/eda/plot")
def eda_plot(body: PlotIn):
    state = get_state(body.session_id or "")
    if state.work_df is None:
        raise HTTPException(400, "データが未ロードです。")

    df = state.work_df
    _ensure_target(df, state)

    plot_type = body.plot_type
    png_path = _new_png_path(plot_type)

    plt.clf()
    plt.figure(figsize=(6, 4))

    # /eda/plot の中の count 分岐を置き換え
    if plot_type == "count":
        col = body.column
        if not col or col not in df.columns:
            raise HTTPException(400, "column を指定してください。")

        s = df[col]

    # 欠損の扱い
        if not body.include_na:
            s = s.dropna()

    # 数値列ならビン分割
        if np.issubdtype(s.dtype, np.number):
        # ビン数のバリデーション（未指定は10）
            bins = body.bins if (body.bins and body.bins > 1) else 10
            strat = (body.bin_strategy or "width").lower()

            try:
                if strat == "quantile":
                # 分位ビン。重複分位で失敗することがあるので duplicates="drop"
                    s_binned = pd.qcut(s, q=bins, duplicates="drop")
                else:
                # 等幅ビン（最小～最大を等間隔）
                    s_binned = pd.cut(s, bins=bins, include_lowest=True)
            except Exception as e:
            # qcut で一意なビンが作れない時は等幅にフォールバック
                s_binned = pd.cut(s, bins=bins, include_lowest=True)

            x_series = s_binned.astype("category")
            order = x_series.cat.categories  # ビン順序を保持
            sns.countplot(x=x_series, order=order)
            plt.xticks(rotation=30, ha="right")
            plt.title(f"Count of {col} (binned: {bins}, {strat})")
            plt.tight_layout()
        else:
        # 非数値列はそのままカテゴリで集計
            x_series = s.astype("category")
        # カテゴリが多すぎる場合は上位Nだけにするなどの拡張も可
            sns.countplot(x=x_series)
            plt.xticks(rotation=15, ha="right")
            plt.title(f"Count of {col}")
            plt.tight_layout()
        plt.savefig(png_path, bbox_inches="tight")
        return {"image_url": f"/static/{os.path.basename(png_path)}"}
    elif plot_type == "pie":  # ← pie_target を汎用 pie に
        col = body.column or state.target
        if col not in df.columns:
            raise HTTPException(400, f"{col} 列が存在しません。")
        df[col].value_counts().plot(kind="pie", autopct="%.1f%%")
        plt.ylabel("")
        plt.title(f"Distribution of {col}")
        plt.tight_layout()
        plt.savefig(png_path, bbox_inches="tight")
        return {"image_url": f"/static/{os.path.basename(png_path)}"}

    elif plot_type == "heatmap":
        num_df = df.select_dtypes(include=[np.number])
        corr = num_df.corr(numeric_only=True)
        sns.heatmap(corr, annot=True, fmt=".2f",cmap="Blues")
        plt.title("Numeric Correlation Heatmap")
        plt.tight_layout()
        plt.savefig(png_path, bbox_inches="tight")
        return {"image_url": f"/static/{os.path.basename(png_path)}"}
    elif plot_type == "by_target":
        col = body.column
        if not col:
            raise HTTPException(400, "column を指定してください。")
        if col not in df.columns:
            raise HTTPException(400, f"{col} 列が存在しません。")

        s = df[col].copy()

        # 欠損の扱い（未指定は含む = True としたいなら既定値を True に）
        if not body.include_na:
            s = s.dropna()

        # ====== ここが追加：数値ならビニング ======
        if np.issubdtype(s.dtype, np.number):
            # bins, bin_strategy は body から（未指定は bins=10, strategy="width"）
            x_series = _bin_series_for_plot(
                s,
                bins=body.bins,
                strat=body.bin_strategy
            )
            order = x_series.cat.categories
        else:
            x_series = s.astype("category")
            order = None

        # 上位カテゴリだけに絞る（数値ビン時は無視してもOK。適用するなら value_counts 基準）
        if body.top_n and body.top_n > 0 and order is None:
            top = x_series.value_counts().nlargest(body.top_n).index
            x_series = x_series.where(x_series.isin(top), other="__OTHER__")

        hue_series = df[state.target].astype("category")

        plt.clf(); plt.figure(figsize=(6,4))
        sns.countplot(x=x_series, hue=hue_series, order=order)
        title = f"{col} by {state.target}"
        if np.issubdtype(df[col].dtype, np.number):
            bs = body.bins if (body.bins and body.bins > 1) else 10
            strat = (body.bin_strategy or "width").lower()
            title += f" (binned: {bs}, {strat})"
        plt.title(title)
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(png_path, bbox_inches="tight")
        return {"image_url": f"/static/{os.path.basename(png_path)}"}


    # --- ここから追加：散布図 ---
    elif plot_type == "scatter":
        if not body.x or not body.y:
            raise HTTPException(400, "scatter には x と y を指定してください。")
        if body.x not in df.columns or body.y not in df.columns:
            raise HTTPException(400, f"指定された x または y 列が存在しません。")

        cols = [body.x, body.y] + ([body.hue] if body.hue else [])
        df_plot = df[cols].dropna()

    # x, y を可能なら数値に変換（文字列数値にも対応）
        def ensure_numeric(s: pd.Series, name: str) -> pd.Series:
            if np.issubdtype(s.dtype, np.number):
                return s
        # 文字列→数値変換を試す（"12.3" など）
            s2 = pd.to_numeric(s, errors="coerce")
            if s2.notna().any():
                return s2
            raise HTTPException(400, f"{name} は数値列ではありません（変換不可）。")

        x_series = ensure_numeric(df_plot[body.x], body.x)
        y_series = ensure_numeric(df_plot[body.y], body.y)

        # hue はカテゴリ扱い（任意）
        hue_series = None
        if body.hue:
            if body.hue not in df_plot.columns:
                raise HTTPException(400, f"hue 列 {body.hue} が存在しません。")
            hue_series = df_plot[body.hue].astype("category")

        # サンプリング（任意）
        if body.sample and len(df_plot) > body.sample:
            df_plot = df_plot.sample(body.sample, random_state=42)
            x_series = x_series.loc[df_plot.index]
            y_series = y_series.loc[df_plot.index]
            if hue_series is not None:
                hue_series = hue_series.loc[df_plot.index]

    # プロット
        plt.clf()
        plt.figure(figsize=(6, 4))
        sns.scatterplot(
            x=x_series, y=y_series,
            hue=hue_series if body.hue else None,
            alpha=body.alpha if body.alpha is not None else 0.7,
            s=body.size if body.size is not None else 25,
            edgecolor="none"
    )
        plt.xlabel(body.x)
        plt.ylabel(body.y)
        title = f"{body.y} vs {body.x}"
        if body.hue:
            title += f" (hue: {body.hue})"
        plt.title(title)
 

        plt.savefig(png_path, bbox_inches="tight")
        return {"image_url": f"/static/{os.path.basename(png_path)}"}


# 静的ファイルを配信
from fastapi.staticfiles import StaticFiles
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
@app.post("/dataset/columns")
def dataset_columns(body: SessionIn):
    state = get_state(body.session_id or "")
    if state.work_df is None:
        raise HTTPException(400, "データが未ロードです。")
    df = state.work_df
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in df.columns if c not in num_cols]
    return {"cols": df.columns.tolist(), "numeric": num_cols, "categorical": cat_cols}

# -------------------------
# 5) 前処理：平均値補完
# -------------------------
@app.post("/preprocess/impute")
def preprocess_impute(body: ImputeIn):
    state = get_state(body.session_id or "")
    df = state.work_df
    if df is None:
        raise HTTPException(400, "データが未ロードです。")
    col = body.column
    if col not in df.columns:
        raise HTTPException(400, f"{col} が存在しません。")

    strategy = body.strategy.lower()
    info = {"column": col, "strategy": strategy}

    # 型チェック（平均・中央値は数値列のみ許可）
    if strategy in ["mean", "median"] and df[col].dtype.kind not in "if":
        raise HTTPException(400, f"{col} は数値列ではありません。{strategy} は数値列のみ対応。")

    if strategy == "mean":
        val = df[col].mean()
        df[col] = df[col].fillna(val)
        info["value"] = float(val) if pd.notna(val) else None

    elif strategy == "median":
        val = df[col].median()
        df[col] = df[col].fillna(val)
        info["value"] = float(val) if pd.notna(val) else None

    elif strategy == "mode":
        # 最頻値（複数あれば先頭）
        modes = df[col].mode(dropna=True)
        if len(modes) == 0:
            raise HTTPException(400, f"{col} の最頻値が計算できません。")
        val = modes.iloc[0]
        df[col] = df[col].fillna(val)
        info["value"] = val if isinstance(val, (int, float)) else str(val)

    elif strategy == "ffill":
        df[col] = df[col].fillna(method="ffill")

    elif strategy == "bfill":
        df[col] = df[col].fillna(method="bfill")

    elif strategy == "constant":
        if body.constant is None and df[col].dtype.kind in "if":
            raise HTTPException(400, "数値列の定数補完には 'constant' を指定してください。")
        const_val = body.constant if body.constant is not None else ""
        df[col] = df[col].fillna(const_val)
        info["value"] = const_val

    else:
        raise HTTPException(400, f"未知のstrategy: {strategy}")

    state.work_df = df
    value_txt = ""
    if "value" in info and info["value"] is not None:
        value_txt = f" {info['value']}"
    message = f"{col} ({strategy}{value_txt})"

    if not hasattr(state, "impute_logs") or state.impute_logs is None:
        state.impute_logs = []
    state.impute_logs.append(message)
    state.df_version += 1
    state.df_updated_at = datetime.utcnow().isoformat() + "Z"

    return {
        "ok": True,
        **info,
        "message": message,
        "logs": state.impute_logs,   # ← リストで返す
            "df_version": state.df_version,          # ★追加
    "df_updated_at": state.df_updated_at, 
    }

# -------------------------
# 6) 前処理：LabelEncoding
# -------------------------
@app.post("/preprocess/scale_one")
def preprocess_scale_one(body: ScaleOneIn):
    state = get_state(body.session_id or "")
    df = state.work_df
    if df is None:
        raise HTTPException(400, "データが未ロードです。")

    col = body.column
    if col not in df.columns:
        raise HTTPException(400, f"{col} が存在しません。")

    # 数値列のみ対象
    if df[col].dtype.kind not in "if":
        raise HTTPException(400, f"{col} は数値列ではありません。スケーリング対象は数値列のみです。")

    # 履歴バッファ（リスト）を用意
    if not hasattr(state, "scale_logs") or state.scale_logs is None:
        state.scale_logs = []

    method = body.method.lower()
    if method == "standard":
        scaler = StandardScaler()
        df[col] = scaler.fit_transform(df[[col]])
        state.scalers[col] = {"method": "standard", "scaler": scaler}
        message = f"{col} (standard: mean→0, std→1)"

    elif method == "minmax":
        scaler = MinMaxScaler()
        df[col] = scaler.fit_transform(df[[col]])
        state.scalers[col] = {"method": "minmax", "scaler": scaler}
        message = f"{col} (minmax: scaled to [0,1])"

    else:
        raise HTTPException(400, f"未知のmethod: {method}")

    state.work_df = df

    # 履歴に追記
    state.scale_logs.append(message)
    state.df_version += 1
    state.df_updated_at = datetime.utcnow().isoformat() + "Z"

    return {
        "ok": True,
        "method": method,
        "column": col,
        "message": message,      # ← Bubble で :plus item しやすい
        "logs": state.scale_logs, # ← 必要ならRGにまとめて表示
            "df_version": state.df_version,          # ★追加
    "df_updated_at": state.df_updated_at, 
    }

from datetime import datetime
from sklearn.preprocessing import LabelEncoder

@app.post("/preprocess/encode")
def preprocess_encode(body: EncodeIn):
    state = get_state(body.session_id or "")
    df = state.work_df
    if df is None:
        raise HTTPException(400, "データが未ロードです。")
    col = body.column
    if col not in df.columns:
        raise HTTPException(400, f"{col} が存在しません。")

    method = body.method.lower()

    # --- 履歴バッファの用意（impute と同じ）-----------------------------
    if not hasattr(state, "encode_logs") or state.encode_logs is None:
        state.encode_logs = []  # 文字列のリスト
    # ------------------------------------------------------------------

    if method == "label":
        le = LabelEncoder()
        df[col] = df[col].astype(str)
        df[col] = le.fit_transform(df[col])
        state.encoders[col] = le
        state.work_df = df

        classes = list(map(str, le.classes_))
        # --- message を作る（impute と同じ形式）------------------------
        message = f"{col} (label: {len(classes)} classes)"
        state.encode_logs.append(message)
        # ---------------------------------------------------------------
        state.df_version += 1
        state.df_updated_at = datetime.utcnow().isoformat() + "Z"

        return {
            "ok": True,
            "column": col,
            "method": "label",
            "classes": classes,
            "message": message,          # ← Bubble で :plus item しやすい
            "logs": state.encode_logs,   # ← まとめて返す（必要なら参照）
            "time": datetime.utcnow().isoformat() + "Z",
                "df_version": state.df_version,          # ★追加
    "df_updated_at": state.df_updated_at, 
        }

    elif method == "onehot":
        cats = df[col].astype(str).unique().tolist()
        if len(cats) > 50:
            raise HTTPException(400, f"{col} のカテゴリ数が多すぎます（{len(cats)}）。one-hotは50以下推奨。")

        dummies = pd.get_dummies(df[col].astype(str), prefix=col, dummy_na=False)
        new_cols = dummies.columns.tolist()
        df = pd.concat([df.drop(columns=[col]), dummies], axis=1)

        state.encoders[col] = {"method": "onehot", "cols": new_cols}
        state.work_df = df

        # --- message ---------------------------------------------------
        message = f"{col} (onehot: {len(new_cols)} cols)"
        state.encode_logs.append(message)
        # ---------------------------------------------------------------
        state.df_version += 1
        state.df_updated_at = datetime.utcnow().isoformat() + "Z"

        return {
            "ok": True,
            "column": col,
            "method": "onehot",
            "created_columns": new_cols,
            "n_created": len(new_cols),
            "message": message,
            "logs": state.encode_logs,
            "time": datetime.utcnow().isoformat() + "Z",
                "df_version": state.df_version,          # ★追加
    "df_updated_at": state.df_updated_at, 
        }

    elif method == "frequency":
        # 相対頻度（0〜1）
        freq = df[col].astype(str).value_counts(normalize=True).to_dict()
        df[col] = df[col].astype(str).map(freq)

        state.encoders[col] = {"method": "frequency", "map": freq}
        state.work_df = df

        # --- message ---------------------------------------------------
        message = f"{col} (frequency: {len(freq)} cats)"
        state.encode_logs.append(message)
        # ---------------------------------------------------------------
        state.df_version += 1
        state.df_updated_at = datetime.utcnow().isoformat() + "Z"

        return {
            "ok": True,
            "column": col,
            "method": "frequency",
            "unique_categories": len(freq),
            "message": message,
            "logs": state.encode_logs,
            "time": datetime.utcnow().isoformat() + "Z",
                "df_version": state.df_version,          # ★追加
    "df_updated_at": state.df_updated_at, 
        }

    else:
        raise HTTPException(400, f"未知のmethod: {method}")
   


# === 新特徴量生成：足し算 ===
@app.post("/feature/add")
def feature_add(body: FeatureAddIn):
    state = get_state(body.session_id or "")
    df = state.work_df
    if df is None:
        raise HTTPException(400, "データが未ロードです。")

    # 新列名チェック
    new_col = body.new_col.strip()
    if not new_col:
        raise HTTPException(400, "新しい特徴量の名前を入力してください。")
    if new_col in df.columns:
        raise HTTPException(400, f"{new_col} はすでに存在します。別の名前にしてください。")

    # オペランド取得
    a = _operand_to_series(body.a_kind, body.a_col, body.a_const, df)
    b = _operand_to_series(body.b_kind, body.b_col, body.b_const, df)

    # 四則演算
    zero_division_fixed = False
    if body.op == "add":
        out = a + b
        msg_op = "＋"
    elif body.op == "sub":
        out = a - b
        msg_op = "−"
    elif body.op == "mul":
        out = a * b
        msg_op = "×"
    elif body.op == "div":
        with np.errstate(divide="ignore", invalid="ignore"):
            out = a / b
        # 無限大・NaN は 0 に置換
        bad_mask = ~np.isfinite(out.fillna(0))
        if bad_mask.any() or out.isna().any():
            out = out.replace([np.inf, -np.inf], np.nan).fillna(0)
            zero_division_fixed = True
        msg_op = "÷"
    else:
        raise HTTPException(400, f"未知のop: {body.op}")

    # 生成して保存
    df[new_col] = out
    state.work_df = df

    # 履歴（任意）
    if not hasattr(state, "feature_logs") or state.feature_logs is None:
        state.feature_logs = []
    # A/B の表示用文字列
    def _disp(kind, col, const):
        return col if kind == "col" else str(const)
    a_disp = _disp(body.a_kind, body.a_col, body.a_const)
    b_disp = _disp(body.b_kind, body.b_col, body.b_const)

    message = f"{new_col} = {a_disp} {msg_op} {b_disp}"
    if zero_division_fixed and body.op == "div":
        message += "（0除算は0に）"
    state.feature_logs.append(message)
    state.df_version += 1
    state.df_updated_at = datetime.utcnow().isoformat() + "Z"

    return {
        "ok": True,
        "new_col": new_col,
        "message": message,
        "logs": state.feature_logs,
            "df_version": state.df_version,          # ★追加
    "df_updated_at": state.df_updated_at, 
    }

# -------------------------
# 7) 特徴量選択
# -------------------------
# ---- 置き換え/追加: 安全な列フィルタ（target除外 & 重複除去） ----
def _safe_feature_cols(df: pd.DataFrame, cols: list[str], target: str) -> list[str]:
    seen = set()
    out = []
    for c in cols:
        if c == target:
            continue
        if c in df.columns and c not in seen:
            seen.add(c)
            out.append(c)
    return out

# ---- 置き換え: /features/set ----
@app.post("/features/set")
def features_set(body: FeatureSetIn):
    state = get_state(body.session_id or "")
    df = state.work_df
    if df is None:
        raise HTTPException(400, "データが未ロードです。")

    raw = body.features if body.features is not None else body.features_csv
    feat_list = _normalize_list_any(raw)

    use_cols = _safe_feature_cols(df, feat_list, state.target)
    if not use_cols:
        raise HTTPException(400, "使える列がありません（targetや存在しない列は除外されます）。")
        # 追加：存在しない列が混じっていないか明示チェック
    missing = [c for c in feat_list if c not in df.columns and c != state.target]
    if missing:
        raise HTTPException(400, f"存在しない列が含まれています: {missing}。/dataset/columns で列名を確認し、全角の引用符などを外してください。")

    state.features = use_cols

    # 特徴量が変わったら、過去の分割＆モデルは無効化（再実行を促す）
    state.split = {}
    state.model = None
    state.model_name = None

    return {"ok": True, "features": use_cols}



def _do_holdout_split(state):
    df = state.work_df
    if df is None:
        raise HTTPException(400, "データが未ロードです。")
    _ensure_target(df, state)
    if not state.features:
        raise HTTPException(400, "featuresが未設定です。/features/set を先に。")

    X = df[state.features].copy()
    y = df[state.target].copy()

    # 目的変数の NaN を必ず除去
    na_mask = y.isna()
    if na_mask.any():
        X = X.loc[~na_mask]
        y = y.loc[~na_mask]

    if len(y) == 0:
        raise HTTPException(400, "目的変数がすべて欠損です。")

    # y を数値化（True/False, "0"/"1" 等も吸収）→整数化
    y = pd.to_numeric(y, errors="coerce")
    if y.isna().any():
        raise HTTPException(400, "目的変数に数値化できない値が含まれています。")
    # 連続値が入っているケースを検出（0/1 以外がある）
    uniq = set(pd.unique(y))
    if not uniq.issubset({0, 1}):
        raise HTTPException(400, f"目的変数が 0/1 の離散値ではありません（値: {sorted(list(uniq))}）。"
                                 "ターゲット列を前処理していないか確認してください。")
    y = y.astype(int)

    stratify_arg = y if state.stratify else None

    try:
        X_tr, X_va, y_tr, y_va = train_test_split(
            X, y, test_size=state.test_size,
            random_state=state.random_state, stratify=stratify_arg
        )
    except ValueError:
        # stratify が無理なら外して再実行
        X_tr, X_va, y_tr, y_va = train_test_split(
            X, y, test_size=state.test_size,
            random_state=state.random_state, stratify=None
        )
        state.stratify = False

    state.split = {
        "X_train": X_tr, "X_valid": X_va,
        "y_train": y_tr, "y_valid": y_va,
        "train_size": len(X_tr), "valid_size": len(X_va),
        "ratio": state.test_size
    }
    return {"ok": True, "train_size": len(X_tr), "valid_size": len(X_va), "ratio": state.test_size}




# 置き換え: /split/holdout を「設定＋分割」まで一発で
@app.post("/split/holdout")
def split_holdout(body: HoldoutIn):
    state = get_state(body.session_id or "")
    df = state.work_df
    if df is None:
        raise HTTPException(400, "データが未ロードです。")

    # 渡ってきた値を優先して state に反映（未指定は既存の state 値を使用）
    if body.test_size is not None:
        state.test_size = float(body.test_size)
    if body.stratify is not None:
        state.stratify = bool(body.stratify)
    if body.random_state is not None:
        state.random_state = int(body.random_state)

    # モードは常に holdout
    state.split_method = "holdout"

    # 実行
    return _do_holdout_split(state)
from sklearn.impute import SimpleImputer

@app.post("/model/decision_tree/train_eval")
def decision_tree_train_eval(body: DecisionTreeTrainEvalIn):
    state = get_state(body.session_id or "")
    df = state.work_df
    if df is None:
        raise HTTPException(400, "データが未ロードです。")
    _ensure_target(df, state)
    if not state.features:
        raise HTTPException(400, "featuresが未設定です。/features/set を先に。")

    # --- 必要なら holdout を作る（test_size=0.2固定） ---
    state.split_method = "holdout"
    state.test_size = 0.2
    state.stratify = True if df[state.target].nunique() <= 20 else False
    state.random_state = 42

    if not state.split:
        _ = _do_holdout_split(state)
        # features が本当に存在するか最終チェック
    missing = [c for c in state.features if c not in df.columns]
    if missing:
        raise HTTPException(400, f"特徴量に存在しない列があります: {missing}。/features/set をやり直してください。")

    X_train = state.split["X_train"].copy(); y_train = state.split["y_train"].copy()
    X_valid = state.split["X_valid"].copy(); y_valid = state.split["y_valid"].copy()
    sorted(pd.unique(state.split["y_train"]))

    # ★ 非数値列が残っていないかチェック（残っていたら前処理を促す）
    non_num = [c for c in X_train.columns if not np.issubdtype(X_train[c].dtype, np.number)]
    if non_num:
        raise HTTPException(
            400,
            f"特徴量が数値化されていません: {non_num}。/preprocess/encode などで数値化してから再実行してください。"
        )

    # ★ NaN を簡易補完（最頻値）。運用で変えたければここを差し替え。
    if X_train.isna().any().any() or X_valid.isna().any().any():
        imputer = SimpleImputer(strategy="most_frequent")
        X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
        X_valid = pd.DataFrame(imputer.transform(X_valid), columns=X_valid.columns, index=X_valid.index)

    # --- 決定木（ハイパラは2つだけ）---
    dt_params: Dict[str, Any] = {"random_state": state.random_state}
    if body.max_depth is not None:
        dt_params["max_depth"] = int(body.max_depth)
    if body.min_samples_leaf is not None:
        dt_params["min_samples_leaf"] = int(body.min_samples_leaf)

    model = DecisionTreeClassifier(**dt_params)
    model.fit(X_train, y_train)

    # 検証スコア
    y_pred = model.predict(X_valid)
    acc = float(accuracy_score(y_valid, y_pred))

    state.model_name = "decision_tree"
    state.model = model

    return {
        "ok": True,
        "model": "decision_tree",
        "params": {
            "max_depth": model.get_params().get("max_depth", None),
            "min_samples_leaf": model.get_params().get("min_samples_leaf", 1),
        },
        "holdout": {
            "test_size": state.test_size,
            "train_size": state.split["train_size"],
            "valid_size": state.split["valid_size"],
            "stratify": state.stratify,
        },
        "valid_accuracy": acc
    }

# === 学習時の前処理をテストにも適用するヘルパー ===
def _apply_trained_preprocess(df: pd.DataFrame, state: SessionState) -> pd.DataFrame:
    out = df.copy()

    # 1) エンコード（学習時に保存した encoders を利用）
    for col, enc in state.encoders.items():
        if isinstance(enc, LabelEncoder):
            # 未知値は一旦文字列化→既知クラス以外は '___UNK___' 等で埋める
            s = out[col].astype(str)
            known = set(enc.classes_.tolist())
            s = s.where(s.isin(known), other=list(known)[0])  # 簡易に既知の先頭に寄せる
            out[col] = enc.transform(s)

        elif isinstance(enc, dict) and enc.get("method") == "onehot":
            base_cols = enc["cols"]  # 学習時に作られた列名リスト
            dummies = pd.get_dummies(out[col].astype(str), prefix=col, dummy_na=False)
            # 学習時に存在した列だけを揃える（無い列は0で作る）
            for c in base_cols:
                if c not in dummies.columns:
                    dummies[c] = 0
            dummies = dummies[base_cols]
            out = pd.concat([out.drop(columns=[col]), dummies], axis=1)

        elif isinstance(enc, dict) and enc.get("method") == "frequency":
            mp = enc["map"]
            out[col] = out[col].astype(str).map(mp).fillna(0.0)

    # 2) スケーリング（学習時に保存した scalers を利用）
    for col, info in state.scalers.items():
        scaler = info["scaler"]
        if col in out.columns:
            # 欠損は 0 埋め等、学習と同じ流儀に合わせる（ここでは0）
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0)
            out[col] = scaler.transform(out[[col]])

    return out
def _styled_html_table(df: pd.DataFrame) -> str:
    # 既存の /data/table_html と同じ雰囲気の見た目
    base_html = df.to_html(index=False, classes="df-table", border=0, escape=False)
    return f"""
    <style>
      .df-wrap {{
        max-height: 520px; overflow-y: auto;
        border: 1px solid #e5e7eb; border-radius: 10px;
        box-shadow: 0 1px 3px rgba(0,0,0,.06); background: white;
      }}
      .df-table {{ width:100%; border-collapse:collapse; font-size:14px; }}
      .df-table thead th {{
        position:sticky; top:0; background:#f8fafc; border-bottom:1px solid #e5e7eb;
        text-align:left; padding:10px 12px; z-index:1;
      }}
      .df-table tbody td {{
        border-bottom:1px solid #f1f5f9; padding:8px 12px; white-space:nowrap;
        text-overflow:ellipsis; overflow:hidden;
      }}
      .df-table tbody tr:nth-child(even) {{ background:#fcfcfd; }}
      .df-table tbody tr:hover {{ background:#f5f7fb; }}
    </style>
    <div class="df-wrap">{base_html}</div>
    """

@app.post("/predict/test_run")  # ← POST に統一
def predict_test_run(body: PredictTestRunIn):
    state = get_state(body.session_id or "")
    df = state.work_df
    if df is None:
        raise HTTPException(400, "データが未ロードです。/dataset/load を先に。")
    _ensure_target(df, state)
    if not state.features:
        raise HTTPException(400, "featuresが未設定です。/features/set を先に。")

    # 学習用とテスト用の切り分け（目的変数が欠損=テスト）
    train_mask = df[state.target].notna()
    test_mask  = ~train_mask
    if test_mask.sum() == 0:
        raise HTTPException(400, "testデータが見つかりません。/dataset/load で 'test' または 'both' を選択してください。")
        # features が本当に存在するか最終チェック
    missing = [c for c in state.features if c not in df.columns]
    if missing:
        raise HTTPException(400, f"特徴量に存在しない列があります: {missing}。/features/set をやり直してください。")

    X_tr = df.loc[train_mask, state.features].copy()
    y_tr = df.loc[train_mask, state.target].copy()
    X_te = df.loc[test_mask,  state.features].copy()

    # ---------------- ここから追加（入れる位置はここ） ----------------
    # y を 0/1 の int に揃える（文字列/True/False/nullable も吸収）
    y_tr = pd.to_numeric(y_tr, errors="coerce")
    if y_tr.isna().any():
        raise HTTPException(400, "学習用の目的変数に数値化できない値があります。")
    uniq = set(pd.unique(y_tr))
    if not uniq.issubset({0, 1}):
        raise HTTPException(400, f"学習用の目的変数が 0/1 ではありません（値: {sorted(list(uniq))}）。"
                                 "ターゲット列を前処理していないか確認してください。")
    y_tr = y_tr.astype(int)

    # 特徴量が数値だけかチェック（残っていたら前処理を促す）
    non_num = [c for c in X_tr.columns if not np.issubdtype(X_tr[c].dtype, np.number)]
    if non_num:
        raise HTTPException(
            400,
            f"特徴量が数値化されていません: {non_num}。/preprocess/encode などで数値化してください。"
        )

    # 欠損があれば簡易補完（最頻値で埋める）
    if X_tr.isna().any().any() or X_te.isna().any().any():
        from sklearn.impute import SimpleImputer  # 既に上で import 済みならこの行は不要
        imp = SimpleImputer(strategy="most_frequent")
        X_tr = pd.DataFrame(imp.fit_transform(X_tr), columns=X_tr.columns, index=X_tr.index)
        X_te = pd.DataFrame(imp.transform(X_te), columns=X_te.columns, index=X_te.index)
    # ---------------- 追加はここまで ----------------------------------

    # 決定木（max_depth / min_samples_leaf のみ許可）
    dt_params = {"random_state": state.random_state}
    if body.max_depth is not None:
        dt_params["max_depth"] = int(body.max_depth)
    if body.min_samples_leaf is not None:
        dt_params["min_samples_leaf"] = int(body.min_samples_leaf)

    model = DecisionTreeClassifier(**dt_params)
    model.fit(X_tr, y_tr)

    pred = model.predict(X_te).astype(int)

    # 表示用（日本語）と提出用（英語）の両方を作る
    out_ja = pd.DataFrame({
        "乗客番号": df.loc[test_mask, "乗客番号"].values,
        TARGET_JA: pred
    })
    out_en = pd.DataFrame({
        "PassengerId": out_ja["乗客番号"].values,
        "Survived":    out_ja[TARGET_JA].values
    })

    # CSV 保存（/static に書き出す）
    fname = f"predict_{uuid.uuid4().hex[:8]}.csv"
    csv_path = os.path.join(STATIC_DIR, fname)
    out_en.to_csv(csv_path, index=False)

    # HTML（max_rows 指定があれば先頭だけ表示）
    show = out_ja.copy()
    if body.max_rows and body.max_rows > 0:
        show = show.head(int(body.max_rows))
    html = _styled_html_table(show)

    # 状態に最後の学習済みモデルを格納（任意）
    state.model = model
    state.model_name = "decision_tree"

    return {
        "ok": True,
        "n_test": int(len(out_ja)),
        "csv_url": f"/static/{fname}",
        "html": html,
        "params": dt_params,
        "features": state.features,
    }
