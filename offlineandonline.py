# app.py
from fastapi import FastAPI, APIRouter, Body, Query, HTTPException
from fastapi.responses import JSONResponse
from pathlib import Path
from typing import Optional, List, Iterable
from collections import defaultdict
import csv, json, os, stat, time

# optional EDA helper
try:
    from services import common_eda as ceda
except Exception:
    ceda = None

from services import leaderboard as lb
from services import mlkit
from services.tasks import region_spec


def make_region_router(region: str) -> APIRouter:
    r = APIRouter(prefix=f"/v1/datasets/{region}", tags=[region])

    # -------- helpers (この関数内で完結させる) --------
    def _feats_by_level(lv: int) -> list[str]:
        f = ["Evaluate", "Submit"]
        if lv >= 2: f += ["EDA", "Model", "Train", "Predict"]
        if lv >= 3: f += ["Preprocess"]
        if lv >= 4: f += ["Scaling"]
        if lv >= 5: f += ["Hyperparam"]
        if lv >= 6: f += ["CV", "Boosting"]
        if lv >= 7: f += ["Ensemble"]
        if lv >= 8: f += ["Stacking"]
        return f

    def _normalize_scaling(x) -> str:
        s = (x or "none").strip().lower()
        return "standard" if s in ("std", "standard", "zscore") else ("minmax" if s in ("minmax", "min-max") else "none")

    def _parse_models(x) -> list[str]:
        if x is None:
            return []
        if isinstance(x, str):
            parts = [p.strip() for p in x.split(",") if p.strip()]
        elif isinstance(x, Iterable):
            parts = [str(p).strip() for p in x if str(p).strip()]
        else:
            parts = [str(x).strip()]
        # 重複除去（順序維持）
        seen, out = set(), []
        for p in parts:
            if p not in seen:
                out.append(p); seen.add(p)
        return out

    def _ceda(name: str, **kwargs):
        if not ceda or not hasattr(ceda, name):
            return None
        from inspect import signature
        sig = signature(getattr(ceda, name))
        filt = {k: v for k, v in kwargs.items() if k in sig.parameters}
        return getattr(ceda, name)(**filt)

    def _safe_rmtree(path: Path) -> bool:
        def _on_err(func, p, exc_info):
            try:
                os.chmod(p, stat.S_IWRITE)
                func(p)
            except Exception:
                pass
        for _ in range(3):
            try:
                path.exists() and __import__("shutil").rmtree(path, onerror=_on_err)
                return True
            except FileNotFoundError:
                return True
            except PermissionError:
                time.sleep(0.2)
        try:
            trash = path.parent / f"__trash_{path.name}_{int(time.time()*1000)}"
            path.rename(trash)
            __import__("shutil").rmtree(trash, onerror=_on_err)
            return True
        except Exception:
            return False

    def _csv_path(region_: str, problem: str) -> Path:
        """フォールバック探索つきで CSV を解決"""
        p = Path(f"storage/datasets/{region_}/{problem}.csv")
        if p.exists():
            return p
        root = Path("storage/datasets")
        if root.exists() and root.is_dir():
            for sub in root.iterdir():
                q = sub / f"{problem}.csv"
                if q.exists():
                    return q
        return p  # 最終的に元を返す

    def _read_rows(p: Path) -> list[dict]:
        if not p.exists():
            return []
        with p.open(encoding="utf-8") as f:
            rdr = csv.DictReader(f)
            return [dict(r) for r in rdr]

    def _infer_types(rows: list[dict]) -> dict[str, str]:
        if not rows:
            return {}
        cols = rows[0].keys()
        out: dict[str, str] = {}
        for c in cols:
            vals = [r.get(c) for r in rows if r.get(c) not in (None, "", "NaN")]
            is_num = True
            for v in vals:
                try:
                    float(v)
                except Exception:
                    is_num = False
                    break
            out[c] = "number" if is_num else "category"
        return out

    # ----- task -----
    @r.get("/task")
    def task():
        spec = region_spec(region) or {}
        lv = int(spec.get("level", 1))
        feats = _feats_by_level(lv)

        raw_probs = spec.get("problems", [])
        problems_detail = []
        for p in raw_probs:
            if isinstance(p, dict):
                d = {"slug": p.get("slug")}
                for k, v in p.items():
                    if k != "slug":
                        d[k] = v
                problems_detail.append(d)
            else:
                slug = str(p)
                problems_detail.append({"slug": slug, "name": slug.title()})

        if not problems_detail:
            problems_detail = [{"slug": region, "name": region.title()}]

        problems = [p["slug"] for p in problems_detail]

        resp = {
            "region": region,
            "level": lv,
            "allowed_features": feats,
            "allowed_models": spec.get("allowed_models", ["LogisticRegression", "DecisionTree", "RandomForest"]),
            "threshold": float(spec.get("threshold", 0.8)),
            "problems": problems,
            "problems_detail": problems_detail,
            "notes": spec.get("notes", ""),
        }
        print("TASK:", region, "keys=", list(resp.keys()))
        return resp

    # ----- train -----
    @r.post("/train")
    def train(
        problem: str,
        model: str = "LogisticRegression",
        features: Optional[str] = Query(None, description="a,b,c 形式。空なら use_all とみなす"),
        use_all: bool = Query(False),
        onehot: bool = Query(True),
        scaling: str = Query("none"),
        l2: float | None = None,
        max_depth: int | None = None,
        n_estimators: int | None = None,
        learning_rate: float | None = None,
        subsample: float | None = None,
        body: dict = Body({}, description="互換のため空JSONでOK"),
    ):
        kw = {
            "features": features,
            "use_all": use_all,
            "onehot": onehot,
            "scaling": scaling,
            "l2": l2,
            "max_depth": max_depth,
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "subsample": subsample,
        }
        try:
            mlkit.train_and_save(region, problem, model, **kw)
            return {"ok": True, "model": model, "scaling": scaling}
        except Exception as e:
            return JSONResponse(status_code=500, content={"detail": f"train error: {e}"})

    # ----- predict -----
    @r.post("/predict")
    def predict(problem: str, model: str = "LogisticRegression", body: dict | None = Body(None)):
        try:
            pred, proba = mlkit.predict_one(region, problem, model, body or {})
            return {"pred": int(pred), "proba": float(proba)}
        except Exception as e:
            return JSONResponse(status_code=500, content={"detail": f"predict error: {e}"})

    # ----- evaluate -----
    @r.get("/evaluate")
    def evaluate(problem: str, model: str = "LogisticRegression"):
        try:
            score = mlkit.evaluate_full(region, problem, model)
            spec = region_spec(region) or {}
            thr = float(spec.get("threshold", 0.8))
            force_pass = (region == "okinawa") or bool(spec.get("force_pass"))
            passed = True if force_pass else (score >= thr)
            return {"score": float(score), "threshold": thr, "passed": passed, "force_pass": force_pass}
        except Exception as e:
            return JSONResponse(status_code=500, content={"detail": f"evaluate error: {e}"})

    # ----- submit -----
    @r.post("/submit")
    def submit(problem: str, model: str = "LogisticRegression", body: dict = Body(...)):
        try:
            player = (body or {}).get("player_id") or "guest"
            score = (body or {}).get("score")
            if score is None:
                score = mlkit.evaluate_full(region, problem, model)
            rec = lb.write_log(region, problem, model, player, float(score), (body or {}).get("meta"))
            return {"ok": True, "entry": rec}
        except Exception as e:
            return JSONResponse(status_code=500, content={"detail": f"submit error: {e}"})

    # ----- leaderboard -----
    @r.get("/leaderboard")
    def leaderboard(problem: str, period: str = "week", limit: int = 50):
        try:
            return lb.top(region, problem, period=period, limit=limit)
        except Exception as e:
            return JSONResponse(status_code=500, content={"detail": f"leaderboard error: {e}"})

    # ----- cross validation -----
    @r.get("/cv")
    def cv(problem: str, model: str = "LogisticRegression", k: int = 5, scaling: str = "none"):
        try:
            return mlkit.cross_validate(region, problem, model, k=k, scaling=scaling)
        except Exception as e:
            return JSONResponse(status_code=500, content={"detail": f"cv error: {e}"})

    # ----- reset -----
    @r.post("/reset")
    def reset(problem: str, model: Optional[str] = Query(None)):
        try:
            base = Path(f"storage/models/{region}/{problem}")
            if model:
                ok = _safe_rmtree(base / model)
            else:
                ok = True
                if base.exists():
                    for child in base.iterdir():
                        ok = _safe_rmtree(child) and ok
                    try:
                        base.rmdir()
                    except Exception:
                        pass
            return {"ok": bool(ok)}
        except Exception as e:
            return JSONResponse(status_code=500, content={"detail": f"reset error: {e}"})

    # ----- Ensemble config / get -----
    @r.post("/ensemble/config")
    def ensemble_config(problem: str, body: dict = Body(...)):
        try:
            base_models = _parse_models(body.get("base_models") or body.get("models"))
            if len(base_models) < 2:
                raise HTTPException(status_code=400, detail="ensemble requires >= 2 base models")

            weights = body.get("weights")
            if weights is not None:
                if not isinstance(weights, list):
                    raise HTTPException(status_code=400, detail="weights must be a list of numbers")
                if len(weights) != len(base_models):
                    raise HTTPException(status_code=400, detail="weights length must match base_models")

            scaling = _normalize_scaling(body.get("scaling"))
            cfg = mlkit.set_ensemble_config(region, problem, base_models=base_models, weights=weights, scaling=scaling)
            return {"ok": True, "config": cfg}
        except HTTPException:
            raise
        except Exception as e:
            return JSONResponse(status_code=500, content={"detail": f"ensemble config error: {e}"})

    @r.get("/ensemble/get")
    def ensemble_get(problem: str):
        try:
            return mlkit.get_ensemble_config(region, problem) or {}
        except Exception as e:
            return JSONResponse(status_code=500, content={"detail": f"ensemble get error: {e}"})

    # ----- Stacking train / get -----
    @r.post("/stacking/train")
    def stacking_train(problem: str, body: dict = Body(...)):
        try:
            base_models = _parse_models(body.get("base_models") or body.get("models"))
            if len(base_models) < 2:
                raise HTTPException(status_code=400, detail="stacking requires >= 2 base models")

            scaling = _normalize_scaling(body.get("scaling"))
            ret = mlkit.train_stacking(region, problem, base_models=base_models, scaling=scaling)
            return {"ok": True, **ret}
        except HTTPException:
            raise
        except Exception as e:
            return JSONResponse(status_code=500, content={"detail": f"stacking train error: {e}"})

    @r.get("/stacking/get")
    def stacking_get(problem: str):
        try:
            cfg = mlkit.get_stacking_config(region, problem) or {}
            meta_p = Path(f"storage/models/{region}/{problem}/_stacking/meta.json")
            meta = json.loads(meta_p.read_text(encoding="utf-8")) if meta_p.exists() else {"weights": []}
            return {"config": cfg, "meta": meta}
        except Exception as e:
            return JSONResponse(status_code=500, content={"detail": f"stacking get error: {e}"})

    # ---------- EDA ----------
    @r.get("/eda/schema")
    def eda_schema(problem: str):
        try:
            p = _csv_path(region, problem)
            rows = _read_rows(p)
            if ceda and hasattr(ceda, "schema"):
                return _ceda("schema", region=region, problem=problem, path=str(p), default_problem=problem)
            types = _infer_types(rows)
            return {"n_rows": len(rows), "columns": [{"name": k, "dtype": v} for k, v in types.items()]}
        except Exception as e:
            return JSONResponse(status_code=500, content={"detail": f"eda schema error: {e}"})

    @r.get("/eda/missing")
    def eda_missing(problem: str):
        try:
            p = _csv_path(region, problem)
            rows = _read_rows(p)
            if ceda and hasattr(ceda, "missing"):
                return _ceda("missing", region=region, problem=problem, path=str(p), default_problem=problem)
            cols = rows[0].keys() if rows else []
            miss = {c: sum(1 for r in rows if r.get(c) in (None, "", "NaN")) for c in cols}
            return {"missing": [{"column": c, "count": miss[c], "ratio": miss[c] / max(1, len(rows))} for c in cols]}
        except Exception as e:
            return JSONResponse(status_code=500, content={"detail": f"eda missing error: {e}"})

    @r.get("/eda/corr")
    def eda_corr(problem: str, sample: int = 0):
        try:
            def _sample():
                return {
                    "columns": ["x1", "x2", "y"],
                    "rows": [{"values": [1.0, 0.2, 0.05]}, {"values": [0.2, 1.0, 0.10]}, {"values": [0.05, 0.10, 1.0]}],
                    "matrix": [[1, 0.2, 0.05], [0.2, 1, 0.10], [0.05, 0.10, 1]],
                }

            if sample:
                return _sample()

            p = _csv_path(region, problem)
            rows = _read_rows(p)
            if not rows:
                return _sample()

            types = _infer_types(rows)
            nums = [c for c, t in types.items() if t == "number"]
            if not nums:
                return _sample()

            vec = {c: [float(r[c]) for r in rows if r.get(c) not in (None, "", "NaN")] for c in nums}

            def pearson(a, b):
                n = min(len(vec[a]), len(vec[b]))
                xs, ys = vec[a][:n], vec[b][:n]
                if n < 2:
                    return 0.0
                mx, my = sum(xs) / n, sum(ys) / n
                vx = sum((x - mx) ** 2 for x in xs)
                vy = sum((y - my) ** 2 for y in ys)
                if vx <= 1e-12 or vy <= 1e-12:
                    return 0.0
                cov = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
                return float(cov / ((vx ** 0.5) * (vy ** 0.5)))

            matrix = [[pearson(a, b) for b in nums] for a in nums]
            rows_out = [{"values": r} for r in matrix]
            return {"columns": nums, "rows": rows_out, "matrix": matrix}
        except Exception as e:
            return JSONResponse(status_code=500, content={"detail": f"eda corr error: {e}"})

    @r.get("/eda/univariate")
    def eda_univariate(problem: str, column: str, bins: int = 10):
        try:
            p = _csv_path(region, problem)
            rows = _read_rows(p)
            if ceda and hasattr(ceda, "univariate"):
                return _ceda("univariate", region=region, problem=problem, path=str(p), column=column, bins=bins, default_problem=problem)
            xs = [float(r[column]) for r in rows if r.get(column) not in (None, "", "NaN")]
            if not xs:
                return {"hist": [], "edges": []}
            lo, hi = min(xs), max(xs)
            bins = max(1, int(bins))
            width = (hi - lo) / bins if hi > lo else 1.0
            edges = [lo + i * width for i in range(bins + 1)]
            hist = [0] * bins
            for v in xs:
                k = min(int((v - lo) / max(width, 1e-9)), bins - 1)
                hist[k] += 1
            return {"hist": hist, "edges": edges, "min": lo, "max": hi, "mean": sum(xs) / len(xs)}
        except Exception as e:
            return JSONResponse(status_code=500, content={"detail": f"eda univariate error: {e}"})

    @r.get("/eda/crosstab")
    def eda_crosstab(problem: str, x: str, y: str):
        try:
            p = _csv_path(region, problem)
            rows = _read_rows(p)
            if ceda and hasattr(ceda, "crosstab"):
                return _ceda("crosstab", region=region, problem=problem, path=str(p), x=x, y=y, default_problem=problem)
            table = defaultdict(lambda: defaultdict(int))
            xs, ys = set(), set()
            for r in rows:
                xv, yv = r.get(x), r.get(y)
                table[xv][yv] += 1
                xs.add(xv)
                ys.add(yv)
            xs = sorted(xs)
            ys = sorted(ys)
            data = [{"x": xv, **{yv: table[xv].get(yv, 0) for yv in ys}} for xv in xs]
            return {"x_levels": xs, "y_levels": ys, "table": data}
        except Exception as e:
            return JSONResponse(status_code=500, content={"detail": f"eda crosstab error: {e}"})

    @r.get("/eda/top_corr_pairs")
    def eda_top_corr_pairs(problem: str, k: int = 10):
        try:
            res = eda_corr(problem)
            cols: List[str] = res.get("columns", []) if isinstance(res, dict) else []
            mat: List[List[float]] = res.get("matrix", []) if isinstance(res, dict) else []
            pairs = []
            for i, a in enumerate(cols):
                for j in range(i + 1, len(cols)):
                    b = cols[j]
                    corr = abs(mat[i][j]) if i < len(mat) and j < len(mat[i]) else 0.0
                    pairs.append({"a": a, "b": b, "corr": corr})
            pairs.sort(key=lambda d: -d["corr"])
            return {"pairs": pairs[: max(1, int(k))]}
        except Exception as e:
            return JSONResponse(status_code=500, content={"detail": f"eda top_corr_pairs error: {e}"})

    @r.get("/eda/outliers")
    def eda_outliers(problem: str, column: str, z: float = 3.0):
        try:
            p = _csv_path(region, problem)
            rows = _read_rows(p)
            if ceda and hasattr(ceda, "outliers"):
                return _ceda("outliers", region=region, problem=problem, path=str(p), column=column, z=z)
            xs = [float(r[column]) for r in rows if r.get(column) not in (None, "", "NaN")]
            if len(xs) < 2:
                return {"idx": [], "values": []}
            m = sum(xs) / len(xs)
            var = sum((v - m) ** 2 for v in xs) / max(1, len(xs) - 1)
            s = (var ** 0.5) or 1.0
            idx, vals = [], []
            j = 0
            for i, r in enumerate(rows):
                v = r.get(column)
                if v in (None, "", "NaN"):
                    continue
                vv = float(v)
                zc = abs((vv - m) / s)
                if zc >= z:
                    idx.append(i)
                    vals.append(vv)
                j += 1
            return {"idx": idx, "values": vals, "mean": m, "std": s, "z": z}
        except Exception as e:
            return JSONResponse(status_code=500, content={"detail": f"eda outliers error: {e}"})

    @r.get("/eda/pca2")
    def eda_pca2(problem: str):
        try:
            p = _csv_path(region, problem)
            if ceda and hasattr(ceda, "pca2"):
                return _ceda("pca2", region=region, problem=problem, path=str(p))
            return JSONResponse(status_code=501, content={"detail": "pca2 not implemented (common_eda.py があれば自動で使われます)"})
        except Exception as e:
            return JSONResponse(status_code=500, content={"detail": f"eda pca2 error: {e}"})

    return r


# ---- FastAPI app 構築（関数定義のあと！） ----
app = FastAPI()

REGIONS = [
    "okinawa", "kanto", "kinki", "chubu",
    "tohoku", "hokkaido", "chugoku", "shikoku", "kyushu",
]

for slug in REGIONS:
    app.include_router(make_region_router(slug))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", reload=True)
