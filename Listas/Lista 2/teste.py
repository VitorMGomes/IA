import sys
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple


def H(y: pd.Series) -> float:
    """Entropia de Shannon (bits)."""
    counts = y.value_counts(dropna=False).astype(float)
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())

def IG(df: pd.DataFrame, attr: str, y_col: str, baseH: float = None) -> float:
    """Ganho de informação de attr sobre y_col (atributo nominal)."""
    if baseH is None:
        baseH = H(df[y_col])
    n = len(df)
    condH = 0.0
    for _, g in df.groupby(attr, dropna=False, observed=True):
        condH += (len(g) / n) * H(g[y_col])
    return baseH - condH

def rank_ig(df: pd.DataFrame, y_col: str, attrs: List[str]) -> List[Tuple[str, float]]:
    """Retorna [(attr, IG)] ordenado do maior para o menor."""
    base = H(df[y_col])
    pares = [(a, IG(df, a, y_col, base)) for a in attrs]
    pares.sort(key=lambda kv: kv[1], reverse=True)
    return pares

def read_any_sep(path: str) -> pd.DataFrame:
    """Lê CSV inferindo separador automaticamente (fallback = vírgula)."""
    try:
        return pd.read_csv(path, sep=None, engine="python")
    except Exception:
        return pd.read_csv(path)

def as_categorical(df: pd.DataFrame, y_col: str) -> pd.DataFrame:
    """Converte preditores para category; mantém alvo como está."""
    for c in df.columns:
        if c != y_col:
            df[c] = df[c].astype("category")
    return df

def print_table(title: str, pares: List[Tuple[str, float]], k: int = None):
    print(f"\n{title}")
    print("  Atributo".ljust(16), "IG")
    print("  " + "-" * 24)
    for i, (a, v) in enumerate(pares):
        if k is not None and i >= k: break
        print(f"  {a.ljust(16)} {v:.6f}")


def raiz_e_segundo_nivel(df: pd.DataFrame, y_col: str) -> Dict:
    """Devolve raiz (maior IG) e, para cada valor do ramo da raiz, o melhor atributo seguinte."""
    attrs = [c for c in df.columns if c != y_col]
    base = H(df[y_col])
    ranking = rank_ig(df, y_col, attrs)
    raiz, ig_raiz = ranking[0]

    por_ramo = {}
    for valor in df[raiz].dropna().unique().tolist():
        sub = df[df[raiz] == valor].copy()
        candidatos = [c for c in attrs if c != raiz]
        if len(sub) == 0 or len(candidatos) == 0:
            por_ramo[valor] = []
            continue
        sub_rank = rank_ig(sub, y_col, candidatos)
        por_ramo[valor] = sub_rank
    return {"ranking_global": ranking, "raiz": (raiz, ig_raiz), "por_ramo": por_ramo}

def main():
    if len(sys.argv) < 3:
        print("Uso: python ig_restaurante_v2.py <csv_path> <coluna_alvo>")
        sys.exit(1)

    csv_path, y_col = sys.argv[1], sys.argv[2]
    df = read_any_sep(csv_path)

    if y_col not in df.columns:
        raise ValueError(f'Coluna alvo "{y_col}" não encontrada. Colunas: {list(df.columns)}')

    df = as_categorical(df, y_col)
    print("Colunas:", list(df.columns))

    resultado = raiz_e_segundo_nivel(df, y_col)

    # ranking global
    print_table("Ganhos de informação (global)", resultado["ranking_global"])
    raiz, ig_raiz = resultado["raiz"]
    print(f"\n>>> Atributo raiz sugerido (ID3): {raiz}  (IG={ig_raiz:.6f})")

    # melhores do 2º nível por ramo
    print("\n=== 2º nível sugerido por ramo da raiz ===")
    for valor, rank in resultado["por_ramo"].items():
        print_table(f'Ramo {raiz} = "{valor}" (top 3)', rank, k=3)
        if len(rank) > 0:
            print(f'>>> Melhor próximo atributo neste ramo: {rank[0][0]}\n')

if __name__ == "__main__":
    main()
