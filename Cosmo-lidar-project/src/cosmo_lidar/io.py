from __future__ import annotations
import re, requests
from urllib.parse import urljoin
from pathlib import Path
# --- parseurs radiosonde (.dat) ---

from pathlib import Path
import re
import pandas as pd
from io import StringIO




def fetch_html(url: str) -> str:
    r = requests.get(url, headers={"User-Agent": "Mozilla/5.0 (cosmo-lidar/0.1)"}, timeout=30)
    r.raise_for_status()
    return r.text

def extract_ut_column_dat_links(html: str, base_url: str):
    """
    1) Isole la <table> qui contient un <th> avec 'UT Launch Date' ET 'Data'
    2) Récupère tous les liens se terminant par .dat DANS cette table
    3) Retourne une liste de dicts: {'label': texte du lien, 'url': url absolue}
    """
    # 1) isole la table (DOTALL + non-greedy). On tolère le * après Date.
    table_pat = re.compile(
        r'(?is)<table\b.*?>.*?UT\s*Launch\s*Date\*?\s*&\s*Data.*?</table>'
    )
    m = table_pat.search(html)
    if not m:
        # fallback: tous les .dat de la page si pas de table détectée
        hrefs = re.findall(r'(?is)href=["\']([^"\']+\.dat)["\']', html)
        return [{"label": None, "url": urljoin(base_url, h)} for h in sorted(set(hrefs))]

    table_html = m.group(0)

    # 2) dans cette table: <a ... href="...dat">LABEL</a>
    a_pat = re.compile(r'(?is)<a[^>]+href=["\']([^"\']+\.dat)["\'][^>]*>(.*?)</a>')
    rows = []
    for href, raw_label in a_pat.findall(table_html):
        label = re.sub(r'(?is)<[^>]+>', ' ', raw_label)      # enlève balises internes
        label = re.sub(r'\s+', ' ', label).strip()
        rows.append({"label": label or None, "url": urljoin(base_url, href)})

    # dédup
    seen, uniq = set(), []
    for r in rows:
        if r["url"] in seen: 
            continue
        seen.add(r["url"])
        uniq.append(r)
    return uniq

"""def download_some(links, dest: str | Path, n: int = 3):
    dest = Path(dest); dest.mkdir(parents=True, exist_ok=True)
    out = []
    for r in links[:n]:
        base = (r["label"] or Path(r["url"]).name).replace(":", "-").replace("/", "-").replace(" ", "_")
        if not base.lower().endswith(".dat"):
            base += ".dat"
        p = dest / base
        resp = requests.get(r["url"], headers={"User-Agent": "Mozilla/5.0 (cosmo-lidar/0.1)"}, timeout=60)
        resp.raise_for_status()
        p.write_bytes(resp.content)
        out.append(p)
    return out"""
def download_some(links, dest: str | Path, n: int | None = 3, timeout: int = 60, verbose: bool = True):
    """
    Télécharge jusqu'à n fichiers depuis `links` (liste de dicts {'label','url'}).
    Ignore silencieusement les liens cassés (404) ou toute erreur réseau/HTTP,
    et passe au suivant.

    Retourne:
        paths_ok (list[Path]): fichiers téléchargés avec succès
        errors   (list[tuple[str, str]]): (url, message d'erreur)
    """
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    if n is None:
        n = len(links)

    paths_ok: list[Path] = []
    errors: list[tuple[str, str]] = []

    session = requests.Session()
    headers = {"User-Agent": "Mozilla/5.0 (cosmo-lidar/0.1)"}

    for r in links[:n]:
        url = r.get("url")
        label = r.get("label") or Path(url).name
        base = (
            label.replace(":", "-")
                 .replace("/", "-")
                 .replace("\\", "-")
                 .replace(" ", "_")
        )
        if not base.lower().endswith(".dat"):
            base += ".dat"
        p = dest / base

        try:
            resp = session.get(url, headers=headers, timeout=timeout)
            # Si 404 (ou autre), raise pour passer au except
            resp.raise_for_status()

            # Écrit le contenu si tout est OK
            p.write_bytes(resp.content)
            paths_ok.append(p)
            if verbose:
                print(f"✓ {url} -> {p} ({len(resp.content)} bytes)")

        except requests.HTTPError as e:
            # Erreurs HTTP (404, 403, 500, etc.)
            errors.append((url, f"HTTPError {resp.status_code}: {e}"))
            if verbose:
                print(f"✗ {url} ignoré (HTTP {resp.status_code})")
            continue
        except requests.RequestException as e:
            # Timeout, connexion, SSL, etc.
            errors.append((url, f"RequestException: {e}"))
            if verbose:
                print(f"✗ {url} ignoré ({e})")
            continue
        except Exception as e:
            # Toute autre erreur (I/O, etc.)
            errors.append((url, f"Unexpected: {e}"))
            if verbose:
                print(f"✗ {url} ignoré (Unexpected: {e})")
            continue

    return paths_ok, errors




from pathlib import Path
import re
import pandas as pd

def _open_text_with_fallback(path: Path) -> tuple[str, str]:
    """
    Ouvre un fichier texte avec fallback d'encodage.
    Retourne (contenu, encoding_utilisé).
    """
    try:
        txt = path.read_text(encoding="utf-8")
        return txt, "utf-8"
    except UnicodeDecodeError:
        try:
            txt = path.read_text(encoding="latin-1")
            return txt, "latin-1"
        except UnicodeDecodeError:
            # Dernier recours : on décode en latin-1 en remplaçant les octets invalides
            txt = path.read_bytes().decode("latin-1", errors="replace")
            return txt, "latin-1(replace)"

def _split_cols(line: str) -> list[str]:
    return [c for c in re.split(r"\s+", line.strip()) if c]

def _find_geop_col(cols: list) -> str:
    """Retourne le nom exact de la colonne GEOP (tolérant)."""
    for c in cols:
        s = str(c).strip().upper()
        if s == "GEOP" or s.startswith("GEOP"):
            return str(c)
    raise KeyError("Colonne GEOP introuvable dans ce fichier.")


def read_radiosonde_dat(
    path: str | Path,
    *,
    skip_rows: int = 4,
    drop_last: int = 0,
    min_geop: float | None = None,
    max_geop: float | None = None,
    geop_step_threshold: float | None = -8.0,
    sort_geop: bool = False,
    parse_time: bool = False,
) -> pd.DataFrame:
    path = Path(path)

    text, enc = _open_text_with_fallback(path)

    # 1) noms de colonnes depuis la 1re ligne
    first_nl = text.find("\n")
    header_line = text if first_nl == -1 else text[:first_nl]
    cols = [str(c) for c in _split_cols(header_line)]  # <- cast en str

    # 2) lecture du tableau
    sio = StringIO(text)
    df = pd.read_csv(
        sio,
        sep=r"\s+",            # <= remplace delim_whitespace
        engine="python",
        header=None,
        names=cols,
        skiprows=skip_rows,
        comment="#",
    )

    # 3) colonnes & lignes vides
    df = df.dropna(axis=1, how="all")
    df.columns = [str(c).strip() for c in df.columns]  # <- force str partout

    if drop_last and drop_last > 0 and drop_last < len(df):
        df = df.iloc[:-drop_last, :]

    if parse_time and "TIME" in df.columns:
        df["TIME"] = pd.to_timedelta(df["TIME"], errors="coerce")

    # 4) GEOP numérique + filtrage + tri
    geop_col = _find_geop_col(list(df.columns))
    df[geop_col] = pd.to_numeric(df[geop_col], errors="coerce")

    if min_geop is not None:
        df = df[df[geop_col] >= float(min_geop)]
        
    if max_geop is not None:
        df = df[df[geop_col] <= float(max_geop)]

    # Optionnel: tronquer la série si la différence successive z[i+1]-z[i] <= geop_step_threshold
    if geop_step_threshold is not None and len(df) > 0:
        

        # reset index pour garantir des indices consécutifs
        df = df.reset_index(drop=True)
        z = df[geop_col].to_numpy(dtype='float64')

        # couper avant la première valeur NaN dans GEOP
        nan_pos = np.where(np.isnan(z))[0]
        if nan_pos.size:
            df = df.iloc[: nan_pos[0]]
            z = df[geop_col].to_numpy(dtype='float64')

        if z.size > 1:
            diffs = np.diff(z)
            # on tronque dès que la différence suivante est <= threshold
            bad = np.where(diffs <= float(geop_step_threshold))[0]
            if bad.size:
                cut = int(bad[0]) + 1
                df = df.iloc[:cut]

    if sort_geop:
        df = df.sort_values(geop_col, kind="mergesort").reset_index(drop=True)

    df["__source_file"] = path.name
    df["__source_dir"] = path.parent.name
    df["__encoding"] = enc
    return df

def read_many_radiosonde(
    paths: list[str | Path],
    *,
    per_file: bool = False,
    save_per_file: bool = False,
    out_dir: str | Path = "../data/processed",
    **kwargs,                    # transmis à read_radiosonde_dat (drop_last, min_geop, sort_geop, parse_time, ...)
) -> pd.DataFrame | list[pd.DataFrame]:
    """
    Lit plusieurs fichiers radiosonde avec read_radiosonde_dat.

    - per_file=False (défaut)  -> retourne un unique DataFrame concaténé
    - per_file=True            -> retourne une liste de DataFrames (un par fichier)
      - si save_per_file=True  -> enregistre chaque DF dans `out_dir` au format parquet

    `kwargs` est passé tel quel à read_radiosonde_dat.
    """
    parts: list[pd.DataFrame] = []
    errors: list[tuple[str, str]] = []

    for p in paths:
        p = Path(p)
        try:
            parts.append(read_radiosonde_dat(p, **kwargs))
        except Exception as e:
            errors.append((p.name, repr(e)))

    if errors:
        print("⚠️ Fichiers ignorés (erreur de parsing) :")
        for fn, err in errors[:15]:
            print(f" - {fn}: {err}")
        if len(errors) > 15:
            print(f"   … (+{len(errors)-15} autres)")

    if per_file:
        if save_per_file:
            out_dir = Path(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            for df in parts:
                # nom basé sur la source si dispo
                if "__source_file" in df.columns and not df["__source_file"].empty:
                    name = f"{df['__source_file'].iloc[0]}_clean.parquet"
                else:
                    name = f"radiosonde_{len(df)}rows_clean.parquet"
                df.to_parquet(out_dir / name, index=False)
        return parts

    # concaténé
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


# utilitaires de persistance
def save_table(df: pd.DataFrame, out: str | Path, fmt: str = "parquet"):
    out = Path(out); out.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "parquet":
        df.to_parquet(out, index=False)
    elif fmt == "csv":
        df.to_csv(out, index=False)
    else:
        raise ValueError("fmt must be 'parquet' or 'csv'")

def load_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError("Fichier inconnu (attendu .parquet ou .csv)")



from pathlib import Path
from typing import Dict, List, Mapping, Optional
import numpy as np
import pyarrow.parquet as pq
import pyarrow as pa

def load_parquet_columns_as_numpy(
    root: Path,
    glob_pattern: str = "*.parquet",
    exclude_prefixes: Optional[List[str]] = ("__source_", "__encoding"),
    sentinels: Optional[Mapping[str, List[float]]] = None,
) -> Dict[Path, Dict[str, np.ndarray]]:
    """
    Parcourt `root` (non-récursif par défaut) et retourne:
        { chemin_fichier: { nom_colonne: np.ndarray } }

    - exclude_prefixes : permet d’ignorer certaines colonnes (ex: métadonnées)
    - sentinels : map de colonnes -> liste de valeurs à convertir en NaN
                  ex: {"AZ":[9999], "EL":[9999], "SPEED":[999], "DIR":[999], "RT":[99.99]}
    """
    root = Path(root)
    results: Dict[Path, Dict[str, np.ndarray]] = {}

    for f in sorted(root.glob(glob_pattern)):
        pf = pq.ParquetFile(f)
        cols = pf.schema_arrow.names
        file_dict: Dict[str, np.ndarray] = {}

        for col in cols:
            if exclude_prefixes and any(col.startswith(pfx) for pfx in exclude_prefixes):
                continue

            # Lecture colonne seule (mémoire efficace)
            arr: pa.ChunkedArray = pf.read(columns=[col]).column(0)

            # Concatène les chunks puis convertit en numpy
            # zero_copy_only=False pour gérer les types qui exigent une copie
            np_col = arr.combine_chunks().to_numpy(zero_copy_only=False)

            # Convertit les strings Arrow en numpy object (déjà le cas)
            # Normalise les bool/int/float en dtypes numpy standards
            if pa.types.is_boolean(arr.type):
                np_col = np_col.astype(np.bool_)
            elif pa.types.is_integer(arr.type):
                # Utilise float64 si présence de NA pour conserver NaN
                np_col = np_col.astype("float64") if arr.null_count > 0 else np_col.astype(np.int64)
            elif pa.types.is_floating(arr.type):
                np_col = np_col.astype(np.float64)
            # pa.string() -> dtype=object par défaut; OK.

            # Applique les sentinelles -> NaN si défini
            if sentinels and col in sentinels and np_col.size:
                mask = np.zeros(np_col.shape, dtype=bool)
                for s in sentinels[col]:
                    mask |= (np_col == s)
                # Cast en float si nécessaire pour pouvoir mettre NaN
                if np.issubdtype(np_col.dtype, np.integer) or np_col.dtype == np.bool_:
                    np_col = np_col.astype("float64")
                np_col = np.where(mask, np.nan, np_col)

            file_dict[col] = np_col

        results[f] = file_dict

    return results



    
def to_float64(arr, sentinels=None):
    """
    Convertit un tableau potentiellement 'object/str' en float64.
    - Nettoie espaces, NBSP, séparateur décimal virgule.
    - Convertit '', 'nan', 'None' -> NaN.
    - Applique les sentinelles (ex. 9999, '999', '99.99') -> NaN.
    """
    a = np.asarray(arr)

    # Déjà numérique ?
    if np.issubdtype(a.dtype, np.number):
        out = a.astype('float64', copy=False)
    else:
        s = a.astype(str)
        # trim
        s = np.char.strip(s)
        # normalise les virgules décimales et enlève espaces fines/insécables
        s = np.char.replace(s, '\xa0', '')
        s = np.char.replace(s, ' ', '')
        s = np.char.replace(s, ',', '.')  # 12,34 -> 12.34

        # valeurs vides -> NaN
        mask_empty = (s == '') | (s == 'nan') | (s == 'None') | (s == 'NaN')
        s[mask_empty] = 'NaN'

        # pandas.to_numeric pour une conversion robuste
        out = pd.to_numeric(pd.Series(s), errors='coerce').to_numpy(dtype='float64')

    # applique les sentinelles éventuelles
    if sentinels:
        # accepte nombres OU strings dans la liste
        sent_float = []
        for v in sentinels:
            try:
                sent_float.append(float(str(v).replace(',', '.')))
            except Exception:
                pass
        if sent_float:
            mask = np.zeros(out.shape, dtype=bool)
            for v in sent_float:
                mask |= np.isclose(out, v, equal_nan=False)
            out = np.where(mask, np.nan, out)

    return out