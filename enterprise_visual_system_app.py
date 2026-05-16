# -*- coding: utf-8 -*-
"""
enterprise_visual_system_app.py
企业生存、薪资估值与空间结构演化可视化系统（一键上传版）

适配你当前这组文件：
1. 企业基础信息_有招聘.xlsx
2. 企业招聘行为表_清洗终.xlsx
3. 北京/深圳/苏州边界文件（建议上传完整 Shapefile 压缩包，包含 shp/shx/dbf/prj）

运行：
streamlit run enterprise_visual_system_app.py

说明：
- 左侧支持“一次上传多个文件”
- 系统会自动识别企业基础信息表、企业招聘行为表、城市边界文件
- 若没有第三题就业中心结果，系统会基于企业经纬度和年度招聘人数自动估计 DBSCAN 就业中心
"""

import os
import re
import math
import zipfile
import tempfile
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

import plotly.express as px
import plotly.graph_objects as go
import networkx as nx

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from wordcloud import WordCloud
from collections import Counter

try:
    import jieba
except Exception:
    jieba = None

try:
    import geopandas as gpd
except Exception:
    gpd = None

try:
    from sklearn.cluster import DBSCAN
except Exception:
    DBSCAN = None

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import accuracy_score, roc_auc_score
except Exception:
    RandomForestClassifier = None
    train_test_split = None
    ColumnTransformer = None
    Pipeline = None
    OneHotEncoder = None
    SimpleImputer = None
    accuracy_score = None
    roc_auc_score = None


# =============================================================================
# 0. 页面设置
# =============================================================================
st.set_page_config(
    page_title="企业生存、薪资估值与空间结构演化可视化系统",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded",
)



# =============================================================================
# 1. 通用函数
# =============================================================================
def read_table(file_obj) -> pd.DataFrame:
    """读取 Excel / CSV。"""
    name = file_obj.name
    suffix = Path(name).suffix.lower()

    if suffix in [".xlsx", ".xls"]:
        return pd.read_excel(file_obj)
    if suffix == ".csv":
        try:
            return pd.read_csv(file_obj, encoding="utf-8-sig")
        except UnicodeDecodeError:
            return pd.read_csv(file_obj, encoding="gbk")
    return pd.DataFrame()



def file_signature(file_obj):
    """根据上传文件内容生成签名，供 Streamlit 缓存使用。"""
    if file_obj is None:
        return None
    pos = file_obj.tell()
    file_obj.seek(0)
    data = file_obj.getvalue()
    file_obj.seek(pos)
    return {
        "name": file_obj.name,
        "bytes": data,
        "md5": hashlib.md5(data).hexdigest()
    }


def table_from_signature(sig):
    """从缓存签名恢复文件并读取。"""
    if sig is None:
        return pd.DataFrame()
    import io
    bio = io.BytesIO(sig["bytes"])
    bio.name = sig["name"]
    return read_table(bio)


@st.cache_data(show_spinner="正在读取并预处理 Excel 数据……")
def cached_prepare_tables(basic_sig, job_sig):
    """缓存企业表和招聘表预处理，筛选条件变化时不会重复读取 Excel。"""
    df_basic_raw = table_from_signature(basic_sig)
    df_job_raw = table_from_signature(job_sig)

    df_basic = prepare_basic(df_basic_raw)
    df_jobs = prepare_jobs(df_job_raw, df_basic)
    df_ey = build_enterprise_year(df_jobs)

    return df_basic, df_jobs, df_ey


@st.cache_data(show_spinner="正在训练随机森林并计算生存概率……")
def cached_rf_survival(df_basic, df_jobs, df_ey):
    """缓存随机森林结果，筛选条件变化时不会重复训练。"""
    return compute_rf_survival_probability(df_basic, df_jobs, df_ey)


@st.cache_data(show_spinner="正在计算就业中心和行业集聚半径……")
def cached_spatial_results(df_ey):
    """缓存 DBSCAN 就业中心和行业集聚半径。"""
    df_centers = auto_dbscan_centers(df_ey)
    df_radius = compute_industry_radius(df_ey)
    return df_centers, df_radius


@st.cache_data(show_spinner="正在读取城市边界文件……")
def cached_read_boundaries(shape_file_sigs, zip_file_sigs):
    """缓存城市边界读取。"""
    import io

    class FakeUpload:
        def __init__(self, sig):
            self.name = sig["name"]
            self._bytes = sig["bytes"]
        def getbuffer(self):
            return memoryview(self._bytes)

    shape_files = [FakeUpload(s) for s in (shape_file_sigs or [])]
    zip_files = [FakeUpload(s) for s in (zip_file_sigs or [])]
    return read_boundaries_from_uploads(shape_files, zip_files)


@st.cache_data(show_spinner=False)
def cached_token_frequencies(text):
    """缓存词云分词结果。"""
    words = tokenize_for_wordcloud(text)
    return dict(Counter(words))


def make_wordcloud_from_freq(freq_dict):
    """从缓存好的词频生成词云图。"""
    fig, ax = plt.subplots(figsize=(12, 6))

    if not freq_dict:
        ax.text(0.5, 0.5, "暂无可生成词云的文本", ha="center", va="center", fontsize=16)
        ax.axis("off")
        return fig

    font_path = get_chinese_font_path()
    if font_path is None:
        ax.text(
            0.5, 0.5,
            "未找到中文字体，词云可能显示为方框。\\n建议使用微软雅黑或黑体。",
            ha="center", va="center", fontsize=14
        )
        ax.axis("off")
        return fig

    wc = WordCloud(
        font_path=get_chinese_font_path(),
        width=1200,
        height=600,
        background_color="white",
        max_words=180,
        collocations=False,
        prefer_horizontal=0.9
    ).generate_from_frequencies(freq_dict)

    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    return fig


def downsample_points(df, max_points=6000, weight_col=None, random_state=42):
    """
    地图点过多时自动抽样，提升渲染速度。
    如果指定 weight_col，会优先保留权重大的一部分。
    """
    if df is None or len(df) <= max_points:
        return df

    if weight_col and weight_col in df.columns:
        d = df.copy()
        d["_w_tmp"] = pd.to_numeric(d[weight_col], errors="coerce").fillna(0)
        top_n = int(max_points * 0.35)
        top = d.sort_values("_w_tmp", ascending=False).head(top_n)
        rest = d.drop(top.index)
        rest_n = max_points - len(top)
        if rest_n > 0 and len(rest) > rest_n:
            rest = rest.sample(rest_n, random_state=random_state)
        out = pd.concat([top, rest], ignore_index=True).drop(columns=["_w_tmp"], errors="ignore")
        return out

    return df.sample(max_points, random_state=random_state)



def normalize_city(x):
    """城市字段标准化。"""
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    if "北京" in s:
        return "北京"
    if "深圳" in s:
        return "深圳"
    if "苏州" in s:
        return "苏州"

    # 对“上海-浦东”“杭州 西湖”等格式，保留前部
    parts = re.split(r"[-_/，,、\s|]+", s)
    if parts:
        p = parts[0].replace("市", "")
        return p
    return s


def extract_year(s):
    """从时间列提取年份。"""
    dt = pd.to_datetime(s, errors="coerce")
    year = dt.dt.year
    if year.notna().sum() == 0:
        year = pd.to_numeric(s.astype(str).str.extract(r"(20\d{2})")[0], errors="coerce")
    return year


def to_num(s):
    """文本转数值。"""
    return pd.to_numeric(
        s.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("万元", "", regex=False)
        .str.replace("万", "", regex=False)
        .str.replace("人", "", regex=False)
        .str.extract(r"([-+]?\d*\.?\d+)")[0],
        errors="coerce"
    )


def find_col(df, candidates):
    """按候选名寻找列。"""
    if df is None or df.empty:
        return None

    cols = list(df.columns)

    for c in candidates:
        if c in cols:
            return c

    for c in candidates:
        for col in cols:
            if str(c).replace(" ", "") == str(col).replace(" ", ""):
                return col

    for c in candidates:
        for col in cols:
            if str(c) in str(col) or str(col) in str(c):
                return col

    return None


def valid_coord(df, lon_col, lat_col):
    lon = pd.to_numeric(df[lon_col], errors="coerce")
    lat = pd.to_numeric(df[lat_col], errors="coerce")
    return lon.between(70, 140) & lat.between(10, 60)



def get_chinese_font_path():
    """
    自动寻找本机中文字体，解决 Matplotlib / WordCloud 中文显示成方框的问题。
    Windows 常见：微软雅黑 msyh.ttc、黑体 simhei.ttf。
    """
    candidates = [
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\msyh.ttf",
        r"C:\Windows\Fonts\simhei.ttf",
        r"C:\Windows\Fonts\simsun.ttc",
        r"C:\Windows\Fonts\simkai.ttf",
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    ]
    for p in candidates:
        if Path(p).exists():
            return p
    return None


def setup_chinese_matplotlib_font():
    """
    全局设置 Matplotlib 中文字体。
    这个函数会让九宫格地图、坐标轴标题、图例标题正常显示中文。
    """
    font_path = get_chinese_font_path()
    if font_path:
        try:
            fm.fontManager.addfont(font_path)
            font_name = fm.FontProperties(fname=font_path).get_name()
            plt.rcParams["font.sans-serif"] = [font_name, "Microsoft YaHei", "SimHei", "Arial Unicode MS"]
            plt.rcParams["font.family"] = "sans-serif"
        except Exception:
            plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
    else:
        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False



def make_download_html(fig):
    return fig.to_html(include_plotlyjs="cdn").encode("utf-8")


def classify_uploaded_files(uploaded_files):
    """
    一次上传多个文件后，自动识别：
    - 企业基础信息表
    - 招聘行为表
    - 第三题就业中心结果，可选
    - 行业集聚半径结果，可选
    - Shapefile / zip 边界文件
    """
    result = {
        "basic_file": None,
        "job_file": None,
        "center_file": None,
        "radius_file": None,
        "shape_files": [],
        "zip_files": [],
        "other_tables": []
    }

    for f in uploaded_files:
        name = f.name
        suffix = Path(name).suffix.lower()

        if suffix == ".zip":
            result["zip_files"].append(f)
            continue

        if suffix in [".shp", ".shx", ".dbf", ".prj", ".cpg"]:
            result["shape_files"].append(f)
            continue

        if suffix not in [".xlsx", ".xls", ".csv"]:
            continue

        # 读取前几行判断字段
        try:
            df_head = read_table(f)
            f.seek(0)
        except Exception:
            result["other_tables"].append(f)
            continue

        cols = set(map(str, df_head.columns))

        # 企业基础信息表特征
        if {"企业名称", "城市", "经度", "纬度"}.issubset(cols) or ("注册资本(万元)" in cols and "行业门类名称" in cols):
            result["basic_file"] = f
            continue

        # 招聘行为表特征
        if "职位刷新时间" in cols or "职位名称" in cols or "平均年薪(万)" in cols or "职位描述_清洗" in cols:
            result["job_file"] = f
            continue

        # 就业中心结果
        if "中心经度" in cols or "中心纬度" in cols or "中心招聘占比" in cols or "聚类标签" in cols:
            result["center_file"] = f
            continue

        # 行业集聚半径结果
        if "50%覆盖半径" in "".join(cols) or "90%覆盖半径" in "".join(cols) or "加权平均半径" in "".join(cols):
            result["radius_file"] = f
            continue

        result["other_tables"].append(f)

    return result


def _safe_read_gdf(path):
    """
    更稳健地读取空间文件：
    1. 优先使用 pyogrio，绕开 fiona.path 版本兼容问题；
    2. 再退回 geopandas 默认引擎；
    3. 若坐标系存在，则统一转 EPSG:4326。
    """
    if gpd is None:
        raise RuntimeError("未安装 geopandas")

    last_err = None

    # 优先使用 pyogrio
    try:
        gdf = gpd.read_file(path, engine="pyogrio")
        if gdf.crs is not None:
            gdf = gdf.to_crs(epsg=4326)
        return gdf
    except Exception as e:
        last_err = e

    # 再尝试默认引擎
    try:
        gdf = gpd.read_file(path)
        if gdf.crs is not None:
            gdf = gdf.to_crs(epsg=4326)
        return gdf
    except Exception as e:
        raise RuntimeError(f"pyogrio 与默认引擎均读取失败。pyogrio错误：{last_err}；默认引擎错误：{e}")


def read_boundaries_from_uploads(shape_files, zip_files):
    """
    读取城市边界。
    修复版：
    - 优先使用 pyogrio 读取，避免 fiona.path 报错；
    - 支持 zip；
    - 支持一次上传同名 .shp/.shx/.dbf/.prj 文件组。
    """
    if gpd is None:
        return None, "未安装 geopandas，暂不能读取城市边界。"

    gdfs = []
    messages = []

    # 读取 zip
    for zf in zip_files:
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                zpath = Path(tmpdir) / zf.name
                zpath.write_bytes(zf.getbuffer())
                with zipfile.ZipFile(zpath, "r") as z:
                    z.extractall(tmpdir)

                shp_paths = list(Path(tmpdir).rglob("*.shp"))
                if not shp_paths:
                    messages.append(f"{zf.name} 中没有找到 .shp 文件。")

                for shp in shp_paths:
                    try:
                        gdf = _safe_read_gdf(shp)
                        gdf["来源文件"] = shp.name
                        gdfs.append(gdf)
                    except Exception as e:
                        messages.append(f"{shp.name} 读取失败：{e}")
        except Exception as e:
            messages.append(f"{zf.name} 解压或读取失败：{e}")

    # 读取散装 shp/shx/dbf/prj
    if shape_files:
        with tempfile.TemporaryDirectory() as tmpdir:
            for f in shape_files:
                (Path(tmpdir) / f.name).write_bytes(f.getbuffer())

            shp_paths = list(Path(tmpdir).glob("*.shp"))
            for shp in shp_paths:
                # 检查同名配套文件
                missing = []
                for ext in [".shx", ".dbf"]:
                    if not (shp.with_suffix(ext)).exists():
                        missing.append(ext)

                if missing:
                    messages.append(
                        f"{shp.name} 缺少配套文件 {', '.join(missing)}。"
                        "Shapefile 不能只上传 .shp，建议上传完整 zip：.shp + .shx + .dbf + .prj。"
                    )
                    continue

                try:
                    gdf = _safe_read_gdf(shp)
                    gdf["来源文件"] = shp.name
                    gdfs.append(gdf)
                except Exception as e:
                    messages.append(
                        f"{shp.name} 读取失败：{e}。"
                        "建议：1）上传完整 zip；2）在终端执行 pip install -U geopandas pyogrio shapely pyproj；"
                        "3）如仍报 fiona.path，可执行 pip install fiona==1.9.6。"
                    )

    if not gdfs:
        return None, "\n".join(messages) if messages else "未上传城市边界文件。"

    try:
        gdf_all = pd.concat(gdfs, ignore_index=True)
        return gdf_all, "\n".join(messages)
    except Exception as e:
        return None, f"边界合并失败：{e}"


# =============================================================================
# 2. 数据处理：适配你当前两张表
# =============================================================================
def prepare_basic(df_basic_raw):
    df = df_basic_raw.copy()

    company_col = find_col(df, ["企业名称", "公司名称"])
    city_col = find_col(df, ["城市", "企业所在地", "企业所在城市"])
    capital_col = find_col(df, ["注册资本(万元)", "注册资本", "注册资本（万元）"])
    status_col = find_col(df, ["企业状态", "存活状态", "状态"])
    establish_col = find_col(df, ["成立日期", "成立时间", "成立年份"])
    industry_col = find_col(df, ["行业门类名称", "行业名称", "行业"])
    lon_col = find_col(df, ["经度", "longitude", "lon"])
    lat_col = find_col(df, ["纬度", "latitude", "lat"])

    if company_col and company_col != "企业名称":
        df["企业名称"] = df[company_col]
    elif "企业名称" not in df.columns:
        df["企业名称"] = [f"企业{i+1}" for i in range(len(df))]

    df["企业所在地"] = df[city_col].apply(normalize_city) if city_col else np.nan
    df["行业"] = df[industry_col].fillna("未知行业").astype(str) if industry_col else "未知行业"

    if capital_col:
        df["注册资本_万元"] = to_num(df[capital_col])
    else:
        df["注册资本_万元"] = np.nan

    if status_col:
        df["企业状态_标准"] = df[status_col].fillna("未知").astype(str)
    else:
        df["企业状态_标准"] = "未知"

    if establish_col:
        est_year = extract_year(df[establish_col])
        df["成立年份"] = est_year
        df["成立年限"] = pd.Timestamp.today().year - df["成立年份"]
    else:
        df["成立年份"] = np.nan
        df["成立年限"] = np.nan

    if lon_col:
        df["经度"] = pd.to_numeric(df[lon_col], errors="coerce")
    else:
        df["经度"] = np.nan

    if lat_col:
        df["纬度"] = pd.to_numeric(df[lat_col], errors="coerce")
    else:
        df["纬度"] = np.nan

    # 如果没有第一题逐企业生存概率，先根据企业状态生成一个展示用生存概率
    status_text = df["企业状态_标准"].astype(str)
    df["生存概率"] = np.where(
        status_text.str.contains("存活|正常|开业|在营|存续", regex=True),
        0.85,
        np.where(
            status_text.str.contains("注销|吊销|倒闭|迁出|撤销", regex=True),
            0.15,
            0.55
        )
    )

    return df


def prepare_jobs(df_job_raw, df_basic):
    df = df_job_raw.copy()

    company_col = find_col(df, ["企业名称", "公司名称"])
    if company_col and company_col != "企业名称":
        df["企业名称"] = df[company_col]

    time_col = find_col(df, ["职位刷新时间", "发布时间", "日期", "时间"])
    place_col = find_col(df, ["工作地点", "岗位所在地", "工作城市"])
    job_col = find_col(df, ["职位名称", "岗位名称", "岗位"])
    func_col = find_col(df, ["职位职能", "岗位类别", "职位类别"])
    recruit_col = find_col(df, ["招聘人数", "招聘人数_最终"])
    salary_col = find_col(df, ["平均年薪(万)", "平均薪资", "薪资", "薪水范围"])
    # 词云优先使用“职位描述_清洗”，避免分词列中的分隔符/异常字符导致显示混乱
    skill_col = find_col(df, ["职位描述_清洗", "职位描述", "职位描述_分词", "技能关键词"])
    edu_col = find_col(df, ["学历", "最低学历"])
    exp_col = find_col(df, ["工作年限", "经验要求"])

    if time_col:
        df["年份"] = extract_year(df[time_col])
    else:
        df["年份"] = np.nan

    if place_col:
        df["岗位所在地"] = df[place_col].apply(normalize_city)
    else:
        df["岗位所在地"] = np.nan

    df["岗位名称"] = df[job_col].fillna("未知岗位").astype(str) if job_col else "未知岗位"
    df["岗位类别"] = df[func_col].fillna(df["岗位名称"]).astype(str) if func_col else df["岗位名称"]

    if recruit_col:
        df["招聘人数_数值"] = to_num(df[recruit_col]).fillna(1).clip(lower=0)
    else:
        df["招聘人数_数值"] = 1

    if salary_col:
        df["平均年薪_万"] = pd.to_numeric(df[salary_col], errors="coerce")
    else:
        df["平均年薪_万"] = np.nan

    if skill_col:
        df["技能文本"] = df[skill_col].fillna("").astype(str)
        df["词云文本"] = df[skill_col].fillna("").astype(str)
    else:
        df["技能文本"] = ""
        df["词云文本"] = ""

    if edu_col:
        df["学历"] = df[edu_col].fillna("未知").astype(str)
    else:
        df["学历"] = "未知"

    if exp_col:
        df["工作年限"] = df[exp_col].fillna("未知").astype(str)
    else:
        df["工作年限"] = "未知"

    # 保留组织文化特征
    for c in ["压力文化词频", "奋斗精神词频", "福利待遇词频"]:
        if c not in df.columns:
            df[c] = 0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    # 合并企业基础信息
    if "企业名称" in df.columns and "企业名称" in df_basic.columns:
        keep_cols = [
            "企业名称", "企业所在地", "行业", "经度", "纬度",
            "注册资本_万元", "成立年限", "企业状态_标准", "生存概率"
        ]
        base = df_basic[keep_cols].drop_duplicates(subset=["企业名称"])
        df = df.merge(base, on="企业名称", how="left", suffixes=("", "_企业"))

    return df


def build_enterprise_year(df_jobs):
    if df_jobs.empty or "企业名称" not in df_jobs.columns:
        return pd.DataFrame()

    df = df_jobs.dropna(subset=["年份"]).copy()
    if df.empty:
        return pd.DataFrame()

    g = df.groupby(["企业名称", "年份"], as_index=False).agg(
        企业所在地=("企业所在地", "first"),
        行业=("行业", "first"),
        经度=("经度", "first"),
        纬度=("纬度", "first"),
        年度招聘人数=("招聘人数_数值", "sum"),
        年度招聘次数=("企业名称", "size"),
        岗位类型数=("岗位类别", pd.Series.nunique),
        平均年薪_万=("平均年薪_万", "mean"),
        生存概率=("生存概率", "first"),
    )
    return g


def auto_dbscan_centers(df_ey, eps_km=6.0, min_samples=3):
    if DBSCAN is None or df_ey.empty:
        return pd.DataFrame()

    need = ["企业所在地", "年份", "经度", "纬度", "年度招聘人数"]
    if any(c not in df_ey.columns for c in need):
        return pd.DataFrame()

    records = []

    for (city, year), g in df_ey.dropna(subset=["经度", "纬度"]).groupby(["企业所在地", "年份"]):
        g = g[g["年度招聘人数"].fillna(0) > 0].copy()
        if len(g) < min_samples:
            continue

        lat0 = g["纬度"].mean()
        x = g["经度"].astype(float) * 111 * math.cos(math.radians(lat0))
        y = g["纬度"].astype(float) * 111
        coords = np.vstack([x, y]).T

        labels = DBSCAN(eps=eps_km, min_samples=min_samples).fit_predict(coords)
        g["cluster"] = labels

        total = g["年度招聘人数"].sum()

        for lab, cg in g[g["cluster"] >= 0].groupby("cluster"):
            w = cg["年度招聘人数"].fillna(0).astype(float)
            if w.sum() > 0:
                lon_c = np.average(cg["经度"], weights=w)
                lat_c = np.average(cg["纬度"], weights=w)
            else:
                lon_c = cg["经度"].mean()
                lat_c = cg["纬度"].mean()

            records.append({
                "城市": city,
                "年份": int(year),
                "中心编号": f"C{int(lab)+1}",
                "中心经度": lon_c,
                "中心纬度": lat_c,
                "中心招聘人数": w.sum(),
                "中心企业数量": len(cg),
                "中心招聘占比": w.sum() / total if total else np.nan
            })

    out = pd.DataFrame(records)
    if out.empty:
        return out

    out = out.sort_values(["城市", "年份", "中心招聘人数"], ascending=[True, True, False])
    out["中心序号"] = out.groupby(["城市", "年份"]).cumcount() + 1
    out["中心编号"] = "C" + out["中心序号"].astype(str)
    return out.drop(columns=["中心序号"])



def compute_rf_survival_probability(df_basic, df_jobs, df_ey):
    """
    用随机森林模型重新估计每家企业的生存概率。

    说明：
    - 目标变量来自企业状态：
      存活/正常/开业/在营/存续 -> 1
      注销/吊销/倒闭/迁出/撤销 -> 0
    - 特征来自企业工商属性 + 招聘行为聚合特征。
    - 如果数据不足、只有单一类别或 sklearn 不可用，则回退到原来的状态展示概率。
    """
    info = {
        "used_rf": False,
        "message": "",
        "feature_count": 0,
        "sample_count": 0,
        "positive_count": 0,
        "negative_count": 0,
        "test_accuracy": None,
        "test_auc": None,
        "top_features": pd.DataFrame()
    }

    if RandomForestClassifier is None:
        info["message"] = "未安装 scikit-learn，暂时使用状态占位概率。"
        return df_basic, info

    if df_basic is None or df_basic.empty or "企业名称" not in df_basic.columns:
        info["message"] = "企业基础信息为空或缺少企业名称，无法训练随机森林。"
        return df_basic, info

    df = df_basic.copy()

    # 1) 构造目标变量
    status = df.get("企业状态_标准", pd.Series(index=df.index, dtype=object)).astype(str)
    y = pd.Series(np.nan, index=df.index)

    alive_pattern = r"存活|正常|开业|在营|存续"
    dead_pattern = r"注销|吊销|倒闭|迁出|撤销|停业|关闭"

    y[status.str.contains(alive_pattern, regex=True, na=False)] = 1
    y[status.str.contains(dead_pattern, regex=True, na=False)] = 0

    # 2) 企业层面招聘特征聚合
    feat = df[["企业名称"]].drop_duplicates().copy()

    base_cols = ["企业名称", "企业所在地", "行业", "注册资本_万元", "成立年限"]
    base_cols = [c for c in base_cols if c in df.columns]
    base_feat = df[base_cols].drop_duplicates(subset=["企业名称"])
    feat = feat.merge(base_feat, on="企业名称", how="left")

    if df_jobs is not None and not df_jobs.empty and "企业名称" in df_jobs.columns:
        job_agg_dict = {}
        if "招聘人数_数值" in df_jobs.columns:
            job_agg_dict["招聘总人数"] = ("招聘人数_数值", "sum")
            job_agg_dict["平均单条招聘人数"] = ("招聘人数_数值", "mean")
        if "平均年薪_万" in df_jobs.columns:
            job_agg_dict["平均年薪_万"] = ("平均年薪_万", "mean")
            job_agg_dict["最高年薪_万"] = ("平均年薪_万", "max")
        if "岗位名称" in df_jobs.columns:
            job_agg_dict["岗位名称数"] = ("岗位名称", pd.Series.nunique)
        if "岗位类别" in df_jobs.columns:
            job_agg_dict["岗位类别数"] = ("岗位类别", pd.Series.nunique)
        for c in ["压力文化词频", "奋斗精神词频", "福利待遇词频"]:
            if c in df_jobs.columns:
                job_agg_dict[c + "_合计"] = (c, "sum")
                job_agg_dict[c + "_均值"] = (c, "mean")
        if "年份" in df_jobs.columns:
            job_agg_dict["招聘年份数"] = ("年份", pd.Series.nunique)

        if job_agg_dict:
            job_feat = df_jobs.groupby("企业名称", as_index=False).agg(**job_agg_dict)
            feat = feat.merge(job_feat, on="企业名称", how="left")

    if df_ey is not None and not df_ey.empty and "企业名称" in df_ey.columns:
        ey_agg_dict = {}
        if "年度招聘人数" in df_ey.columns:
            ey_agg_dict["年度招聘人数均值"] = ("年度招聘人数", "mean")
            ey_agg_dict["年度招聘人数最大值"] = ("年度招聘人数", "max")
            ey_agg_dict["年度招聘人数标准差"] = ("年度招聘人数", "std")
        if "年度招聘次数" in df_ey.columns:
            ey_agg_dict["年度招聘次数均值"] = ("年度招聘次数", "mean")
        if "岗位类型数" in df_ey.columns:
            ey_agg_dict["岗位类型数均值"] = ("岗位类型数", "mean")

        if ey_agg_dict:
            ey_feat = df_ey.groupby("企业名称", as_index=False).agg(**ey_agg_dict)
            feat = feat.merge(ey_feat, on="企业名称", how="left")

    model_df = feat.merge(
        pd.DataFrame({"企业名称": df["企业名称"], "target": y}),
        on="企业名称",
        how="left"
    )
    model_df = model_df.dropna(subset=["target"]).drop_duplicates(subset=["企业名称"])

    info["sample_count"] = int(len(model_df))
    info["positive_count"] = int((model_df["target"] == 1).sum())
    info["negative_count"] = int((model_df["target"] == 0).sum())

    if model_df["target"].nunique() < 2:
        info["message"] = "企业状态只有单一类别，无法训练随机森林，暂时使用状态占位概率。"
        return df_basic, info

    if len(model_df) < 30:
        info["message"] = "可用于训练的企业样本过少，无法稳定训练随机森林，暂时使用状态占位概率。"
        return df_basic, info

    # 3) 选择特征
    exclude_cols = {"企业名称", "target"}
    feature_cols = [c for c in model_df.columns if c not in exclude_cols]

    numeric_cols = [
        c for c in feature_cols
        if pd.api.types.is_numeric_dtype(model_df[c])
    ]
    categorical_cols = [c for c in feature_cols if c not in numeric_cols]

    if not numeric_cols and not categorical_cols:
        info["message"] = "没有可用特征，无法训练随机森林。"
        return df_basic, info

    info["feature_count"] = len(numeric_cols) + len(categorical_cols)

    # 兼容不同 sklearn 版本的 OneHotEncoder 参数
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    preprocess = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric_cols),
            ("cat", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", ohe)
            ]), categorical_cols)
        ],
        remainder="drop"
    )

    clf = RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        min_samples_leaf=2,
        random_state=42,
        class_weight="balanced_subsample",
        n_jobs=-1
    )

    model = Pipeline(steps=[
        ("preprocess", preprocess),
        ("rf", clf)
    ])

    X = model_df[feature_cols]
    y_model = model_df["target"].astype(int)

    # 4) 简单留出集评价，仅用于页面说明
    can_split = min(y_model.value_counts()) >= 5 and len(model_df) >= 80
    if can_split:
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_model,
                test_size=0.25,
                random_state=42,
                stratify=y_model
            )
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            prob = model.predict_proba(X_test)[:, list(model.classes_).index(1)]
            info["test_accuracy"] = float(accuracy_score(y_test, pred))
            try:
                info["test_auc"] = float(roc_auc_score(y_test, prob))
            except Exception:
                info["test_auc"] = None
        except Exception:
            pass

    # 5) 用全量样本训练，并对所有企业预测生存概率
    model.fit(X, y_model)

    all_feat = feat.copy()
    for c in feature_cols:
        if c not in all_feat.columns:
            all_feat[c] = np.nan

    proba = model.predict_proba(all_feat[feature_cols])
    class_list = list(model.classes_)
    if 1 in class_list:
        p_alive = proba[:, class_list.index(1)]
    else:
        p_alive = np.full(len(all_feat), np.nan)

    prob_map = pd.Series(p_alive, index=all_feat["企业名称"]).to_dict()
    df["生存概率"] = df["企业名称"].map(prob_map).fillna(df["生存概率"])
    df["生存概率"] = pd.to_numeric(df["生存概率"], errors="coerce").clip(0, 1)

    # 6) 特征重要度
    try:
        rf = model.named_steps["rf"]
        pre = model.named_steps["preprocess"]
        feature_names = []
        feature_names.extend(numeric_cols)

        if categorical_cols:
            cat_pipe = pre.named_transformers_["cat"]
            ohe_step = cat_pipe.named_steps["onehot"]
            cat_names = ohe_step.get_feature_names_out(categorical_cols).tolist()
            feature_names.extend(cat_names)

        imp = pd.DataFrame({
            "特征": feature_names,
            "重要度": rf.feature_importances_
        }).sort_values("重要度", ascending=False).head(20)
        info["top_features"] = imp
    except Exception:
        pass

    info["used_rf"] = True
    info["message"] = "已使用随机森林模型重新估计企业生存概率。"
    return df, info


def compute_industry_radius(df_ey):
    """行业空间集聚半径：简化经纬度距离版，单位 km。"""
    if df_ey.empty:
        return pd.DataFrame()

    need = ["企业所在地", "年份", "行业", "经度", "纬度", "年度招聘人数"]
    if any(c not in df_ey.columns for c in need):
        return pd.DataFrame()

    records = []
    d = df_ey.dropna(subset=["经度", "纬度", "行业"]).copy()
    d = d[d["年度招聘人数"].fillna(0) > 0]

    for (city, year, industry), g in d.groupby(["企业所在地", "年份", "行业"]):
        if len(g) < 2:
            continue

        w = g["年度招聘人数"].fillna(0).astype(float).values
        if w.sum() <= 0:
            continue

        lon_c = np.average(g["经度"].astype(float), weights=w)
        lat_c = np.average(g["纬度"].astype(float), weights=w)

        lon = np.radians(g["经度"].astype(float).values)
        lat = np.radians(g["纬度"].astype(float).values)
        lon0 = math.radians(lon_c)
        lat0 = math.radians(lat_c)

        a = np.sin((lat - lat0) / 2) ** 2 + np.cos(lat0) * np.cos(lat) * np.sin((lon - lon0) / 2) ** 2
        dist = 6371 * 2 * np.arcsin(np.sqrt(a))

        order = np.argsort(dist)
        dist_sorted = dist[order]
        w_sorted = w[order]
        cum = np.cumsum(w_sorted) / w_sorted.sum()

        def radius(q):
            idx = np.searchsorted(cum, q, side="left")
            idx = min(idx, len(dist_sorted) - 1)
            return float(dist_sorted[idx])

        records.append({
            "城市": city,
            "年份": int(year),
            "行业": industry,
            "行业中心经度": lon_c,
            "行业中心纬度": lat_c,
            "50%覆盖半径_km": radius(0.5),
            "80%覆盖半径_km": radius(0.8),
            "90%覆盖半径_km": radius(0.9),
            "加权平均半径_km": float(np.average(dist, weights=w)),
            "招聘总人数": float(w.sum()),
            "企业数量": len(g)
        })

    return pd.DataFrame(records)


# =============================================================================
# 3. 可视化函数
# =============================================================================
def plot_enterprise_map(df_basic, gdf_boundary=None):
    df = df_basic.dropna(subset=["经度", "纬度"]).copy()
    df = df[valid_coord(df, "经度", "纬度")]
    df = downsample_points(df, max_points=7000, weight_col="生存概率")

    fig = px.scatter_mapbox(
        df,
        lat="纬度",
        lon="经度",
        color="生存概率",
        size="生存概率",
        hover_name="企业名称",
        hover_data=["企业所在地", "行业", "注册资本_万元", "成立年限", "企业状态_标准"],
        color_continuous_scale="RdYlGn",
        size_max=14,
        mapbox_style="carto-positron",
        zoom=4,
        title="产业主体来源地图：企业所在地 × 生存概率"
    )

    if not df.empty:
        fig.update_layout(mapbox_center={"lat": df["纬度"].mean(), "lon": df["经度"].mean()})

    if gdf_boundary is not None and not gdf_boundary.empty:
        try:
            fig.add_choroplethmapbox(
                geojson=gdf_boundary.__geo_interface__,
                locations=gdf_boundary.index,
                z=[0] * len(gdf_boundary),
                colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(0,0,0,0)"]],
                marker_line_width=1.5,
                marker_line_color="black",
                showscale=False,
                hoverinfo="skip"
            )
        except Exception:
            pass

    fig.update_layout(height=650, margin=dict(l=0, r=0, t=50, b=0))
    return fig


def plot_city_count(df_basic, gdf_boundary=None):
    city_count = df_basic.groupby("企业所在地").size().reset_index(name="企业数量")

    # 有边界时尝试面图，没有边界就柱状图
    if gdf_boundary is not None and not gdf_boundary.empty:
        try:
            gdf = gdf_boundary.copy()
            city_col = None
            for c in gdf.columns:
                if any(k in str(c).lower() for k in ["name", "city", "市", "名称"]):
                    city_col = c
                    break
            if city_col is None:
                gdf["城市名"] = gdf["来源文件"].apply(normalize_city) if "来源文件" in gdf.columns else ""
            else:
                gdf["城市名"] = gdf[city_col].apply(normalize_city)

            merged = gdf.merge(city_count, left_on="城市名", right_on="企业所在地", how="left")
            merged["企业数量"] = merged["企业数量"].fillna(0)

            fig = px.choropleth_mapbox(
                merged,
                geojson=merged.__geo_interface__,
                locations=merged.index,
                color="企业数量",
                hover_name="城市名",
                mapbox_style="carto-positron",
                center={"lat": 32.5, "lon": 113.0},
                zoom=3.2,
                color_continuous_scale="Blues",
                title="北京、苏州、深圳企业数量"
            )
            fig.update_layout(height=500, margin=dict(l=0, r=0, t=50, b=0))
            return fig
        except Exception:
            pass

    fig = px.bar(city_count, x="企业所在地", y="企业数量", text="企业数量", title="北京、苏州、深圳企业数量")
    return fig


def plot_job_heat(df_jobs, metric):
    df = df_jobs.dropna(subset=["经度", "纬度"]).copy()
    df = df[valid_coord(df, "经度", "纬度")]
    df = downsample_points(df, max_points=8000, weight_col=metric)

    if df.empty:
        return None

    if metric not in df.columns:
        metric = "招聘人数_数值"

    fig = px.density_mapbox(
        df,
        lat="纬度",
        lon="经度",
        z=metric,
        radius=22,
        mapbox_style="carto-positron",
        center={"lat": df["纬度"].mean(), "lon": df["经度"].mean()},
        zoom=5,
        title=f"岗位需求热力图：{metric}"
    )
    fig.update_layout(height=650, margin=dict(l=0, r=0, t=50, b=0))
    return fig



def get_chinese_font_path():
    """
    自动寻找本机中文字体，解决词云中文显示成方框的问题。
    Windows 常见：微软雅黑 msyh.ttc、黑体 simhei.ttf。
    """
    candidates = [
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\msyh.ttf",
        r"C:\Windows\Fonts\simhei.ttf",
        r"C:\Windows\Fonts\simsun.ttc",
        r"C:\Windows\Fonts\simkai.ttf",
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    ]
    for p in candidates:
        if Path(p).exists():
            return p
    return None


STOPWORDS_CN = {
    "我们", "你们", "他们", "以及", "或者", "进行", "相关", "负责", "工作", "职位",
    "岗位", "要求", "能力", "优先", "具有", "具备", "熟悉", "了解", "掌握", "使用",
    "公司", "团队", "完成", "根据", "提供", "以上", "以下", "经验", "学历", "本科",
    "大专", "招聘", "描述", "任职", "职责", "良好", "较强", "一定", "能够", "可以",
    "需要", "包括", "参与", "协助", "支持", "客户", "项目", "产品", "业务", "岗位职责",
    "任职资格", "职位要求"
}


def tokenize_for_wordcloud(text):
    """
    使用“职位描述_清洗”生成词云：
    - 如果安装了 jieba，用 jieba 分词；
    - 如果未安装，则用正则提取中文/英文词。
    """
    text = str(text).replace("/", " ").replace("\\n", " ").replace("\\t", " ")
    text = re.sub(r"[0-9]+", " ", text)
    text = re.sub(r"[^\u4e00-\u9fa5A-Za-z+#.]+", " ", text)

    if jieba is not None:
        words = jieba.lcut(text)
    else:
        words = re.findall(r"[\u4e00-\u9fa5]{2,}|[A-Za-z][A-Za-z+#.]{1,}", text)

    clean_words = []
    for w in words:
        w = str(w).strip()
        if len(w) < 2:
            continue
        if w in STOPWORDS_CN:
            continue
        if re.fullmatch(r"[A-Za-z]+", w) and len(w) <= 1:
            continue
        clean_words.append(w)
    return clean_words


def make_wordcloud(text):
    fig, ax = plt.subplots(figsize=(12, 6))

    words = tokenize_for_wordcloud(text)
    if not words:
        ax.text(0.5, 0.5, "暂无可生成词云的文本", ha="center", va="center", fontsize=16)
        ax.axis("off")
        return fig

    freq = Counter(words)
    font_path = get_chinese_font_path()

    if font_path is None:
        ax.text(
            0.5, 0.5,
            "未找到中文字体，词云可能显示为方框。\n建议安装/使用微软雅黑或黑体。",
            ha="center", va="center", fontsize=14
        )
        ax.axis("off")
        return fig

    wc = WordCloud(
        font_path=get_chinese_font_path(),
        width=1200,
        height=600,
        background_color="white",
        max_words=180,
        collocations=False,
        prefer_horizontal=0.9
    ).generate_from_frequencies(freq)

    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    return fig


def hex_to_rgba(hex_color, alpha=0.38):
    """把 #RRGGBB 转成 rgba，用于桑基图边颜色。"""
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def plot_sankey(df, source, target, value, title):
    """
    彩色桑基图：
    - 节点自动使用多色调色板；
    - 流向边颜色跟随来源节点，并设置透明度；
    - hover 显示来源、去向和流量。
    """
    d = df.copy()
    d[source] = d[source].fillna("未知").astype(str)
    d[target] = d[target].fillna("未知").astype(str)
    d[value] = pd.to_numeric(d[value], errors="coerce").fillna(1)

    flow = d.groupby([source, target], as_index=False)[value].sum()
    flow = flow[flow[value] > 0]
    flow = flow.sort_values(value, ascending=False).head(80)

    nodes = pd.unique(pd.concat([flow[source], flow[target]], ignore_index=True)).tolist()
    idx = {n: i for i, n in enumerate(nodes)}

    # 比较适合 PPT 展示的明亮色板
    palette = [
        "#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F",
        "#EDC948", "#B07AA1", "#FF9DA7", "#9C755F", "#BAB0AC",
        "#1F77B4", "#FF7F0E", "#2CA02C", "#D62728", "#9467BD",
        "#8C564B", "#E377C2", "#7F7F7F", "#BCBD22", "#17BECF",
        "#6A5ACD", "#20B2AA", "#FF6347", "#3CB371", "#DAA520"
    ]

    node_colors = [palette[i % len(palette)] for i in range(len(nodes))]
    node_color_map = {node: node_colors[i] for i, node in enumerate(nodes)}

    source_ids = [idx[x] for x in flow[source]]
    target_ids = [idx[x] for x in flow[target]]
    values = flow[value].tolist()

    # 边颜色跟随来源节点，透明一些更美观
    link_colors = [hex_to_rgba(node_color_map[s], alpha=0.35) for s in flow[source]]

    customdata = np.stack([flow[source], flow[target]], axis=-1)

    fig = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(
            pad=20,
            thickness=20,
            line=dict(color="rgba(60,60,60,0.35)", width=0.5),
            label=nodes,
            color=node_colors,
            hovertemplate="节点：%{label}<extra></extra>"
        ),
        link=dict(
            source=source_ids,
            target=target_ids,
            value=values,
            color=link_colors,
            customdata=customdata,
            hovertemplate=(
                "%{customdata[0]} → %{customdata[1]}<br>"
                "流量：%{value:,.0f}<extra></extra>"
            )
        )
    ))

    fig.update_layout(
        title=title,
        height=650,
        font=dict(size=13),
        margin=dict(l=10, r=10, t=55, b=10),
        paper_bgcolor="white",
        plot_bgcolor="white"
    )
    return fig


def plot_flight_map(df_jobs, df_centers):
    city_coord = {
        "北京": (116.4074, 39.9042),
        "苏州": (120.5853, 31.2989),
        "深圳": (114.0579, 22.5431),
    }

    flow = df_jobs.groupby(["企业所在地", "岗位所在地"], as_index=False)["招聘人数_数值"].sum()
    flow = flow.sort_values("招聘人数_数值", ascending=False).head(60)

    fig = go.Figure()

    for _, r in flow.iterrows():
        s, t = r["企业所在地"], r["岗位所在地"]
        if s not in city_coord or t not in city_coord:
            continue

        lon0, lat0 = city_coord[s]
        lon1, lat1 = city_coord[t]
        width = max(1, min(8, math.log1p(r["招聘人数_数值"])))

        fig.add_trace(go.Scattermapbox(
            lon=[lon0, lon1],
            lat=[lat0, lat1],
            mode="lines",
            line=dict(width=width),
            hovertext=f"{s} → {t}<br>招聘人数：{r['招聘人数_数值']:.0f}",
            hoverinfo="text",
            showlegend=False
        ))

    node_df = pd.DataFrame([
        {"城市": c, "经度": city_coord[c][0], "纬度": city_coord[c][1]}
        for c in city_coord
    ])

    fig.add_trace(go.Scattermapbox(
        lon=node_df["经度"],
        lat=node_df["纬度"],
        mode="markers+text",
        marker=dict(size=18),
        text=node_df["城市"],
        textposition="top center",
        name="城市节点"
    ))

    if df_centers is not None and not df_centers.empty:
        fig.add_trace(go.Scattermapbox(
            lon=df_centers["中心经度"],
            lat=df_centers["中心纬度"],
            mode="markers+text",
            marker=dict(size=12, symbol="star"),
            text=df_centers["中心编号"],
            textposition="bottom center",
            hovertext=[
                f"{r['城市']} {r['年份']} {r['中心编号']}<br>"
                f"中心招聘人数：{r['中心招聘人数']:.0f}<br>"
                f"中心招聘占比：{r['中心招聘占比']:.2%}"
                for _, r in df_centers.iterrows()
            ],
            name="就业中心"
        ))

    fig.update_layout(
        title="企业所在地 → 岗位所在地地图飞线图（叠加就业中心）",
        mapbox_style="carto-positron",
        mapbox_zoom=3.2,
        mapbox_center={"lat": 31.8, "lon": 113.5},
        height=650,
        margin=dict(l=0, r=0, t=50, b=0)
    )

    return fig


def plot_city_network(df_jobs):
    flow = df_jobs.groupby(["企业所在地", "岗位所在地"], as_index=False)["招聘人数_数值"].sum()
    flow = flow[flow["招聘人数_数值"] > 0].head(100)

    G = nx.DiGraph()
    for _, r in flow.iterrows():
        G.add_edge(r["企业所在地"], r["岗位所在地"], weight=float(r["招聘人数_数值"]))

    if len(G.nodes) == 0:
        return None

    pos = nx.spring_layout(G, seed=42, k=0.8)

    edge_x, edge_y = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    strength = dict(G.degree(weight="weight"))
    max_s = max(strength.values()) if strength else 1

    node_x, node_y, text, size = [], [], [], []
    for n in G.nodes():
        x, y = pos[n]
        node_x.append(x)
        node_y.append(y)
        text.append(n)
        size.append(16 + 35 * strength.get(n, 1) / max_s)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines", line=dict(width=1), hoverinfo="none"))
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        marker=dict(size=size),
        text=text,
        textposition="top center"
    ))
    fig.update_layout(
        title="城市招聘流向网络图",
        height=600,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        showlegend=False
    )
    return fig


def plot_industry_radius(df_radius):
    if df_radius is None or df_radius.empty:
        return None

    fig = px.scatter_mapbox(
        df_radius,
        lat="行业中心纬度",
        lon="行业中心经度",
        color="行业",
        size="90%覆盖半径_km",
        hover_name="行业",
        hover_data=["城市", "年份", "50%覆盖半径_km", "80%覆盖半径_km", "90%覆盖半径_km", "招聘总人数", "企业数量"],
        mapbox_style="carto-positron",
        zoom=4,
        center={"lat": df_radius["行业中心纬度"].mean(), "lon": df_radius["行业中心经度"].mean()},
        title="行业招聘加权中心与覆盖半径"
    )
    fig.update_layout(height=650, margin=dict(l=0, r=0, t=50, b=0))
    return fig



def get_city_zoom_center(city, df_points=None, gdf_boundary=None):
    """让地图只聚焦当前城市，避免全国视角太大。"""
    city_defaults = {
        "北京": {"lat": 39.9042, "lon": 116.4074, "zoom": 8.2},
        "苏州": {"lat": 31.2989, "lon": 120.5853, "zoom": 8.5},
        "深圳": {"lat": 22.5431, "lon": 114.0579, "zoom": 9.0},
    }

    if city in city_defaults:
        return {"lat": city_defaults[city]["lat"], "lon": city_defaults[city]["lon"]}, city_defaults[city]["zoom"]

    if df_points is not None and not df_points.empty and {"经度", "纬度"}.issubset(df_points.columns):
        d = df_points.dropna(subset=["经度", "纬度"])
        if not d.empty:
            return {"lat": d["纬度"].mean(), "lon": d["经度"].mean()}, 9

    return {"lat": 31.5, "lon": 113.5}, 4


def judge_core_structure(df_centers):
    """
    根据第三题口径进行单核/多核判断：
    - 主要就业中心：中心招聘占比 >= 5%
    - 单核结构：主要中心数=1，且最大中心占比 >= 50%
    - 强主中心+次中心：主要中心数>=2，且最大中心占比 >= 50%
    - 多核结构：主要中心数>=2，且最大中心占比 < 50%
    """
    if df_centers is None or df_centers.empty:
        return pd.DataFrame()

    d = df_centers.copy()
    if "中心招聘占比" not in d.columns:
        if "中心招聘人数" in d.columns:
            d["中心招聘占比"] = d.groupby(["城市", "年份"])["中心招聘人数"].transform(lambda x: x / x.sum())
        else:
            return pd.DataFrame()

    records = []
    for (city, year), g in d.groupby(["城市", "年份"]):
        g = g.copy()
        main = g[g["中心招聘占比"].fillna(0) >= 0.05]
        n_main = len(main)
        max_share = g["中心招聘占比"].max()
        total_centers = len(g)

        if n_main <= 0:
            structure = "未识别出主要中心"
        elif n_main == 1 and max_share >= 0.50:
            structure = "单核结构"
        elif n_main >= 2 and max_share >= 0.50:
            structure = "强主中心+次中心"
        elif n_main >= 2 and max_share < 0.50:
            structure = "多核结构"
        else:
            structure = "弱中心结构"

        records.append({
            "城市": city,
            "年份": int(year) if pd.notna(year) else year,
            "就业中心总数": total_centers,
            "主要就业中心数_占比不低于5%": n_main,
            "最大中心招聘占比": max_share,
            "空间结构判断": structure
        })

    out = pd.DataFrame(records).sort_values(["城市", "年份"])
    return out


def plot_city_year_center_map(df_ey, df_centers, gdf_boundary, city, year):
    """城市-年份就业中心地图：企业点 + 就业中心点 + 城市边界。"""
    pts = df_ey.copy()
    if "企业所在地" in pts.columns:
        pts = pts[pts["企业所在地"] == city]
    if "年份" in pts.columns:
        pts = pts[pts["年份"] == year]
    pts = pts.dropna(subset=["经度", "纬度"]).copy()
    if not pts.empty:
        pts = pts[valid_coord(pts, "经度", "纬度")]
        pts = downsample_points(pts, max_points=5000, weight_col="年度招聘人数")

    centers = df_centers.copy() if df_centers is not None else pd.DataFrame()
    if not centers.empty:
        centers = centers[(centers["城市"] == city) & (centers["年份"] == year)].copy()

    center, zoom = get_city_zoom_center(city, pts, gdf_boundary)

    fig = go.Figure()

    # 城市边界
    if gdf_boundary is not None and not gdf_boundary.empty:
        try:
            gdf = gdf_boundary.copy()
            # 根据来源文件尽量筛选当前城市边界
            if "来源文件" in gdf.columns:
                gdf_city = gdf[gdf["来源文件"].astype(str).apply(lambda x: city in normalize_city(x) or city in x)]
                if gdf_city.empty:
                    gdf_city = gdf
            else:
                gdf_city = gdf

            fig.add_choroplethmapbox(
                geojson=gdf_city.__geo_interface__,
                locations=gdf_city.index,
                z=[0] * len(gdf_city),
                colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(0,0,0,0)"]],
                marker_line_width=1.4,
                marker_line_color="black",
                showscale=False,
                hoverinfo="skip",
                name="城市边界"
            )
        except Exception:
            pass

    if not pts.empty:
        size_col = "年度招聘人数" if "年度招聘人数" in pts.columns else None
        marker_size = None
        if size_col:
            val = pd.to_numeric(pts[size_col], errors="coerce").fillna(0)
            if val.max() > 0:
                marker_size = 6 + 18 * np.sqrt(val / val.max())
            else:
                marker_size = 6
        else:
            marker_size = 6

        fig.add_trace(go.Scattermapbox(
            lon=pts["经度"],
            lat=pts["纬度"],
            mode="markers",
            marker=dict(size=marker_size, opacity=0.45),
            hovertext=[
                f"企业：{r.get('企业名称', '')}<br>"
                f"行业：{r.get('行业', '')}<br>"
                f"年度招聘人数：{r.get('年度招聘人数', '')}<br>"
                f"生存概率：{r.get('生存概率', '')}"
                for _, r in pts.iterrows()
            ],
            hoverinfo="text",
            name="企业点"
        ))

    if not centers.empty:
        fig.add_trace(go.Scattermapbox(
            lon=centers["中心经度"],
            lat=centers["中心纬度"],
            mode="markers+text",
            marker=dict(size=18, symbol="star"),
            text=centers["中心编号"],
            textposition="top center",
            hovertext=[
                f"{r['城市']} {r['年份']} {r['中心编号']}<br>"
                f"中心招聘人数：{r.get('中心招聘人数', np.nan):.0f}<br>"
                f"中心企业数量：{r.get('中心企业数量', np.nan)}<br>"
                f"中心招聘占比：{r.get('中心招聘占比', np.nan):.2%}"
                for _, r in centers.iterrows()
            ],
            hoverinfo="text",
            name="就业中心"
        ))

    fig.update_layout(
        title=f"{city}{int(year)}年就业中心识别地图",
        mapbox_style="carto-positron",
        mapbox_center=center,
        mapbox_zoom=zoom,
        height=620,
        margin=dict(l=0, r=0, t=50, b=0),
        legend=dict(orientation="h", y=0.02, x=0.02)
    )
    return fig


def plot_core_evolution(struct_df):
    """单核/多核三年演化分析图。"""
    if struct_df is None or struct_df.empty:
        return None, None

    fig1 = px.line(
        struct_df,
        x="年份",
        y="主要就业中心数_占比不低于5%",
        color="城市",
        markers=True,
        title="三年演化：主要就业中心数量变化"
    )

    fig2 = px.line(
        struct_df,
        x="年份",
        y="最大中心招聘占比",
        color="城市",
        markers=True,
        title="三年演化：最大中心招聘占比变化"
    )
    fig2.update_yaxes(tickformat=".0%")
    return fig1, fig2


def plot_center_strength_evolution(df_centers):
    """各城市就业中心招聘强度三年变化。"""
    if df_centers is None or df_centers.empty:
        return None
    d = df_centers.copy()
    d["中心标签"] = d["城市"].astype(str) + "-" + d["中心编号"].astype(str)
    fig = px.line(
        d,
        x="年份",
        y="中心招聘人数",
        color="中心标签",
        markers=True,
        facet_col="城市",
        facet_col_wrap=3,
        title="三年演化：各就业中心招聘强度变化"
    )
    fig.update_layout(height=520)
    return fig


def radius_to_lonlat_circle(lon, lat, radius_km, n=160):
    """把 km 半径近似转成经纬度圆，用于九宫格展示。"""
    angles = np.linspace(0, 2 * np.pi, n)
    lat_radius = radius_km / 111.0
    lon_radius = radius_km / (111.0 * max(np.cos(np.radians(lat)), 0.2))
    xs = lon + lon_radius * np.cos(angles)
    ys = lat + lat_radius * np.sin(angles)
    return xs, ys


def plot_industry_radius_nine_grid(df_ey, df_radius, gdf_boundary, industry, cities, years):
    """
    行业集聚半径九宫格：城市 × 年份。
    包含企业点、行业加权中心、50/80/90覆盖半径。
    """
    if df_radius is None or df_radius.empty:
        return None

    if not cities:
        cities = sorted(df_radius["城市"].dropna().unique().tolist())
    if not years:
        years = sorted(df_radius["年份"].dropna().unique().tolist())

    cities = cities[:3]
    years = sorted(years)[:3]

    if len(cities) == 0 or len(years) == 0:
        return None

    nrows, ncols = len(cities), len(years)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.8 * nrows), squeeze=False)

    for i, city in enumerate(cities):
        for j, year in enumerate(years):
            ax = axes[i][j]

            pts = df_ey.copy()
            if "企业所在地" in pts.columns:
                pts = pts[pts["企业所在地"] == city]
            if "年份" in pts.columns:
                pts = pts[pts["年份"] == year]
            if "行业" in pts.columns:
                pts = pts[pts["行业"] == industry]
            pts = pts.dropna(subset=["经度", "纬度"]).copy()

            rad = df_radius[
                (df_radius["城市"] == city) &
                (df_radius["年份"] == year) &
                (df_radius["行业"] == industry)
            ]

            # 画边界
            if gdf_boundary is not None and not gdf_boundary.empty:
                try:
                    gdf = gdf_boundary.copy()
                    if "来源文件" in gdf.columns:
                        gdf_city = gdf[gdf["来源文件"].astype(str).apply(lambda x: city in normalize_city(x) or city in x)]
                        if gdf_city.empty:
                            gdf_city = gdf
                    else:
                        gdf_city = gdf
                    gdf_city.boundary.plot(ax=ax, linewidth=0.8, color="black")
                except Exception:
                    pass

            if not pts.empty:
                size = pd.to_numeric(pts.get("年度招聘人数", pd.Series([1] * len(pts))), errors="coerce").fillna(1)
                size = 10 + 70 * np.sqrt(size / size.max()) if size.max() > 0 else 12
                ax.scatter(pts["经度"], pts["纬度"], s=size, alpha=0.45, label="企业点")

            if not rad.empty:
                r = rad.iloc[0]
                lon_c = r["行业中心经度"]
                lat_c = r["行业中心纬度"]
                ax.scatter([lon_c], [lat_c], s=90, marker="*", label="行业中心")

                for col, label, lw in [
                    ("50%覆盖半径_km", "50%", 1.4),
                    ("80%覆盖半径_km", "80%", 1.2),
                    ("90%覆盖半径_km", "90%", 1.0),
                ]:
                    if col in r and pd.notna(r[col]):
                        xs, ys = radius_to_lonlat_circle(lon_c, lat_c, float(r[col]))
                        ax.plot(xs, ys, linewidth=lw, label=label)

            ax.set_title(f"{city}-{int(year)}")
            ax.set_xlabel("经度")
            ax.set_ylabel("纬度")
            ax.grid(alpha=0.25)

            # 聚焦当前城市/行业点，避免范围过大
            if not pts.empty:
                pad_x = max((pts["经度"].max() - pts["经度"].min()) * 0.15, 0.03)
                pad_y = max((pts["纬度"].max() - pts["纬度"].min()) * 0.15, 0.03)
                ax.set_xlim(pts["经度"].min() - pad_x, pts["经度"].max() + pad_x)
                ax.set_ylim(pts["纬度"].min() - pad_y, pts["纬度"].max() + pad_y)

    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=5)
    fig.suptitle(f"行业集聚半径九宫格：{industry}", fontsize=16)
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])
    return fig



def plot_city_year_industry_map(df_ey, df_radius, gdf_boundary, city, year, industry):
    """
    第三题增强版：
    一个城市 + 一个年份 + 一个行业 = 一张地图
    展示：
    - 城市边界
    - 企业点（该城市、该年份、该行业）
    - 行业招聘加权中心
    - 50% / 80% / 90% 覆盖半径
    """
    if df_ey is None or df_ey.empty or df_radius is None or df_radius.empty:
        return None

    pts = df_ey.copy()
    if "企业所在地" in pts.columns:
        pts = pts[pts["企业所在地"] == city]
    if "年份" in pts.columns:
        pts = pts[pts["年份"] == year]
    if "行业" in pts.columns:
        pts = pts[pts["行业"] == industry]
    pts = pts.dropna(subset=["经度", "纬度"]).copy()

    if not pts.empty:
        pts = pts[valid_coord(pts, "经度", "纬度")]
        pts = downsample_points(pts, max_points=4000, weight_col="年度招聘人数")

    rad = df_radius[
        (df_radius["城市"] == city) &
        (df_radius["年份"] == year) &
        (df_radius["行业"] == industry)
    ].copy()

    center, zoom = get_city_zoom_center(city, pts, gdf_boundary)
    fig = go.Figure()

    # 城市边界
    if gdf_boundary is not None and not gdf_boundary.empty:
        try:
            gdf = gdf_boundary.copy()
            if "来源文件" in gdf.columns:
                gdf_city = gdf[gdf["来源文件"].astype(str).apply(lambda x: city in normalize_city(x) or city in x)]
                if gdf_city.empty:
                    gdf_city = gdf
            else:
                gdf_city = gdf

            fig.add_choroplethmapbox(
                geojson=gdf_city.__geo_interface__,
                locations=gdf_city.index,
                z=[0] * len(gdf_city),
                colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(0,0,0,0)"]],
                marker_line_width=1.5,
                marker_line_color="black",
                showscale=False,
                hoverinfo="skip",
                name="城市边界"
            )
        except Exception:
            pass

    # 企业点
    if not pts.empty:
        size_raw = pd.to_numeric(pts.get("年度招聘人数", pd.Series([1] * len(pts))), errors="coerce").fillna(1)
        size = 7 + 16 * np.sqrt(size_raw / size_raw.max()) if size_raw.max() > 0 else 8

        fig.add_trace(go.Scattermapbox(
            lon=pts["经度"],
            lat=pts["纬度"],
            mode="markers",
            marker=dict(size=size, color="rgba(31,119,180,0.55)"),
            hovertext=[
                f"企业：{r.get('企业名称', '')}<br>"
                f"行业：{r.get('行业', '')}<br>"
                f"年份：{r.get('年份', '')}<br>"
                f"年度招聘人数：{r.get('年度招聘人数', '')}<br>"
                f"平均年薪：{r.get('平均年薪_万', '')}"
                for _, r in pts.iterrows()
            ],
            hoverinfo="text",
            name="企业点"
        ))

    # 行业中心与集聚半径
    if not rad.empty:
        r = rad.iloc[0]
        lon_c = float(r["行业中心经度"])
        lat_c = float(r["行业中心纬度"])

        # 中心点
        fig.add_trace(go.Scattermapbox(
            lon=[lon_c],
            lat=[lat_c],
            mode="markers+text",
            marker=dict(size=16, color="orange", symbol="star"),
            text=[industry],
            textposition="top center",
            hovertext=[(
                f"行业中心：{industry}<br>"
                f"城市：{city}<br>"
                f"年份：{year}<br>"
                f"招聘总人数：{r.get('招聘总人数', np.nan):.0f}<br>"
                f"企业数量：{r.get('企业数量', np.nan)}<br>"
                f"50%覆盖半径：{r.get('50%覆盖半径_km', np.nan):.2f} km<br>"
                f"80%覆盖半径：{r.get('80%覆盖半径_km', np.nan):.2f} km<br>"
                f"90%覆盖半径：{r.get('90%覆盖半径_km', np.nan):.2f} km"
            )],
            hoverinfo="text",
            name="行业中心"
        ))

        for col, label, color in [
            ("50%覆盖半径_km", "50%覆盖半径", "#1f77b4"),
            ("80%覆盖半径_km", "80%覆盖半径", "#ff7f0e"),
            ("90%覆盖半径_km", "90%覆盖半径", "#2ca02c"),
        ]:
            if col in r and pd.notna(r[col]):
                xs, ys = radius_to_lonlat_circle(lon_c, lat_c, float(r[col]))
                fig.add_trace(go.Scattermapbox(
                    lon=xs.tolist(),
                    lat=ys.tolist(),
                    mode="lines",
                    line=dict(width=2, color=color),
                    hoverinfo="text",
                    hovertext=[f"{label}：{float(r[col]):.2f} km"] * len(xs),
                    name=label
                ))

    # 自动聚焦范围
    if not pts.empty:
        center = {"lat": pts["纬度"].mean(), "lon": pts["经度"].mean()}

    fig.update_layout(
        title=f"{city} - {year} - {industry}：行业集聚地图",
        mapbox_style="carto-positron",
        mapbox_center=center,
        mapbox_zoom=zoom,
        height=680,
        margin=dict(l=0, r=0, t=55, b=0),
        legend=dict(
            orientation="h",
            y=0.02,
            x=0.01,
            bgcolor="rgba(255,255,255,0.75)"
        )
    )
    return fig


# 设置 Matplotlib 中文字体，避免九宫格等静态图中文显示为方框
setup_chinese_matplotlib_font()

# =============================================================================
# 4. Streamlit 主界面
# =============================================================================
st.title("🏙️ 企业生存、薪资估值与空间结构演化可视化系统")
st.caption("加速版：数据读取、随机森林、DBSCAN、行业半径和词云分词均已缓存；筛选条件需点击“应用筛选”后刷新。")

with st.sidebar:
    if st.button("清除缓存并重新计算"):
        st.cache_data.clear()
        st.rerun()

with st.sidebar:
    st.header("① 一次上传数据")
    uploaded_files = st.file_uploader(
        "请一次选择多个文件上传",
        type=["xlsx", "xls", "csv", "zip", "shp", "shx", "dbf", "prj", "cpg"],
        accept_multiple_files=True
    )

    st.markdown(
        """
        **建议一次上传：**
        - 企业基础信息_有招聘.xlsx
        - 企业招聘行为表_清洗终.xlsx
        - 城市边界 zip，或完整 Shapefile 文件组  
          `.shp + .shx + .dbf + .prj`
        """
    )

if not uploaded_files:
    st.info("请先在左侧一次上传企业基础信息表、企业招聘行为表和边界文件。")
    st.stop()

files = classify_uploaded_files(uploaded_files)

if files["basic_file"] is None:
    st.error("未识别到企业基础信息表。请确认文件中包含：企业名称、城市、经度、纬度、行业门类名称等字段。")
    st.stop()

if files["job_file"] is None:
    st.error("未识别到企业招聘行为表。请确认文件中包含：企业名称、工作地点、职位名称、招聘人数、平均年薪(万) 等字段。")
    st.stop()

basic_sig = file_signature(files["basic_file"])
job_sig = file_signature(files["job_file"])

df_basic, df_jobs, df_ey = cached_prepare_tables(basic_sig, job_sig)

# 使用第一题表现更好的随机森林模型，重新估计企业生存概率；结果缓存
df_basic, rf_info = cached_rf_survival(df_basic, df_jobs, df_ey)

# 将随机森林生存概率同步回招聘表和企业-年份表
if "企业名称" in df_basic.columns:
    _prob_map = df_basic.set_index("企业名称")["生存概率"].to_dict()
    if "企业名称" in df_jobs.columns:
        df_jobs["生存概率"] = df_jobs["企业名称"].map(_prob_map).fillna(df_jobs.get("生存概率"))
    if "企业名称" in df_ey.columns:
        df_ey["生存概率"] = df_ey["企业名称"].map(_prob_map).fillna(df_ey.get("生存概率"))

# DBSCAN 与行业半径结果缓存
df_centers, df_radius = cached_spatial_results(df_ey)

shape_sigs = [file_signature(f) for f in files["shape_files"]]
zip_sigs = [file_signature(f) for f in files["zip_files"]]
gdf_boundary, boundary_msg = cached_read_boundaries(shape_sigs, zip_sigs)
if boundary_msg:
    with st.expander("城市边界文件读取提示", expanded=False):
        st.write(boundary_msg)

with st.sidebar:
    st.header("② 筛选条件")
    st.caption("为提升速度，修改筛选后请点击“应用筛选”。")

    city_options = sorted(df_basic["企业所在地"].dropna().unique().tolist())
    job_city_options = sorted(df_jobs["岗位所在地"].dropna().unique().tolist())
    industry_options = sorted(df_basic["行业"].dropna().unique().tolist())
    year_options = sorted([int(x) for x in df_jobs["年份"].dropna().unique().tolist()])

    default_industry = industry_options[:10] if len(industry_options) > 10 else industry_options

    # 初始化 session_state
    st.session_state.setdefault("selected_cities", city_options)
    st.session_state.setdefault("selected_job_cities", job_city_options[:10])
    st.session_state.setdefault("selected_industries", default_industry)
    st.session_state.setdefault("selected_years", year_options[-3:] if len(year_options) >= 3 else year_options)

    module = st.radio(
        "模块切换",
        [
            "模块一：产业主体来源地图",
            "模块二：岗位需求热力与技能词云",
            "模块三：企业所在地 → 岗位所在地流向网络",
            "模块四：行业维度人才需求网络",
            "模块五：第三题空间结构专题",
            "数据预览与导出"
        ]
    )

    with st.form("filter_form"):
        selected_cities_tmp = st.multiselect(
            "企业所在地",
            city_options,
            default=[x for x in st.session_state["selected_cities"] if x in city_options]
        )

        selected_job_cities_tmp = st.multiselect(
            "岗位所在地",
            job_city_options,
            default=[x for x in st.session_state["selected_job_cities"] if x in job_city_options]
        )

        selected_industries_tmp = st.multiselect(
            "行业",
            industry_options,
            default=[x for x in st.session_state["selected_industries"] if x in industry_options]
        )

        selected_years_tmp = st.multiselect(
            "年份",
            year_options,
            default=[x for x in st.session_state["selected_years"] if x in year_options]
        )

        apply_filter = st.form_submit_button("应用筛选")

    if apply_filter:
        st.session_state["selected_cities"] = selected_cities_tmp
        st.session_state["selected_job_cities"] = selected_job_cities_tmp
        st.session_state["selected_industries"] = selected_industries_tmp
        st.session_state["selected_years"] = selected_years_tmp

    selected_cities = st.session_state["selected_cities"]
    selected_job_cities = st.session_state["selected_job_cities"]
    selected_industries = st.session_state["selected_industries"]
    selected_years = st.session_state["selected_years"]

# 应用筛选
df_basic_f = df_basic.copy()
if selected_cities:
    df_basic_f = df_basic_f[df_basic_f["企业所在地"].isin(selected_cities)]
if selected_industries:
    df_basic_f = df_basic_f[df_basic_f["行业"].isin(selected_industries)]

df_jobs_f = df_jobs.copy()
if selected_cities:
    df_jobs_f = df_jobs_f[df_jobs_f["企业所在地"].isin(selected_cities)]
if selected_job_cities:
    df_jobs_f = df_jobs_f[df_jobs_f["岗位所在地"].isin(selected_job_cities)]
if selected_industries:
    df_jobs_f = df_jobs_f[df_jobs_f["行业"].isin(selected_industries)]
if selected_years:
    df_jobs_f = df_jobs_f[df_jobs_f["年份"].isin(selected_years)]

df_centers_f = df_centers.copy()
if not df_centers_f.empty:
    if selected_cities:
        df_centers_f = df_centers_f[df_centers_f["城市"].isin(selected_cities)]
    if selected_years:
        df_centers_f = df_centers_f[df_centers_f["年份"].isin(selected_years)]

df_radius_f = df_radius.copy()
if not df_radius_f.empty:
    if selected_cities:
        df_radius_f = df_radius_f[df_radius_f["城市"].isin(selected_cities)]
    if selected_years:
        df_radius_f = df_radius_f[df_radius_f["年份"].isin(selected_years)]
    if selected_industries:
        df_radius_f = df_radius_f[df_radius_f["行业"].isin(selected_industries)]

# 顶部指标
st.subheader("数据概览")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("企业数量", f"{len(df_basic_f):,}")
c2.metric("招聘记录", f"{len(df_jobs_f):,}")
c3.metric("招聘人数合计", f"{df_jobs_f['招聘人数_数值'].sum():,.0f}")
c4.metric("平均年薪（万）", f"{df_jobs_f['平均年薪_万'].mean():.2f}" if df_jobs_f["平均年薪_万"].notna().any() else "暂无")
c5.metric("就业中心数", f"{len(df_centers_f):,}")

if 'rf_info' in globals():
    if rf_info.get("used_rf"):
        msg = f"✅ 生存概率来源：随机森林模型；训练样本 {rf_info.get('sample_count')} 家企业，其中存活 {rf_info.get('positive_count')} 家、非存活 {rf_info.get('negative_count')} 家。"
        if rf_info.get("test_accuracy") is not None:
            msg += f" 留出集 Accuracy={rf_info.get('test_accuracy'):.3f}"
        if rf_info.get("test_auc") is not None:
            msg += f"，AUC={rf_info.get('test_auc'):.3f}"
        st.success(msg)
    else:
        st.warning("⚠️ 生存概率暂未使用随机森林：" + str(rf_info.get("message", "")))


# =============================================================================
# 5. 模块展示
# =============================================================================
if module == "模块一：产业主体来源地图":
    st.header("模块一：产业主体来源地图")
    st.write("使用企业所在地，展示企业数量、行业结构、注册资本、成立年限、存活状态，并用生存概率映射点的颜色和大小。")

    col1, col2 = st.columns([2, 1])
    with col1:
        fig = plot_enterprise_map(df_basic_f, gdf_boundary)
        st.plotly_chart(fig, use_container_width=True)
        st.download_button("下载产业主体地图 HTML", make_download_html(fig), "产业主体来源地图.html", "text/html")

    with col2:
        fig_count = plot_city_count(df_basic_f, gdf_boundary)
        st.plotly_chart(fig_count, use_container_width=True)

        industry_count = df_basic_f["行业"].value_counts().head(15).reset_index()
        industry_count.columns = ["行业", "企业数量"]
        fig_ind = px.bar(industry_count, x="企业数量", y="行业", orientation="h", title="行业结构 Top15")
        st.plotly_chart(fig_ind, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        fig_cap = px.box(df_basic_f, x="企业所在地", y="注册资本_万元", color="企业所在地", title="注册资本分布")
        st.plotly_chart(fig_cap, use_container_width=True)

    with col4:
        fig_age = px.histogram(df_basic_f, x="成立年限", color="企业所在地", nbins=30, title="成立年限分布")
        st.plotly_chart(fig_age, use_container_width=True)

    status_df = df_basic_f.groupby(["企业所在地", "企业状态_标准"]).size().reset_index(name="企业数量")
    fig_status = px.bar(status_df, x="企业所在地", y="企业数量", color="企业状态_标准", barmode="group", title="存活状态分布")
    st.plotly_chart(fig_status, use_container_width=True)

    if 'rf_info' in globals() and rf_info.get("used_rf") and not rf_info.get("top_features", pd.DataFrame()).empty:
        st.subheader("随机森林生存预测模型：特征重要度 Top20")
        imp = rf_info["top_features"].copy()
        fig_imp = px.bar(imp, x="重要度", y="特征", orientation="h", title="随机森林特征重要度")
        fig_imp.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig_imp, use_container_width=True)


elif module == "模块二：岗位需求热力与技能词云":
    st.header("模块二：岗位画像分析：按岗位筛选")
    st.write(
        "本模块使用独立岗位筛选逻辑。即使左侧全局行业筛选过窄，也可以在这里直接选择岗位分析，"
        "避免出现“已经选了但模块二没有内容”的情况。"
    )

    if df_jobs.empty:
        st.warning("招聘行为表为空，模块二无法展示。")
        st.stop()

    # -------------------------------------------------------------------------
    # 0. 模块二独立筛选：只默认继承年份和岗位所在地，不强制继承行业
    # -------------------------------------------------------------------------
    with st.expander("模块二筛选设置", expanded=True):
        st.caption("如果模块二没有结果，优先检查这里。建议先取消行业限制，只按岗位和年份查看。")

        c_filter1, c_filter2, c_filter3 = st.columns(3)

        with c_filter1:
            use_global_year = st.checkbox("使用左侧年份筛选", value=True)
        with c_filter2:
            use_global_job_city = st.checkbox("使用左侧岗位所在地筛选", value=True)
        with c_filter3:
            use_global_industry = st.checkbox("使用左侧行业筛选", value=False)

    df_module2 = df_jobs.copy()

    if use_global_year and selected_years:
        df_module2 = df_module2[df_module2["年份"].isin(selected_years)]

    if use_global_job_city and selected_job_cities:
        df_module2 = df_module2[df_module2["岗位所在地"].isin(selected_job_cities)]

    if use_global_industry and selected_industries:
        df_module2 = df_module2[df_module2["行业"].isin(selected_industries)]

    st.caption(
        f"模块二当前可用招聘记录：{len(df_module2):,} 条；"
        f"岗位数：{df_module2['岗位名称'].nunique() if '岗位名称' in df_module2.columns else 0:,} 个。"
    )

    if df_module2.empty:
        st.warning(
            "模块二当前筛选后没有招聘记录。请取消“使用左侧岗位所在地筛选”或“使用左侧行业筛选”，"
            "或者回到左侧修改筛选条件并点击“应用筛选”。"
        )
        st.stop()

    # -------------------------------------------------------------------------
    # 1. 岗位选择区：支持关键词搜索 + TopN候选
    # -------------------------------------------------------------------------
    st.markdown("### 1. 选择岗位")

    keyword = st.text_input(
        "岗位关键词搜索",
        value="",
        placeholder="例如：软件、数据、销售、开发、产品、财务"
    )

    job_base = df_module2.copy()
    if keyword.strip():
        job_base = job_base[job_base["岗位名称"].astype(str).str.contains(keyword.strip(), case=False, na=False)]

    if job_base.empty:
        st.warning("没有匹配该关键词的岗位。请换一个关键词，或清空关键词。")
        st.stop()

    job_summary_all = (
        job_base.groupby("岗位名称", as_index=False)
        .agg(
            招聘人数=("招聘人数_数值", "sum"),
            招聘记录数=("岗位名称", "size"),
            平均年薪_万=("平均年薪_万", "mean")
        )
        .sort_values("招聘人数", ascending=False)
    )

    show_all_jobs = st.checkbox(
        "显示全部岗位候选",
        value=True,
        help="勾选后会列出当前筛选条件下的全部岗位；如果岗位太多导致页面变慢，可以取消勾选并只显示 Top N。"
    )

    if show_all_jobs:
        candidate_jobs = job_summary_all["岗位名称"].tolist()
        st.caption(f"当前已列出全部岗位：{len(candidate_jobs):,} 个")
    else:
        top_n = st.slider("显示候选岗位数量", min_value=10, max_value=500, value=100, step=10)
        candidate_jobs = job_summary_all.head(top_n)["岗位名称"].tolist()
        st.caption(f"当前仅列出招聘人数排名前 {len(candidate_jobs):,} 个岗位")

    default_jobs = candidate_jobs[:1] if candidate_jobs else []

    selected_jobs = st.multiselect(
        "选择要分析的岗位",
        candidate_jobs,
        default=default_jobs,
        help="默认按招聘人数从高到低排序。可以先用上方关键词搜索缩小范围，再选择岗位。"
    )

    with st.expander("查看当前候选岗位清单"):
        st.dataframe(job_summary_all[job_summary_all["岗位名称"].isin(candidate_jobs)], use_container_width=True)
        st.download_button(
            "下载当前候选岗位清单 CSV",
            job_summary_all.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig"),
            "模块二_岗位候选清单.csv",
            "text/csv"
        )

    if not selected_jobs:
        st.info("请至少选择一个岗位。")
        st.stop()

    df_job_selected = df_module2[df_module2["岗位名称"].isin(selected_jobs)].copy()

    if df_job_selected.empty:
        st.warning("已选择岗位，但筛选后没有记录。请取消部分筛选条件。")
        st.stop()

    # -------------------------------------------------------------------------
    # 2. 岗位核心指标
    # -------------------------------------------------------------------------
    st.markdown("### 2. 岗位核心指标")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("所选岗位数", f"{len(selected_jobs)}")
    c2.metric("招聘记录数", f"{len(df_job_selected):,}")
    c3.metric("招聘人数合计", f"{df_job_selected['招聘人数_数值'].sum():,.0f}")
    c4.metric(
        "平均年薪（万）",
        f"{df_job_selected['平均年薪_万'].mean():.2f}" if df_job_selected["平均年薪_万"].notna().any() else "暂无"
    )
    c5.metric("涉及企业数", f"{df_job_selected['企业名称'].nunique():,}" if "企业名称" in df_job_selected.columns else "暂无")

    # -------------------------------------------------------------------------
    # 3. 岗位空间热力图
    # -------------------------------------------------------------------------
    st.markdown("### 3. 所选岗位需求空间热力图")
    metric = st.selectbox(
        "热力图指标",
        ["招聘人数_数值", "平均年薪_万"],
        help="招聘人数表示岗位需求强度；平均年薪表示岗位薪资空间差异。"
    )

    fig_heat = plot_job_heat(df_job_selected, metric)
    if fig_heat:
        fig_heat.update_layout(title=f"所选岗位需求热力图：{', '.join(selected_jobs[:3])}" + (" 等" if len(selected_jobs) > 3 else ""))
        st.plotly_chart(fig_heat, use_container_width=True)
        st.download_button("下载岗位热力图 HTML", make_download_html(fig_heat), "所选岗位需求热力图.html", "text/html")
    else:
        st.warning(
            "当前所选岗位没有可用经纬度，无法绘制岗位热力图。"
            "请确认企业基础信息表中有经度、纬度，并且招聘表能按企业名称匹配到企业表。"
        )

    # -------------------------------------------------------------------------
    # 4. 城市对比：招聘人数、平均薪资、高薪占比
    # -------------------------------------------------------------------------
    st.markdown("### 4. 不同城市的岗位需求与薪资对比")
    col1, col2 = st.columns(2)

    with col1:
        city_job = df_job_selected.groupby("岗位所在地").agg(
            招聘人数=("招聘人数_数值", "sum"),
            招聘记录数=("岗位所在地", "size"),
            平均年薪=("平均年薪_万", "mean"),
            企业数=("企业名称", "nunique")
        ).reset_index().sort_values("招聘人数", ascending=False)

        if city_job.empty:
            st.info("没有城市对比数据。")
        else:
            fig_city_job = px.bar(
                city_job,
                x="岗位所在地",
                y="招聘人数",
                color="平均年薪",
                text="招聘人数",
                title="所选岗位：各城市招聘人数与平均薪资"
            )
            fig_city_job.update_traces(texttemplate="%{text:.0f}")
            st.plotly_chart(fig_city_job, use_container_width=True)

    with col2:
        if df_job_selected["平均年薪_万"].notna().any():
            q75 = df_job_selected["平均年薪_万"].quantile(0.75)
            tmp = df_job_selected.copy()
            tmp["是否高薪"] = tmp["平均年薪_万"] >= q75
            high = tmp.groupby("岗位所在地")["是否高薪"].mean().reset_index()
            high["高薪岗位占比"] = high["是否高薪"] * 100
            high = high.sort_values("高薪岗位占比", ascending=False)

            fig_high = px.bar(
                high,
                x="岗位所在地",
                y="高薪岗位占比",
                text="高薪岗位占比",
                title="所选岗位：各城市高薪岗位占比"
            )
            fig_high.update_traces(texttemplate="%{text:.1f}%")
            st.plotly_chart(fig_high, use_container_width=True)
        else:
            st.info("该岗位缺少薪资数据，暂不能计算高薪岗位占比。")

    # -------------------------------------------------------------------------
    # 5. 薪资分布、学历/经验结构
    # -------------------------------------------------------------------------
    st.markdown("### 5. 所选岗位薪资与要求结构")
    col3, col4 = st.columns(2)

    with col3:
        if df_job_selected["平均年薪_万"].notna().any():
            fig_salary = px.box(
                df_job_selected,
                x="岗位所在地",
                y="平均年薪_万",
                color="岗位所在地",
                title="所选岗位：不同城市薪资分布"
            )
            st.plotly_chart(fig_salary, use_container_width=True)
        else:
            st.info("该岗位缺少薪资数据，暂不能绘制薪资分布。")

    with col4:
        exp_count = (
            df_job_selected.groupby("工作年限")["招聘人数_数值"]
            .sum()
            .sort_values(ascending=False)
            .head(12)
            .reset_index()
        )
        fig_exp = px.bar(
            exp_count,
            x="招聘人数_数值",
            y="工作年限",
            orientation="h",
            title="所选岗位：经验要求结构"
        )
        st.plotly_chart(fig_exp, use_container_width=True)

    col5, col6 = st.columns(2)
    with col5:
        edu_count = (
            df_job_selected.groupby("学历")["招聘人数_数值"]
            .sum()
            .sort_values(ascending=False)
            .head(12)
            .reset_index()
        )
        fig_edu = px.bar(
            edu_count,
            x="招聘人数_数值",
            y="学历",
            orientation="h",
            title="所选岗位：学历要求结构"
        )
        st.plotly_chart(fig_edu, use_container_width=True)

    with col6:
        industry_count = (
            df_job_selected.groupby("行业")["招聘人数_数值"]
            .sum()
            .sort_values(ascending=False)
            .head(12)
            .reset_index()
        )
        fig_ind_job = px.bar(
            industry_count,
            x="招聘人数_数值",
            y="行业",
            orientation="h",
            title="所选岗位：行业来源结构"
        )
        st.plotly_chart(fig_ind_job, use_container_width=True)

    # -------------------------------------------------------------------------
    # 6. 技能词云与技能词频
    # -------------------------------------------------------------------------
    st.markdown("### 6. 所选岗位技能画像")
    col7, col8 = st.columns([1.2, 1])

    with col7:
        text = " ".join(df_job_selected["词云文本"].dropna().astype(str).tolist())
        freq_dict = cached_token_frequencies(text)
        fig_wc = make_wordcloud_from_freq(freq_dict)
        st.pyplot(fig_wc, use_container_width=True)

    with col8:
        freq_dict = cached_token_frequencies(" ".join(df_job_selected["词云文本"].dropna().astype(str).tolist()))
        if freq_dict:
            word_df = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1], reverse=True)[:20], columns=["关键词", "出现次数"])
            fig_word = px.bar(
                word_df,
                x="出现次数",
                y="关键词",
                orientation="h",
                title="所选岗位关键词 Top20"
            )
            fig_word.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig_word, use_container_width=True)
        else:
            st.info("没有可统计的关键词。")

    # -------------------------------------------------------------------------
    # 7. 岗位明细表
    # -------------------------------------------------------------------------
    with st.expander("查看所选岗位明细数据"):
        show_cols = [
            "企业名称", "企业所在地", "岗位所在地", "行业", "岗位名称",
            "岗位类别", "招聘人数_数值", "平均年薪_万", "学历", "工作年限",
            "职位刷新时间", "词云文本", "技能文本"
        ]
        show_cols = [c for c in show_cols if c in df_job_selected.columns]
        st.dataframe(df_job_selected[show_cols].head(500), use_container_width=True)
        st.download_button(
            "下载所选岗位明细 CSV",
            df_job_selected.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig"),
            "所选岗位明细.csv",
            "text/csv"
        )


elif module == "模块三：企业所在地 → 岗位所在地流向网络":
    st.header("模块三：企业所在地 → 岗位所在地流向网络")
    st.write("构建有向网络，展示桑基图、地图飞线图、城市网络图，并叠加自动识别的就业中心。")

    tab1, tab2, tab3, tab4 = st.tabs(["桑基图", "地图飞线图", "城市网络图", "就业中心表"])

    with tab1:
        fig_sankey = plot_sankey(df_jobs_f, "企业所在地", "岗位所在地", "招聘人数_数值", "企业所在地 → 岗位所在地流向桑基图")
        st.plotly_chart(fig_sankey, use_container_width=True)
        st.download_button("下载桑基图 HTML", make_download_html(fig_sankey), "企业岗位流向桑基图.html", "text/html")

    with tab2:
        fig_flight = plot_flight_map(df_jobs_f, df_centers_f)
        st.plotly_chart(fig_flight, use_container_width=True)
        st.download_button("下载飞线图 HTML", make_download_html(fig_flight), "地图飞线图.html", "text/html")

    with tab3:
        fig_net = plot_city_network(df_jobs_f)
        if fig_net:
            st.plotly_chart(fig_net, use_container_width=True)
        else:
            st.warning("暂无可生成城市网络图的数据。")

    with tab4:
        if df_centers_f.empty:
            st.info("当前筛选条件下没有识别到就业中心。")
        else:
            st.dataframe(df_centers_f, use_container_width=True)
            st.download_button(
                "下载就业中心结果 CSV",
                df_centers_f.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig"),
                "就业中心结果.csv",
                "text/csv"
            )


elif module == "模块四：行业维度人才需求网络":
    st.header("模块四：行业维度人才需求网络")
    st.write("构建行业 → 岗位类别人才需求网络，并展示行业招聘加权中心与覆盖半径。")

    tab1, tab2, tab3 = st.tabs(["行业 → 岗位桑基图", "行业集聚半径地图", "行业半径表"])

    with tab1:
        fig_ind_sankey = plot_sankey(df_jobs_f, "行业", "岗位类别", "招聘人数_数值", "行业维度人才需求网络：行业 → 岗位类别")
        st.plotly_chart(fig_ind_sankey, use_container_width=True)
        st.download_button("下载行业网络 HTML", make_download_html(fig_ind_sankey), "行业人才需求网络.html", "text/html")

    with tab2:
        fig_radius = plot_industry_radius(df_radius_f)
        if fig_radius:
            st.plotly_chart(fig_radius, use_container_width=True)
            st.download_button("下载行业半径地图 HTML", make_download_html(fig_radius), "行业集聚半径地图.html", "text/html")
        else:
            st.warning("暂无行业集聚半径数据。")

    with tab3:
        if df_radius_f.empty:
            st.info("暂无行业集聚半径结果。")
        else:
            st.dataframe(df_radius_f, use_container_width=True)
            st.download_button(
                "下载行业集聚半径 CSV",
                df_radius_f.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig"),
                "行业集聚半径结果.csv",
                "text/csv"
            )



elif module == "模块五：第三题空间结构专题":
    st.header("模块五：第三题空间结构专题")
    st.write(
        "本模块集中展示第三题结果：城市年份就业中心地图、单核/多核结构判断、"
        "三年演化分析，以及行业集聚半径九宫格。地图仅聚焦当前上传或筛选的城市，避免显示范围过大。"
    )

    if df_ey.empty:
        st.warning("企业-年份招聘强度表为空，无法展示第三题空间结构专题。")
        st.stop()

    # 只使用当前上传数据中真实存在的城市与年份
    topic_cities = sorted(df_ey["企业所在地"].dropna().unique().tolist())
    if selected_cities:
        topic_cities = [c for c in topic_cities if c in selected_cities]

    topic_years = sorted([int(y) for y in df_ey["年份"].dropna().unique().tolist()])
    if selected_years:
        topic_years = [y for y in topic_years if y in selected_years]

    if not topic_cities or not topic_years:
        st.warning("当前筛选条件下没有可展示的城市或年份。")
        st.stop()

    struct_df = judge_core_structure(df_centers_f if not df_centers_f.empty else df_centers)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "城市年份就业中心地图",
        "单核/多核判断",
        "三年演化分析",
        "行业集聚半径九宫格",
        "城市+年份+行业单张地图"
    ])

    with tab1:
        st.subheader("城市年份就业中心地图")
        c1, c2 = st.columns(2)
        with c1:
            map_city = st.selectbox("选择城市", topic_cities, key="center_map_city")
        with c2:
            map_year = st.selectbox("选择年份", topic_years, index=len(topic_years)-1, key="center_map_year")

        fig_center_map = plot_city_year_center_map(
            df_ey,
            df_centers if not df_centers.empty else df_centers_f,
            gdf_boundary,
            map_city,
            map_year
        )
        st.plotly_chart(fig_center_map, use_container_width=True)
        st.download_button(
            "下载城市年份就业中心地图 HTML",
            make_download_html(fig_center_map),
            f"{map_city}_{map_year}_就业中心地图.html",
            "text/html"
        )

        centers_show = df_centers[
            (df_centers["城市"] == map_city) &
            (df_centers["年份"] == map_year)
        ] if not df_centers.empty else pd.DataFrame()

        if not centers_show.empty:
            st.markdown("#### 当前城市年份就业中心明细")
            st.dataframe(centers_show, use_container_width=True)
        else:
            st.info("当前城市年份没有识别到就业中心，可能是招聘点过少或 DBSCAN 参数下未形成密集中心。")

    with tab2:
        st.subheader("单核 / 强主中心+次中心 / 多核结构判断")
        if struct_df.empty:
            st.warning("暂无就业中心结果，无法进行结构判断。")
        else:
            struct_show = struct_df.copy()
            if topic_cities:
                struct_show = struct_show[struct_show["城市"].isin(topic_cities)]
            if topic_years:
                struct_show = struct_show[struct_show["年份"].isin(topic_years)]

            st.dataframe(struct_show, use_container_width=True)

            fig_struct = px.bar(
                struct_show,
                x="年份",
                y="最大中心招聘占比",
                color="空间结构判断",
                facet_col="城市",
                text="空间结构判断",
                title="各城市年份单核/多核结构判断"
            )
            fig_struct.update_yaxes(tickformat=".0%")
            fig_struct.update_layout(height=500)
            st.plotly_chart(fig_struct, use_container_width=True)

            st.download_button(
                "下载单核多核判断结果 CSV",
                struct_show.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig"),
                "单核多核判断结果.csv",
                "text/csv"
            )

    with tab3:
        st.subheader("三年演化分析")
        if struct_df.empty:
            st.warning("暂无结构判断结果，无法绘制三年演化图。")
        else:
            struct_show = struct_df.copy()
            if topic_cities:
                struct_show = struct_show[struct_show["城市"].isin(topic_cities)]
            if topic_years:
                struct_show = struct_show[struct_show["年份"].isin(topic_years)]

            fig_n, fig_share = plot_core_evolution(struct_show)
            if fig_n:
                st.plotly_chart(fig_n, use_container_width=True)
            if fig_share:
                st.plotly_chart(fig_share, use_container_width=True)

            center_evo = df_centers.copy()
            if not center_evo.empty:
                center_evo = center_evo[center_evo["城市"].isin(topic_cities)]
                center_evo = center_evo[center_evo["年份"].isin(topic_years)]
                fig_strength = plot_center_strength_evolution(center_evo)
                if fig_strength:
                    st.plotly_chart(fig_strength, use_container_width=True)

            st.markdown("#### 可用于论文/报告的自动解读")
            for city in topic_cities:
                tmp = struct_show[struct_show["城市"] == city].sort_values("年份")
                if tmp.empty:
                    continue
                first = tmp.iloc[0]
                last = tmp.iloc[-1]
                st.write(
                    f"**{city}**：从 {int(first['年份'])} 年到 {int(last['年份'])} 年，"
                    f"主要就业中心数由 {int(first['主要就业中心数_占比不低于5%'])} 个变化为 "
                    f"{int(last['主要就业中心数_占比不低于5%'])} 个；"
                    f"最大中心招聘占比由 {first['最大中心招聘占比']:.1%} 变化为 "
                    f"{last['最大中心招聘占比']:.1%}。"
                    f"空间结构由“{first['空间结构判断']}”演化为“{last['空间结构判断']}”。"
                )

    with tab4:
        st.subheader("行业集聚半径地图：九宫格带企业点")
        if df_radius.empty:
            st.warning("暂无行业集聚半径结果，无法绘制九宫格。")
        else:
            available_industries = sorted(df_radius["行业"].dropna().unique().tolist())
            if selected_industries:
                available_industries = [x for x in available_industries if x in selected_industries] or available_industries

            industry_choice = st.selectbox(
                "选择行业",
                available_industries,
                key="radius_grid_industry"
            )

            st.caption(
                "九宫格默认最多展示 3 个城市 × 3 个年份。企业点大小表示年度招聘人数，星形表示行业招聘加权中心，圆圈表示 50%、80%、90% 覆盖半径。"
            )

            fig_grid = plot_industry_radius_nine_grid(
                df_ey,
                df_radius,
                gdf_boundary,
                industry_choice,
                topic_cities[:3],
                topic_years[:3]
            )

            if fig_grid:
                st.pyplot(fig_grid, use_container_width=True)
            else:
                st.warning("当前行业在所选城市年份下没有足够数据绘制九宫格。")

            radius_table = df_radius[
                (df_radius["行业"] == industry_choice) &
                (df_radius["城市"].isin(topic_cities)) &
                (df_radius["年份"].isin(topic_years))
            ].copy()
            st.markdown("#### 当前行业集聚半径明细")
            st.dataframe(radius_table, use_container_width=True)

            st.download_button(
                "下载当前行业集聚半径 CSV",
                radius_table.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig"),
                f"{industry_choice}_行业集聚半径.csv",
                "text/csv"
            )


    with tab5:
        st.subheader("城市 + 年份 + 行业：单张地图展示")
        st.write("该图按照第三题的思路，只展示一个城市、一个年份、一个行业，适合直接截图放入论文或 PPT。")

        if df_radius.empty:
            st.warning("暂无行业集聚半径结果，无法绘制该地图。")
        else:
            c1, c2, c3 = st.columns(3)

            with c1:
                city_choice_single = st.selectbox(
                    "选择城市",
                    topic_cities,
                    key="single_map_city"
                )

            year_candidates_single = sorted(
                df_radius[df_radius["城市"] == city_choice_single]["年份"].dropna().unique().tolist()
            )
            if not year_candidates_single:
                year_candidates_single = topic_years

            with c2:
                year_choice_single = st.selectbox(
                    "选择年份",
                    year_candidates_single,
                    key="single_map_year"
                )

            industry_candidates_single = sorted(
                df_radius[
                    (df_radius["城市"] == city_choice_single) &
                    (df_radius["年份"] == year_choice_single)
                ]["行业"].dropna().unique().tolist()
            )
            if not industry_candidates_single:
                industry_candidates_single = sorted(df_radius["行业"].dropna().unique().tolist())

            with c3:
                industry_choice_single = st.selectbox(
                    "选择行业",
                    industry_candidates_single,
                    key="single_map_industry"
                )

            fig_single = plot_city_year_industry_map(
                df_ey=df_ey,
                df_radius=df_radius,
                gdf_boundary=gdf_boundary,
                city=city_choice_single,
                year=year_choice_single,
                industry=industry_choice_single
            )

            if fig_single:
                st.plotly_chart(fig_single, use_container_width=True)
                st.download_button(
                    "下载当前单张地图 HTML",
                    make_download_html(fig_single),
                    f"{city_choice_single}_{year_choice_single}_{industry_choice_single}_行业集聚地图.html",
                    "text/html"
                )

                detail_table = df_radius[
                    (df_radius["城市"] == city_choice_single) &
                    (df_radius["年份"] == year_choice_single) &
                    (df_radius["行业"] == industry_choice_single)
                ].copy()

                if not detail_table.empty:
                    st.markdown("#### 当前地图对应的行业集聚半径明细")
                    st.dataframe(detail_table, use_container_width=True)
                else:
                    st.info("当前城市、年份、行业没有行业集聚半径结果。")
            else:
                st.warning("当前城市、年份、行业组合没有足够数据绘制地图。")



elif module == "数据预览与导出":
    st.header("数据预览与导出")
    tab1, tab2, tab3, tab4 = st.tabs(["企业基础信息", "招聘行为信息", "企业-年份表", "字段说明"])

    with tab1:
        st.dataframe(df_basic.head(300), use_container_width=True)
        st.download_button(
            "下载企业基础信息_系统处理后.csv",
            df_basic.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig"),
            "企业基础信息_系统处理后.csv",
            "text/csv"
        )

    with tab2:
        st.dataframe(df_jobs.head(300), use_container_width=True)
        st.download_button(
            "下载招聘行为_系统处理后.csv",
            df_jobs.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig"),
            "招聘行为_系统处理后.csv",
            "text/csv"
        )

    with tab3:
        st.dataframe(df_ey.head(300), use_container_width=True)
        st.download_button(
            "下载企业年份招聘强度表.csv",
            df_ey.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig"),
            "企业年份招聘强度表.csv",
            "text/csv"
        )

    with tab4:
        st.markdown(
            """
            ### 系统使用字段

            **企业基础信息表需要字段：**
            - 企业名称
            - 城市
            - 注册资本(万元)
            - 企业状态
            - 成立日期
            - 行业门类名称 / 行业名称
            - 经度
            - 纬度

            **招聘行为表需要字段：**
            - 企业名称
            - 职位刷新时间
            - 工作地点
            - 职位名称
            - 职位职能
            - 招聘人数
            - 平均年薪(万)
            - 职位描述_分词 / 职位描述_清洗

            **说明：**
            - 岗位热力图当前使用企业经纬度作为岗位需求承载位置，因为招聘表中没有岗位经纬度。
            - 如果你后续有岗位所在地经纬度，可以扩展字段后改为岗位坐标。
            - 城市边界建议上传完整 Shapefile zip，不建议只上传 .shp。
            """
        )
