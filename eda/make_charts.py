"""Графики EDA для README.

    python eda/make_charts.py [--data eda/chartdata.json] [--out eda]

Вход — `chartdata.json`, свод разведочного анализа: он посчитан из `train.parquet`
скриптами `01_prep_and_checks.py` … `05_chartdata.py` и лежит рядом, чтобы графики
пересобирались за секунду и без 180 МБ сырых данных.

Локальный якорь 378 (2026-01-14), горизонт 30 дней — копия боевой постановки:
история до якоря, цель на следующие 30 дней. Настоящее целевое окно
2026-02-14…2026-03-15 в обучающих данных отсутствует, поэтому всё, что здесь
сказано про цель, измерено на этом якоре.
"""
import argparse, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter

# Палитра. Три категориальных тона проверены на цветовую слепоту: минимальная пара
# ΔE = 15,2 (OKLab ×100) при симуляции дейтеран-, прот- и тританопии, норма ≥ 8;
# у всех пар ΔE нормального зрения ≥ 21. Последовательная шкала — один тон.
C1, C2, C3 = "#1a5b86", "#bf4a2e", "#c9a227"
INK, SEC, MUT = "#16191d", "#4a5058", "#767d86"
GRID, SURF, FILL = "#e4e7ea", "#ffffff", "#ccd4da"
SEQ = LinearSegmentedColormap.from_list("seq", ["#f4f7fa", "#9cbcd4", "#3d7ba4", "#0e3a58"])

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 10.5,
    "axes.edgecolor": GRID, "axes.linewidth": 0.8, "axes.titlelocation": "left",
    "axes.titlepad": 9, "axes.labelcolor": SEC, "text.color": INK,
    "xtick.color": MUT, "ytick.color": MUT, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "grid.color": GRID, "grid.linewidth": 0.7, "legend.frameon": False,
    "legend.fontsize": 8.5, "lines.linewidth": 2.0, "lines.solid_capstyle": "round",
    "svg.fonttype": "none", "figure.dpi": 110,
})


def frame(ax, axis="y"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.set_axisbelow(True)
    if axis:
        ax.grid(axis=axis)
    ax.tick_params(length=0)


def save(fig, out, name):
    fig.tight_layout()
    fig.savefig(os.path.join(out, name), format="svg", bbox_inches="tight")
    plt.close(fig)
    print("   ", name)


def ru(x, nd=1):
    return f"{x:.{nd}f}".replace(".", ",")


def main():
    H = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(H, "chartdata.json"))
    ap.add_argument("--out", default=H)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    D = json.load(open(a.data))
    print("строю графики из", os.path.basename(a.data))

    # ---------------------------------------------------------- 1. цель
    td = D["target_dist"]
    lab = [r["bucket"].replace("(", "").replace("]", "").replace(",", "–") for r in td]
    lab[0] = "0"
    shu = [r["share"] * 100 for r in td]
    shg = [r["gmv_share"] * 100 for r in td]
    fig, ax = plt.subplots(figsize=(7.8, 3.4))
    x = np.arange(len(lab)); w = 0.38
    ax.bar(x - w / 2 - 0.012, shu, w, color=C1, label="доля клиентов")
    ax.bar(x + w / 2 + 0.012, shg, w, color=C3, label="доля GMV")
    ax.set_xticks(x); ax.set_xticklabels(lab, rotation=25, ha="right")
    ax.set_ylabel("%"); ax.set_xlabel("GMV клиента за 30 дней горизонта")
    ax.set_title("Цель: 45,9% клиентов не купят ничего, а 0,03% дают 2,5% денег")
    ax.annotate(f"{ru(shu[0])}% клиентов\nи ровно 0% выручки", (0, shu[0]), (0.7, shu[0] - 3),
                color=C1, fontsize=8.5, arrowprops=dict(arrowstyle="-", color=C1, lw=0.8))
    ax.legend(loc="upper right"); frame(ax)
    save(fig, a.out, "01_target.svg")

    # ---------------------------------------------------------- 2. Лоренц
    lz = D["lorenz"]
    pop = np.array(lz["pop"]) * 100; gmv = np.array(lz["gmv"]) * 100
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    ax.plot([0, 100], [0, 100], lw=1.0, ls=(0, (4, 3)), color=MUT)
    ax.plot(100 - pop[::-1], 100 - gmv[::-1], color=C1)
    con = {r["top_share_users"]: r["gmv_share"] for r in D["concentration"]}
    for f_, t_ in ((0.01, "верхний 1%"), (0.1, "верхние 10%"), (0.2, "верхние 20%")):
        if f_ in con:
            ax.plot([f_ * 100], [con[f_] * 100], "o", ms=5, color=C1)
            ax.annotate(f"{t_} → {ru(con[f_]*100,0)}% GMV", (f_ * 100, con[f_] * 100),
                        (6, -3), textcoords="offset points", color=SEC, fontsize=8.5)
    ax.set_xlabel("верхние N% клиентов"); ax.set_ylabel("доля GMV, %")
    ax.set_title("Концентрация: хвост и решает метрику")
    frame(ax)
    save(fig, a.out, "02_lorenz.svg")

    # ---------------------------------------------------------- 3. дневной ряд
    # Две панели, а не две шкалы на одной оси: подённый GMV и сумма за 30 дней
    # различаются в тридцать раз, и совмещение их на общей оси было бы обманом.
    dd = D["daily"]; rr = D["roll30"]
    dt = np.array(dd["date"], dtype="datetime64[D]")
    rt = np.array(rr["date"], dtype="datetime64[D]")
    fig, axes = plt.subplots(2, 1, figsize=(9.4, 4.6), sharex=True)
    axes[0].plot(dt, np.array(dd["gmv"]) / 1e3, lw=0.9, color=C1)
    axes[0].set_ylabel("GMV за день, тыс.")
    axes[0].set_title("409 дней истории: рост вдвое, декабрьский пик, январский провал")
    axes[1].plot(rt, np.array(rr["gmv"]) / 1e6, lw=2.0, color=C1)
    axes[1].set_ylabel("GMV за 30 дней, млн")
    axes[1].set_xlabel("та же величина, что и цель: сумма за скользящее окно в 30 дней")
    for ax in axes:
        ax.axvline(np.datetime64("2025-04-01"), color=C2, lw=1.1, ls=(0, (4, 3)))
        ax.axvspan(np.datetime64("2026-01-15"), np.datetime64("2026-02-13"),
                   color=C3, alpha=0.20, lw=0)
        frame(ax)
    lo, hi = axes[0].get_ylim()
    axes[0].annotate("апрель 2025:\nперелом каналов", (np.datetime64("2025-04-06"), hi),
                     color=C2, fontsize=8.5, va="top")
    lo1 = axes[1].get_ylim()[0]
    axes[1].annotate("окно цели\nлокального якоря", (np.datetime64("2026-01-11"), lo1),
                     (0, 4), textcoords="offset points",
                     color="#8a6f13", fontsize=8.5, ha="right", va="bottom")
    save(fig, a.out, "03_daily.svg")

    # ---------------------------------------------------------- 4. рост площадки
    mo = [r for r in D["monthly"] if r["days"] >= 28]   # 2026-02 оборван якорем
    ms = [r["month"][2:] for r in mo]
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.0))
    for ax, vals, ylab, col in (
            (axes[0], [r["gmv"] / 1e6 for r in mo], "GMV за месяц, млн", C1),
            (axes[1], [r["uniq_buyers"] / 1e3 for r in mo], "покупателей, тыс.", C1),
            (axes[2], [r["aov"] for r in mo], "средний чек", C2)):
        ax.plot(ms, vals, color=col, marker="o", ms=3.5)
        ax.set_ylabel(ylab); ax.tick_params(axis="x", rotation=60); frame(ax)
        ax.set_xticks(ms[::3])
    fig.suptitle("Рост идёт числом покупателей, а не величиной покупки",
                 x=0.005, ha="left", fontsize=10.5, color=INK)
    save(fig, a.out, "04_growth.svg")

    # ---------------------------------------------------------- 5. давность
    br = D["by_recency"]
    lb = [("0 дн" if r["recency_bucket"] == "0-0"
           else r["recency_bucket"].replace("-", "–") + " дн") for r in br]
    pp = [r["p_target_pos"] * 100 for r in br]
    nn = [r["n_users"] for r in br]
    fig, ax = plt.subplots(figsize=(6.6, 3.2))
    ax.bar(lb, pp, color=C1, width=0.6)
    for i, (p_, n_) in enumerate(zip(pp, nn)):
        ax.annotate(f"{ru(p_)}%", (i, p_), (0, 4), textcoords="offset points",
                    ha="center", fontsize=8.5, color=SEC)
        ax.annotate(f"{n_:,}".replace(",", " "), (i, 1.5), ha="center", fontsize=8, color="#ffffff")
    ax.set_ylabel("доля купивших в горизонте, %")
    ax.set_xlabel("дней с последнего визита на якоре 378 (внутри столбца — клиентов)")
    ax.set_title("Давность визита — сильнейший одиночный признак")
    frame(ax)
    save(fig, a.out, "05_recency.svg")

    # ---------------------------------------------------------- 6. матрица переходов
    tr = D["transition"]
    M = np.array(tr["matrix"]) * 100
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    ax.grid(False)
    im = ax.imshow(M, cmap=SEQ, aspect="auto", vmin=0, vmax=M.max())
    ax.set_xticks(range(len(tr["labels"]))); ax.set_xticklabels(tr["labels"], rotation=45, ha="right")
    ax.set_yticks(range(len(tr["labels"]))); ax.set_yticklabels(tr["labels"])
    ax.set_xlabel("GMV в горизонте"); ax.set_ylabel("GMV за предыдущие 30 дней")
    ax.set_title("Куда переходит клиент: диагональ размыта, а нуль липкий")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if M[i, j] >= 4:
                ax.text(j, i, f"{M[i,j]:.0f}", ha="center", va="center", fontsize=7.5,
                        color="#ffffff" if M[i, j] > M.max() * 0.55 else SEC)
    cb = fig.colorbar(im, ax=ax, fraction=0.042, pad=0.03)
    cb.set_label("% строки", color=SEC, fontsize=8.5)
    cb.outline.set_visible(False); cb.ax.tick_params(length=0, labelsize=8, colors=MUT)
    for s in ax.spines.values(): s.set_visible(False)
    ax.tick_params(length=0)
    save(fig, a.out, "06_transition.svg")

    # ---------------------------------------------------------- 7. устойчивость блоков
    bl = D["blocks"]
    nm = [r["block"] for r in bl]
    fig, ax = plt.subplots(figsize=(7.6, 3.0))
    ax.plot(nm, [r["share_active"] * 100 for r in bl], color=C1, marker="o", ms=4)
    ax.plot(nm, [r["share_buyers"] * 100 for r in bl], color=C2, marker="o", ms=4)
    ax.annotate("заходили", (0, bl[0]["share_active"] * 100), (6, 6),
                textcoords="offset points", color=C1, fontsize=8.5)
    ax.annotate("покупали", (0, bl[0]["share_buyers"] * 100), (6, 6),
                textcoords="offset points", color=C2, fontsize=8.5)
    ax.set_ylim(0, 108); ax.set_ylabel("доля клиентов, %")
    ax.set_xlabel("блок по 30 дней, B-0 — последний перед 2026-02-13")
    ax.set_title("Клиенты отобраны по активности: три последних блока — ровно 100%")
    ax.axvspan(9.5, len(nm) - 0.5, color=C3, alpha=0.16, lw=0)
    ax.annotate("окно отбора:\n100,0000%, а в B-3 уже 93,3%",
                (len(nm) - 1.2, 100), (-8, -30), textcoords="offset points",
                ha="right", color="#8a6f13", fontsize=8.5,
                arrowprops=dict(arrowstyle="-", color="#8a6f13", lw=0.8))
    frame(ax)
    save(fig, a.out, "07_blocks.svg")

    # ---------------------------------------------------------- 8. бейзлайны
    RU = {"const 0": "константа 0",
          "optimal constant (8.4)": "оптимальная константа 8,4",
          "prev30 (naive AR = sample_submit)": "prev30 как есть — сабмит организаторов",
          "prev60 / 2": "prev60 / 2", "prev90 / 3": "prev90 / 3", "prev180 / 6": "prev180 / 6",
          "mean(prev30, prev60/2, prev90/3)": "среднее трёх окон",
          "alpha*prev30 (alpha=0.30)": "0,30 · prev30",
          "lifetime mean per 30d": "среднее за месяц жизни",
          "expm1(0.54*log1p(prev30)+0.95)": "лог-линейная регрессия по prev30",
          "oracle: mean log-target per prev30 bucket": "оракул: корзина prev30",
          "oracle: prev30 bucket x recency bucket": "оракул: prev30 × давность"}
    bs = [(RU.get(r["baseline"], r["baseline"]), r["rmsle"]) for r in D["baselines"]]
    bs = sorted(bs, key=lambda t: -t[1]) + [("наше решение stack_v23_dir", 1.6452202009)]
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    nmb = [b[0] for b in bs]; vv = [b[1] for b in bs]
    cols = [FILL] * len(vv); cols[-1] = C1
    for i, n_ in enumerate(nmb):
        if n_.startswith("оракул"): cols[i] = C3
    ax.barh(nmb, vv, color=cols, height=0.64)
    for i, v in enumerate(vv):
        ax.annotate(f"{v:.4f}".replace(".", ","), (v, i), (5, 0), textcoords="offset points",
                    va="center", fontsize=8, color=SEC)
    ax.set_xlim(1.55, 3.42); ax.set_xlabel("RMSLE, меньше — лучше")
    frame(ax, axis="x")
    fig.suptitle("Оракулы на агрегатах не доходят до 1,73 — задача не в подборе правила",
                 x=0.005, ha="left", fontsize=10.5, color=INK)
    save(fig, a.out, "08_baselines.svg")

    # ---------------------------------------------------------- 9. лестница
    lad = [("v9", 1.6463024839), ("v11", 1.6462436395), ("v12", 1.6460012027),
           ("v14", 1.6457757641), ("v16", 1.6456395215), ("v17", 1.6455060133),
           ("v18", 1.6454944639), ("v19", 1.6454440380), ("v20", 1.6453934846),
           ("v21", 1.6453455262), ("v22", 1.6452809664), ("v23_dir", 1.6452202009)]
    PEN = 3.29e-05
    nm = [l[0] for l in lad]; sc = np.array([l[1] for l in lad])
    step = -np.diff(sc); net = np.cumsum(step - PEN)
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.4))
    axes[0].plot(nm, sc, color=C1, marker="o", ms=4)
    axes[0].set_ylabel("публичный RMSLE")
    axes[0].set_title("Лестница: одно измеренное направление на ступень")
    axes[1].bar(nm[1:], (step - PEN) * 1e4, width=0.6,
                color=[C1 if v > 0 else C2 for v in step - PEN], label="чистый шаг")
    axes[1].plot(nm[1:], net * 1e4, color=C3, marker="o", ms=3.5, lw=1.6, label="накопленный итог")
    axes[1].axhline(0, color=MUT, lw=0.9)
    axes[1].set_ylim(top=max(net.max(), (step - PEN).max()) * 1e4 * 1.30)
    axes[1].annotate(f"накоплено +{ru(net[-1]*1e4,2)}", (len(nm) - 2, net[-1] * 1e4), (-4, -24),
                     textcoords="offset points", ha="right", color="#8a6f13", fontsize=8.5)
    axes[1].annotate("v17→v18 не отбил штраф", (5, (step - PEN)[5] * 1e4), (10, 42),
                     textcoords="offset points", color=C2, fontsize=8.5,
                     arrowprops=dict(arrowstyle="-", color=C2, lw=0.8))
    axes[1].set_ylabel("×10⁻⁴ RMSLE")
    axes[1].set_title("Шаг за вычетом штрафа за подгонку 3,29·10⁻⁵")
    axes[1].legend(loc="upper left", ncol=2)
    for ax in axes:
        ax.tick_params(axis="x", rotation=45); frame(ax)
    save(fig, a.out, "09_ladder.svg")

    # ---------------------------------------------------------- 10. день недели
    dw = D["dow"]
    names = ["пн", "вт", "ср", "чт", "пт", "сб", "вс"]
    idx = np.array([r["gmv_index"] for r in dw]) * 100 - 100
    fig, ax = plt.subplots(figsize=(5.6, 2.9))
    ax.bar(names, idx, color=[C1 if v >= 0 else C2 for v in idx], width=0.6)
    for i, v in enumerate(idx):
        ax.annotate(f"{v:+.1f}".replace(".", ","), (i, v), (0, 4 if v >= 0 else -13),
                    textcoords="offset points", ha="center", fontsize=8, color=SEC)
    ax.axhline(0, color=MUT, lw=0.9)
    ax.set_ylabel("отклонение GMV от среднего, %")
    ax.set_title("Недельный ритм ±2%: месячная сумма его усредняет")
    frame(ax)
    save(fig, a.out, "10_dow.svg")


if __name__ == "__main__":
    main()
