import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── 1. Генерація синтетичного датасету (аналог California Housing) ────────────
np.random.seed(42)
N = 20_640

df = pd.DataFrame({
    "longitude":          np.random.uniform(-124.35, -114.31, N),
    "latitude":           np.random.uniform(32.54,   41.95,   N),
    "housing_median_age": np.random.randint(1, 52, N).astype(np.float64),
    "total_rooms":        np.random.randint(2, 39320, N).astype(np.float64),
    "total_bedrooms":     np.random.randint(1, 6445, N).astype(np.float64),
    "population":         np.random.randint(3, 35682, N).astype(np.float64),
    "households":         np.random.randint(1, 6082,  N).astype(np.float64),
    "median_income":      np.random.uniform(0.5, 15.0, N),
    "price":              np.random.uniform(14_999, 500_001, N),
})

# ── 2. Еталонне середнє до конвертації ───────────────────────────────────────
price_mean_f64 = df["price"].mean()

# ── 3. Пам'ять оригіналу ─────────────────────────────────────────────────────
mem_orig       = df.memory_usage(deep=True)
total_mem_orig = mem_orig.sum()

print("=" * 60)
print("ВИХІДНИЙ ДАТАСЕТ")
print("=" * 60)
print(f"Рядків: {len(df):,}  |  Колонок: {len(df.columns)}")
print("\nТипи даних (оригінал):")
print(df.dtypes)

print(f"\n{'─'*60}")
print("ПАМ'ЯТЬ ДО КОНВЕРТАЦІЇ (float64)")
print(f"{'─'*60}")
for col in df.columns:
    mem = mem_orig[col] / 1024
    print(f"  {col:<22} {mem:>8.2f} KB  [{df[col].dtype}]")
print(f"  {'РАЗОМ':<22} {total_mem_orig/1024:>8.2f} KB")

# ── 4. Конвертація ───────────────────────────────────────────────────────────
df_conv    = df.copy()
int_cols   = ["total_rooms", "total_bedrooms", "population", "households"]
float_cols = [c for c in df.columns if c not in int_cols]

for col in float_cols:
    df_conv[col] = df_conv[col].astype(np.float32)
for col in int_cols:
    df_conv[col] = df_conv[col].astype(np.int32)

# ── 5. Пам'ять після конвертації ─────────────────────────────────────────────
mem_conv       = df_conv.memory_usage(deep=True)
total_mem_conv = mem_conv.sum()

print(f"\n{'─'*60}")
print("ПАМ'ЯТЬ ПІСЛЯ КОНВЕРТАЦІЇ (float32 / int32)")
print(f"{'─'*60}")

before_list, after_list = [], []
for col in df.columns:
    before = mem_orig[col] / 1024
    after  = mem_conv[col] / 1024
    before_list.append(before)
    after_list.append(after)
    saved = (1 - after / before) * 100
    print(f"  {col:<22} {before:>7.2f} KB -> {after:>7.2f} KB  "
          f"[{df[col].dtype} -> {df_conv[col].dtype}]  -{saved:.0f}%")
print(f"  {'РАЗОМ':<22} {total_mem_orig/1024:>7.2f} KB -> {total_mem_conv/1024:>7.2f} KB"
      f"  -{(1-total_mem_conv/total_mem_orig)*100:.1f}%")

# ── 6. Похибка середнього price ───────────────────────────────────────────────
price_mean_f32 = float(df_conv["price"].mean())
abs_error      = abs(price_mean_f64 - price_mean_f32)
rel_error      = abs_error / price_mean_f64 * 100

print(f"\n{'─'*60}")
print("ПОХИБКА СЕРЕДНЬОГО ЗНАЧЕННЯ 'price'")
print(f"{'─'*60}")
print(f"  Середнє (float64): {price_mean_f64:.10f}")
print(f"  Середнє (float32): {price_mean_f32:.10f}")
print(f"  Абсолютна похибка: {abs_error:.2e}")
print(f"  Відносна похибка:  {rel_error:.8f} %")

# ── 7. Підсумок ───────────────────────────────────────────────────────────────
saved_kb  = (total_mem_orig - total_mem_conv) / 1024
saved_pct = (1 - total_mem_conv / total_mem_orig) * 100

print(f"\n{'='*60}")
print("ПІДСУМОК")
print(f"{'='*60}")
print(f"  Зекономлено пам'яті:  {saved_kb:.1f} KB  ({saved_pct:.1f}%)")
print(f"  Похибка mean price:   {abs_error:.2e}  ({rel_error:.8f}%)")
print(f"\n  Висновок: float32 економить ~50% пам'яті числових колонок,")
print(f"  при цьому похибка середнього price ~ {rel_error:.6f}% -- практично нульова.")
print("=" * 60)

# ══════════════════════════════════════════════════════════════════════════════
# ── 8. ГРАФІКИ ───────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

cols  = list(df.columns)
x     = np.arange(len(cols))
width = 0.38

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Порівняння типів даних: float64 vs float32/int32",
             fontsize=15, fontweight="bold", y=1.01)

# ── Графік 1: Пам'ять по колонках ────────────────────────────────────────────
ax1 = axes[0]
bars1 = ax1.bar(x - width/2, before_list, width, label="float64",      color="#4C72B0", alpha=0.9)
bars2 = ax1.bar(x + width/2, after_list,  width, label="float32/int32", color="#55A868", alpha=0.9)
ax1.set_title("Пам'ять по колонках (KB)", fontweight="bold")
ax1.set_xticks(x)
ax1.set_xticklabels(cols, rotation=40, ha="right", fontsize=8)
ax1.set_ylabel("KB")
ax1.legend()
ax1.grid(axis="y", alpha=0.3)
for bar in bars1:
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f"{bar.get_height():.0f}", ha="center", va="bottom", fontsize=6.5, color="#4C72B0")
for bar in bars2:
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f"{bar.get_height():.0f}", ha="center", va="bottom", fontsize=6.5, color="#55A868")

# ── Графік 2: Загальна пам'ять (pie) ─────────────────────────────────────────
ax2 = axes[1]
used   = total_mem_conv / 1024
saved  = saved_kb
colors = ["#55A868", "#C0C0C0"]
wedges, texts, autotexts = ax2.pie(
    [used, saved],
    labels=["Після\nконвертації", "Зекономлено"],
    autopct="%1.1f%%",
    colors=colors,
    startangle=90,
    wedgeprops=dict(edgecolor="white", linewidth=2),
    textprops=dict(fontsize=10)
)
for at in autotexts:
    at.set_fontweight("bold")
ax2.set_title(f"Загальна пам'ять\n(було: {total_mem_orig/1024:.0f} KB)", fontweight="bold")
patch1 = mpatches.Patch(color=colors[0], label=f"Після: {used:.0f} KB")
patch2 = mpatches.Patch(color=colors[1], label=f"Економія: {saved:.0f} KB")
ax2.legend(handles=[patch1, patch2], loc="lower center",
           bbox_to_anchor=(0.5, -0.12), fontsize=9)

# ── Графік 3: Середнє price — float64 vs float32 ─────────────────────────────
ax3 = axes[2]
means  = [price_mean_f64, price_mean_f32]
labels = ["float64\n(еталон)", "float32\n(конвертоване)"]
bcolors= ["#4C72B0", "#DD8452"]
bars3  = ax3.bar(labels, means, color=bcolors, alpha=0.9, width=0.4, edgecolor="white")

delta = abs_error * 5000
ax3.set_ylim(min(means) - delta, max(means) + delta * 3)
ax3.set_title("Середнє значення 'price'\n(float64 vs float32)", fontweight="bold")
ax3.set_ylabel("Середня ціна ($)")
ax3.grid(axis="y", alpha=0.3)
for bar, val in zip(bars3, means):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + delta * 0.3,
             f"${val:,.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
ax3.annotate(
    f"Абс. похибка:\n{abs_error:.2e}\nВідн.: {rel_error:.6f}%",
    xy=(1, price_mean_f32),
    xytext=(1.35, price_mean_f32 + delta * 1.5),
    fontsize=8.5, color="#C44E52",
    arrowprops=dict(arrowstyle="->", color="#C44E52"),
    bbox=dict(boxstyle="round,pad=0.3", fc="#FFF0F0", ec="#C44E52")
)

plt.tight_layout()
plt.savefig("dtype_comparison.png", dpi=150, bbox_inches="tight")
print("\n Графік збережено: dtype_comparison.png")
plt.show()