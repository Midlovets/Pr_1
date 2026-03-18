import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ── Генерація синтетичних даних ─────────────────────────────────────────────
np.random.seed(42)
n = 730
dates = pd.date_range(start='2022-01-01', periods=n, freq='D')

trend_true  = np.linspace(100, 160, n) + 0.01 * np.arange(n) ** 1.05
seasonal    = 30 * np.sin(2 * np.pi * np.arange(n) / 365) + \
              10 * np.sin(2 * np.pi * np.arange(n) / 7)
noise       = np.random.normal(0, 8, n)
consumption = trend_true + seasonal + noise

ts = pd.Series(consumption, index=dates, name='Споживання, кВт·год')

# ── Декомпозиція ─────────────────────────────────────────────────────────────
result = seasonal_decompose(ts, model='additive', period=365)
trend_c    = result.trend.dropna()
seasonal_c = result.seasonal.dropna()
resid_c    = result.resid.dropna()
common     = trend_c.index.intersection(seasonal_c.index).intersection(resid_c.index)
trend_c    = trend_c[common]
seasonal_c = seasonal_c[common]
resid_c    = resid_c[common]
observed_c = ts[common]

# ── Прогнозування ────────────────────────────────────────────────────────────
split = len(common) - 60
X_tr = np.arange(split).reshape(-1, 1)
y_tr = trend_c.iloc[:split].values
lr   = LinearRegression().fit(X_tr, y_tr)

X_te       = np.arange(split, len(common)).reshape(-1, 1)
fc_trend   = lr.predict(X_te)
fc_seasonal= seasonal_c.iloc[split:].values
forecast   = fc_trend + fc_seasonal
actual     = observed_c.iloc[split:].values
error      = actual - forecast

mae  = mean_absolute_error(actual, forecast)
rmse = np.sqrt(mean_squared_error(actual, forecast))
r2   = r2_score(actual, forecast)
mape = np.mean(np.abs(error / actual)) * 100

# ══════════════════════════════════════════════════════════════════════════════
# РИСУНОК 1 — Декомпозиція
# ══════════════════════════════════════════════════════════════════════════════
C = dict(obs='#2563EB', trend='#D97706', seas='#059669',
         resid='#DC2626', fill='#EFF6FF', line='#E2E8F0',
         text='#1E293B', muted='#64748B')

fig1, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
fig1.patch.set_facecolor('white')
fig1.suptitle('Декомпозиція часового ряду енергоспоживання',
              fontsize=15, fontweight='bold', color=C['text'], y=0.98)

panels = [
    (observed_c, 'Вихідний ряд',  C['obs'],   True),
    (trend_c,    'Тренд',         C['trend'],  False),
    (seasonal_c, 'Сезонність',    C['seas'],   True),
    (resid_c,    'Залишок',       C['resid'],  False),
]

for ax, (data, title, color, fill) in zip(axes, panels):
    ax.set_facecolor('white')
    for sp in ax.spines.values():
        sp.set_color(C['line'])
    ax.tick_params(colors=C['muted'], labelsize=9)
    ax.yaxis.set_tick_params(labelsize=9)
    ax.grid(axis='y', color=C['line'], linewidth=0.8, linestyle='-')
    ax.set_axisbelow(True)

    ax.plot(data.index, data.values, color=color, linewidth=1.3, zorder=3)
    if fill:
        ax.fill_between(data.index, data.values, alpha=0.08, color=color)

    # Осьова лінія для залишку
    if title == 'Залишок':
        ax.axhline(0, color=C['muted'], linewidth=0.8, linestyle='--', alpha=0.6)
        for bar in ax.collections:
            pass
        pos = data.values > 0
        ax.fill_between(data.index, data.values, 0,
                        where=pos,  alpha=0.15, color=C['seas'])
        ax.fill_between(data.index, data.values, 0,
                        where=~pos, alpha=0.15, color=C['resid'])

    ax.set_ylabel(title, fontsize=10, color=C['text'], fontweight='500',
                  labelpad=6)

axes[-1].xaxis.set_tick_params(labelrotation=0)
fig1.tight_layout(rect=[0, 0, 1, 0.97])
fig1.savefig('e:/Pr_1/fig1_decomposition.png',
             dpi=150, bbox_inches='tight', facecolor='white')
plt.close(fig1)
print('✓ fig1_decomposition.png')


# ══════════════════════════════════════════════════════════════════════════════
# РИСУНОК 2 — Прогнозування
# ══════════════════════════════════════════════════════════════════════════════
fig2 = plt.figure(figsize=(14, 8))
fig2.patch.set_facecolor('white')
gs = gridspec.GridSpec(2, 2, figure=fig2, hspace=0.42, wspace=0.3)

ax_main = fig2.add_subplot(gs[0, :])
ax_err  = fig2.add_subplot(gs[1, 0])
ax_scat = fig2.add_subplot(gs[1, 1])

fig2.suptitle('Прогнозування енергоспоживання (Тренд + Сезонність)',
              fontsize=15, fontweight='bold', color=C['text'], y=0.98)

def style(ax):
    ax.set_facecolor('white')
    for sp in ax.spines.values(): sp.set_color(C['line'])
    ax.tick_params(colors=C['muted'], labelsize=9)
    ax.grid(color=C['line'], linewidth=0.8, linestyle='-')
    ax.set_axisbelow(True)

# ─ Головний графік ─
style(ax_main)
train_idx = observed_c.index[:split]
test_idx  = observed_c.index[split:]

ax_main.plot(train_idx, observed_c.values[:split],
             color=C['obs'], linewidth=1.2, label='Тренінг (факт)', alpha=0.8)
ax_main.plot(test_idx, actual,
             color=C['seas'], linewidth=2.0, label='Тест (факт)', zorder=4)
ax_main.plot(test_idx, forecast,
             color=C['trend'], linewidth=2.2, linestyle='--',
             label='Прогноз (Тренд+Сезон)', zorder=5)

# Смуга розбиття
ax_main.axvline(test_idx[0], color=C['muted'], linewidth=1.2,
                linestyle=':', alpha=0.7)
ax_main.text(test_idx[0], ax_main.get_ylim()[0] if ax_main.get_ylim()[0] > 0 else 60,
             '  ← тренінг | тест →', fontsize=8.5, color=C['muted'],
             va='bottom')

ax_main.set_ylabel('кВт·год', fontsize=10, color=C['text'])
ax_main.legend(fontsize=9, framealpha=0.9, edgecolor=C['line'])

# Метрики у кутку
metrics_txt = (f"MAE  = {mae:.3f}\n"
               f"RMSE = {rmse:.3f}\n"
               f"MAPE = {mape:.2f}%\n"
               f"R²    = {r2:.4f}")
ax_main.text(0.01, 0.97, metrics_txt, transform=ax_main.transAxes,
             fontsize=9, va='top', ha='left', color=C['text'],
             fontfamily='monospace',
             bbox=dict(facecolor='white', edgecolor=C['line'],
                       boxstyle='round,pad=0.5', alpha=0.95))

# ─ Похибка ─
style(ax_err)
colors_err = [C['seas'] if e >= 0 else C['resid'] for e in error]
ax_err.bar(range(len(error)), error, color=colors_err, width=0.9, alpha=0.75)
ax_err.axhline(0, color=C['muted'], linewidth=1.0)
ax_err.set_xlabel('День тесту', fontsize=9, color=C['muted'])
ax_err.set_ylabel('Факт − Прогноз, кВт·год', fontsize=9, color=C['text'])
ax_err.set_title('Похибка прогнозу', fontsize=11, color=C['text'], pad=8)

# ─ Scatter факт vs прогноз ─
style(ax_scat)
ax_scat.scatter(actual, forecast, color=C['obs'], s=30, alpha=0.7, zorder=3)
mn, mx = min(actual.min(), forecast.min()), max(actual.max(), forecast.max())
pad = (mx - mn) * 0.05
ax_scat.plot([mn-pad, mx+pad], [mn-pad, mx+pad],
             color=C['resid'], linewidth=1.5, linestyle='--',
             label='Ідеальний прогноз', zorder=4)
ax_scat.set_xlabel('Факт, кВт·год', fontsize=9, color=C['muted'])
ax_scat.set_ylabel('Прогноз, кВт·год', fontsize=9, color=C['text'])
ax_scat.set_title(f'Факт vs Прогноз  (R²={r2:.4f})',
                  fontsize=11, color=C['text'], pad=8)
ax_scat.legend(fontsize=8, edgecolor=C['line'])

fig2.savefig('e:/Pr_1/fig2_forecast.png',
             dpi=150, bbox_inches='tight', facecolor='white')
plt.close(fig2)
print('✓ fig2_forecast.png')


# ══════════════════════════════════════════════════════════════════════════════
# РИСУНОК 3 — Статистика компонентів
# ══════════════════════════════════════════════════════════════════════════════
fig3, axes3 = plt.subplots(1, 3, figsize=(14, 4))
fig3.patch.set_facecolor('white')
fig3.suptitle('Аналіз компонентів декомпозиції',
              fontsize=14, fontweight='bold', color=C['text'], y=1.02)

# Внесок у дисперсію
def var(a): m=a.mean(); return ((a-m)**2).mean()
tv, sv, rv = var(trend_c.values), var(seasonal_c.values), var(resid_c.values)
total = tv + sv + rv
labels3 = ['Тренд', 'Сезонність', 'Залишок']
vals3   = [tv/total*100, sv/total*100, rv/total*100]
colors3 = [C['trend'], C['seas'], C['resid']]

style(axes3[0])
bars = axes3[0].barh(labels3, vals3, color=colors3, height=0.5, alpha=0.85)
for bar, v in zip(bars, vals3):
    axes3[0].text(v + 0.5, bar.get_y() + bar.get_height()/2,
                  f'{v:.1f}%', va='center', fontsize=10,
                  color=C['text'], fontweight='500')
axes3[0].set_xlim(0, 105)
axes3[0].set_xlabel('Частка дисперсії, %', fontsize=9, color=C['muted'])
axes3[0].set_title('Внесок у дисперсію', fontsize=11, color=C['text'], pad=8)
axes3[0].grid(axis='x', color=C['line'], linewidth=0.8)
axes3[0].grid(axis='y', visible=False)

# Гістограма залишку
style(axes3[1])
axes3[1].hist(resid_c.values, bins=30, color=C['obs'], alpha=0.75,
              edgecolor='white', linewidth=0.5)
axes3[1].axvline(resid_c.mean(), color=C['resid'], linewidth=1.8,
                 linestyle='--', label=f'Середнє={resid_c.mean():.2f}')
axes3[1].set_xlabel('Значення залишку', fontsize=9, color=C['muted'])
axes3[1].set_ylabel('Частота', fontsize=9, color=C['text'])
axes3[1].set_title('Розподіл залишку', fontsize=11, color=C['text'], pad=8)
axes3[1].legend(fontsize=8, edgecolor=C['line'])

# Метрики у вигляді таблиці
style(axes3[2])
axes3[2].axis('off')
rows = [
    ['MAE',   f'{mae:.3f}',  'кВт·год'],
    ['RMSE',  f'{rmse:.3f}', 'кВт·год'],
    ['MAPE',  f'{mape:.2f}%',''],
    ['R²',    f'{r2:.4f}',   ''],
    ['Тренд ↑', f'+{(trend_c.iloc[-1]-trend_c.iloc[0]):.1f}', 'кВт·год'],
    ['Амп. сезону', f'±{seasonal_c.values.std():.1f}', 'кВт·год'],
]
tbl = axes3[2].table(cellText=rows,
                     colLabels=['Метрика', 'Значення', 'Одиниці'],
                     cellLoc='center', loc='center',
                     colWidths=[0.38, 0.32, 0.30])
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
for (r, c), cell in tbl.get_celld().items():
    cell.set_edgecolor(C['line'])
    if r == 0:
        cell.set_facecolor('#F1F5F9')
        cell.set_text_props(fontweight='bold', color=C['text'])
    else:
        cell.set_facecolor('white')
        cell.set_text_props(color=C['text'])
    cell.set_height(0.13)
axes3[2].set_title('Зведені метрики', fontsize=11, color=C['text'], pad=8)

fig3.tight_layout()
fig3.savefig('e:/Pr_1/fig3_stats.png',
             dpi=150, bbox_inches='tight', facecolor='white')
plt.close(fig3)
print('✓ fig3_stats.png')

print('\nВсі графіки збережено!')
print(f'  MAE={mae:.3f}  RMSE={rmse:.3f}  MAPE={mape:.2f}%  R²={r2:.4f}')