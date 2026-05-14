import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ====================== 【严格按论文】参数定义 ======================
TASK_NUM = 1110
MANPOWER_LIMIT = 20
UTIL_LOW = 0.8
UTIL_HIGH = 0.9
POP_SIZE = 50
GEN_NUM = 100
CROSS_RATE = 0.8
MUTA_RATE = 0.05
LAMBDA = 100
C_PENALTY = 1e6

# ====================== 1. 读取数据【修正为当前目录路径】 ======================
df = pd.read_excel("aircraft_maintenance_raw_tasks.xlsx", engine="openpyxl")

# 提取核心列
df = df[["Event/Description", "EstMH (All)", "Interval"]].copy()

# 处理工时 MH_j
def parse_mh(x):
    if pd.isna(x): return 0
    s = str(x).strip()
    if "days" in s: s = s.split("days")[-1].strip()
    if ":" in s:
        h,m = s.split(":")[:2]
        return int(h)+int(m)/60
    try: return float(s)
    except: return 0

df["MH_j"] = df["EstMH (All)"].apply(parse_mh)

# 处理最大间隔
def parse_max_gap(x):
    if pd.isna(x) or x=="N/A": return 200
    try: return max(int(x),1)
    except: return 200

df["max_interval"] = df["Interval"].apply(parse_max_gap)
df = df[df["MH_j"]>0].head(TASK_NUM).reset_index(drop=True)

# ====================== ✅ 输出清洗后数据（控制台+本地导出） ======================
print("="*60)
print("📊 Cleaned Data Statistics (清洗后数据)")
print("="*60)
print(f"Total Valid Tasks: {len(df)}")
print(f"Total Man-Hours: {df['MH_j'].sum():.2f}")
print(f"Average Man-Hours: {df['MH_j'].mean():.2f}")
print(f"Average Max Interval: {df['max_interval'].mean():.2f} days")
print("\nSample Data (Top 5):")
print(df[["Event/Description", "MH_j", "max_interval"]].head())
print("="*60)

# 导出清洗后数据到【当前文件夹】
df.to_csv("cleaned_maintenance_tasks.csv", index=False, encoding="utf-8-sig")
print("✅ Cleaned data saved to: cleaned_maintenance_tasks.csv\n")

# 提取算法数组
MH = df["MH_j"].values
MAX_INTERVAL = df["max_interval"].values

# ====================== 2. 染色体初始化（满足80%-90%利用率） ======================
def init_pop():
    pop = []
    for _ in range(POP_SIZE):
        chrom = []
        for j in range(TASK_NUM):
            min_gap = max(int(MAX_INTERVAL[j] * UTIL_LOW), 1)
            max_gap = int(MAX_INTERVAL[j] * UTIL_HIGH)
            gap = np.random.randint(min_gap, max_gap+1)
            chrom.append(gap)
        pop.append(chrom)
    return np.array(pop)

# ====================== 3. 论文原版适应度函数 ======================
def fitness(pop):
    fit_scores = []
    for chrom in pop:
        cost = 0
        for j in range(TASK_NUM):
            gap = chrom[j]
            cost += MH[j] / gap if gap>0 else 1e9
        
        day_load = {}
        zero_cnt = 0
        for j in range(TASK_NUM):
            gap = chrom[j]
            if gap <= 0:
                zero_cnt +=1
                continue
            d = int(gap)
            day_load[d] = day_load.get(d, 0) + MH[j]
        
        overload_pen = 0
        for d, load in day_load.items():
            if load > MANPOWER_LIMIT:
                overload_pen += (load - MANPOWER_LIMIT)**2
        overload_pen *= LAMBDA
        zero_pen = C_PENALTY * zero_cnt
        
        total = cost + overload_pen + zero_pen
        fit_scores.append(-total)
    return np.array(fit_scores)

# ====================== 4. 遗传操作 ======================
def select(pop, fit):
    fit = fit - np.min(fit) + 1e-8
    idx = np.random.choice(POP_SIZE, POP_SIZE, p=fit/fit.sum())
    return pop[idx]

def crossover(c1,c2):
    if np.random.rand() < CROSS_RATE:
        pt = np.random.randint(0, TASK_NUM)
        c1[pt:], c2[pt:] = c2[pt:], c1[pt:]
    return c1,c2

def mutate(c):
    for j in range(TASK_NUM):
        if np.random.rand() < MUTA_RATE:
            min_g = max(int(MAX_INTERVAL[j]*0.8),1)
            max_g = int(MAX_INTERVAL[j]*0.9)
            c[j] = np.random.randint(min_g, max_g+1)
    return c

# ====================== 5. 运行算法 ======================
pop = init_pop()
best_history = []

print("🚀 Running Genetic Algorithm...")
for gen in range(GEN_NUM):
    fit = fitness(pop)
    best = np.max(fit)
    best_history.append(best)
    pop = select(pop, fit)
    for i in range(0, POP_SIZE, 2):
        c1,c2 = pop[i], pop[i+1]
        c1,c2 = crossover(c1,c2)
        pop[i] = mutate(c1)
        pop[i+1] = mutate(c2)
    if gen%10==0:
        print(f"Generation {gen} | Best Fitness: {best:.2f}")

# 最优解
best_idx = np.argmax(fitness(pop))
best_chrom = pop[best_idx]

# ====================== 6. 结果统计 ======================
day_load = {}
util_list = []
total_mh = 0
zero_cnt = 0

for j in range(TASK_NUM):
    g = best_chrom[j]
    if g<=0:
        zero_cnt +=1
        continue
    day_load[int(g)] = day_load.get(int(g),0) + MH[j]
    util_list.append(g / MAX_INTERVAL[j])
    total_mh += MH[j]

avg_util = np.mean(util_list)*100
over_days = [d for d,l in day_load.items() if l>20]
compliant_rate = len([u for u in util_list if 0.8<=u<=0.9])/len(util_list)*100

print("\n" + "="*60)
print("✅ Final Optimization Results")
print(f"Total Tasks: {TASK_NUM}")
print(f"Average Utilization: {avg_util:.2f}% (Target: 80%-90%)")
print(f"Compliant Tasks: {compliant_rate:.2f}%")
print(f"Average Daily Workload: {total_mh/len(day_load):.2f} Man-Hours")
print(f"Over-Limit Days (>20H): {len(over_days)}")
print("="*60)

# ====================== 7. 可视化 + 保存图片到当前文件夹 ======================
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 收敛曲线
plt.figure(figsize=(10,4))
plt.plot(best_history)
plt.title("GA Convergence Curve")
plt.xlabel("Generations")
plt.ylabel("Fitness")
plt.grid()
plt.tight_layout()
plt.savefig("convergence_curve.png", dpi=300)
plt.show()

# 人力负载图
days = sorted(day_load.keys())[:30]
loads = [day_load[d] for d in days]
plt.figure(figsize=(10,4))
plt.bar(days, loads)
plt.axhline(20, c='r', ls='--', label="20 Man-Hours Limit")
plt.title("30-Day Workload Distribution")
plt.xlabel("Day")
plt.ylabel("Man-Hours")
plt.legend()
plt.grid(axis='y')
plt.tight_layout()
plt.savefig("daily_workload_30days.png", dpi=300)
plt.show()