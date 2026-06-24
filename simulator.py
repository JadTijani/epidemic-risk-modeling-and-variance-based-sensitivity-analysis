import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from matplotlib.widgets import Slider, CheckButtons
import datetime as dt
import matplotlib.dates as mdates
import tkinter as tk
from tkinter import ttk, messagebox
from SALib.sample import sobol as sobol_sample
from SALib.analyze import sobol as sobol_analyze
import seaborn as sns
import concurrent.futures
import multiprocessing

START_DATE = dt.datetime(2020, 1, 24)
try:
    matplotlib.use('TkAgg')
except:
    pass

default_p = {}

tooltips = {
    's_beta': "Beta 0: Initial contagiousness (Transmission rate).",
    's_sigma': "Incubation: Time to become infectious (E->I).",
    's_gamma': "Recovery: Time to recover (I->R).",
    's_omega': "Immunity Loss: Return R -> S .",
    's_mut': "Mutation: Virus becomes more contagious over time.",
    's_fear': "Fear: Contact reduction when cases rise.",
    's_thresh': "Lockdown Threshold: % triggering lockdown.",
    's_alpha': "Lockdown Force: Effectiveness/Strictness of confinement.",
    's_qcap': "Hospital Capacity: Max % tested/isolated.",
    's_fatigue': "Fatigue: Lockdown effectiveness decreases over time."
}


def setup_parameters_gui():
    root = tk.Tk()
    root.title("SEIR Model Initial Configuration")

    user_cancelled = [True]

    params_config = {
        "Global & Vaccination": [
            ('Total Population (N)', 'N', 100000),
            ('Max Time (Days)', 'T_max', 1400),
            ('Time Step (dt)', 'dt', 0.1),
            ('Vax Start (Day)', 'vax_start', 365),
            ('Vax Rate (/day)', 'vax_rate', 0.003),
        ],
        "Biology (Input in Days)": [
            ('Incubation (Days)', 'sigma_days', 5.2),
            ('Recovery (Days)', 'gamma_days', 10.0),
            ('Immunity Duration (Days)', 'omega_days', 180.0),
            ('Quarantine Duration (Days)', 'gamma_Q_days', 14.0),
            ('Beta 0 (Base R0)', 'beta_0', 0.27),
            ('Mutation Rate', 'mutation_rate', 0.0001),
        ],
        "Behavior & Costs": [
            ('Fear Factor (fear_k)', 'fear_k', 0.7),
            ('Habituation (awareness)', 'awareness_slope', 0.0001),
            ('Holiday Boost', 'holiday_boost', 1.3),
            ('Lockdown Cost (cost_L)', 'cost_L', 1.0),
            ('Sickness Cost (cost_I)', 'cost_I', 10.0),
        ],
        "Dynamic Lockdown": [
            ('Max Test Capacity', 'delta_max', 0.4),
            ('Quarantine Capacity (%)', 'Q_capacity_pct', 0.04),
            ('Lockdown Threshold (%)', 'lockdown_threshold_pct', 0.5),
            ('Beds Threshold Base', 'beds_threshold_base', 0.6),
            ('Lockdown Delay', 'lockdown_delay', 7),
            ('Lockdown Strength (alpha_0)', 'alpha_0', 1.0),
            ('Lockdown Leakage (mu_0)', 'mu_0', 0.5),
            ('Fatigue Rate', 'fatigue_rate', 0.05),
            ('Fatigue Recovery', 'fatigue_recovery', 0.005),
        ],
        "Fixed Lockdown (Legacy)": [
            ('Lockdown Cycles', 'legacy_cycles', 2),
            ('Lockdown Duration (Days)', 'legacy_dur', 30),
            ('Pause between lockdowns', 'legacy_pause', 90),
        ],
        "Waves & Evasion (ex: Omicron)": [
            ('Wave 1 Start (Day)', 'var1_t', 250),
            ('Wave 1 Boost (x Beta)', 'var1_boost', 1.5),
            ('Wave 2 Start (Day)', 'var2_t', 450),
            ('Wave 2 Boost (x Beta)', 'var2_boost', 1.8),
            ('Wave 3 Start (Day)', 'var3_t', 680),
            ('Wave 3 Boost (x Beta)', 'var3_boost', 4.5),
            ('Evasion Start (Day)', 'evasion_t', 680),
            ('Evasion Duration (Days)', 'evasion_dur', 170),
            ('Evasion Rate (% R/day)', 'evasion_rate_pct', 0.5),
        ]
    }

    entries = {}

    def on_submit():
        try:
            for key, entry in entries.items():
                val = float(entry.get())
                if key.endswith('_days'):
                    real_key = key.replace('_days', '')
                    default_p[real_key] = 1.0 / val if val > 0 else 0.0
                else:
                    if key in ['N', 'T_max', 'lockdown_delay', 'legacy_cycles', 'legacy_dur', 'legacy_pause', 'var1_t',
                               'var2_t', 'var3_t', 'evasion_t', 'evasion_dur']:
                        default_p[key] = int(val)
                    else:
                        default_p[key] = val
            user_cancelled[0] = False
            root.destroy()
        except ValueError:
            messagebox.showerror("Error", "Please enter valid numbers only.")

    row, col = 0, 0
    for category, params in params_config.items():
        frame = ttk.LabelFrame(root, text=category, padding=10)
        frame.grid(row=row, column=col, padx=10, pady=10, sticky="n")

        for i, (label_text, key, default_val) in enumerate(params):
            ttk.Label(frame, text=label_text).grid(row=i, column=0, sticky="w", pady=2)
            entry = ttk.Entry(frame, width=10)
            entry.insert(0, str(default_val))
            entry.grid(row=i, column=1, sticky="e", pady=2)
            entries[key] = entry

        col += 1
        if col >= 3:
            col = 0
            row += 1

    submit_btn = ttk.Button(root, text="Validate and Launch Simulation", command=on_submit)
    submit_btn.grid(row=row + 1, column=0, columnspan=3, pady=15)

    root.mainloop()

    if user_cancelled[0]:
        print("Simulation cancelled.")
        sys.exit()


def deriv(y, t, p, use_dyn_L, use_dyn_F):
    S, E, I, Q, R, L, F_acc, Econ = y
    cap_abs = p['N'] * (p['Q_capacity_pct'] / 100.0)

    variant_boost = 1.0
    if t > p['var1_t']: variant_boost = p['var1_boost']
    if t > p['var2_t']: variant_boost = p['var2_boost']
    if t > p['var3_t']: variant_boost = p['var3_boost']

    testing_drop = 1.0
    if t > 1200:
        testing_drop = 1.0 - 0.8 * (1 / (1 + np.exp(-0.02 * (t - 1000))))

    day_of_year = t % 365
    holiday_mult = 1.0
    if day_of_year > 330 or day_of_year < 15:
        holiday_mult = p['holiday_boost']

    variant_boost = variant_boost * testing_drop
    seasonality = 1 + 0.35 * np.cos(2 * np.pi * t / 365)

    beta_cal = p['beta_0'] * variant_boost * seasonality * holiday_mult

    fear_fatigue = np.exp(-p['awareness_slope'] * t)
    current_fear = p['fear_k'] * fear_fatigue
    beta_t = beta_cal * np.exp(-current_fear * (I / p['N']))

    immune_evasion_flow = 0.0
    if p['evasion_t'] < t < (p['evasion_t'] + p['evasion_dur']):
        immune_evasion_flow = R * (p['evasion_rate_pct'] / 100.0)

    vax_efficiency = 1.0
    vax_hospital_protection = 1.0
    if t > p['evasion_t']:
        vax_efficiency = 0.3
        vax_hospital_protection = 0.5

    vax_flow = 0
    if t > p['vax_start']:
        vax_flow = p['vax_rate'] * S * vax_efficiency

    active_target = 0.0
    if use_dyn_L:
        incidence_relative = I / p['N']
        occ_rate = Q / (cap_abs + 1e-9)

        threshold_infected = p['lockdown_threshold_pct'] / 100.0
        threshold_beds = p['beds_threshold_base'] + (R / p['N']) * 0.2
        target_incidence = 1 / (1 + np.exp(-1000 * (incidence_relative - threshold_infected)))
        target_lits = 1 / (1 + np.exp(-15 * (occ_rate - threshold_beds)))

        active_target = max(target_incidence, target_lits)
        if t < p['lockdown_delay']: active_target = 0.0
    else:
        starts = np.arange(20, p['T_max'], p['legacy_dur'] + p['legacy_pause'])[:int(max(1, p['legacy_cycles']))]
        in_period = 0.0
        for s in starts:
            arg_in = np.clip(-1.0 * (t - s), -100, 100)
            arg_out = np.clip(-1.0 * (t - (s + p['legacy_dur'])), -100, 100)
            in_period += (1 / (1 + np.exp(arg_in))) - (1 / (1 + np.exp(arg_out)))
        active_target = np.clip(in_period, 0, 1)

    fatigue_effect = np.exp(-F_acc) if use_dyn_F else 1.0
    curr_alpha = p['alpha_0'] * active_target * fatigue_effect
    curr_mu = p['mu_0'] * (1 - active_target)

    fill_ratio = Q / (cap_abs + 1e-9)
    sat_factor = 1 / (1 + np.exp(np.clip(5.0 * (fill_ratio - 1), -100, 100)))
    system_rampup = 1 / (1 + np.exp(-0.2 * (t - 30)))
    curr_delta = p['delta_max'] * system_rampup * max(0.05, sat_factor)

    force_inf = (beta_t * I / p['N']) + 5e-5
    flow_S_E = force_inf * S
    flow_L_E = (force_inf * 0.3) * L

    dSdt = -flow_S_E - curr_alpha * S + curr_mu * L + p['omega'] * R - vax_flow + immune_evasion_flow
    dLdt = curr_alpha * S - curr_mu * L - flow_L_E
    dEdt = (flow_S_E + flow_L_E) - p['sigma'] * E

    effective_delta = curr_delta * vax_hospital_protection
    dIdt = p['sigma'] * E - (p['gamma'] + effective_delta) * I
    dQdt = effective_delta * I - p['gamma_Q'] * Q
    dRdt = p['gamma'] * I + p['gamma_Q'] * Q - p['omega'] * R + vax_flow - immune_evasion_flow

    dF_acc = p['fatigue_rate'] * (L / p['N']) - p['fatigue_recovery'] * F_acc
    dEcon = (L / p['N'] * p['cost_L']) + (I / p['N'] * p['cost_I'])

    return dSdt, dEdt, dIdt, dQdt, dRdt, dLdt, dF_acc, dEcon


def run_scenario_process(args):
    i, X, p_dict = args
    p_local = p_dict.copy()

    p_local['delta_max'] = X[0]
    p_local['Q_capacity_pct'] = X[1]
    p_local['lockdown_threshold_pct'] = X[2]
    p_local['beds_threshold_base'] = X[3]
    p_local['lockdown_delay'] = X[4]
    p_local['alpha_0'] = X[5]
    p_local['mu_0'] = X[6]
    p_local['fatigue_rate'] = X[7]
    p_local['fatigue_recovery'] = X[8]
    p_local['fear_k'] = X[9]
    p_local['awareness_slope'] = X[10]
    p_local['holiday_boost'] = X[11]

    t_steps = int(p_local['T_max'] / p_local['dt'])
    t_space = np.linspace(0, p_local['T_max'], t_steps)

    I_start = 50
    E_start = 200
    S_start = p_local['N'] - I_start - E_start
    y0 = (S_start, E_start, I_start, 0, 0, 0, 0, 0)

    res = odeint(deriv, y0, t_space, args=(p_local, True, True))

    peak_I = np.max(res[:, 2])
    total_Econ = res[-1, 7]
    total_Recovered = res[-1, 4]
    peak_Q = np.max(res[:, 3])
    peak_time = np.argmax(res[:, 2]) * p_local['dt']
    total_L_vol = np.sum(res[:, 5]) * p_local['dt']

    return i, peak_I, total_Econ, total_Recovered, peak_Q, peak_time, total_L_vol


def run_mathematical_sensitivity():
    try:
        from SALib.sample import sobol as sobol_sample
        from SALib.analyze import sobol as sobol_analyze
        import seaborn as sns
        import concurrent.futures
        import multiprocessing
    except ImportError:
        print("\n[INFO] Install SALib/seaborn: pip install SALib seaborn")
        return

    print("\n" + "=" * 60)
    print(" STARTING SOBOL ANALYSIS: 6-DIMENSIONAL MATRICES + ST BARS")
    print("=" * 60)

    problem = {
        'num_vars': 12,
        'names': [
            'Test_Cap', 'Hosp_Cap', 'Lock_Thresh', 'Beds_Thresh',
            'Delay', 'Lock_Force', 'Leakage', 'Fatigue_Rate',
            'Fatigue_Recov', 'Fear_Factor', 'Habituation', 'Holidays'
        ],
        'bounds': [
            [0.2, 0.8],
            [0.05, 0.5],
            [0.5, 3.0],
            [0.4, 0.8],
            [0, 21],
            [0.5, 2.5],
            [0.2, 0.7],
            [0.01, 0.10],
            [0.0, 0.02],
            [0.0, 5.0],
            [0.0001, 0.001],
            [1.0, 1.3]
        ]
    }

    param_values = sobol_sample.sample(problem, 512)
    total_sims = len(param_values)

    Y_peak = np.zeros(total_sims)
    Y_econ = np.zeros(total_sims)
    Y_rec = np.zeros(total_sims)
    Y_hosp = np.zeros(total_sims)
    Y_time = np.zeros(total_sims)
    Y_lock = np.zeros(total_sims)

    tasks = [(i, X, default_p.copy()) for i, X in enumerate(param_values)]
    cpu_count = multiprocessing.cpu_count()

    print(f"Calculating {total_sims} multi-dimensional simulations...")
    with concurrent.futures.ProcessPoolExecutor(max_workers=cpu_count) as executor:
        futures = {executor.submit(run_scenario_process, task): task for task in tasks}
        completed = 0
        for future in concurrent.futures.as_completed(futures):
            idx, p_I, t_E, t_R, p_Q, p_T, t_L = future.result()
            Y_peak[idx] = p_I
            Y_econ[idx] = t_E
            Y_rec[idx] = t_R
            Y_hosp[idx] = p_Q
            Y_time[idx] = p_T
            Y_lock[idx] = t_L
            completed += 1
            if completed % 100 == 0 or completed == total_sims:
                print(f"\r[Progress] {completed}/{total_sims} calculated...", end="", flush=True)

    print("\n\nGenerating graphs...")

    def clean_sobol(Si):
        Si['S1'] = np.clip(Si['S1'], 0.0, 1.0)
        Si['S2'] = np.clip(Si['S2'], 0.0, 1.0)
        Si['ST'] = np.clip(Si['ST'], 0.0, 1.0)
        return Si

    Si_peak = clean_sobol(sobol_analyze.analyze(problem, Y_peak, print_to_console=False))
    Si_econ = clean_sobol(sobol_analyze.analyze(problem, Y_econ, print_to_console=False))
    Si_rec = clean_sobol(sobol_analyze.analyze(problem, Y_rec, print_to_console=False))
    Si_hosp = clean_sobol(sobol_analyze.analyze(problem, Y_hosp, print_to_console=False))
    Si_time = clean_sobol(sobol_analyze.analyze(problem, Y_time, print_to_console=False))
    Si_lock = clean_sobol(sobol_analyze.analyze(problem, Y_lock, print_to_console=False))

    names = problem['names']
    x = np.arange(len(names))

    fig_bar, ax_bar = plt.subplots(figsize=(18, 8))
    width = 0.14

    ax_bar.bar(x - 2.5 * width, Si_peak['ST'], width, label='1. Peak Infected (Red)', color='crimson')
    ax_bar.bar(x - 1.5 * width, Si_hosp['ST'], width, label='2. Quarantine Saturation (Purple)', color='rebeccapurple')
    ax_bar.bar(x - 0.5 * width, Si_econ['ST'], width, label='3. Economic Cost (Yellow)', color='goldenrod')
    ax_bar.bar(x + 0.5 * width, Si_rec['ST'], width, label='4. Immunity Outcome (Green)', color='teal')
    ax_bar.bar(x + 1.5 * width, Si_time['ST'], width, label='5. Peak Date (Blue)', color='royalblue')
    ax_bar.bar(x + 2.5 * width, Si_lock['ST'], width, label='6. Lockdown Volume (Gray)', color='dimgray')

    ax_bar.set_ylabel("Total Importance Index (ST)", fontweight='bold')
    ax_bar.set_title("Each parameter controls a different dimension of the epidemic", fontsize=16, fontweight='bold')
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(names, rotation=45, ha='right', fontweight='bold', fontsize=11)
    ax_bar.legend(fontsize=11, loc='upper left', bbox_to_anchor=(1, 1))
    ax_bar.grid(axis='y', alpha=0.3)
    fig_bar.tight_layout()

    fig_heat, axes = plt.subplots(2, 3, figsize=(24, 14))
    axes = axes.flatten()

    cibles = [
        ("1. PEAK INFECTED (Height)", Si_peak, 'Reds'),
        ("2. QUARANTINE SATURATION", Si_hosp, 'Purples'),
        ("3. ECONOMIC COST", Si_econ, 'YlOrBr'),
        ("4. IMMUNITY OUTCOME", Si_rec, 'PuBuGn'),
        ("5. PEAK DATE (Wave Speed)", Si_time, 'Blues'),
        ("6. LOCKDOWN VOLUME", Si_lock, 'Greys')
    ]

    for ax_h, (titre, Si, color_map) in zip(axes, cibles):
        s2_matrix = Si['S2'].copy()
        for i in range(len(names)):
            s2_matrix[i, i] = Si['S1'][i]

        mask = np.tril(np.ones_like(s2_matrix, dtype=bool), k=-1)

        sns.heatmap(s2_matrix, mask=mask, annot=True, cmap=color_map,
                    xticklabels=names, yticklabels=names, fmt=".2f", ax=ax_h,
                    cbar_kws={"shrink": .8}, vmin=0, vmax=np.nanmax(s2_matrix))
        ax_h.set_title(titre, fontweight='bold', fontsize=12)
        ax_h.tick_params(axis='x', rotation=45)

    fig_heat.suptitle("Complete Analysis: Interactions (S2) and Direct Effects (S1)", fontsize=18, fontweight='bold')
    fig_heat.tight_layout()

    plt.show()


if __name__ == '__main__':
    setup_parameters_gui()

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10))
    plt.subplots_adjust(left=0.32, bottom=0.05, hspace=0.4, top=0.95)

    ax_desc = plt.axes([0.35, 0.01, 0.6, 0.03], facecolor='#f0f0f0')
    ax_desc.set_xticks([])
    ax_desc.set_yticks([])
    desc_text = ax_desc.text(0.5, 0.5, ".", ha='center', va='center', fontsize=10)


    def create_slider(name, label, y_pos, valmin, valmax, valinit, fmt='%1.2f'):
        ax = plt.axes([0.08, y_pos, 0.16, 0.015])
        return Slider(ax, label, valmin, valmax, valinit=valinit, valfmt=fmt)


    s_widgets = {}
    s_widgets['s_beta'] = create_slider('s_beta', 'Beta 0', 0.92, 0.1, 0.8, default_p['beta_0'])
    s_widgets['s_sigma'] = create_slider('s_sigma', 'Incub. (Rate)', 0.89, 0.05, 0.5, default_p['sigma'])
    s_widgets['s_gamma'] = create_slider('s_gamma', 'Recov. (Rate)', 0.86, 0.05, 0.5, default_p['gamma'])
    s_widgets['s_omega'] = create_slider('s_omega', 'Immunity Loss', 0.83, 0.0, 0.02, default_p['omega'], '%1.4f')
    s_widgets['s_mut'] = create_slider('s_mut', 'Mutation', 0.80, 0.0, 0.001, default_p['mutation_rate'], '%1.4f')
    s_widgets['s_fear'] = create_slider('s_fear', 'Fear K', 0.74, 0, 50, default_p['fear_k'], '%1.1f')
    s_widgets['s_thresh'] = create_slider('s_thresh', 'L Threshold(%)', 0.71, 0.1, 10.0,
                                          default_p['lockdown_threshold_pct'], '%1.1f %%')
    s_widgets['s_alpha'] = create_slider('s_alpha', 'Lockdown Force', 0.68, 0.1, 3.0, default_p['alpha_0'])
    s_widgets['s_qcap'] = create_slider('s_qcap', 'Q Cap.(%)', 0.65, 0.1, 5.0, default_p['Q_capacity_pct'], '%1.1f %%')
    s_widgets['s_fatigue'] = create_slider('s_fatigue', 'Fatigue', 0.62, 0.0, 0.1, default_p['fatigue_rate'], '%1.3f')
    s_widgets['s_cycles'] = create_slider('s_cycles', 'L Cycles', 0.54, 1, 10, default_p['legacy_cycles'], '%0.f')
    s_widgets['s_dur'] = create_slider('s_dur', 'L Duration', 0.51, 5, 120, default_p['legacy_dur'], '%0.f')
    s_widgets['s_pause'] = create_slider('s_pause', 'L Pause', 0.48, 5, 120, default_p['legacy_pause'], '%0.f')

    rax = plt.axes([0.08, 0.38, 0.16, 0.08], facecolor='#f7f7f7')
    check = CheckButtons(rax, ('Auto-Lockdown', 'Dyn. Fatigue'), (True, True))


    def run_sim(event=None):
        p = default_p.copy()
        p['beta_0'] = s_widgets['s_beta'].val
        p['sigma'] = s_widgets['s_sigma'].val
        p['gamma'] = s_widgets['s_gamma'].val
        p['omega'] = s_widgets['s_omega'].val
        p['mutation_rate'] = s_widgets['s_mut'].val
        p['fear_k'] = s_widgets['s_fear'].val
        p['lockdown_threshold_pct'] = s_widgets['s_thresh'].val
        p['alpha_0'] = s_widgets['s_alpha'].val
        p['Q_capacity_pct'] = s_widgets['s_qcap'].val
        p['fatigue_rate'] = s_widgets['s_fatigue'].val
        p['legacy_cycles'] = s_widgets['s_cycles'].val
        p['legacy_dur'] = s_widgets['s_dur'].val
        p['legacy_pause'] = s_widgets['s_pause'].val

        use_L, use_F = check.get_status()
        t_space = np.linspace(0, p['T_max'], int(p['T_max'] / p['dt']))
        t_plot = mdates.date2num([START_DATE + dt.timedelta(days=float(x)) for x in t_space])

        I_start = 50
        E_start = 200
        S_start = p['N'] - I_start - E_start
        y0 = (S_start, E_start, I_start, 0, 0, 0, 0, 0)

        res = odeint(deriv, y0, t_space, args=(p, use_L, use_F))
        S, E, I, Q, R, L, F, Econ = res.T

        variant_boost_vec = np.ones_like(t_space)
        variant_boost_vec[t_space > p['var1_t']] = p['var1_boost']
        variant_boost_vec[t_space > p['var2_t']] = p['var2_boost']
        variant_boost_vec[t_space > p['var3_t']] = p['var3_boost']

        testing_drop_vec = np.where(t_space > 900, 1.0 - 0.8 * (1 / (1 + np.exp(-0.02 * (t_space - 1000)))), 1.0)
        variant_boost_vec = variant_boost_vec * testing_drop_vec

        seasonality_vec = 1 + 0.35 * np.cos(2 * np.pi * t_space / 365)
        beta_cal_vec = p['beta_0'] * variant_boost_vec * seasonality_vec

        fear_fatigue_vec = np.exp(-0.002 * t_space)
        current_fear_vec = p['fear_k'] * fear_fatigue_vec
        beta_final_vec = beta_cal_vec * np.exp(-current_fear_vec * (I / p['N']))

        cap_abs = p['N'] * (p['Q_capacity_pct'] / 100.0)
        ramp_arg = np.clip(-0.2 * (t_space - 30), -100, 100)
        system_rampup = 1 / (1 + np.exp(ramp_arg))

        fill_ratio = Q / (cap_abs + 1e-9)
        sat_factor = 1 / (1 + np.exp(np.clip(5.0 * (fill_ratio - 1), -100, 100)))
        delta_vec = p['delta_max'] * system_rampup * np.maximum(0.05, sat_factor)

        vax_hosp_prot_vec = np.ones_like(t_space)
        vax_hosp_prot_vec[t_space > p['evasion_t']] = 0.5

        effective_delta_vec = delta_vec * vax_hosp_prot_vec
        susceptible_pool = (S + 0.3 * L) / p['N']
        removal_rate = p['gamma'] + effective_delta_vec

        Rt_vec = (beta_final_vec * susceptible_pool) / removal_rate

        ax1.clear()
        ax2.clear()
        ax3.clear()
        if hasattr(fig, 'axes_twin_eco'): fig.axes_twin_eco.remove()

        ax1.plot(t_plot, S, color='dodgerblue', alpha=0.8, label='Susceptibles (S)')
        ax1.plot(t_plot, R, color='forestgreen', alpha=0.8, label='Recovered (R)')
        ax1.plot(t_plot, L, 'k--', alpha=0.7, lw=1.5, label='Confined (L)')
        ax1.set_title(f"Global Populations (Total: {p['N']})")
        ax1.legend(loc='upper right', fontsize='small')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylabel("Individuals")

        ax2.plot(t_plot, I, 'r', lw=2, label='Infected (I)')
        ax2.plot(t_plot, E, color='orange', linestyle='--', alpha=0.8, label='Exposed (E)')
        ax2.fill_between(t_plot, 0, Q, color='purple', alpha=0.3, label='Quarantine (Q)')

        if use_L:
            thresh_line = p['N'] * (p['lockdown_threshold_pct'] / 100.0)
            ax2.axhline(thresh_line, color='red', linestyle=':', alpha=0.5, label='Lockdown Threshold')

        ax2.set_title("Active infectious dynamics")
        ax2.legend(loc='upper right', fontsize='small')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylabel("Active Cases")

        ax3.plot(t_plot, Rt_vec, color='darkcyan', lw=1.5, label='Rt')
        ax3.axhline(1.0, color='black', linestyle='--')
        ax3.set_ylabel("Rt (Reprod. rate)", color='darkcyan', fontweight='bold')
        ax3.tick_params(axis='y', labelcolor='darkcyan')
        ax3.set_ylim(0, max(4.0, np.max(Rt_vec) * 1.1))

        ax3_eco = ax3.twinx()
        fig.axes_twin_eco = ax3_eco
        ax3_eco.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False, labeltop=False)
        ax3_eco.xaxis.set_visible(False)
        ax3_eco.plot(t_plot, Econ, 'brown', lw=2, label='Cost')
        ax3_eco.set_ylabel("Economic cost", color='brown')
        ax3_eco.tick_params(axis='y', labelcolor='brown')

        ax3.set_title("Key indicators")
        ax3.grid(True, alpha=0.3)

        locator = mdates.AutoDateLocator()
        formatter = mdates.DateFormatter('%b %Y')
        for ax in (ax1, ax2, ax3):
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(formatter)
        ax3_eco.xaxis.set_major_locator(locator)
        ax3_eco.xaxis.set_major_formatter(formatter)
        plt.draw()


    def on_hover(event):
        for key, widget in s_widgets.items():
            if event.inaxes == widget.ax:
                desc_text.set_text(tooltips.get(key, ""))
                fig.canvas.draw_idle()
                return
        desc_text.set_text("Hover the mouse")
        fig.canvas.draw_idle()


    fig.canvas.mpl_connect('motion_notify_event', on_hover)

    for s in s_widgets.values():
        s.on_changed(run_sim)
    check.on_clicked(run_sim)

    run_sim(None)

    plt.show()

    run_mathematical_sensitivity()
