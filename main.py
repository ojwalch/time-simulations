from enum import Enum
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
import matplotlib.font_manager as font_manager
from matplotlib.ticker import FuncFormatter

from matplotlib import rcParams

rcParams['font.family'] = 'Arial'
rcParams['font.size'] = 12
rcParams['axes.titlesize'] = 14
rcParams['axes.labelsize'] = 14


class TimingStrategy(Enum):
    MORNING = "Morning"
    EVENING = "Evening"


am_color = [204/255, 85/255, 0]
pm_color = "#000080"

morning_efficacy = 0.8
evening_efficacy = 1.0
morning_group = 9849
evening_group = 9537

morning_timing_nonadherence_rate = 0.225
evening_timing_nonadherence_rate = 0.39

adherence_related_efficacy_loss = 0.07


def run_trial(subject_count, strategy, noise_sigma=0):
    all_efficacies = []

    for subject in range(subject_count):
        if strategy == TimingStrategy.MORNING:
            efficacy_for_person = morning_efficacy + \
                np.random.normal(0, noise_sigma, 1)

        if strategy == TimingStrategy.EVENING:
            efficacy_for_person = evening_efficacy + \
                np.random.normal(0, noise_sigma, 1)

        all_efficacies.append(efficacy_for_person)

    return np.array(all_efficacies)


def run_trial_with_nonadherence_to_time(subject_count, strategy, noise_sigma=0):
    all_efficacies = []

    # Assume they switch at some random point in the first quarter of the trial
    change_point = 0.25 * np.random.rand()

    for subject in range(subject_count):
        if strategy == TimingStrategy.MORNING:
            if np.random.rand() < morning_timing_nonadherence_rate:
                efficacy_for_person = morning_efficacy * change_point + evening_efficacy * \
                    (1 - change_point) + np.random.normal(0, noise_sigma, 1)
                # efficacy_for_person = evening_efficacy + np.random.normal(0, noise_sigma, 1)  # Assumes instant changeover to new time
            else:
                efficacy_for_person = morning_efficacy + \
                    np.random.normal(0, noise_sigma, 1)

        if strategy == TimingStrategy.EVENING:
            if np.random.rand() < evening_timing_nonadherence_rate:
                efficacy_for_person = evening_efficacy * change_point + morning_efficacy * \
                    (1 - change_point) + np.random.normal(0, noise_sigma, 1)
                # efficacy_for_person = morning_efficacy + np.random.normal(0, noise_sigma, 1) # Assumes instant changeover to new time
            else:
                efficacy_for_person = evening_efficacy + \
                    np.random.normal(0, noise_sigma, 1)

        all_efficacies.append(efficacy_for_person)

    return np.array(all_efficacies)


def run_trial_with_nonadherence_to_time_and_drug(subject_count, strategy, noise_sigma=0):
    all_efficacies = []

    # Assume they switch at some random point in the first quarter of the trial
    change_point = 0.25 * np.random.rand()

    for subject in range(subject_count):
        if strategy == TimingStrategy.MORNING:
            if np.random.rand() < morning_timing_nonadherence_rate:
                efficacy_for_person = morning_efficacy * change_point + evening_efficacy * \
                    (1 - change_point) + np.random.normal(0, noise_sigma, 1)
                # efficacy_for_person = evening_efficacy + np.random.normal(0, noise_sigma, 1)  # Instant changeover to different dosing time

                # Reduce efficacy for nonadherence since they are dosing in the evening:
                efficacy_for_person = efficacy_for_person - adherence_related_efficacy_loss

            else:
                efficacy_for_person = morning_efficacy + \
                    np.random.normal(0, noise_sigma, 1)

        if strategy == TimingStrategy.EVENING:
            if np.random.rand() < evening_timing_nonadherence_rate:
                efficacy_for_person = evening_efficacy * change_point + morning_efficacy * \
                    (1 - change_point) + np.random.normal(0, noise_sigma, 1)
                # efficacy_for_person = morning_efficacy + np.random.normal(0, noise_sigma, 1)  # Instant changeover to different dosing time
            else:
                efficacy_for_person = evening_efficacy + \
                    np.random.normal(0, noise_sigma, 1)

                # Reduce efficacy for nonadherence since they are dosing in the evening:
                efficacy_for_person = efficacy_for_person - adherence_related_efficacy_loss

        all_efficacies.append(efficacy_for_person)

    return np.array(all_efficacies)


def simulate_perfect_adherence():
    morning_doser_efficacies = run_trial(morning_group, TimingStrategy.MORNING)
    evening_doser_efficacies = run_trial(evening_group, TimingStrategy.EVENING)

    title = "Perfect adherence"
    fig, ax = plt.subplots()
    plot_bars(ax, morning_doser_efficacies,
              evening_doser_efficacies, title=title)
    plt.tight_layout()
    plt.savefig(f"output/{title}.png", dpi=300)

    print(np.mean(morning_doser_efficacies))
    print(np.mean(evening_doser_efficacies))


def simulate_nonadherence_to_time():
    morning_doser_efficacies = run_trial_with_nonadherence_to_time(
        morning_group, TimingStrategy.MORNING)
    evening_doser_efficacies = run_trial_with_nonadherence_to_time(
        evening_group, TimingStrategy.EVENING)

    title = "Nonadherent to time"
    fig, ax = plt.subplots()
    plot_bars(ax, morning_doser_efficacies,
              evening_doser_efficacies, title=title)
    plt.tight_layout()
    plt.savefig(f"output/{title}.png", dpi=300)

    print(np.mean(morning_doser_efficacies))
    print(np.mean(evening_doser_efficacies))


def simulate_nonadherence_to_time_and_drug():
    morning_doser_efficacies = run_trial_with_nonadherence_to_time_and_drug(
        morning_group, TimingStrategy.MORNING)
    evening_doser_efficacies = run_trial_with_nonadherence_to_time_and_drug(
        evening_group, TimingStrategy.EVENING)

    title = "Nonadherent to time and drug"
    fig, ax = plt.subplots()
    plot_bars(ax, morning_doser_efficacies,
              evening_doser_efficacies, title=title)
    plt.tight_layout()
    plt.savefig(f"output/{title}.png", dpi=300)

    print(np.mean(morning_doser_efficacies))
    print(np.mean(evening_doser_efficacies))


def plot_bars(ax, morning_doser_efficacies, evening_doser_efficacies, title="", draw_xlabel=True):
    # Calculating means for morning and evening doser efficacies
    morning_mean = np.mean(morning_doser_efficacies)
    evening_mean = np.mean(evening_doser_efficacies)

    _ = ax.bar([0], [morning_mean], color=am_color, label='AM')
    _ = ax.bar([1], [evening_mean], color=pm_color, label='PM')

    ax.set_ylim([0, 1.05])
    font_size = 22
    title_font_size = 28
    label_font_size = 24
    tick_font_size = 20

    # Adding text on top of the bars with a slight offset above the bar for visibility
    ax.text(0, morning_mean / 2, f'{morning_mean:.2f}',
            ha='center', va='bottom', color='white', fontsize=font_size)
    ax.text(1, evening_mean / 2, f'{evening_mean:.2f}',
            ha='center', va='bottom', color='white', fontsize=font_size)

    # Setting labels and styles
    ax.set_ylabel('Effectiveness', fontsize=tick_font_size, fontname='Arial')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['AM', 'PM'], fontname='Arial',
                       fontsize=label_font_size)
    if draw_xlabel:
        ax.set_xlabel('Assigned group',
                      fontsize=tick_font_size, fontname='Arial')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', which='major', labelsize=label_font_size)
    ax.set_title(title, fontname='Arial', fontsize=title_font_size)


def draw_prior_dosing(ax, am_pct, draw_legend=False):
    trial_start = [0.7, 0.4, 0.9]
    drug_start = [0.6, 0.8, 0.3]
    x_start = 0.4
    ax.add_patch(patches.Rectangle((x_start, 0.6), 1, 0.2, facecolor=am_color))
    ax.text(0, 0.67, "Avg. AM History", fontname='Arial', fontsize=14)
    ax.text(0, 0.335, "Avg. PM History", fontname='Arial', fontsize=14)

    ax.add_patch(patches.Rectangle((x_start, 0.3),
                 am_pct, 0.2, facecolor=am_color))
    ax.add_patch(patches.Rectangle((x_start + am_pct, 0.3),
                 1 - am_pct, 0.2, facecolor=pm_color))

    ax.plot([x_start], [0.1], 'o', color=drug_start,
            markersize=10, label='Drug start')  # drug start
    ax.plot([x_start + am_pct], [0.2], 'o', color=trial_start,
            markersize=10, label='Trial start')  # study start
    ax.set_xlim(0, 1 + x_start)
    ax.set_ylim(0, 1)
    font_props = font_manager.FontProperties(family='Arial', size=12)

    if draw_legend:
        ax.legend(bbox_to_anchor=(1.0, -0.1), prop=font_props)

    ax.axis('off')


def make_prior_dosing_history_figure():
    fig = plt.figure(figsize=(13, 8))
    gs = gridspec.GridSpec(3, 2, width_ratios=[1.5, 1], figure=fig)

    am_pcts = [0.0, 0.5, 0.75]
    evening_efficacies_weighted = [
        am_pct * morning_efficacy + (1 - am_pct) * evening_efficacy for am_pct in am_pcts]

    # Drawing and plotting in the grids
    for i, am_pct in enumerate(am_pcts):
        ax1 = fig.add_subplot(gs[i, 0])
        draw_prior_dosing(ax1, am_pct, draw_legend=i == 2)
        ax2 = fig.add_subplot(gs[i, 1])
        plot_bars(ax2, morning_efficacy,
                  evening_efficacies_weighted[i], draw_xlabel=i == 2)

    # Adjust layout and save
    plt.tight_layout(pad=3.0)
    plt.savefig("output/Prior dosing history.png", dpi=300)
    plt.close()


def make_half_life_figure():
    solve_system(want_individual_plots=True)
    solve_system(want_individual_plots=False)


def rk4_step(func, y, t, dt, *args):
    y = np.array(y)
    k1 = dt * func(y, t, *args)
    k2 = dt * func(y + 0.5 * k1, t + 0.5 * dt, *args)
    k3 = dt * func(y + 0.5 * k2, t + 0.5 * dt, *args)
    k4 = dt * func(y + k3, t + dt, *args)
    return y + (k1 + 2 * k2 + 2 * k3 + k4) / 6


def system(y, t, k1, k2, k3):
    # System of equations for the DNA damage and repaid
    drug, target, interaction = y
    dDrug = -k1 * drug  # decay rate of Z after the pulse
    dTarget = np.sin(t * (2 * np.pi) / 24)  # Assumed oscillatory target
    dInteraction = drug * target * k2 + -interaction * k3

    return np.array([dDrug, dTarget, dInteraction])


def solve_system(step_size=1, want_individual_plots=True):
    fig, ax = plt.subplots()

    for half_life in [1.5, 12]:

        print(f"Running half life: {half_life}")
        k1 = np.log(2) / half_life  # Clearance rate of drug
        k2 = 0.3
        k3 = 0.01
        start_time = 0  # Start at t = 0
        end_time = 172  # Hours
        dt = 0.01  # Step size in hours

        time_points = np.arange(start_time, end_time, dt)

        # Initial conditions
        drug_baseline = 0  # At the start, there is no drug

        # At the start there is a nonzero (but arbitrary) amount of target
        target_baseline = 1
        interaction_baseline = 0  # At the start there is no interaction

        pulse_times = np.arange(1, 26, step_size)
        solution = np.zeros((len(time_points), 3))
        solution[0] = [drug_baseline, target_baseline, interaction_baseline]

        efficacy_profile_for_dose = []
        for pulse_time in pulse_times:
            for i in range(1, len(time_points)):
                last_state = np.copy(solution[i - 1])

                # Add instantaneous delta of drug at pulse time
                if np.abs(time_points[i - 1] - pulse_time) < dt / 2:
                    last_state[0] = 1.0  # Arbitrary choice of 1.0 for impulse

                solution[i] = rk4_step(
                    system, last_state, time_points[i - 1], dt, k1, k2, k3)

            if want_individual_plots:
                plot_single_pulse_time(
                    time_points, solution, pulse_time, half_life)
            efficacy_profile_for_dose.append(np.mean(solution[:, 2]))

        rescaled_efficacy = np.array(
            efficacy_profile_for_dose) / np.max(efficacy_profile_for_dose)
        ax.plot(rescaled_efficacy,
                label=f"Half life: {half_life} hrs", linewidth=1.5, color=[0, 0.8 * (1 - 1.0 * (half_life / 12)), 0.8 * (half_life / 12)])

        ax.set_xlabel('Wall Clock Time of Dose', fontsize=16)
        ax.set_ylabel('Relative efficacy', fontsize=16)
        ax.set_ylim([0, 1.08])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout()
        if want_individual_plots:
            plt.close()

    if not want_individual_plots:
        plt.legend(frameon=False)
        plt.savefig(
            "output/Effect of half-life on time of day efficacy.png", dpi=300)
        plt.show()


def plot_single_pulse_time(time_points, solution, pulse_time, half_life):
    drug_color = '#3B8EA5'
    target_color = '#AB3428'
    interaction_color = '#E6AF2E'

    fig, ax = plt.subplots(3, 1)
    ax[0].plot(time_points, solution[:, 0], label='Drug', color=drug_color)
    ax[1].plot(time_points, solution[:, 1], label='Target', color=target_color)
    ax[2].plot(time_points, solution[:, 2],
               label='Interaction', color=interaction_color)

    ax[0].set_xlabel('Time (hours)')
    ax[0].set_ylabel('Amount of \ndrug (a.u.)', fontsize=12)
    ax[0].spines['top'].set_visible(False)
    ax[0].spines['right'].set_visible(False)

    ax[1].set_xlabel('Time (hours)')
    ax[1].set_ylabel('Amount of \nTarget (a.u.)', fontsize=12)
    ax[1].spines['top'].set_visible(False)
    ax[1].spines['right'].set_visible(False)

    ax[2].set_xlabel('Time (hours)')
    ax[2].set_ylabel('Amount of \ninteraction (a.u.)', fontsize=12)
    ax[2].spines['top'].set_visible(False)
    ax[2].spines['right'].set_visible(False)

    # ax[0].set_title("Dosing " + str(pulse_time) +
    #                 " hrs after target minimum", fontsize=12)
    plt.savefig("output/" + str(half_life) + "_" +
                str(pulse_time).zfill(2) + ".png", dpi=300)
    plt.close()


def make_typical_dayworker_sampling_figure(amplitude, vertical_shift, phase_shift, title="",  save_name=""):
    mean = 21.22
    standard_deviation = 1.22
    plot_distribution_of_dlmos_and_efficacy(
        mean, standard_deviation, amplitude, vertical_shift, phase_shift, title)
    plt.savefig(f"output/Efficacy {title} {save_name}.png", dpi=300)


def make_more_disrupted_sampling_figure(amplitude, vertical_shift, phase_shift, title="", save_name=""):
    mean = 21.22
    standard_deviation = 8
    plot_distribution_of_dlmos_and_efficacy(
        mean, standard_deviation, amplitude, vertical_shift, phase_shift, title)
    plt.savefig(f"output/Efficacy {title} {save_name}.png", dpi=300)


def plot_distribution_of_dlmos_and_efficacy(mean, standard_deviation, amplitude, vertical_shift, phase_shift, title):
    color = [0.2, 0.6, 0.5]
    num_samples = 1000
    dlmos = np.random.normal(mean, standard_deviation, num_samples)
    fig, ax = plt.subplots()

    n, bins, patches = ax.hist(dlmos, color=color)

    ax.set_xlabel('DLMO (hours after midnight)', fontsize=26)
    ax.set_ylabel('Count', fontsize=18)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.tick_params(axis='both', which='major', labelsize=22)
    formatter = FuncFormatter(lambda x, pos: f"{int(x % 24)}")
    ax.xaxis.set_major_formatter(formatter)

    ax.set_xlim(6, 36)
    plt.tight_layout()
    plt.savefig(f"output/{title} DLMO Distribution.png", dpi=300)
    plt.close()

    dlmos_morning = np.random.normal(mean, standard_deviation, num_samples)
    offsets_morning = dlmos_morning - mean
    dlmos_evening = np.random.normal(mean, standard_deviation, num_samples)
    offsets_evening = dlmos_evening - mean

    morning_time = 8
    evening_time = 22
    efficacy_morning = efficacy_curve(
        morning_time + offsets_morning, amplitude, vertical_shift, phase_shift)
    efficacy_evening = efficacy_curve(
        evening_time + offsets_evening, amplitude, vertical_shift, phase_shift)

    print(f"Mean morning efficacy: {np.mean(efficacy_morning)}")
    print(f"Mean evening efficacy: {np.mean(efficacy_evening)}")

    fig, ax = plt.subplots()
    plot_bars(ax, efficacy_morning, efficacy_evening,
              title="", draw_xlabel=False)
    plt.tight_layout()


def efficacy_curve(x, amplitude, vertical_shift, phase_shift):
    period = 24

    return amplitude * np.sin(2 * np.pi / period * (x - phase_shift)) + vertical_shift


def plot_efficacy(ax, x, y, color=[0, 0, 0, 0.3], linewidth=2, want_dots=False, label=""):
    ax.plot(x, y, label='Effectiveness', color=color, linewidth=linewidth)

    if want_dots:
        dot_locations = [8, 22]
        dot_y_values = [np.interp(dot, x, y) for dot in dot_locations]
        ax.plot(dot_locations, dot_y_values,
                'o', color=color, markersize=10)

        ax.plot([0, 24], [dot_y_values[0], dot_y_values[0]], 'k--', linewidth=1)
        ax.plot([0, 24], [dot_y_values[1], dot_y_values[1]], 'k--', linewidth=1)

        percent_difference = (
            (dot_y_values[1] - dot_y_values[0]) / dot_y_values[0]) * 100

        txt_string = f"{int(np.round(percent_difference))}% observed\nAM/PM difference"
        annotation_font_size = 11
        x_loc = 28
        txt_string = f"{int(np.round(percent_difference))}%"
        annotation_font_size = 20

        bracket_x_loc = 25.1
        ax.text(
            bracket_x_loc, np.mean(dot_y_values), "}", fontsize=300 * (dot_y_values[1] - dot_y_values[0]), color=color, horizontalalignment='center', verticalalignment='center')

        x_loc = 27.5

        ax.text(
            x_loc, np.mean(dot_y_values), txt_string, fontsize=annotation_font_size, color=color, horizontalalignment='center', verticalalignment='center')

    ax.text(
        8, 1.1, "AM Dose\nWindow", fontsize=11, color=am_color, horizontalalignment='center', verticalalignment='center')
    ax.text(
        22, 1.1, "PM Dose\nWindow", fontsize=11, color=pm_color, horizontalalignment='center', verticalalignment='center')

    rect_min = patches.Rectangle(
        (6, -1.5), 4, 3, linewidth=1, edgecolor='none', facecolor=am_color, alpha=0.2)
    ax.add_patch(rect_min)

    rect_max = patches.Rectangle(
        (20, -1.5), 4, 3, linewidth=1, edgecolor='none', facecolor=pm_color, alpha=0.2)
    ax.add_patch(rect_max)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_xlim(0, 24)
    ax.set_ylim(0, 1.03)
    ax.set_xlabel('Wall clock time (hours after midnight)', fontsize=18)
    ax.set_ylabel('Effectiveness', fontsize=18)
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)


def show_chronotypes_on_different_schedules(base_amplitude,
                                            base_vertical_shift,
                                            base_phase_shift,
                                            is_early,
                                            noise_amount,
                                            file_name=""):

    fig, ax = plt.subplots()
    N = 20
    offset = 3 if is_early else -3
    save_name = "Early Chronotype" if is_early else "Late Chronotypes"
    color = am_color if is_early else pm_color
    for i in range(N):

        x = np.linspace(0, 24, 1000)
        y = efficacy_curve(x, amplitude=base_amplitude,
                           vertical_shift=base_vertical_shift,
                           phase_shift=base_phase_shift - np.random.normal(offset, noise_amount))
        if i < N-1:
            ax.plot(x, y, color=color, linewidth=1)
    plot_efficacy(ax, x, y, color=color, linewidth=1)
    plt.title(file_name + " " + save_name, fontsize=22)
    plt.tight_layout()
    plt.savefig("output/" + file_name + "_" + save_name + ".png", dpi=300)
    plt.close()


def plot_hypothetical_efficacies():
    evening_optimal_amplitude = 0.0857
    evening_optimal_vertical_shift = 0.9
    evening_optimal_phase_shift = 15.5
    evening_optimal_color = [.6, 0.35, 0.4, 1.0]  # Pink color
    fig, ax = plt.subplots()
    x = np.linspace(0, 24, 1000)
    y = efficacy_curve(x, amplitude=evening_optimal_amplitude,
                       vertical_shift=evening_optimal_vertical_shift,
                       phase_shift=evening_optimal_phase_shift)
    plot_efficacy(ax, x, y, color=evening_optimal_color,
                  want_dots=True, label="Evening optimal,")
    plt.tight_layout()
    plt.savefig("output/Evening optimal efficacy curve.png", dpi=300)

    midday_optimal_amplitude = 0.38
    midday_optimal_vertical_shift = 0.605
    midday_optimal_phase_shift = 24 - 14.517
    midday_optimal_color = [0.0, 0.5, 0.0, 1.0]  # Green color

    x = np.linspace(0, 24, 1000)
    y = efficacy_curve(x, amplitude=midday_optimal_amplitude,
                       vertical_shift=midday_optimal_vertical_shift,
                       phase_shift=midday_optimal_phase_shift)
    plot_efficacy(ax, x, y, color=midday_optimal_color,
                  want_dots=True,  label="Midday optimal,")
    plt.tight_layout()
    fig.set_size_inches(6, 6)
    plt.savefig("output/Two hypothetical efficacy curves.png", dpi=300)
    plt.close()

    make_typical_dayworker_sampling_figure(amplitude=evening_optimal_amplitude,
                                           vertical_shift=evening_optimal_vertical_shift,
                                           phase_shift=evening_optimal_phase_shift,
                                           title="Homogeneous", save_name="Evening Optimal")

    make_more_disrupted_sampling_figure(amplitude=evening_optimal_amplitude,
                                        vertical_shift=evening_optimal_vertical_shift,
                                        phase_shift=evening_optimal_phase_shift,
                                        title="Disrupted", save_name="Evening Optimal")

    make_typical_dayworker_sampling_figure(amplitude=midday_optimal_amplitude,
                                           vertical_shift=midday_optimal_vertical_shift,
                                           phase_shift=midday_optimal_phase_shift,
                                           title="Homogeneous", save_name="Midday Optimal")

    make_more_disrupted_sampling_figure(amplitude=midday_optimal_amplitude,
                                        vertical_shift=midday_optimal_vertical_shift,
                                        phase_shift=midday_optimal_phase_shift,
                                        title="Disrupted", save_name="Midday Optimal")

    make_bar_plots_for_different_strategies(evening_optimal_amplitude,
                                            evening_optimal_vertical_shift,
                                            evening_optimal_phase_shift,
                                            midday_optimal_amplitude,
                                            midday_optimal_vertical_shift,
                                            midday_optimal_phase_shift)


def make_bar_plots_for_different_strategies(evening_optimal_amplitude,
                                            evening_optimal_vertical_shift,
                                            evening_optimal_phase_shift,
                                            midday_optimal_amplitude,
                                            midday_optimal_vertical_shift,
                                            midday_optimal_phase_shift):
    # Sampling times for early and late types
    early_am_time = 8 + 2
    early_pm_time = 22 + 2
    late_am_time = 8 - 2
    late_pm_time = 22 - 2

    # Calculate efficacy for evening optimal curve
    early_am_efficacy_evening = efficacy_curve(
        early_am_time, evening_optimal_amplitude, evening_optimal_vertical_shift, evening_optimal_phase_shift)
    early_pm_efficacy_evening = efficacy_curve(
        early_pm_time, evening_optimal_amplitude, evening_optimal_vertical_shift, evening_optimal_phase_shift)
    late_am_efficacy_evening = efficacy_curve(
        late_am_time, evening_optimal_amplitude, evening_optimal_vertical_shift, evening_optimal_phase_shift)
    late_pm_efficacy_evening = efficacy_curve(
        late_pm_time, evening_optimal_amplitude, evening_optimal_vertical_shift, evening_optimal_phase_shift)

    # Calculate efficacy for midday optimal curve
    early_am_efficacy_midday = efficacy_curve(
        early_am_time, midday_optimal_amplitude, midday_optimal_vertical_shift, midday_optimal_phase_shift)
    early_pm_efficacy_midday = efficacy_curve(
        early_pm_time, midday_optimal_amplitude, midday_optimal_vertical_shift, midday_optimal_phase_shift)
    late_am_efficacy_midday = efficacy_curve(
        late_am_time, midday_optimal_amplitude, midday_optimal_vertical_shift, midday_optimal_phase_shift)
    late_pm_efficacy_midday = efficacy_curve(
        late_pm_time, midday_optimal_amplitude, midday_optimal_vertical_shift, midday_optimal_phase_shift)

    # Plot bar chart for evening optimal curve
    fig, ax = plt.subplots()
    ax.bar([0, 2], [early_am_efficacy_evening, early_pm_efficacy_evening],
           color=am_color, label="AM Dosing")
    ax.bar([1, 3], [late_am_efficacy_evening, late_pm_efficacy_evening],
           color=pm_color, label="PM Dosing")
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(
        ["Early\nChronotype", "Early\nChronotype", "Late\nChronotype", "Late\nChronotype"], fontsize=16)
    ax.set_ylabel("Effectiveness", fontsize=20)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    ax.set_yticklabels([0, 0.2, 0.4, 0.6, 0.8, 1], fontsize=24)
    # ax.set_title("Evening Optimal Curve")
    # ax.legend(frameon=False, fontsize=20)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig("output/Evening_Optimal_Bar_Plot.png", dpi=300)
    plt.close()

    # Plot bar chart for midday optimal curve
    fig, ax = plt.subplots()
    ax.bar([0, 2], [early_am_efficacy_midday, early_pm_efficacy_midday],
           color=am_color, label="AM Dosing")
    ax.bar([1, 3], [late_am_efficacy_midday, late_pm_efficacy_midday],
           color=pm_color, label="PM Dosing")
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(
        ["Early\nChronotype", "Early\nChronotype", "Late\nChronotype", "Late\nChronotype"], fontsize=16)
    ax.set_ylabel("Effectiveness", fontsize=20)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    ax.set_yticklabels([0, 0.2, 0.4, 0.6, 0.8, 1], fontsize=24)

    # ax.set_title("Midday Optimal Curve")
    ax.legend(frameon=False, fontsize=20)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig("output/Midday_Optimal_Bar_Plot.png", dpi=300)
    plt.close()


if __name__ == '__main__':
    plot_hypothetical_efficacies()
    # simulate_perfect_adherence()
    # simulate_nonadherence_to_time()
    # simulate_nonadherence_to_time_and_drug()
    # make_prior_dosing_history_figure()
    # make_half_life_figure()
