import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import pygal
from collections import Counter

import stats
from stats import StatsGetter
from readInData import get_data_for_stats

def plot_last_trials(SG, BINS = 'auto'):
    ##########

    sorted_last_trials = sorted(SG.last_trials, key=SG.make_phaseround_comparable)
    sorted_last_trials.remove('')
    n, bins, patches = plt.hist(x=sorted_last_trials, bins=BINS, color='#0504aa',
                                alpha=0.7, rwidth=0.85)
    plt.xlabel('Trial')
    plt.ylabel('Frequency')
    plt.title('The trials in which people stopped playing')
    maxfreq = n.max()
    plt.tick_params(axis='x', rotation=45)
    plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
    plt.show()

    ###########

    phase_only = [re.sub('\d', '', x) for x in sorted_last_trials]
    n, bins, patches = plt.hist(x=phase_only, bins=BINS, color='#0504aa',
                                alpha=0.7, rwidth=0.85)
    plt.xlabel('Phase')
    plt.ylabel('Frequency')
    plt.title('The phase in which people stopped playing')
    maxfreq = n.max()
    plt.tick_params(axis='x')
    plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
    plt.show()

    # barbins = sorted(set(sorted_last_trials), key=SG.make_phaseround_comparable)

    # depression_groups = {'severe' : sorted(SG.get_last_trials(depression_group='severe'), key=SG.make_phaseround_comparable),
    #                      "moderated" : sorted(SG.get_last_trials(depression_group='moderately severe'), key=SG.make_phaseround_comparable),
    #                      "minimal" : sorted(SG.get_last_trials(depression_group='moderated'), key=SG.make_phaseround_comparable),
    #                      "no depression": sorted(SG.get_last_trials(depression_group='minimal'), key=SG.make_phaseround_comparable)}
    #
    # bar_chart = pygal.Bar()
    # for depression_group in depression_groups.keys():
    #     bar_chart.add(depression_group, depression_groups[depression_group])
    # bar_chart.render_to_png('last_phase_by_depression.png')

# filter by nan removes 6 people 100 --> 94
# filter by >70 responses wrong reduces 9 more 94 --> 85
def num_correct_df(SG, phase = None):

    categories = ["gender","gender",
                  "age","age","age","age","age","age","age",
                  "education","education","education","education","education"]

    values = ["F", "M",
              "<18", "18-24", "25-34", "35-54", "55-64", "65-74", "75+",
              "no high school", "high school", "BSc", "MSc", "PhD"]
    smooth = True

    mean, median, percent, num, variance = [], [], [], [], []
    clean_ids = SG._clean_ids
    for cat, val in zip(categories, values):
        current_ids = SG.filter_ids(IDS=clean_ids, what_element=cat, what_value=val)
        mean.append(SG.mean_correct_per_person(current_ids, phase,smooth))
        median.append(SG.median_correct_per_person(current_ids, phase, smooth))
        variance.append(np.var(SG.count_correct_all(current_ids, phase, smooth)))
        num_nan = len(SG.filter_ids(IDS = clean_ids, what_element=cat, what_value=np.nan))
        fraction = (len(current_ids)) / (len(clean_ids) - num_nan)
        percent.append(round(100*(fraction)))
        num.append(len(current_ids))

    dep_cat = ["depression level", "depression level", "depression level", "depression level"]
    dep_val = ["minimal", "moderated", "moderately severe", "severe"]

    for val in dep_val:
        current_ids = SG.filter_by_depression(IDS=clean_ids, group=val)
        mean.append(SG.mean_correct_per_person(current_ids, phase, smooth))
        median.append(SG.median_correct_per_person(current_ids, phase, smooth))
        variance.append(np.var(SG.count_correct_all(current_ids, phase, smooth)))
        fraction = (len(current_ids)) / len(clean_ids)
        percent.append(round(100 * (fraction)))
        num.append(len(current_ids))

    categories += dep_cat
    values += dep_val

    mean.append(SG.mean_correct_per_person(clean_ids, phase, smooth))
    median.append(SG.median_correct_per_person(clean_ids, phase, smooth))
    variance.append(np.var(SG.count_correct_all(clean_ids, phase, smooth)))
    categories.append("all")
    values.append("all")
    percent.append(100)
    num.append(len(clean_ids))

    ncdf = pd.DataFrame({"category": categories,
                         "value" : values,
                         "mean correct" : mean,
                         "median correct" : median,
                         "percent" : percent,
                         "number": num
                         })
    return ncdf

def num_wrong_df(SG, phase = None):

    categories = ["gender","gender",
                  "age","age","age","age","age","age","age",
                  "education","education","education","education","education"]

    values = ["F", "M",
              "<18", "18-24", "25-34", "35-54", "55-64", "65-74", "75+",
              "no high school", "high school", "BSc", "MSc", "PhD"]
    smooth = True

    mean, median, percent, num = [], [], [], []
    clean_ids = SG._clean_ids
    for cat, val in zip(categories, values):
        current_ids = SG.filter_ids(IDS=clean_ids, what_element=cat, what_value=val)
        mean.append(SG.mean_wrong_per_person(current_ids, phase,smooth))
        median.append(SG.median_wrong_per_person(current_ids, phase, smooth))
        num_nan = len(SG.filter_ids(IDS = clean_ids, what_element=cat, what_value=np.nan))
        fraction = (len(current_ids)) / (len(clean_ids) - num_nan)
        percent.append(round(100*(fraction)))
        num.append(len(current_ids))

    dep_cat = ["depression level", "depression level", "depression level", "depression level"]
    dep_val = ["minimal", "moderated", "moderately severe", "severe"]

    for val in dep_val:
        current_ids = SG.filter_by_depression(IDS=clean_ids, group=val)
        mean.append(SG.mean_wrong_per_person(current_ids, phase, smooth))
        median.append(SG.median_wrong_per_person(current_ids, phase, smooth))
        fraction = (len(current_ids)) / len(clean_ids)
        percent.append(round(100 * (fraction)))
        num.append(len(current_ids))

    categories += dep_cat
    values += dep_val

    mean.append(SG.mean_wrong_per_person(clean_ids, phase, smooth))
    median.append(SG.median_wrong_per_person(clean_ids, phase, smooth))
    categories.append("all")
    values.append("all")
    percent.append(100)
    num.append(len(clean_ids))

    ncdf = pd.DataFrame({"category": categories,
                         "value" : values,
                         "mean correct" : mean,
                         "median correct" : median,
                         "percent" : percent,
                         "number": num
                         })
    return ncdf

def print_correct_per_phase(SG):
    print("mean correct in")
    print("GameA :", SG.mean_correct_per_person(SG.ids(), "GameA"),
          "GameB :", SG.mean_correct_per_person(SG.ids(), "GameB"),
          "GameC :", SG.mean_correct_per_person(SG.ids(), "GameC"),)

    print("median correct in")
    print("GameA :", SG.median_correct_per_person(SG.ids(), "GameA"),
          "GameB :", SG.median_correct_per_person(SG.ids(), "GameB"),
          "GameC :", SG.median_correct_per_person(SG.ids(), "GameC"),)

def category_counts(SG):
    ids = SG.ids()
    male = [iidd for iidd in ids if SG._card_data[iidd][SG._card_data.phaseround == 'Q00'].values[0] == '0']
    female = [iidd for iidd in ids if SG._card_data[iidd][SG._card_data.phaseround == 'Q00'].values[0] == '1']

    gc = {"male" : len(male), "female" : len(female)}

    age0_u18 = [iidd for iidd in ids if SG._card_data[iidd][SG._card_data.phaseround == 'Q01'].values[0] == '0']
    age1_o18 = [iidd for iidd in ids if SG._card_data[iidd][SG._card_data.phaseround == 'Q01'].values[0] == '1']
    age2_o25 = [iidd for iidd in ids if SG._card_data[iidd][SG._card_data.phaseround == 'Q01'].values[0] == '2']
    age3_035 = [iidd for iidd in ids if SG._card_data[iidd][SG._card_data.phaseround == 'Q01'].values[0] == '3']
    age4_o55 = [iidd for iidd in ids if SG._card_data[iidd][SG._card_data.phaseround == 'Q01'].values[0] == '4']
    age5_o65 = [iidd for iidd in ids if SG._card_data[iidd][SG._card_data.phaseround == 'Q01'].values[0] == '5']

    agec = {"<18" : len(age0_u18),
          "18-24" : len(age1_o18),
          "25-34" : len(age2_o25),
          "35-54" : len(age3_035),
          "55-64" : len(age4_o55),
          "65-74" : len(age5_o65),
          }

    edu_noHS = [iidd for iidd in ids if SG._card_data[iidd][SG._card_data.phaseround == 'Q02'].values[0] == '0']
    edu_HS = [iidd for iidd in ids if SG._card_data[iidd][SG._card_data.phaseround == 'Q02'].values[0] == '1']
    edu_BSC = [iidd for iidd in ids if SG._card_data[iidd][SG._card_data.phaseround == 'Q02'].values[0] == '2']
    edu_MSC = [iidd for iidd in ids if SG._card_data[iidd][SG._card_data.phaseround == 'Q02'].values[0] == '3']
    edu_phd = [iidd for iidd in ids if SG._card_data[iidd][SG._card_data.phaseround == 'Q02'].values[0] == '4']

    educ = {"no high school" : len(edu_noHS),
          "high school" : len(edu_HS),
          "BSc" : len(edu_BSC),
          "MSc" : len(edu_MSC),
          "PhD" : len(edu_phd),
          }
    return gc, agec, educ

def plot_num_ofNaNs(SG):
    na_counts = SG.count_nas_all()
    plt.plot(sorted(na_counts))
    plt.title("number of missing values")
    plt.xlabel("number of people")
    plt.ylabel("number of missing values")
    plt.show()

def plot_num_correct(SG, IDS):
    # correct_counts = SG.count_correct_all(SG.ids())
    smooth_correct_counts = SG.count_correct_all(IDS,smooth=True)
    # plt.plot(sorted(correct_counts))
    plt.plot(sorted(smooth_correct_counts))
    plt.title("number of correct values")
    plt.xlabel("number of people")
    plt.ylabel("number of correct values")
    plt.show()

def plot_depression_scores(SG):
    depression_scores = [SG.score_depression(id) for id in SG.ids()]
    plt.plot(sorted(depression_scores)[::-1])
    plt.title("depression scores")
    plt.xlabel("number of people")
    plt.ylabel("depression score")
    plt.show()

def plot_age_pie(SG):
    from pygal.style import Style
    colors = ['#f5da42','#f59e42', '#f04371', '#bb74fc', '#7aabff', '#6afcb6']
    custom_style = Style(
        colors=colors)
    ids = SG.ids()
    pie_chart = pygal.Pieg(inner_radius=.4, half_pie=True, style=custom_style)
    pie_chart.title = 'Ages in our cohort'
    for age_group in ['<18', '18-24', '25-34', '35-54', '55-64', '65-74']:
        pie_chart.add(age_group, len(SG.filter_ids(ids, 'age', age_group)))
    pie_chart.render_to_file('agepie.svg')

def plot_edu_pie(SG):
    from pygal.style import Style
    colors = ['#f5da42','#f59e42', '#f04371', '#bb74fc', '#7aabff', '#6afcb6']
    custom_style = Style(colors=colors)
    ids = SG.ids()
    pie_chart = pygal.Pie(inner_radius=.4, half_pie=True, style=custom_style)
    pie_chart.title = 'Education levels in our cohort'
    for edu_group in ['no high school', 'high school', 'BSc', 'MSc', 'PhD']:
        nm = len(SG.filter_ids(ids, 'education', edu_group))
        print(nm)
        pie_chart.add(edu_group, nm)
    pie_chart.render_to_file('edupie.svg')

def success_by_depression_group(SG):
    # Overall
    # Counter(success_class) = {1: 86, 0: 14}
    success_class = [SG.classify_success_person(pid, threshold=90, smooth=True) for pid in SG._clean_ids]
    depression_class = [SG.score2depressiongroup(SG.score_depression(pid)) for pid in SG._clean_ids]

    success_by_depression = pd.DataFrame({"success":success_class, "depression":depression_class, "id": SG._clean_ids})

    min = Counter(success_by_depression['success'][success_by_depression.depression == 'minimal'])
    mod = Counter(success_by_depression['success'][success_by_depression.depression == 'moderated'])
    modsev = Counter(success_by_depression['success'][success_by_depression.depression == 'moderately severe'])
    sev= Counter(success_by_depression['success'][success_by_depression.depression == 'severe'])

    return

def scores_by_depression_group(SG):
    # Overall

    scores_correct = SG.count_correct_all(SG._clean_ids,smooth=True)
    scores_wrong = SG.count_wrong_all(SG._clean_ids,smooth=True)

    depression_class = [SG.score2depressiongroup(SG.score_depression(pid)) for pid in SG._clean_ids]

    score_by_depression = pd.DataFrame({
                                        "scoreCorrect":scores_correct,
                                        "scoreWrong":scores_wrong,
                                        "depression":depression_class,
                                        "id": SG._clean_ids
                                        })

    return score_by_depression


# FEEDBACK

# Liked both : 14, liked of of Q/game : 3, disliked both : 4

# Understood completely: 9, I feel I understood most of the rules but was uncertain at times : 10,
# "I understood the rules of the first game round and part of the second round but not the rest." : 1 ,
# I did not understand : 0

# how did you use the feedback

# 17 people:
# "I carefully analysed both the 'correct' and 'wrong' feedback to learn whether I had to match by color, shape or number.
# I was only somewhat confused by the tricky feedback you included in the latter rounds."
# 5 people: "The feedback was somewhat useful but I did not always trust it.",

# 14 people : "I realise this is not a competition or a diagnostic test but nonetheless felt motivated by 'correct' feedback, while 'wrong' feedback made me think harder which I enjoyed."
# 6 people : "I was happy when I got right answers but somewhat frustrated by wrong ones.",
# 1 person : "I felt bad everytime I got something wrong, as felt I was doing badly.",
# 1 person : didn't care


def plot_responses(SG):
    # 1 Little interest or pleasure
    responses_Q1 = [SG._data[SG._data.id == pid].response[np.logical_and(SG._data.phase == 'Q', SG._data.trial == 3)].values[0]
                    for pid in SG._clean_ids]

    #SG._data.response[np.logical_and(SG._data.phase == 'Q', SG._data.trial == 3)].values
    # 2 Feeling down, depressed, or hopeless
    responses_Q2 = SG._data.response[np.logical_and(SG._data.phase == 'Q', SG._data.trial == 4)].values
    # 3 sleep
    responses_Q3 = SG._data.response[np.logical_and(SG._data.phase == 'Q', SG._data.trial == 5)].values
    # 4 Feeling tired or having little energy?
    responses_Q4 = SG._data.response[np.logical_and(SG._data.phase == 'Q', SG._data.trial == 6)].values
    # 5 Poor appetite or overeating?
    responses_Q5 = SG._data.response[np.logical_and(SG._data.phase == 'Q', SG._data.trial == 7)].values
    # Feeling bad about yourself - or that you are a failure
    responses_Q6 = SG._data.response[np.logical_and(SG._data.phase == 'Q', SG._data.trial == 8)].values
    # Trouble concentrating
    responses_Q7 = SG._data.response[np.logical_and(SG._data.phase == 'Q', SG._data.trial == 9)].values
    # Moving or speaking so slowly that other people could have noticed? Or the opposite - being so fidgety or restless that you have been moving around a lot more than usual?
    responses_Q8 = SG._data.response[np.logical_and(SG._data.phase == 'Q', SG._data.trial == 10)].values
    # Thoughts that you would be better off dead,
    responses_Q9 = SG._data.response[np.logical_and(SG._data.phase == 'Q', SG._data.trial == 11)].values

if __name__ == '__main__':
    filename = 'gamestats26_05.csv'
    data, col_data, correct_data, combined_data = get_data_for_stats(filename)
    SG = StatsGetter(data, col_data, correct_data,combined_data, filename)
    #plot_last_trials(SG)
    nc_df = num_correct_df(SG)
    nc_df.to_csv("num_correct_per_category.csv")
    gc, agec, educ = category_counts(SG)
