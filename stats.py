import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from collections import Counter

from readInData import get_data_for_stats

filename = 'gamestats26_05.csv'

# len(A[A.phase == 'Q'])      = 12
# len(A[A.phase == 'GameA'])  = 47
# len(A[A.phase == 'GameB'])  = 75
# len(A[A.phase == 'GameC'])  = 42

# missing questionnaire but, depression - 46161
# missing questionnaire but seem minimal anyway : 59414 (?), 22808 (?), 28313, 5478
# only education missing - 73461
# only demographics missing - 76220
# non salvagable - 46539, 62008, 71134, 54630
# other - 30801

# Want to Check :

# 1) Distribution of num of correct trials - overall, per phase, per subgroup
# 2) Distribution of ages, genders, academic background
# 3) correlation of age and success rate
# 4) Remove people with too much wrong
# 5) distribution of wrong
# 6) most common response with wrong

class StatsGetter:

    def __init__(self, data, card_data, correct_data, combined_col_data, filename='gamestats18_05.csv'):
        self._filename = filename
        self._data = data
        self.set_last_trials()
        self.total_participants_num = len(set(data.id))

        self._card_data = card_data
        self._correct_df = correct_data
        self._combined_df = combined_col_data
        self._ids = get_ids(card_data)#list(set(self.data.id))
        self._phase_change_inds = [0, 12, 59, 134, 176]
        self._phases = ['Q', 'GameA', 'GameB', 'GameC']

        self.other_ids_to_remove = ['74978', # chose by no rule 6 times, 61 mistakes
                                    '72093', # chose by no rule 8 times, 68 mistakes
                                    '71149', # chose by no rule 25 times, 85 mistakes
                                    '25580', # chose by no rule 6 times, 75 mistakes
                                    '15311', # chose by no rule 10 times, 69 mistakes
                                    ]

        self.num_finished = np.sum(np.array(self.last_trials) == 'GameC41')
        self.genders = card_data[self._ids][card_data.phaseround == 'Q00']
        self.agegroups = card_data[self._ids][card_data.phaseround == 'Q01']

        self.smooth_out_nans()
        self.set_cleaned_ids()
        self.num_participated = len(self._ids)

    def set_last_trials(self):
        self.last_trials = self.get_last_trials()

    def get_last_trials(self, depression_group=None):
        last_trials = []
        orig_ids = list(set(self._data.id))
        ids = self.filter_by_depression(orig_ids, depression_group)
        for id in set(ids):
            last_trials.append(self.get_last_trial(self._data, id))
        return last_trials

    def ids(self):
        return self._ids

    def card_data(self):
        return self._card_data

    def num_participants(self):
        return len(self._ids)

    def classify_success_person(self, person_id, threshold=90, smooth=False, correct=True):
        if correct and self.count_correct_person(person_id, smooth=smooth) >= threshold:
            return 1
        elif (not correct) and self.count_wrong_person(person_id, smooth=smooth) <= threshold:
            return 1
        return 0

    def classify_success_all(self, IDS=None, threshold=90, smooth=False):
        return [self.classify_success_person(pid, threshold=threshold, smooth=smooth) for pid in IDS]

    def count_nas_person(self, person_id, phase=None):
        if phase is not None:
            # count number of nas only for this phase
            return len(self._card_data[person_id][np.logical_and(self._card_data.phase == phase, pd.isna(self._card_data[person_id]))])
        # else, count for all phases but the feedback
        return len(self._card_data[person_id][np.logical_and(self._card_data.phase != 'F', pd.isna(self._card_data[person_id]))])

    def count_nas_all(self, IDS=None, phase=None):
        if IDS is None: IDS = self._ids
        return [self.count_nas_person(person_id=iidd, phase=phase) for iidd in IDS]

    def count_correct_person(self, person_id, phase=None, smooth=False):
        if smooth:
            return self.count_smooth_correct_per_person(person_id, phase)
        if phase is not None:
            return self._correct_df[person_id][self._correct_df.phase == phase].sum()
        return self._correct_df[person_id][7:171].sum()

    def count_smooth_correct_per_person(self, person_id, phase=None):
        if phase is not None:
            return self._combined_df[person_id +'correct'][self._combined_df.phase == phase].sum()
        return self._combined_df[person_id +'correct'][7:171].sum()

    def count_smooth_wrong_per_person(self, person_id, phase=None):
        if phase is not None:
            return self._combined_df[person_id +'correct'][self._combined_df.phase == phase].sum()
        return np.logical_not(self._combined_df[person_id +'correct'][7:171]).sum()

    def count_correct_all(self, IDS=None, phase=None, smooth=False):
        if IDS is None: IDS = self._ids
        return [self.count_correct_person(person_id=iidd, phase=phase, smooth=smooth) for iidd in IDS]

    def count_wrong_person(self, person_id, phase=None, smooth=False):
        if smooth:
            return self.count_smooth_wrong_per_person(person_id, phase)
        if phase is not None:
            return np.logical_not(self._correct_df[person_id][self._correct_df.phase == phase]).sum()
        return np.logical_not(self._correct_df[person_id][7:171]).sum()

    def count_wrong_all(self, IDS=None, phase=None, smooth=False):
        if IDS is None: IDS = self._ids
        return [self.count_wrong_person(person_id=iidd, phase=phase) for iidd in IDS]

    def mean_correct_per_person(self, IDS, phase=None,smooth=False):
        return np.mean(self.count_correct_all(IDS=IDS, phase=phase,smooth=smooth))

    def median_correct_per_person(self, IDS, phase=None,smooth=False):
        return np.median(self.count_correct_all(IDS=IDS, phase=phase,smooth=smooth))

    def mean_wrong_per_person(self, IDS, phase=None,smooth=False):
        return np.mean(self.count_wrong_all(IDS=IDS, phase=phase,smooth=smooth))

    def median_wrong_per_person(self, IDS, phase=None,smooth=False):
        return np.median(self.count_wrong_all(IDS=IDS, phase=phase,smooth=smooth))

    def set_cleaned_ids(self):
        self._clean_ids = self.filter_by_nans(self._ids)
        self._clean_ids = self.filter_by_wrong(self._clean_ids, thresh=70)
        self._clean_ids = [k for k in self._clean_ids if k not in self.other_ids_to_remove]

    def filter_ids(self, IDS, what_element, what_value):

        condition_phaseround, condition_value = self.get_condition(what_element, what_value)
        new_ids = []
        for id in IDS:
            val_of_id = self._card_data[id][self._card_data.phaseround == condition_phaseround].values
            if len(val_of_id) == 1:
                if pd.isna(what_value) and pd.isna(val_of_id[0]):
                        new_ids.append(id)
                elif val_of_id[0] ==  condition_value:
                        new_ids.append(id)
        return new_ids

    def filter_by_answer(self, IDS, condition_phaseround, zero=False):
        new_ids = []
        for id in IDS:
            val_of_id = self._card_data[id][self._card_data.phaseround == condition_phaseround].values
            if len(val_of_id) == 1:
                if zero and val_of_id[0] ==  '0':
                    new_ids.append(id)
                elif (not zero) and val_of_id[0] != '0':
                    new_ids.append(id)
        return new_ids

    def filter_by_depression(self, IDS, group=None):
        if group is None:
            return  IDS
        new_ids = []
        for id in IDS:
            score = self.score_depression(id)
            depression_group = self.score2depressiongroup(score)
            if depression_group == group:
                new_ids.append(id)
        return new_ids

    def filter_by_nans(self, IDS, thresh=40):
        return [id for id in IDS if self.count_nas_person(id) <= thresh]

    def filter_by_wrong(self, IDS, thresh=70):
        return [id for id in IDS if self.count_wrong_person(id) < thresh]

    def get_condition(self, what_element, what_value):
        condition_phaseround, condition_value = '', what_value
        if what_element == 'gender':
            condition_phaseround = 'Q00'
            condition_value = self.gender2val(what_value)
        elif what_element == 'age':
            condition_phaseround = 'Q01'
            condition_value = self.age2val(what_value)
        elif what_element == 'education':
            condition_phaseround = 'Q02'
            condition_value = self.education2val(what_value)
        return condition_phaseround, condition_value

    def gender2val(self, what):
        if pd.isna(what):
            return what
        if what.lower() in ['m', 'male']:
            return str(0)
        elif what.lower() in ['f', 'female']:
            return str(1)
        elif what.lower() in ['o', 'other']:
            return str(2)
        elif what.lower() in ['p', 'prefer not to say', 'prefer not']:
            return str(3)
        return np.nan

    def age2val(self, what):
        if pd.isna(what):
            return what
        if what in ['<18', '< 18']:
            return str(0)
        elif what in ['18-24']:
            return str(1)
        elif what in ['25-34']:
            return str(2)
        elif what in ['35-54']:
            return str(3)
        elif what in ['55-64']:
            return str(4)
        elif what in ['65-74']:
            return str(5)
        elif what in ['75+']:
            return str(6)
        return np.nan

    def education2val(self, what):
        if pd.isna(what):
            return what
        if what.lower() in ['no high school', 'less than high school']:
            return str(0)
        elif what.lower() in ['high school', 'high school diploma']:
            return str(1)
        elif what.lower() in ['bsc', 'bachelors']:
            return str(2)
        elif what.lower() in ['msc', 'masters']:
            return str(3)
        elif what.lower() in ['phd']:
            return str(4)
        return np.nan

    def get_last_trial(self, data, person_id):
        person_data = data[data['id'] == person_id]
        person_phases = set(person_data.phase)
        for phs in ['GameC', 'GameB', 'GameA', 'Q']:
            if phs in person_phases:
                return phs + self.get_trial_str(max(person_data.trial[person_data.phase == phs]))
        return ''

    def get_trial_str(self, trial):
        trialstr = str(int(trial))
        if len(trialstr) == 1:
            return '0' + trialstr
        return trialstr

    def make_phaseround_comparable(self, phsr):
        return phsr.replace('Q', '0').replace('GameA', '1').replace('GameB', '2').replace('GameC', '3')

    def score2depressiongroup(self, score):
        if score < 5:
            return "minimal"
        elif score < 10:
            return "moderated"
        elif score < 15:
            return "moderately severe"
        else:
            return "severe"

    def score_depression(self, subject):
        subject_PHQ9_answers = self._data.loc[(self._data["id"] == subject) &
                                          (self._data["phase"] == "Q") &
                                          (self._data["trial"] > 2)]
        subject_PHQ9_answers = [int(k) for k in subject_PHQ9_answers.response.values]
        return sum(subject_PHQ9_answers)

    def smooth_out_nans(self):
        for id in self._ids:
            for idx in self._combined_df.index:
                if self._combined_df.loc[idx, 'phase'] == 'F' or self._combined_df.loc[idx, 'phase'] == 'Q':
                    continue
                if pd.isna(self._combined_df.loc[idx, id + 'rule']):
                    if (self._combined_df.loc[idx - 1, id + 'rule'] == self._combined_df.loc[idx + 1, id + 'rule']):
                        self._combined_df.loc[idx, id + 'rule'] = self._combined_df.loc[idx - 1, id + 'rule']
                        self._combined_df.loc[idx, id + 'correct'] = (self._combined_df.loc[idx, 'trueRule'] == self._combined_df.loc[idx, id + 'rule'] )

def get_ids(col_data):
    ids = []
    for c in col_data.columns:
        try:
            a = int(c)
            ids.append(str(a))
        except:
            a = ''
    return ids


if __name__ == '__main__':

    data, col_data, correct_data, combined_col_data = get_data_for_stats(filename)
    SG = StatsGetter(data, col_data, correct_data,combined_col_data,  filename)

    IDS = [c for c in SG.card_data().columns if c not in
           ['phaseround', 'cardsLeft', 'cardsRight', 'card',
            'Unnamed: 0', 'trueRule', 'phase', 'round']]

    correctcols = [c for c in  SG._combined_df.columns if 'correct' in c]
    newcdf = SG._combined_df[np.logical_and(SG._combined_df.phase != 'Q', SG._combined_df.phase != 'F')].copy()
    newcdf = newcdf[correctcols].copy()
    plt.plot(newcdf.sum(axis=1))
    for ph in SG.__phase_change_inds:
        plt.axvline(ph, color = 'red')