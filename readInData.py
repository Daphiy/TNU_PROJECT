"""
Load data from game.
Each row corresponds to an action in the game.

The format of each row is :
    id , phase , round , response , time

id = a unique ID that identifies each respondant
phase = questionnaire / one of the three game phases
round = round # in current phase
response = pile chosen in wisconsin (0,1,2,3), or True (1)  / False (0) in twist
time = time it took to respond in this round

GameA : wisconsin1 , GameB : twist , GameC : wisconsin2
"""
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
import matplotlib.pyplot as plt
import numpy as np
from random import randint, randrange

filename = 'gamestats24_05.csv'

#### Heplful data structures

PHASES  = ['Q','GameA','GameB','GameC']

RULES = {'number' : lambda card1, card2 : (card1[0] == card2[0]),
         'color'  : lambda card1, card2 : (card1[1] == card2[1]),
         'shape'  : lambda card1, card2 : (card1[2:] == card2[2:]),}

CARD2PILE = {'1Ystar':1, '2Rsquare':2, '3Btriangle':3, '4Gcircle':4}
PILE2CARD  = {1:'1Ystar', 2:'2Rsquare' , 3:'3Btriangle', 4:'4Gcircle'}
RULE2NUM = {'number' : 1 , "color" : 2, "shape" : 3, np.nan : 1}
NUM2RULE = {1 : 'number' , 2 : "color", 3 : "shape", np.nan : "number"}

#### load data

def load_data(filename):
    data = pd.read_csv(filename)
    split_data = data["Log"].str.split(",", n = 4, expand = True)
    split_data.columns = ['id', 'phase', 'round', 'response', 'time']
    return split_data

def get_trials(fwisc1='trialsWisconsin1.json', ftwist='trialsTwist.json', fwisc2='trialsWisconsin2.json'):
    trials_wisconsin1 = pd.read_json(fwisc1)['Trials']
    trials_twist = pd.read_json(ftwist)['Trials']
    trials_wisconsin2 = pd.read_json(fwisc2)['Trials']
    return trials_wisconsin1, trials_twist, trials_wisconsin2

def get_card_df(trials_wisconsin1, trials_twist, trials_wisconsin2):
    """
    Get df with cards used in each trial, from the trials_wisconsin 1 & 2 and trials_twist json files
     * * the files trials_wisconsin1, trials_twist, trials_wisconsin2 should be initialized globally * *
    """
    card_df = pd.DataFrame({'card':
                                trials_wisconsin1['cardsDeck'] +
                                trials_wisconsin2['cardsDeck'],
                            'phaseround':
                                ['GameA' + get_round_str(k) for k, __ in enumerate(trials_wisconsin1['cardsDeck'])] +
                                ['GameC' + get_round_str(k) for k, __ in enumerate(trials_wisconsin2['cardsDeck'])]
                            }
                           )
    cardLR_df = pd.DataFrame({'cardsLeft': trials_twist['cardsLeft'],
                                'cardsRight': trials_twist['cardsRight'],
                                'phaseround': ['GameB' + get_round_str(k) for k, __ in enumerate(trials_twist['cardsLeft'])]
                               })
    return card_df, cardLR_df

def columnar_data(data, rules, trials_wisconsin1, trials_twist, trials_wisconsin2):
    """
    Transform such that each column is a participant, and add cards from trials json files

    """
    data = data.copy()
    data['phaseround'] = data[['phase','round']].apply(create_phase_round_col, axis=1) # create a one-column index

    column_data = data.pivot(index='phaseround', columns='id', values='response')

    card_df, cardLR_df = get_card_df(trials_wisconsin1, trials_twist, trials_wisconsin2)

    column_card_data = pd.merge(column_data, cardLR_df, on='phaseround', how='outer')
    column_card_data = pd.merge(column_card_data, card_df, on='phaseround', how='outer')
    column_card_data = pd.merge(column_card_data, rules, on="phaseround", how="outer")

    column_card_data["phase"] = column_card_data["phaseround"].apply(phaseround_to_phase)
    column_card_data["round"] = column_card_data["phaseround"].apply(phaseround_to_round)

    return column_card_data

def last_round_included_data(data):
    lrids = get_last_round_ids(data)
    lrids = [idx for idx in lrids if idx != '44223']
    lrdata = data[np.isin(data.id, lrids)]
    return lrdata

def filter_ourselves(data):
    data = data[data.id != 0]
    ### TO DO : Also filter if Q answers start with 2, 0, 4 (prefer not to say, <18, PhD)
    return data

def create_columnar_df(filename,
                       only_if_last_round_included = True,
                       rules_filename = "input_HGF.csv",
                       fwisc1='trialsWisconsin1.json',
                       ftwist='trialsTwist.json',
                       fwisc2='trialsWisconsin2.json'):
    trials_wisconsin1, trials_twist, trials_wisconsin2 = get_trials(fwisc1, ftwist, fwisc2)
    rules = pd.read_csv(rules_filename)

    data = load_data(filename)
    data = filter_ourselves(data)

    data['round'] = data['round'].apply(get_round_str)

    if only_if_last_round_included:
        data = last_round_included_data(data)

    card_data = columnar_data(data, rules, trials_wisconsin1, trials_twist, trials_wisconsin2)

    return card_data

def create_rules_columnar_df(card_data, IDS):
    rules_df = card_data[['phase', 'round', 'trueRule']].copy()
    for i, id in enumerate(IDS):
        if id in card_data.columns:
            rules_df[id] = card_data[[id, 'cardsLeft', 'cardsRight', 'card', 'phase', 'trueRule']].apply(response_rule,axis = 1)
            rules_df[id + '_response'] = card_data[id]
    return rules_df

def create_all_col_df(card_data):
    IDS = get_ids(card_data)
    all_df = card_data[['phase', 'round', 'trueRule']].copy()
    for i, id in enumerate(IDS):
        if id in card_data.columns:
            all_df[id + 'rule'] = card_data[[id, 'cardsLeft', 'cardsRight', 'card', 'phase', 'trueRule']].apply(response_rule,axis = 1)
            all_df[id + 'correct'] = card_data[[id, 'cardsLeft', 'cardsRight', 'card', 'phase', 'trueRule']].apply(is_correct_response,axis = 1)
            all_df[id + '_response'] = card_data[id]
    return all_df

def add_rule_column(data, IDS):
    for i, id in enumerate(IDS):
        if id in card_data.columns:
            data[id+'_response_rule'] = card_data[[id, 'cardsLeft', 'cardsRight', 'card', 'phase', 'trueRule']].apply(response_rule,axis = 1)
    return

def create_correct_columnar_df(card_data):
    IDS = get_ids(card_data)
    correct_df = card_data[['phase', 'round', 'trueRule']].copy()
    for i, id in enumerate(IDS):
        if id in card_data.columns:
            correct_df[id] = card_data[[id, 'cardsLeft', 'cardsRight', 'card', 'phase', 'trueRule']].apply(
                is_correct_response,
                axis=1)
    return correct_df

#### Helpers

def get_ids(data):
    if 'id' in data.columns:
        ids = set(data['id'])
    else:
        ids = []
        for c in data.columns:
            try:
                a = int(c)
                ids.append(str(a))
            except:
                a = ''
    return ids

def get_person_data(data, person_id):
    return data[data['id'] == person_id]

def get_last_round_ids(data):
    """
    Assumes rounds are already in 2-digit format: 00, 01, 02, 03 (versus 0,1,2,3)
    """
    last_round = data[np.logical_and(data.phase == 'GameC',
                                     data['round'] > '37'
                                     )
                     ]
    return last_round.id

def validate_trials_file(trials, twist = False):
    if twist:
        tw = pd.DataFrame({"answers": trials['answers'], "cardsLeft": trials['cardsLeft'], 'cardsRight' : trials['cardsRight']})
        tw['idx'] = [k for k in tw.index]
        return tw
    wisc = pd.DataFrame({"answers": trials['answers'], "cards": trials['cardsDeck']})
    wisc["pile"] = wisc.answers.apply(lambda x: PILE2CARD[x])
    wisc['idx'] = [k for k in wisc.index]
    wisc['rule'] = wisc.apply(get_rule, axis=1)
    return wisc

def phaseround_to_phase(phaseround):
    """
    Take a string that represents phase and round (e.g GameB15, Q45)
     and return phase only  (Q, F, GameA, GameB, GameC)
    """
    if phaseround.startswith('G'):
        return phaseround[0:5]
    return phaseround[0]

def phaseround_to_round(phaseround):
    """
    Take a string that represents phase and round (e.g GameB15, Q45)
     and return round only  (e.g. 0, 11, 15, 45)
    """
    if phaseround.startswith('G'):
        return phaseround[5:]
    return phaseround[1:]

def get_rules_for_match(card1, card2, isMatch = True, integers = False):
    """
    input : two cards and whether they match by the response
    output : the rules by which the cards match/don't match
    """
    matching_rules = []
    for rule in RULES:
        if isMatch and RULES[rule](card1, card2):
            # cards match and response says they match
            matching_rules.append(rule)
        elif (not isMatch) and (not RULES[rule](card1, card2)):
            # cards don't match and response says they don't match
            matching_rules.append(rule)
    if integers:
        matching_rules = [RULE2NUM[rl] for rl in matching_rules]
    return matching_rules

def some_wrong_rule(trueRule):
    if trueRule == 1:
        return 2
    return 1

def get_round_str(r):
    roundstr = str(r)
    if len(roundstr) == 1:
        roundstr = '0' + roundstr
    return roundstr

#### Function for applying to the data frame

def get_pile(phaseresponse):
    """
    made for df[["phase", "response"]].apply()
    returns the pile card chosen for wisconsin1 & 2, and nan otherwise
    """
    phase, response = phaseresponse
    if phase == 'GameA' or phase == 'GameC':
        return PILE2CARD[int(response) + 1]
    return np.nan

def get_rule(x):
    rules = get_rules_for_match(x['pile'], x['cards'])
    if len(rules) != 1:
        print(x['cards'], x['idx'])
    if len(rules) < 1:
        return "None"
    return rules[0]

def response_rule(df):
    """
    Usage :
        df[["response", "cardsLeft", "cardsRight", "card", "phase", "trueRule"]].apply(response_rule, axis = 1)

    Returns :
        the rule used for generating the response
    """
    response, cardsLeft, cardsRight, card, phase, trueRule = df

    if (phase == 'F' or phase == 'Q'):
        return np.nan

    if (pd.isna(response)):
        # no response
        return  np.nan

    if phase == 'GameB':    card1, card2, isMatch = cardsLeft, cardsRight, bool(int(response))
    else : card1, card2, isMatch = card, PILE2CARD[int(response) + 1], True

    if (pd.isna(card1) or pd.isna(card2)):
        # no cards
        return np.nan

    matching_rules = get_rules_for_match(card1, card2, isMatch, integers = True)

    if len(matching_rules) < 1:
        # person chose by no rule - return an arbitrary wrong rule
        print("person chose by no rule", df)
        return some_wrong_rule(trueRule)

    elif len(matching_rules) == 1:
        # only one matching rule - return it
        return matching_rules[0]

    elif trueRule in matching_rules:
        # if the true rule is one of the matching - return it
        return trueRule

    # else return one of the matching rules randomly
    idx = np.random.choice(len(matching_rules))
    return matching_rules[idx]

def is_correct_response(df):

    response, cardsLeft, cardsRight, card, phase, trueRule = df

    if phase == 'Q' or phase == 'F':
        # wrong phase
        return np.nan

    if (pd.isna(response)) or (pd.isna(trueRule)):
        # no response or no rule
        return  np.nan

    trueRule = NUM2RULE[trueRule]

    if phase == 'GameB':
        if RULES[trueRule](cardsLeft, cardsRight):
            # there really is a match
            return bool(int(response)) #True if response = 1(match), else False
        else:
            # there really isn't a match
            return (not bool(int(response))) #True if response = 0(no match), else False

    elif phase == 'GameA' or phase == 'GameC':
        # return whether the chosen card & deck card match by the true rule
        responseCard = PILE2CARD[int(response) + 1]
        return RULES[trueRule](card, responseCard)

def create_phase_round_col(phr):
    """
    Usage :
        df[["phase", "round"]].apply(create_phase_round_col, axis = 1)
    """
    ph, r = phr
    r = get_round_str(r)
    return ph + r

####

def one_hot_encode(deckCard, ruleCheckCard):
    rule = []
    for j in range(len(deckCard)):
        matching_rules = get_rules_for_match(deckCard[j],ruleCheckCard[j])
        rule.append(matching_rules)

    df = pd.Series(rule)
    mlb = MultiLabelBinarizer()
    df = pd.DataFrame(mlb.fit_transform(df),
                       columns=mlb.classes_,
                       index=df.index)
    return df

def get_rule_twist_column(deckCard, ruleCheckCard, isMatch):
    """
    input : whole column
    """
    rule = []
    for j in range(len(deckCard)):
        matching_rules = get_rules_for_match(deckCard[j],ruleCheckCard[j])
        if isMatch[j]:
            rule.append(matching_rules)
        else:
            rule.append([rl for rl in RULES.keys() if rl not in matching_rules])
    return rule

def result_game_id(filename, game, identity):
    trials_wisconsin1, trials_twist, trials_wisconsin2 = get_trials(fwisc1='trialsWisconsin1.json',
                                                                    ftwist='trialsTwist.json',
                                                                    fwisc2='trialsWisconsin2.json')

    if game == 'GameA':   trial = trials_wisconsin1
    elif game == 'GameB':    trial = trials_twist
    elif game == 'GameC':   trial = trials_wisconsin2

    split_data = load_data(filename)

    result = split_data.loc[split_data['phase'] == game, ('phase', 'id', 'time','response')]
    result = result.loc[split_data['id'] == identity, ('phase', 'id','time', 'response' )]

    if game == 'GameB':
        card_left = trial['cardsLeft']
        card_right = trial['cardsRight']
        answer = result['response'].tolist()
        correctAnswer = trial['answers']

        df = one_hot_encode(card_left, card_right)

        result['correct'] = correctAnswer
        result['card_left'] = card_left  # add column
        result['card_right'] = card_right  # add column
        result['number'] = df['number'].tolist()
        result['color'] = df['color'].tolist()
        result['shape'] = df['shape'].tolist()



    else:
        deckCard = trial['cardsDeck']
        answer = result['response'].tolist()
        correctAnswer = trial['answers']

        correct = []
        for i in range(len(correctAnswer)):
            correct.append(PILE2CARD[int(correctAnswer[i])])

        pile_card = []
        for card in range(len(answer)):
            pile_card.append(PILE2CARD[int(answer[card]) + 1])

        result.loc[:, 'response'] = pile_card  # Used this format as I replace numeric response with full card
        result['DeckCard'] = deckCard  # add column
        result['Correct'] = correct  # add column

        df1 = one_hot_encode(deckCard, pile_card)
        df2 = one_hot_encode(deckCard, correct)

        result['number_user'] = df1['number'].tolist()
        result['color_user'] = df1['color'].tolist()
        result['shape_user'] = df1['shape'].tolist()
        result['number_rule'] = df2['number'].tolist()
        result['color_rule'] = df2['color'].tolist()
        result['shape_rule'] = df2['shape'].tolist()

    return result

def rules_input_wisconsin(result):
    df = result[['number_rule', 'color_rule', 'shape_rule']]
    df = df.to_numpy()
    u_input = np.zeros((len(df), 1))
    u_input[[i for i in np.where(df[:, 0] == 1)], 0] = 1
    u_input[[i for i in np.where(df[:, 1] == 1)], 0] = 2
    u_input[[i for i in np.where(df[:, 2] == 1)], 0] = 3

    return u_input

def rules_response_wisconsin(result):
    df = result[['number_user', 'color_user', 'shape_user']]
    df = df.to_numpy()
    y_response = np.zeros((len(df), 1))
    for i in range(len(df)):
        if (df[i, 0] == 1 and df[i, 1] == 0 and df[i, 2] == 0):
            y_response[i, 0] = 1
        elif (df[i, 1] == 1 and df[i, 0] == 0 and df[i, 2] == 0):
            y_response[i, 0] = 2
        elif (df[i, 2] == 1 and df[i, 0] == 0 and df[i, 1] == 0):
            y_response[i, 0] = 3
        elif (df[i, 0] == 1 and df[i, 1] == 1 and df[i, 2] == 0):
            y_response[i, 0] = randint(1, 2)
        elif (df[i, 0] == 1 and df[i, 1] == 0 and df[i, 2] == 1):
            y_response[i, 0] = randrange(1, 4, 2)
        elif (df[i, 0] == 0 and df[i, 1] == 1 and df[i, 2] == 1):
            y_response[i, 0] = randint(2, 3)
    return y_response

def read_ids():
    severe = pd.read_csv('severeID.csv', ',', header=None, names=['id'])
    moderatelySevere = pd.read_csv('moderatelySevere.csv', ',', header=None, names=['id'])
    moderated = pd.read_csv('moderated.csv', ',', header=None, names=['id'])
    minimal = pd.read_csv('minimal.csv', ',', header=None, names=['id'])

    # Convert the Dataframes to list
    IDSsevere = severe['id'].astype(str).to_list()
    IDSmoderatelysevere = moderatelySevere['id'].astype(str).to_list()
    IDSmoderated = moderated['id'].astype(str).to_list()
    IDSminimal = minimal['id'].astype(str).to_list()

    return IDSsevere, IDSmoderatelysevere, IDSmoderated, IDSminimal

def has_missing_Q(cdata, id):
    Qphases = ['Q0' + str(k) for k in range(10)] + ['Q10', 'Q11']
    for phsr in Qphases:
        if pd.isna(cdata[id][cdata.phaseround == phsr].values[0]):
            return True
    return False

def patch_up_Q(card_data):
    """
    Patch up specific questionnaires, where the result is not affected by the one answer missing
    """

    default_val = str(int(0))

    idx = card_data.index[card_data.phaseround == 'Q02']
    card_data.loc[idx, '73461'] = default_val # is under 18 so I defaulted to no high school yet (:

    idx = card_data.index[card_data.phaseround == 'Q04']
    card_data.loc[idx, '46161'] = default_val
    card_data.loc[idx, '5478'] = default_val

    idx = card_data.index[card_data.phaseround == 'Q05']
    card_data.loc[idx, '28313'] = default_val

    return card_data

def filter_missing_questionnaire(col_data, ids):
    """
    Input : col_data - data in the columnar format
    Return col_data after filtering out people with incomplete questionnaire
    """
    to_remove = []
    for id in ids:
        if has_missing_Q(col_data[[id, 'phaseround']], id):
            print(id)
            to_remove.append(id)
    col_data = col_data.drop(to_remove, axis = 1)
    return col_data

def str_to_int(x):
    try :
        return int(x)
    except:
        return x

def get_data_for_stats(filename):
    """
    Get three data frames for collecting statistics later
    """
    data = load_data(filename)
    data['trial'] = data['round'].apply(str_to_int)

    col_data = create_columnar_df(filename)
    col_data = patch_up_Q(col_data)

    correct_data = create_correct_columnar_df(col_data)
    combined_col_data = create_all_col_df(col_data)
    return data, col_data, correct_data, combined_col_data

def smooth_out_nans(person_data):

    return

if __name__ == '__main__':
    #IDS = ['14355', '35308', '58403', '21572', '72137', '31634', '65926', '48130']

    # Reading the csv files with the IDs
    IDSsevere, IDSmoderatelysevere, IDSmoderated, IDSminimal = read_ids()

    card_data = create_columnar_df(filename)
    rules_data = create_rules_columnar_df(card_data[7:171], IDSmoderatelysevere)
    correct_df = create_correct_columnar_df(card_data)
    #Uncomment lines below for mode

    rules_for_mode = rules_data.drop(columns = ['phase', 'round', 'trueRule'])
    mode = rules_for_mode.mode(axis=1, dropna=False)
    #mode.to_csv('moderately_severe_mode.csv', index=False, sep=',')
    #print(mode)
    #print((IDSsevere))


    # np.set_printoptions(threshold=sys.maxsize)
    # print(results_matrix)
        # pd.Series(rule_for_id).to_csv('rules_for_responses\\rule_for_' + id + '.csv')
        # with open("rules_for_responses\\rule_for" + str(id) + ".txt", "w")  as f:
        #     lines = ("\n").join([str(k) for k in rule_for_id])
        #     f.writelines(lines)


    # for id in IDSsevere:
    #     rule_for_id = card_data[[id, 'cardsLeft', 'cardsRight', 'card', 'phase', 'trueRule']][7:171].apply(response_rule,axis=1)
    #     rule_for_id = rule_for_id  # only indices relevant for games
    #     pd.Series(rule_for_id).to_csv('rules_for_responses\\rule_for_' + id + '.csv')
    #     with open("rules_for_responses\\rule_for" + str(id) + ".txt", "w")  as f:
    #         lines = ("\n").join([str(k) for k in rule_for_id])
    #         f.writelines(lines)