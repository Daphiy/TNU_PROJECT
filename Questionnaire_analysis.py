import pandas as pd
import numpy as np

dataframe = pd.read_csv('gamestats18_05v2.csv', ';', names=["participantID", "Section", "QuestionTrial", "Answer", "Time"])  #, sep= ";"
#dataframe = dataframe.split(",", n = 4, expand = True)
print(dataframe)

# #dataframe.columns = ["participantID", "Section", "QuestionTrial", "Answer", "Time"]
#
# print(dataframe)
# #print()
subjects = dataframe.participantID.unique()
subjects = list(set(dataframe.participantID.tolist()))
#print(subjects)

subject_scores_df = pd.DataFrame(columns=["participantID",
                                          "sex",
                                          "age",
                                          "education",
                                          "Score",
                                          "DepressionSeverity"])

for sub in subjects:
    current_questions = dataframe.loc[(dataframe["participantID"] == sub) &
                                      (dataframe["Section"] == "Q") &
                                      (dataframe["QuestionTrial"] > 2)]
    #print(current_questions)
    # Check if subject has answered all the questions
    if len(current_questions.index) < 9:
        continue

    sex = dataframe.loc[(dataframe["participantID"] == sub) &
                        (dataframe["Section"] == "Q") &
                        (dataframe["QuestionTrial"] == 0)]
    if sex.empty:
        sex = "NA"
    else:
        sex = sex.Answer.values[0]

    age = dataframe.loc[(dataframe["participantID"] == sub) &
                        (dataframe["Section"] == "Q") &
                        (dataframe["QuestionTrial"] == 1)]
    if age.empty:
        age = "NA"
    else:
        age = age.Answer.values[0]

    education = dataframe.loc[(dataframe["participantID"] == sub) &
                              (dataframe["Section"] == "Q") &
                              (dataframe["QuestionTrial"] == 2)]
    if education.empty:
        education = "NA"
    else:
        education = education.Answer.values[0]

    #skip our testing triels
    if age == 0 and education == 4:
        print("Skip that subject!\n")
        continue
    #calculate the parameters for each partecipants
    current_sub_score = current_questions.Answer.sum()
    if current_sub_score < 5:
        severity = "minimal"
    elif current_sub_score < 10:
        severity = "moderated"
    elif current_sub_score < 15:
        severity = "moderately severe"
    else:
        severity = "severe depression"

    if sex == 0:
        gender = "male"
    elif sex == 1:
        gender = "female"
    elif sex == 2:
        gender = "other"
    else:
        gender = "prefer not to say"

    if age == 0:
        Age = "<18"
    elif age == 1:
        Age = "18-24"
    elif sex == 2:
        Age = "25-34"
    elif sex == 3:
        Age = "35-54"
    elif sex == 4:
        Age = "55-74"
    else:
        Age = "75+"

    if education == 0:
        Education = "less then high school diploma"
    elif education == 1:
        Education = "high school diploma"
    elif education == 2:
        Education = "bachelor"
    elif education == 3:
        Education = "master"
    else:
        Education  = "Phd"

    subject_scores_df = subject_scores_df.append({'participantID': sub,
                                                  'sex': gender,
                                                  'age': Age,
                                                  'education': Education,
                                                  'Score': current_sub_score,
                                                  'DepressionSeverity': severity}, ignore_index=True)


print(subject_scores_df)
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(subject_scores_df.sort_values('Score', ascending = False))
df = pd.DataFrame(subject_scores_df.sort_values('Score', ascending = False)) #dataFrame sorted by level of depression

print(subject_scores_df['DepressionSeverity']=='moderated')

severe = subject_scores_df.loc[subject_scores_df['DepressionSeverity'] == "severe depression", 'participantID']
moderatelySevere = subject_scores_df.loc[subject_scores_df['DepressionSeverity'] == "moderately severe", 'participantID']
moderated = subject_scores_df.loc[subject_scores_df['DepressionSeverity'] == "moderated", 'participantID']
minimal = subject_scores_df.loc[subject_scores_df['DepressionSeverity'] == "minimal", 'participantID']

severe.to_csv('severeID.csv', index = False, sep =',')
moderatelySevere.to_csv('moderatelySevere.csv', index = False, sep =',')
moderated.to_csv('moderated.csv', index = False, sep =',')
minimal.to_csv('minimal.csv', index = False, sep =',')


#subject_scores_df.to_csv("subject_scores.csv", index=False, sep=";")
#df.to_csv("subject_scores.csv", index=False, sep=";")


#feedback or people
#current_feedback = dataframe.loc[(dataframe["participantID"] == sub) &
                                     # (dataframe["Section"] == "F")]
#print(current_feedback)



#subject_scores_df.set_index(["participantID", "DepressionSeverity"]).count(level="severe depression")