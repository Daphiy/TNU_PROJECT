%importing all the responses per each subgroup
response_user_none = importdata('hgf_input\rules_for_response_none.csv');
response_user_mild = importdata('hgf_input\rules_for_response_mild.csv');
response_user_moderated = importdata('hgf_input\rules_for_response_moderated.csv');
response_user_moderatelysevere= importdata('hgf_input\rules_for_response_moderatelysevere.csv');

%loading our input responses to the hgf 
u = load('hgf_input\input_hgf.txt'); 

%% First step is to obtain the posteriors to use as priors for fitting each individual 

est = tapas_fitModel([],...
                     u,...
                     'tapas_hgf_categorical_config',...
                     'tapas_bayes_optimal_categorical_config',... 
                     'tapas_quasinewton_optim_config') 
                  

%plotting ideal response to the set shifting task 
tapas_hgf_categorical_plotTraj(est)
%plotting the correlation plot between the parameters 
tapas_fit_plotCorr(est) 

%% part 2 - fitting subgroups
%use as priors the parameters obtained in step1 that have been stored in
%the file 'tapas_hgf_categorical_config_game'
%% A) Fit No depression with general priors
for i = 1:size(response_user_none,2)
    
   est_none(i) = tapas_fitModel(u,...
                     response_user_none(2:end,i),...
                     'tapas_hgf_categorical_config_game',...
                     'tapas_bayes_optimal_categorical_config',... 
                     'tapas_quasinewton_optim_config') 
end
%% B) Plot all participants - no depression subgroup 
 for i = 1:39
    tapas_hgf_categorical_plotTraj(est_none(i))
  end
%% C) Calculate and plot the bayesian average - no depression subgroup 
bpa_none = tapas_bayesian_parameter_average(est_none(1),est_none(2),est_none(3),est_none(4),est_none(5),est_none(6),est_none(7),est_none(8),est_none(9),est_none(10),est_none(11),est_none(12),est_none(13),est_none(14),est_none(15),est_none(16),est_none(17),est_none(18),est_none(19),est_none(20),est_none(21),est_none(22),est_none(23), est_none(24),est_none(25), est_none(26),est_none(27),est_none(28),est_none(29),est_none(30),est_none(31),est_none(32),est_none(35),est_none(36),est_none(37),est_none(38),est_none(39));
tapas_hgf_categorical_plotTraj(bpa_none)
tapas_fit_plotCorr(bpa_none)
%% D) Extract parameters - no depression subgroup 
k_none = zeros(39,1);
for i = 1:39
    k_none(i)=est_none(i).p_prc.ka;
end    
writematrix(k_none,'kappa_none.csv') 

om_none = zeros(39,1);
for i = 1:39
    om_none(i)=est_none(i).p_prc.om;
end    
writematrix(om_none,'om_none.csv')

th_none = zeros(39,1);
for i = 1:39
    th_none(i)=est_none(i).p_prc.th;
end    
writematrix(th_none,'none.csv')

%% A) Fitting Mild Depression with general prior
for i = 1:size(response_user_mild,2)
   est_mild_g(i) = tapas_fitModel(u,...
                     response_user_mild(2:end,i),...
                     'tapas_hgf_categorical_config_game',...
                     'tapas_bayes_optimal_categorical_config',... 
                     'tapas_quasinewton_optim_config') 
end
%% %% B) Plot all participants - mild depression subgroup
for i = 1:size(response_user_mild,2)
    tapas_hgf_categorical_plotTraj(est_mild_g(i))
end
%% C) calculate and plot ayesian Parameter Averaging - Mild depression subgroup
bpa_mild_g = tapas_bayesian_parameter_average(est_mild_g(1),est_mild_g(2),est_mild_g(3),est_mild_g(4),est_mild_g(5),est_mild_g(6),est_mild_g(7),est_mild_g(8),est_mild_g(10),est_mild_g(11),est_mild_g(12),est_mild_g(13),est_mild_g(14),est_mild_g(15),est_mild_g(16),est_mild_g(17),est_mild_g(18),est_mild_g(19),est_mild_g(20),est_mild_g(21),est_mild_g(22),est_mild_g(23),est_mild_g(24),est_mild_g(25),est_mild_g(26),est_mild_g(27),est_mild_g(28),est_mild_g(29),est_mild_g(30));
tapas_hgf_categorical_plotTraj(bpa_mild_g)
tapas_fit_plotCorr(bpa_mild_g)

%% D) Extract parameters per person - Mild depression subgroup
k_mild = zeros(30,1);
for i = 1:30
    k_mild(i)=est_mild_g(i).p_prc.ka;
end    
writematrix(k_mild,'k_mild.csv') 

om_mild = zeros(30,1);
for i = 1:30
    om_mild(i)=est_mild_g(i).p_prc.om;
end    
writematrix(om_mild,'om_mild.csv')

th_mild = zeros(30,1);
for i = 1:30
    th_mild(i)=est_mild_g(i).p_prc.th;
end    
writematrix(th_mild,'th_mild.csv')

%% A) Fitting Moderated depression participants with general prior
for i = 1:size(response_user_moderated,2)
   est_moderated_g(i) = tapas_fitModel(u,...
                     response_user_moderated(2:end,i),...
                     'tapas_hgf_categorical_config_game',...
                     'tapas_bayes_optimal_categorical_config',... 
                     'tapas_quasinewton_optim_config') 
end

%% B) Plot all participants - Moderated depression subgroup
for i = 1:10
    tapas_hgf_categorical_plotTraj(est_moderated_g(i))
    tapas_fit_plotCorr(est_moderated_g(i))
end
%% C) Calculate and plot Bayesian Parameter Averaging - Moderated depression subgroup
bpa_moderateds_g = tapas_bayesian_parameter_average(est_moderated_g(1),est_moderated_g(2),est_moderated_g(3),est_moderated_g(4),est_moderated_g(5),est_moderated_g(6),est_moderated_g(7),est_moderated_g(8),est_moderated_g(10));
tapas_hgf_categorical_plotTraj(bpa_moderateds_g)
tapas_fit_plotCorr(bpa_moderateds_g)
%% D) Extract parameters per person - Moderated depression subgroup
k_moderateds = zeros(10,1);
for i = 1:10
    k_moderateds(i)=est_moderated_g(i).p_prc.ka
end    
writematrix(k_moderateds,'kappa_modesev.csv') 

om_moderateds = zeros(10,1);
for i = 1:10
    om_moderateds(i)=est_moderated_g(i).p_prc.om
end    
writematrix(om_moderateds,'omega_modesev.csv')

th_moderateds = zeros(10,1);
for i = 1:10
    th_moderateds(i)=est_moderated_g(i).p_prc.th
end    
writematrix(th_moderateds,'theta_modesev.csv')


%% A) Fit Moderated/Severe depression subgroup with general priors
for i = 1:size(response_user_moderatelysevere,2)
   est_moderatelysevere_g(i) = tapas_fitModel(u,...   
                     response_user_moderatelysevere(2:end,i),...
                     'tapas_hgf_categorical_config_game',...
                     'tapas_bayes_optimal_categorical_config',... 
                     'tapas_quasinewton_optim_config') 
end
%% B) Plot all participants - Moderated/Severe depression subgroup
for i = 1:3 
    tapas_hgf_categorical_plotTraj(est_moderatelysevere_g(i))
    tapas_fit_plotCorr(est_moderatelysevere_g(i))
end
%% C) Bayesian Parameter Averaging - Moderated/Severe depression subgroup
%Average for only three cases across wide PHQ9 scores makes little sense
bpa_moderatelysevere = tapas_bayesian_parameter_average(est_moderatelysevere_g(1), est_moderatelysevere_g(2),est_moderatelysevere_g(3));
tapas_hgf_categorical_plotTraj(bpa_moderatelysevere)
tapas_fit_plotCorr(bpa_moderatelysevere)
%% D) Extract parameters per person - Moderated/Severe 
om_moderatelysevere = zeros(3,1);
om_moderatelysevere(1)=est_moderatelysevere_g(1).p_prc.om
om_moderatelysevere(2)=est_moderatelysevere_g(2).p_prc.om
om_moderatelysevere(3)=est_moderatelysevere_g(3).p_prc.om
writematrix(om_moderatelysevere,'omega_moderatelysevere.csv')

th_moderatelysevere = zeros(3,1);
th_moderatelysevere(1)=est_moderatelysevere_g(1).p_prc.th
th_moderatelysevere(2)=est_moderatelysevere_g(2).p_prc.th
th_moderatelysevere(3)=est_moderatelysevere_g(3).p_prc.th
writematrix(th_moderatelysevere,'theta_moderatelysevere.csv')

%%
%Box plots for the bayes_optimal_categorical
%read csv containg all omegas por subgroup 
boxnone = readmatrix('mino'); 
boxmild = readmatrix('modo');
boxmods = readmatrix('mso'); %contains moderaed, moderately severe and severe

g1 = repmat({'None'},size(boxnone,1),1);
g2 = repmat({'Mild'},(size(boxmild,1)),1);
g3 = repmat({'Moderated to severe'},(size(boxmods,1)),1);
g = [g1; g2; g3];

%Boxplot correct
x_correct = [boxnone;boxmild;boxmods];
boxplot(x_correct,g)   
title('ω Values')
ylabel('tonic volatility ω')
xlabel('Level of depression')

%%
%Box plots for the softmax
%read csv containg all omegas por subgroup 
boxnone2 = readmatrix('mino2'); 
boxmild2 = readmatrix('modo2');
boxmods2 = readmatrix('mso2'); %contains moderaed, moderately severe and severe

g1 = repmat({'None'},size(boxnone2,1),1);
g2 = repmat({'Mild'},(size(boxmild2,1)),1);
g3 = repmat({'Moderated to severe'},(size(boxmods2,1)),1);
g = [g1; g2; g3];

%Boxplot correct
x_correct = [boxnone2;boxmild2;boxmods2];
boxplot(x_correct,g)    
title('ω Values')
ylabel('tonic volatility ω')
xlabel('Level of depression')



