clc
clear
%variables
load('datos_wdbc');
datos = trn.xc;
sizerow = trn.n;
sizecol = trn.l;
labels = trn.y;

%Elige el porcentaje que desees.
trn_input=input('Seleccione el porcentaje: (1)70-30, (2)80-20, (3)90-10: ');
if trn_input ==  1
    percent=0.7;
elseif trn_input == 2
    percent=0.8;
else
    percent=0.9;
end

%% TRAINING %%
%Porcentaje en training
training_percent =  floor(sizerow * percent);

%Labels training
label_training = labels((1:training_percent), :);

%labels trn values and indexes
label_tr_c1 = label_training(label_training==1);
lbl_tr_c1_indx = find(label_training==1);

label_tr_c2 = label_training(label_training==2);
lbl_tr_c2_indx = find(label_training==2);

%Clasificación a priori
class_priori_1 = length(label_tr_c1)/length(label_training);
class_priori_2 = length(label_tr_c2)/length(label_training);

%matrices de datos de training
training_set = datos((1:training_percent), :);

tr_data_c1 = training_set(lbl_tr_c1_indx, :);
tr_data_c2 = training_set(lbl_tr_c2_indx, :);

%matriz de covarianza de todo el dataset de training
sigma = cov(training_set);

%matrices de covarianza c1 y c2
sigma_c1 = cov(tr_data_c1);
sigma_c2 = cov(tr_data_c2);

%% Se sanean los valores negativos, y se reconstruye la matriz. %%
ssaneado = sanear(sigma);
ssaneado_c1 = sanear(sigma_c1);
ssaneado_c2 = sanear(sigma_c2);

%  X representa cada columna de cada dataset de clases
%  M representa la media de cada columna de cada dataset de clases
%  Como resultado de QDC y LDC se obtienen dos vectores escalares, uno por cada clase

%% TEST %%
label_test = labels((training_percent+1:end), :);
test_percent = sizerow - training_percent;
test_set = datos((training_percent+1:end), :);

%% Se calcula QDC por dataset_c1 y dataset_c2 %%
QDC_c1 = [];
QDC_c2 = [];

testlen = length(test_set(:, 1));
M_C1 = mean(tr_data_c1);
M_C2 = mean(tr_data_c2);

E_inv_C1 = inv(ssaneado_c1);
E_inv_C2 = inv(ssaneado_c2);

E_C1 = ssaneado_c1;
E_C2 = ssaneado_c2;

for i = 1:testlen
    X = test_set(i, :);
    QDC_c1(i) = -0.5 * X * E_inv_C1 * X' + M_C1 * E_inv_C1 * X'  - 0.5 * M_C1 * E_inv_C1 * M_C1'  - 0.5 * log(det(E_C1)) + log(class_priori_1);
    QDC_c2(i) = -0.5 * X * E_inv_C2 * X' + M_C2 * E_inv_C2 * X'  - 0.5 * M_C2 * E_inv_C2 * M_C2'  - 0.5 * log(det(E_C2)) + log(class_priori_2);
end

%% Se calcula LDC por dataset_c1 y dataset_c2 %%
LDC_c1 = [];
LDC_c2 = [];

for i = 1:testlen
    X = test_set(i, :);
    LDC_c1(i) = M_C1 * inv(ssaneado) * X' -0.5 * M_C1 * inv(ssaneado) * M_C1' + log(class_priori_1);
    LDC_c2(i) = M_C2 * inv(ssaneado) * X' -0.5 * M_C2 * inv(ssaneado) * M_C2' + log(class_priori_2);
end

%Una vez que se calcula QDC y DLC se estima cual clase es mayor
lbl_test_QDC = [];
lbl_test_LDC = [];
for m=1:length(QDC_c1)
    if QDC_c1(m)>QDC_c2(m)
        lbl_test_QDC(m)=0;
    else
        lbl_test_QDC(m)=1;
    end
    
    if LDC_c1(m)>LDC_c2(m)
        lbl_test_LDC(m)=0;
    else
        lbl_test_LDC(m)=1;
    end
end

%Tasa de reconocimiento QDC y LDC, sin regularización
positiveLDC = [];
positiveQDC = [];
labelTst = label_test';
labelTst(labelTst == 1) = 0;
labelTst(labelTst == 2) = 1;
TstTasaQDC = [];
TstTasaLDC = [];

 
for a = 1:length(lbl_test_QDC)
    if lbl_test_QDC(1, a) == labelTst(1, a)
        positiveQDC(a)=1;
    end
    if lbl_test_LDC(1, a) == labelTst(1, a)
        positiveLDC(a)=1;
    end
end
TstTasaQDC = (sum(positiveQDC)/length(lbl_test_QDC)) * 100;
TstTasaLDC = (sum(positiveLDC)/length(lbl_test_LDC)) * 100;

%% Regularización
%QDC
alpha= [0: 0.1: 1];
QDC_c1_reg = [];
QDC_c2_reg = [];
QDC_c1_end= [];
QDC_c2_end = [];

for a = 1: length(alpha)
    QDC_Reg_C1 = alpha(a) * ssaneado_c1 + (1 - alpha(a)) * ssaneado;
    QDC_Reg_C2 = alpha(a) * ssaneado_c2 + (1 - alpha(a)) * ssaneado;
        
    for i = 1:testlen
        X = test_set(i, :);
        QDC_c1_reg(i) = -0.5 * X * inv(QDC_Reg_C1) * X' + M_C1 * inv(QDC_Reg_C1) * X'  - 0.5 * M_C1 * inv(QDC_Reg_C1) * M_C1'  - 0.5 * log(det(QDC_Reg_C1)) + log(class_priori_1);
        QDC_c2_reg(i) = -0.5 * X * inv(QDC_Reg_C2) * X' + M_C2 * inv(QDC_Reg_C2) * X'  - 0.5 * M_C2 * inv(QDC_Reg_C2) * M_C2'  - 0.5 * log(det(QDC_Reg_C2)) + log(class_priori_2);
    end
    
    QDC_c1_end(a, :) = QDC_c1_reg;
    QDC_c2_end(a, :) = QDC_c2_reg;
end

lbl_reg_QDC = [];
reg_QDC = [];

%Clasificación de test regularizado para QDC
for v = 1:length(alpha)
    for m=1:length(QDC_c1_reg)
    
        if QDC_c1_end(v, m)>QDC_c2_end(v, m)
            reg_QDC(m)=0;
        else
            reg_QDC(m)=1;
        end
    end
    lbl_reg_QDC(v, :) = reg_QDC;
end

%Tasa de reconocimiento QDC
aciertosQDC = [];
lblQDCBinary = label_test';
lblQDCBinary(lblQDCBinary == 1) = 0;
lblQDCBinary(lblQDCBinary == 2) = 1;
tasa_QDC = [];

for b = 1:length(alpha) 
    for a = 1:length(lbl_reg_QDC)
        if lbl_reg_QDC(b, a) == lblQDCBinary(1, a)
            aciertosQDC(a)=1;
        end
    end
    tasa_QDC(b) = (sum(aciertosQDC)/length(lbl_reg_QDC)) * 100;
    aciertosQDC = [];
end

trans_alpha = alpha';

%% GRÁFICAS %%
subplot(2, 1, 1);
plot(trans_alpha(:, 1), tasa_QDC, 'r -o');hold on;
xlabel('Valor de Alpha');
ylabel('% de reconocimiento');
title('Tasa de Reconocimiento QDC');


%LDC
gama = [0: 0.1: 1];
LDC_c1_reg = [];
LDC_c2_reg = [];
LDC_c1_end = [];
LDC_c2_end = [];


for b = 1: length(gama)
    LDC_Reg = gama(b) * ssaneado + (1 - gama(b)) * (std(training_set).^2);
    
    for i = 1:testlen
        X = test_set(i, :);
        LDC_c1_reg(i) = M_C1 * inv(LDC_Reg) * X' -0.5 * M_C1 * inv(LDC_Reg) * M_C1' + log(class_priori_1);
        LDC_c2_reg(i) = M_C2 * inv(LDC_Reg) * X' -0.5 * M_C2 * inv(LDC_Reg) * M_C2' + log(class_priori_2);
    end
    LDC_c1_end(b, :) = LDC_c1_reg;
    LDC_c2_end(b, :) = LDC_c2_reg;
end

lbl_reg_LDC = [];
reg_LDC = [];

%Clasificación de test regularizado para LDC
for v = 1:length(gama)
    for m=1:length(LDC_c1_reg)
    
        if LDC_c1_end(v, m)>LDC_c2_end(v, m)
            reg_LDC(m)=0;
        else
            reg_LDC(m)=1;
        end
    end
    lbl_reg_LDC(v, :) = reg_LDC;
end

%Tasa de reconocimiento LDC
aciertosLDC = [];
lblLDCBinary = label_test';
lblLDCBinary(lblLDCBinary == 1) = 0;
lblLDCBinary(lblLDCBinary == 2) = 1;
tasa_LDC = [];

for b = 1:length(gama) 
    for a = 1:length(lbl_reg_LDC)
        if lbl_reg_LDC(b, a) == lblLDCBinary(1, a)
            aciertosLDC(a)=1;
        end
    end
    tasa_LDC(b) = (sum(aciertosLDC)/length(lbl_reg_LDC)) * 100;
    aciertosLDC = [];
end

trans_gamma = gama';

subplot(2, 1, 2);
plot(trans_gamma(:, 1), tasa_LDC, 'b -o');hold on;
xlabel('Valor de Gamma');
ylabel('% de reconocimiento');
title('Tasa de Reconocimiento LDC');
hold off

disp(("La tasa de reconocimiento sin regularizar de QDC es de: " + TstTasaQDC));
disp(("La tasa de reconocimiento sin regularizar de LDC es de: " + TstTasaLDC));

%Nota: Cuando alpha es 0 y gama es 1, los valores de QDC y LDC deben ser iguales
