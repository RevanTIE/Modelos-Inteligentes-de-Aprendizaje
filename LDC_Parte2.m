clc
clear
trnClases=[];

%% TRAINING %%
As = imageDatastore('circulos_cuadrados/training');
matrixA=[];
matrixv = [];
v=[];
    for i = 1:size(As.Files,1)
        img = readimage(As,i);
        img = rgb2gray(img); %%Conversión de la imagen RGB a escala de grises.
        img= im2double(img);
        s = size(img);
        S = s(1)*s(2);
        str=strjoin(As.Files(i));
        fileTrn(i, :)=strsplit(str,'\'); %Para MacOs este signo debe ser '/'
        V = reshape(img,1,S);
        matrixA=[matrixA;V];
    end

v=zeros(size(matrixA,1),1);
v(v==0)=1;
nomFilesTrn = [fileTrn(:, end)];
firstLetter = extractBefore(nomFilesTrn, 2);
betaClases = strrep(firstLetter,'c','0');
trnClases = strrep(betaClases,'o','1');
clases = str2double(trnClases);

%% TEST %%
test = imageDatastore('circulos_cuadrados/test');
for i = 1:size(test.Files,1)
    img_test=readimage(test,i);
    img_test = rgb2gray(img_test); %%Conversión de la imagen RGB a escala de grises.
    img_test= im2double(img_test);
    s = size(img_test);
    S = s(1)*s(2);
    str=strjoin(test.Files(i));
    fileTst(i, :)=strsplit(str,'\'); %Para MacOs este signo debe ser '/'
    V = reshape(img_test,1,S);
    matrixv=[matrixv;V];
end

nomFilesTst = [fileTst(:, end)];
trnX = pca(matrixA);

%Training set y Test set respectivamente
prodTrn = matrixA * trnX;
prodTst = matrixv * trnX;

%labels trn values and indexes

label_tr_c1 = clases(clases==0);
lbl_tr_c1_indx = find(clases==0);

label_tr_c2 = clases(clases==1);
lbl_tr_c2_indx = find(clases==1);

%Clasificación a priori
class_priori_1 = length(label_tr_c1)/length(clases);
class_priori_2 = length(label_tr_c2)/length(clases);

%Training data para c1 y c2
tr_data_c1 = prodTrn(lbl_tr_c1_indx, :);
tr_data_c2 = prodTrn(lbl_tr_c2_indx, :);

%matriz de covarianza de todo el dataset de training
sigma = cov(prodTrn);

%% Se sanean los valores negativos, y se reconstruye la matriz. %%
ssaneado = sanear(sigma);

%% Se calcula LDC por dataset_c1 y dataset_c2 %%
LDC_c1 = [];
LDC_c2 = [];
testlen = length(prodTst(:, 1));
M_C1 = mean(tr_data_c1);
M_C2 = mean(tr_data_c2);

for i = 1:1:testlen
    X = prodTst(i, :);
    LDC_c1(i) = M_C1 * inv(ssaneado) * X' -0.5 * M_C1 * inv(ssaneado) * M_C1' + log(class_priori_1);
    LDC_c2(i) = M_C2 * inv(ssaneado) * X' -0.5 * M_C2 * inv(ssaneado) * M_C2' + log(class_priori_2);
end

%Clasificación de test
lbl_test_LDC = [];
for m=1:length(LDC_c1)
    
    if LDC_c1(m)>LDC_c2(m)
        lbl_test_LDC(m)=0;
    else
        lbl_test_LDC(m)=1;
    end
end

%Tasa de reconocimiento
test_real = [1 0 1 0];
aciertos = [];
for a = 1:length(lbl_test_LDC)
    if lbl_test_LDC(a) == test_real(a)
        aciertos(a)=1;
    end
end

tasa_rec = (length(aciertos)/length(lbl_test_LDC)) * 100;
disp("La tasa de reconocimiento es de " + tasa_rec + "%");

rec_circulo = [];
for n= 1: length(lbl_test_LDC)
    if lbl_test_LDC(n) == 0
        rec_circulo = "Rectángulo";
    else
        rec_circulo = "Óvalo";
    end
    disp(nomFilesTst(n) + " es un " + rec_circulo);
end


%% GRÁFICAS %%
subplot(2, 1, 1);
plot(lbl_test_LDC, 'r-o');hold on;
title('Valores reconocidos');
subplot(2, 1, 2);

plot(test_real', 'b-o');hold on;
title('Valores reales');
hold off