function data =  transfer()
    fileFolder = fullfile("I:\1�ѷ���\CrackForest-dataset\groundTruth\");
    dirOutput = dir(fullfile(fileFolder,"*.mat"));
    for i =  1:size(dirOutput)
        str = sprintf("%03d",i);
        data = load("../"+str+".mat");
        data = data.groundTruth.Boundaries;
        break;
         xlswrite("../xls_Boundaries/"+ str +".xlsx",data);
    end
    