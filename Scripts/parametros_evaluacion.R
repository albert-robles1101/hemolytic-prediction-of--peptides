library(readxl)
library(pROC)
library(mltools)
library(dplyr)
library(caret)

setwd("C:/Users/alber/Documents/resultados_finales/predicciones")
HAPPENN_7_35_dataset <- read_excel("HAPPENN_7_35_dataset.xlsx")
HemoPi1_train_7_35_dataset <- read_excel("HemoPi1_train_7_35_dataset.xlsx")
HemoPi2_train_7_35_dataset <- read_excel("HemoPi2_train_7_35_dataset.xlsx")
HemoPi3_train_7_35_dataset <- read_excel("HemoPi3_train_7_35_dataset.xlsx")
HemoPi1_test_7_35_dataset <- read_excel("HemoPi1_test_7_35_dataset.xlsx")
HemoPi2_test_7_35_dataset <- read_excel("HemoPi2_test_7_35_dataset.xlsx")
HemoPi3_test_7_35_dataset <- read_excel("HemoPi3_test_7_35_dataset.xlsx")

#Plisson Models: LDA
confusionMatrix(data = as.factor(HAPPENN_7_35_dataset$LDA_Plisson), 
                reference = as.factor(HAPPENN_7_35_dataset$value))
mcc(pred = as.factor(HAPPENN_7_35_dataset$LDA_Plisson), actual = as.factor(HAPPENN_7_35_dataset$value))

confusionMatrix(data = as.factor(HemoPi1_train_7_35_dataset$LDA_Plisson), 
                reference = as.factor(HemoPi1_train_7_35_dataset$value))
mcc(pred = as.factor(HemoPi1_train_7_35_dataset$LDA_Plisson), actual = as.factor(HemoPi1_train_7_35_dataset$value))

confusionMatrix(data = as.factor(HemoPi2_train_7_35_dataset$LDA_Plisson), 
                reference = as.factor(HemoPi2_train_7_35_dataset$value))
mcc(pred = as.factor(HemoPi2_train_7_35_dataset$LDA_Plisson), actual = as.factor(HemoPi2_train_7_35_dataset$value))

confusionMatrix(data = as.factor(HemoPi3_train_7_35_dataset$LDA_Plisson), 
                reference = as.factor(HemoPi3_train_7_35_dataset$value))
mcc(pred = as.factor(HemoPi3_train_7_35_dataset$LDA_Plisson), actual = as.factor(HemoPi3_train_7_35_dataset$value))

confusionMatrix(data = as.factor(HemoPi1_test_7_35_dataset$LDA_Plisson), 
                reference = as.factor(HemoPi1_test_7_35_dataset$value))
mcc(pred = as.factor(HemoPi1_test_7_35_dataset$LDA_Plisson), actual = as.factor(HemoPi1_test_7_35_dataset$value))

confusionMatrix(data = as.factor(HemoPi2_test_7_35_dataset$LDA_Plisson), 
                reference = as.factor(HemoPi2_test_7_35_dataset$value))
mcc(pred = as.factor(HemoPi2_test_7_35_dataset$LDA_Plisson), actual = as.factor(HemoPi2_test_7_35_dataset$value))

confusionMatrix(data = as.factor(HemoPi3_test_7_35_dataset$LDA_Plisson), 
                reference = as.factor(HemoPi3_test_7_35_dataset$value))
mcc(pred = as.factor(HemoPi3_test_7_35_dataset$LDA_Plisson), actual = as.factor(HemoPi3_test_7_35_dataset$value))

#Plisson Models: GBC_26
confusionMatrix(data = as.factor(HAPPENN_7_35_dataset$GBC_26_Plisson), 
                reference = as.factor(HAPPENN_7_35_dataset$value))
mcc(pred = as.factor(HAPPENN_7_35_dataset$GBC_26_Plisson), actual = as.factor(HAPPENN_7_35_dataset$value))

confusionMatrix(data = as.factor(HemoPi1_train_7_35_dataset$GBC_26_Plisson), 
                reference = as.factor(HemoPi1_train_7_35_dataset$value))
mcc(pred = as.factor(HemoPi1_train_7_35_dataset$GBC_26_Plisson), actual = as.factor(HemoPi1_train_7_35_dataset$value))

confusionMatrix(data = as.factor(HemoPi2_train_7_35_dataset$GBC_26_Plisson), 
                reference = as.factor(HemoPi2_train_7_35_dataset$value))
mcc(pred = as.factor(HemoPi2_train_7_35_dataset$GBC_26_Plisson), actual = as.factor(HemoPi2_train_7_35_dataset$value))

confusionMatrix(data = as.factor(HemoPi3_train_7_35_dataset$GBC_26_Plisson), 
                reference = as.factor(HemoPi3_train_7_35_dataset$value))
mcc(pred = as.factor(HemoPi3_train_7_35_dataset$GBC_26_Plisson), actual = as.factor(HemoPi3_train_7_35_dataset$value))

confusionMatrix(data = as.factor(HemoPi1_test_7_35_dataset$GBC_26_Plisson), 
                reference = as.factor(HemoPi1_test_7_35_dataset$value))
mcc(pred = as.factor(HemoPi1_test_7_35_dataset$GBC_26_Plisson), actual = as.factor(HemoPi1_test_7_35_dataset$value))

confusionMatrix(data = as.factor(HemoPi2_test_7_35_dataset$GBC_26_Plisson), 
                reference = as.factor(HemoPi2_test_7_35_dataset$value))
mcc(pred = as.factor(HemoPi2_test_7_35_dataset$GBC_26_Plisson), actual = as.factor(HemoPi2_test_7_35_dataset$value))

confusionMatrix(data = as.factor(HemoPi3_test_7_35_dataset$GBC_26_Plisson), 
                reference = as.factor(HemoPi3_test_7_35_dataset$value))
mcc(pred = as.factor(HemoPi3_test_7_35_dataset$GBC_26_Plisson), actual = as.factor(HemoPi3_test_7_35_dataset$value))


#Plisson Models: GBC_56
confusionMatrix(data = as.factor(HAPPENN_7_35_dataset$GBC_56_Plisson), 
                reference = as.factor(HAPPENN_7_35_dataset$value))
mcc(pred = as.factor(HAPPENN_7_35_dataset$GBC_56_Plisson), actual = as.factor(HAPPENN_7_35_dataset$value))

confusionMatrix(data = as.factor(HemoPi1_train_7_35_dataset$GBC_56_Plisson), 
                reference = as.factor(HemoPi1_train_7_35_dataset$value))
mcc(pred = as.factor(HemoPi1_train_7_35_dataset$GBC_56_Plisson), actual = as.factor(HemoPi1_train_7_35_dataset$value))

confusionMatrix(data = as.factor(HemoPi2_train_7_35_dataset$GBC_56_Plisson), 
                reference = as.factor(HemoPi2_train_7_35_dataset$value))
mcc(pred = as.factor(HemoPi2_train_7_35_dataset$GBC_56_Plisson), actual = as.factor(HemoPi2_train_7_35_dataset$value))

confusionMatrix(data = as.factor(HemoPi3_train_7_35_dataset$GBC_56_Plisson), 
                reference = as.factor(HemoPi3_train_7_35_dataset$value))
mcc(pred = as.factor(HemoPi3_train_7_35_dataset$GBC_56_Plisson), actual = as.factor(HemoPi3_train_7_35_dataset$value))

confusionMatrix(data = as.factor(HemoPi1_test_7_35_dataset$GBC_56_Plisson), 
                reference = as.factor(HemoPi1_test_7_35_dataset$value))
mcc(pred = as.factor(HemoPi1_test_7_35_dataset$GBC_56_Plisson), actual = as.factor(HemoPi1_test_7_35_dataset$value))

confusionMatrix(data = as.factor(HemoPi2_test_7_35_dataset$GBC_56_Plisson), 
                reference = as.factor(HemoPi2_test_7_35_dataset$value))
mcc(pred = as.factor(HemoPi2_test_7_35_dataset$GBC_56_Plisson), actual = as.factor(HemoPi2_test_7_35_dataset$value))

confusionMatrix(data = as.factor(HemoPi3_test_7_35_dataset$GBC_56_Plisson), 
                reference = as.factor(HemoPi3_test_7_35_dataset$value))
mcc(pred = as.factor(HemoPi3_test_7_35_dataset$GBC_56_Plisson), actual = as.factor(HemoPi3_test_7_35_dataset$value))


#HAPPENN_class
confusionMatrix(data = as.factor(HAPPENN_7_35_dataset$HAPPENN_class), 
                reference = as.factor(HAPPENN_7_35_dataset$value))
mcc(pred = as.factor(HAPPENN_7_35_dataset$HAPPENN_class), actual = as.factor(HAPPENN_7_35_dataset$value))

confusionMatrix(data = as.factor(HemoPi1_train_7_35_dataset$HAPPENN_class), 
                reference = as.factor(HemoPi1_train_7_35_dataset$value))
mcc(pred = as.factor(HemoPi1_train_7_35_dataset$HAPPENN_class), actual = as.factor(HemoPi1_train_7_35_dataset$value))

confusionMatrix(data = as.factor(HemoPi2_train_7_35_dataset$HAPPENN_class), 
                reference = as.factor(HemoPi2_train_7_35_dataset$value))
mcc(pred = as.factor(HemoPi2_train_7_35_dataset$HAPPENN_class), actual = as.factor(HemoPi2_train_7_35_dataset$value))

confusionMatrix(data = as.factor(HemoPi3_train_7_35_dataset$HAPPENN_class), 
                reference = as.factor(HemoPi3_train_7_35_dataset$value))
mcc(pred = as.factor(HemoPi3_train_7_35_dataset$HAPPENN_class), actual = as.factor(HemoPi3_train_7_35_dataset$value))

confusionMatrix(data = as.factor(HemoPi1_test_7_35_dataset$HAPPENN_class), 
                reference = as.factor(HemoPi1_test_7_35_dataset$value))
mcc(pred = as.factor(HemoPi1_test_7_35_dataset$HAPPENN_class), actual = as.factor(HemoPi1_test_7_35_dataset$value))

confusionMatrix(data = as.factor(HemoPi2_test_7_35_dataset$HAPPENN_class), 
                reference = as.factor(HemoPi2_test_7_35_dataset$value))
mcc(pred = as.factor(HemoPi2_test_7_35_dataset$HAPPENN_class), actual = as.factor(HemoPi2_test_7_35_dataset$value))

confusionMatrix(data = as.factor(HemoPi3_test_7_35_dataset$HAPPENN_class), 
                reference = as.factor(HemoPi3_test_7_35_dataset$value))
mcc(pred = as.factor(HemoPi3_test_7_35_dataset$HAPPENN_class), actual = as.factor(HemoPi3_test_7_35_dataset$value))


#HemoPi
confusionMatrix(data = as.factor(HAPPENN_7_35_dataset$HemoPi), 
                reference = as.factor(HAPPENN_7_35_dataset$value))
mcc(pred = as.factor(HAPPENN_7_35_dataset$HemoPi), actual = as.factor(HAPPENN_7_35_dataset$value))

confusionMatrix(data = as.factor(HemoPi1_train_7_35_dataset$HemoPi), 
                reference = as.factor(HemoPi1_train_7_35_dataset$value))
mcc(pred = as.factor(HemoPi1_train_7_35_dataset$HemoPi), actual = as.factor(HemoPi1_train_7_35_dataset$value))

confusionMatrix(data = as.factor(HemoPi2_train_7_35_dataset$HemoPi), 
                reference = as.factor(HemoPi2_train_7_35_dataset$value))
mcc(pred = as.factor(HemoPi2_train_7_35_dataset$HemoPi), actual = as.factor(HemoPi2_train_7_35_dataset$value))

confusionMatrix(data = as.factor(HemoPi3_train_7_35_dataset$HemoPi), 
                reference = as.factor(HemoPi3_train_7_35_dataset$value))
mcc(pred = as.factor(HemoPi3_train_7_35_dataset$HemoPi), actual = as.factor(HemoPi3_train_7_35_dataset$value))

confusionMatrix(data = as.factor(HemoPi1_test_7_35_dataset$HemoPi), 
                reference = as.factor(HemoPi1_test_7_35_dataset$value))
mcc(pred = as.factor(HemoPi1_test_7_35_dataset$HemoPi), actual = as.factor(HemoPi1_test_7_35_dataset$value))

confusionMatrix(data = as.factor(HemoPi2_test_7_35_dataset$HemoPi), 
                reference = as.factor(HemoPi2_test_7_35_dataset$value))
mcc(pred = as.factor(HemoPi2_test_7_35_dataset$HemoPi), actual = as.factor(HemoPi2_test_7_35_dataset$value))

confusionMatrix(data = as.factor(HemoPi3_test_7_35_dataset$HemoPi), 
                reference = as.factor(HemoPi3_test_7_35_dataset$value))
mcc(pred = as.factor(HemoPi3_test_7_35_dataset$HemoPi), actual = as.factor(HemoPi3_test_7_35_dataset$value))


#HemoPred
confusionMatrix(data = as.factor(HAPPENN_7_35_dataset$HemoPred), 
                reference = as.factor(HAPPENN_7_35_dataset$value))
mcc(pred = as.factor(HAPPENN_7_35_dataset$HemoPred), actual = as.factor(HAPPENN_7_35_dataset$value))

confusionMatrix(data = as.factor(HemoPi1_train_7_35_dataset$HemoPred), 
                reference = as.factor(HemoPi1_train_7_35_dataset$value))
mcc(pred = as.factor(HemoPi1_train_7_35_dataset$HemoPred), actual = as.factor(HemoPi1_train_7_35_dataset$value))

confusionMatrix(data = as.factor(HemoPi2_train_7_35_dataset$HemoPred), 
                reference = as.factor(HemoPi2_train_7_35_dataset$value))
mcc(pred = as.factor(HemoPi2_train_7_35_dataset$HemoPred), actual = as.factor(HemoPi2_train_7_35_dataset$value))

confusionMatrix(data = as.factor(HemoPi3_train_7_35_dataset$HemoPred), 
                reference = as.factor(HemoPi3_train_7_35_dataset$value))
mcc(pred = as.factor(HemoPi3_train_7_35_dataset$HemoPred), actual = as.factor(HemoPi3_train_7_35_dataset$value))

confusionMatrix(data = as.factor(HemoPi1_test_7_35_dataset$HemoPred), 
                reference = as.factor(HemoPi1_test_7_35_dataset$value))
mcc(pred = as.factor(HemoPi1_test_7_35_dataset$HemoPred), actual = as.factor(HemoPi1_test_7_35_dataset$value))

confusionMatrix(data = as.factor(HemoPi2_test_7_35_dataset$HemoPred), 
                reference = as.factor(HemoPi2_test_7_35_dataset$value))
mcc(pred = as.factor(HemoPi2_test_7_35_dataset$HemoPred), actual = as.factor(HemoPi2_test_7_35_dataset$value))

confusionMatrix(data = as.factor(HemoPi3_test_7_35_dataset$HemoPred), 
                reference = as.factor(HemoPi3_test_7_35_dataset$value))
mcc(pred = as.factor(HemoPi3_test_7_35_dataset$HemoPred), actual = as.factor(HemoPi3_test_7_35_dataset$value))

#HLPpredFuse
confusionMatrix(data = as.factor(HAPPENN_7_35_dataset$HLPpredFuse), 
                reference = as.factor(HAPPENN_7_35_dataset$value))
mcc(pred = as.factor(HAPPENN_7_35_dataset$HLPpredFuse), actual = as.factor(HAPPENN_7_35_dataset$value))

confusionMatrix(data = as.factor(HemoPi1_train_7_35_dataset$HLPpredFuse), 
                reference = as.factor(HemoPi1_train_7_35_dataset$value))
mcc(pred = as.factor(HemoPi1_train_7_35_dataset$HLPpredFuse), actual = as.factor(HemoPi1_train_7_35_dataset$value))

confusionMatrix(data = as.factor(HemoPi2_train_7_35_dataset$HLPpredFuse), 
                reference = as.factor(HemoPi2_train_7_35_dataset$value))
mcc(pred = as.factor(HemoPi2_train_7_35_dataset$HLPpredFuse), actual = as.factor(HemoPi2_train_7_35_dataset$value))

confusionMatrix(data = as.factor(HemoPi3_train_7_35_dataset$HLPpredFuse), 
                reference = as.factor(HemoPi3_train_7_35_dataset$value))
mcc(pred = as.factor(HemoPi3_train_7_35_dataset$HLPpredFuse), actual = as.factor(HemoPi3_train_7_35_dataset$value))

confusionMatrix(data = as.factor(HemoPi1_test_7_35_dataset$HLPpredFuse), 
                reference = as.factor(HemoPi1_test_7_35_dataset$value))
mcc(pred = as.factor(HemoPi1_test_7_35_dataset$HLPpredFuse), actual = as.factor(HemoPi1_test_7_35_dataset$value))

confusionMatrix(data = as.factor(HemoPi2_test_7_35_dataset$HLPpredFuse), 
                reference = as.factor(HemoPi2_test_7_35_dataset$value))
mcc(pred = as.factor(HemoPi2_test_7_35_dataset$HLPpredFuse), actual = as.factor(HemoPi2_test_7_35_dataset$value))

confusionMatrix(data = as.factor(HemoPi3_test_7_35_dataset$HLPpredFuse), 
                reference = as.factor(HemoPi3_test_7_35_dataset$value))
mcc(pred = as.factor(HemoPi3_test_7_35_dataset$HLPpredFuse), actual = as.factor(HemoPi3_test_7_35_dataset$value))

