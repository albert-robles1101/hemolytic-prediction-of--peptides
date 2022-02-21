#Alberto Robles
#09 de febrero 2022
#Review: A new era of computational tools for prediction of hemolytic 2 
#        activity of therapeutic peptides 

setwd("C:/Users/alber/Respaldo/Universidad/Ikiam/9no semestre/Review peptidos hemolíticos/Bases de datos/Analisis")

library(Biostrings)
library(dplyr)
library(tidyr)
library(seqinr)
library(Peptides)
library(xlsx)
library(stringr)

leer_fasta <-function(x){
  dataset <- read.fasta(x, as.string = T)
  seq_name = names(dataset)
  sequence = 0
  for (i in 1:length(dataset)) {
    sequence <- rbind(sequence, dataset[[i]][1])
  }
  sequence = sequence[-1,]
  dataset <- data.frame(seq_name, data.frame(sequence))
  dataset$sequence <- str_to_upper(dataset$sequence)
  return(dataset)
}

eliminar_duplicados <-function(x){
  dataset <- x[!duplicated(x$sequence), ]
  return(dataset)
}

positivo_negativo <- function(x, y){
  value <- factor(rep(y, times = length(x$sequence)))
  dataset <- cbind(x, value)
  return(dataset)
}

aminoacids_7_35 <-function(database){
  database <- filter(database, Length > 6 & Length < 36)
  return(database)
}

HAPPENN_dataset <- leer_fasta("HAPPENN_dataset.fasta")
HAPPENN_dataset <- eliminar_duplicados(HAPPENN_dataset)
HAPPENN_dataset <- cbind(HAPPENN_dataset, Length = lengthpep(HAPPENN_dataset$sequence))
HAPPENN_dataset <- aminoacids_7_35(HAPPENN_dataset)

hemopi1_train_pos <- leer_fasta("hemopi1_train_pos.fasta")
hemopi1_train_pos <- eliminar_duplicados(hemopi1_train_pos)
hemopi1_train_pos <- positivo_negativo(hemopi1_train_pos, 1)
hemopi1_train_neg <- leer_fasta("hemopi1_train_neg.fasta")
hemopi1_train_neg <- eliminar_duplicados(hemopi1_train_neg)
hemopi1_train_neg <- positivo_negativo(hemopi1_train_neg, 0)
hemopi1_test_pos <- leer_fasta("hemopi1_test_pos.fasta")
hemopi1_test_pos <- eliminar_duplicados(hemopi1_test_pos)
hemopi1_test_pos <- positivo_negativo(hemopi1_test_pos, 1)
hemopi1_test_neg <- leer_fasta("hemopi1_test_neg.fasta")
hemopi1_test_neg <- eliminar_duplicados(hemopi1_test_neg)
hemopi1_test_neg <- positivo_negativo(hemopi1_test_neg, 0)

hemopi2_train_pos <- leer_fasta("hemopi2_train_pos.fasta")
hemopi2_train_pos <- eliminar_duplicados(hemopi2_train_pos)
hemopi2_train_pos <- positivo_negativo(hemopi2_train_pos, 1)
hemopi2_train_neg <- leer_fasta("hemopi2_train_neg.fasta")
hemopi2_train_neg <- eliminar_duplicados(hemopi2_train_neg)
hemopi2_train_neg <- positivo_negativo(hemopi2_train_neg, 0)
hemopi2_test_pos <- leer_fasta("hemopi2_test_pos.fasta")
hemopi2_test_pos <- eliminar_duplicados(hemopi2_test_pos)
hemopi2_test_pos <- positivo_negativo(hemopi2_test_pos, 1)
hemopi2_test_neg <- leer_fasta("hemopi2_test_neg.fasta")
hemopi2_test_neg <- eliminar_duplicados(hemopi2_test_neg)
hemopi2_test_neg <- positivo_negativo(hemopi2_test_neg, 0)

hemopi3_train_pos <- leer_fasta("hemopi3_train_pos.fasta")
hemopi3_train_pos <- eliminar_duplicados(hemopi3_train_pos)
hemopi3_train_pos <- positivo_negativo(hemopi3_train_pos, 1)
hemopi3_train_neg <- leer_fasta("hemopi3_train_neg.fasta")
hemopi3_train_neg <- eliminar_duplicados(hemopi3_train_neg)
hemopi3_train_neg <- positivo_negativo(hemopi3_train_neg, 0)
hemopi3_test_pos <- leer_fasta("hemopi3_test_pos.fasta")
hemopi3_test_pos <- eliminar_duplicados(hemopi3_test_pos)
hemopi3_test_pos <- positivo_negativo(hemopi3_test_pos, 1)
hemopi3_test_neg <- leer_fasta("hemopi3_test_neg.fasta")
hemopi3_test_neg <- eliminar_duplicados(hemopi3_test_neg)
hemopi3_test_neg <- positivo_negativo(hemopi3_test_neg, 0)

#HemoPi Data train
HemoPi1_train <- rbind(hemopi1_train_pos, hemopi1_train_neg)
HemoPi1_train <- cbind(HemoPi1_train, Length = lengthpep(HemoPi1_train$sequence))
HemoPi1_train <- aminoacids_7_35(HemoPi1_train)

HemoPi2_train <- rbind(hemopi2_train_pos, hemopi2_train_neg)
HemoPi2_train <- cbind(HemoPi2_train, Length = lengthpep(HemoPi2_train$sequence))
HemoPi2_train <- aminoacids_7_35(HemoPi2_train)

HemoPi3_train <- rbind(hemopi3_train_pos, hemopi3_train_neg)
HemoPi3_train <- cbind(HemoPi3_train, Length = lengthpep(HemoPi3_train$sequence))
HemoPi2_train <- aminoacids_7_35(HemoPi2_train)

#HemoPi Data test
HemoPi1_test <- rbind(hemopi1_test_pos, hemopi1_test_neg)
HemoPi1_test <- cbind(HemoPi1_test, Length = lengthpep(HemoPi1_test$sequence))
HemoPi1_test <- aminoacids_7_35(HemoPi1_test)

HemoPi2_test <- rbind(hemopi2_test_pos, hemopi2_test_neg)
HemoPi2_test <- cbind(HemoPi2_test, Length = lengthpep(HemoPi2_test$sequence))
HemoPi2_test <- aminoacids_7_35(HemoPi2_test)

HemoPi3_test <- rbind(hemopi3_test_pos, hemopi3_test_neg)
HemoPi3_test <- cbind(HemoPi3_test, Length = lengthpep(HemoPi3_test$sequence))
HemoPi3_test <- aminoacids_7_35(HemoPi3_test)

#Guardando archivos
write.xlsx(HAPPENN_dataset, 
           file = "C:/Users/alber/Documents/HAPPENN_7_35_dataset.xlsx")
write.fasta(sequences = as.list(HAPPENN_dataset$sequence),names = HAPPENN_dataset$seq_name, 
            file.out="C:/Users/alber/Respaldo/Universidad/Ikiam/9no semestre/Review peptidos hemolíticos/Bases de datos/Analisis/HAPPENN_7_35_dataset.fasta")

write.xlsx(HemoPi1_train, 
           file = "C:/Users/alber/Documents/HemoPi1_train_7_35_dataset.xlsx")
write.fasta(sequences = as.list(HemoPi1_train$sequence),names = HemoPi1_train$seq_name, 
            file.out="C:/Users/alber/Respaldo/Universidad/Ikiam/9no semestre/Review peptidos hemolíticos/Bases de datos/Analisis/HemoPi1_train_7_35_dataset.fasta")

write.xlsx(HemoPi2_train, 
           file = "C:/Users/alber/Documents/HemoPi2_train_7_35_dataset.xlsx")
write.fasta(sequences = as.list(HemoPi2_train$sequence),names = HemoPi2_train$seq_name, 
            file.out="C:/Users/alber/Respaldo/Universidad/Ikiam/9no semestre/Review peptidos hemolíticos/Bases de datos/Analisis/HemoPi2_train_7_35_dataset.fasta")

write.xlsx(HemoPi3_train, 
           file = "C:/Users/alber/Documents/HemoPi3_train_7_35_dataset.xlsx")
write.fasta(sequences = as.list(HemoPi3_train$sequence),names = HemoPi3_train$seq_name, 
            file.out="C:/Users/alber/Respaldo/Universidad/Ikiam/9no semestre/Review peptidos hemolíticos/Bases de datos/Analisis/HemoPi3_train_7_35_dataset.fasta")

write.xlsx(HemoPi1_test, 
           file = "C:/Users/alber/Documents/HemoPi1_test_7_35_dataset.xlsx")
write.fasta(sequences = as.list(HemoPi1_test$sequence),names = HemoPi1_test$seq_name, 
            file.out="C:/Users/alber/Respaldo/Universidad/Ikiam/9no semestre/Review peptidos hemolíticos/Bases de datos/Analisis/HemoPi1_test_7_35_dataset.fasta")

write.xlsx(HemoPi2_test, 
           file = "C:/Users/alber/Documents/HemoPi2_test_7_35_dataset.xlsx")
write.fasta(sequences = as.list(HemoPi2_test$sequence),names = HemoPi2_test$seq_name, 
            file.out="C:/Users/alber/Respaldo/Universidad/Ikiam/9no semestre/Review peptidos hemolíticos/Bases de datos/Analisis/HemoPi2_test_7_35_dataset.fasta")

write.xlsx(HemoPi3_test, 
           file = "C:/Users/alber/Documents/HemoPi3_test_7_35_dataset.xlsx")
write.fasta(sequences = as.list(HemoPi3_test$sequence),names = HemoPi3_test$seq_name, 
            file.out="C:/Users/alber/Respaldo/Universidad/Ikiam/9no semestre/Review peptidos hemolíticos/Bases de datos/Analisis/HemoPi3_test_7_35_dataset.fasta")